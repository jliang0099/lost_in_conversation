"""
model_hf.py — drop-in replacement for model_openai.py
======================================================
Keeps the exact same public API:
    generate(messages, model=..., ...)  -> str | dict
    generate_json(messages, ...)        -> dict

Two backends (set via HF_BACKEND env var or backend= kwarg):
    "local"  — transformers pipeline on GPU  (default, best for V100/L40)
    "api"    — HuggingFace Inference API     (no GPU needed, set HF_API_TOKEN)

Usage examples
--------------
# Local (auto-detects GPU):
    export HF_BACKEND=local
    python model_hf.py

# HF Inference API:
    export HF_BACKEND=api
    export HF_API_TOKEN=hf_xxx
    python model_hf.py
"""

import os
import re
import json
import time
from typing import Optional
import torch
from activation_tracker import ActivationTracker

# ── env config ────────────────────────────────────────────────────────────────
HF_BACKEND   = os.environ.get("HF_BACKEND",   "local")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# GOAL_ELICITING_SUFFIX = (
#     "Given the conversation so far, complete this sentence in one line:\n"
#     '"The question I am ultimately trying to answer is: "'
# )


# ── prompt formatting (identical to original) ─────────────────────────────────

def format_messages(messages: list[dict], variables: dict = {}) -> list[dict]:
    """Replace [[KEY]] placeholders in the last user message."""
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if keys_still_in_prompt:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages


# ── local backend helpers ─────────────────────────────────────────────────────

_model_cache: dict = {}

def _get_local_model(model_name: str):
    """Load (and cache) model + tokenizer directly — NOT via pipeline."""
    if model_name not in _model_cache:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"[HF-local] Loading '{model_name}' …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
        print(f"[HF-local] '{model_name}' ready.")
    return _model_cache[model_name]


def _apply_chat_template(messages: list[dict], tokenizer) -> str:
    """Use tokenizer chat template if available, else fall back to ChatML."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # ChatML fallback
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# def _sanitize_messages_local(messages: list[dict]) -> list[dict]:
#     """
#     Some open models don't support a leading system message.
#     Merge it into the first user message when that's the case.
#     (Optional — remove if your model handles system messages fine.)
#     """
#     if messages and messages[0]["role"] == "system" and len(messages) > 1 and messages[1]["role"] == "user":
#         sys_content = messages[0]["content"]
#         messages = list(messages)  # don't mutate original
#         messages[1] = {**messages[1], "content": f"[System]: {sys_content}\n{messages[1]['content']}"}
#         messages = messages[1:]
#     return messages


# def _build_goal_eliciting_messages(messages: list[dict]) -> list[dict]:
#     """Append a short goal-eliciting suffix to the final user input for prefill only."""
#     goal_messages = list(messages)

#     for i in range(len(goal_messages) - 1, -1, -1):
#         if goal_messages[i]["role"] == "user":
#             goal_messages[i] = {
#                 **goal_messages[i],
#                 "content": f"{goal_messages[i]['content']}\n\n{GOAL_ELICITING_SUFFIX}",
#             }
#             return goal_messages

#     goal_messages.append({"role": "user", "content": GOAL_ELICITING_SUFFIX})
#     return goal_messages


# ── HF Model class ────────────────────────────────────────────────────────────

class HF_Model:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or HF_BACKEND

    # ── internal generation ────────────────────────────────────────────────
    def _generate_local(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_tokens: int,
        is_first_turn: bool,
        is_last_turn: bool,
        activation_tracker=None
    ) -> dict:

        model, tokenizer = _get_local_model(model_name)
        # msgs = _sanitize_messages_local(messages)
        
        prompt = _apply_chat_template(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        
        # ── 第1步：prefill，采样"理解状态" ──────────────────────────────────
        # 只在需要 tracker 时才跑这一步，避免多余开销
        if activation_tracker is not None:
            # goal_msgs = _build_goal_eliciting_messages(msgs)
            # goal_prompt = _apply_chat_template(goal_msgs, tokenizer)
            # goal_inputs = tokenizer(goal_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                prefill_out = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # prefill_out.hidden_states: tuple[layer] of (batch, seq_len, hidden)
            # 取最后一个输入 token 的表征 → 代表"模型读完整个对话后的理解"
            hs = list(prefill_out.hidden_states)  # list[layer] of (1, seq_len, hidden)

            if is_first_turn:
                activation_tracker.set_goal(hs)
            else:
                activation_tracker.record_activation(hs)
                
        # ── 第2步：正常生成，不需要 hidden states ───────────────────────────
        # do_sample = temperature > 0.01
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # 固定低温，保持输出稳定；实际温度控制在 prefill 阶段
                # do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                # output_hidden_states 不再需要
            )
        
        elapsed = time.time() - t0

        # 只取新生成的 token（去掉 prompt 部分）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        approx_tokens = len(generated_ids)

        return {
            "message": text,
            "total_tokens": approx_tokens,
            "prompt_tokens": inputs["input_ids"].shape[1],
            "prompt_tokens_cached": 0,
            "completion_tokens": approx_tokens,
            "total_usd": 0.0,
            "elapsed_sec": elapsed,
        }

    # ── public interface (matches OpenAI_Model.generate) ──────────────────

    def generate(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        timeout: int = 120,             # kept for API compatibility, used as patience
        max_retries: int = 3,
        temperature: float = 1.0,
        is_json: bool = False,
        return_metadata: bool = False,
        max_tokens: Optional[int] = None,
        variables: dict = {},
        backend: Optional[str] = None,  # per-call override
        
        is_first_turn: Optional[bool] = None,  # custom kwarg for simulator
        is_last_turn: Optional[bool] = None,   # custom kwarg for simulator
        activation_tracker=None,  # custom kwarg for simulator
        
    ) -> str | dict:
        """
        Drop-in replacement for OpenAI_Model.generate().

        Returns:
            str   if return_metadata=False
            dict  if return_metadata=True  →  same keys as original
        """
        messages = format_messages(list(messages), variables)  # don't mutate caller's list
        
        effective_backend = backend or self.backend
        max_tokens = max_tokens or 1000

        for attempt in range(max_retries):
            try:
                if effective_backend == "api":
                    result = self._generate_api(messages, model, temperature, max_tokens, is_json)
                else:
                    result = self._generate_local(messages, model, temperature, max_tokens, is_first_turn, is_last_turn, activation_tracker)
                break
            except Exception as e:
                if attempt >= max_retries - 1:
                    raise RuntimeError(f"[HF] Failed after {max_retries} retries: {e}") from e
                wait = 4 * (attempt + 1)
                print(f"[HF] Error on attempt {attempt+1}: {e}. Retrying in {wait}s …")
                time.sleep(wait)

        if not return_metadata:
            return result["message"]
        return result

    def generate_json(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> dict:
        """
        Like generate() but parses the response as JSON.
        Strips markdown fences if the model wraps output in ```json … ```.
        """
        kwargs["return_metadata"] = True
        kwargs["is_json"] = True
        result = self.generate(messages, model, **kwargs)

        raw = result["message"]
        # Strip optional ```json … ``` fences
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"\s*```$", "", clean)

        result["message"] = json.loads(clean)
        return result


# ── module-level singletons — same names as in model_openai.py ───────────────

model = HF_Model()
generate      = model.generate
generate_json = model.generate_json


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--backend", default=None, choices=["local", "api"])
    args = parser.parse_args()

    if args.backend:
        os.environ["HF_BACKEND"] = args.backend

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Tell me a one-sentence joke about Lancaster University."},
    ]

    print(f"\n[test] backend={os.environ.get('HF_BACKEND','local')}  model={args.model}\n")

    # Plain response
    resp = generate(test_messages, model=args.model, return_metadata=False)
    print(f"[plain]    {resp}\n")

    # With metadata
    resp_meta = generate(test_messages, model=args.model, return_metadata=True)
    print(f"[metadata] {resp_meta}\n")

    # JSON mode
    json_messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in valid JSON."},
        {"role": "user",   "content": 'Return a JSON object with keys "city" and "country" for Lancaster.'},
    ]
    resp_json = generate_json(json_messages, model=args.model)
    print(f"[json]     {resp_json}\n")