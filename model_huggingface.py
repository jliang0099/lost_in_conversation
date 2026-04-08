import os
import re
import json
import time
from typing import Optional
import torch
from activation_tracker import ActivationTracker
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# vLLM routing table
#   key   : model name (must match what callers pass as model_name)
#   value : (host, port) of the corresponding vLLM server
#
# Layout assumed:
#   GPU 0-1  →  20B model  →  vLLM on port 5002
#   GPU 2    →  8B  model  →  vLLM on port 5001
#   GPU 3    →  hidden-state extraction (HF, loaded below)
# ---------------------------------------------------------------------------
_VLLM_REGISTRY: dict[str, tuple[str, int]] = {
    "meta-llama/Llama-3.1-8B-Instruct":  ("127.0.0.1", 5001),
    "microsoft/phi-4":         ("127.0.0.1", 5002),
    # Add more models here as needed, e.g.:
    # "another/model": ("127.0.0.1", 5003),
}

# GPU reserved exclusively for hidden-state extraction
_ACTIVATION_GPU = "cuda:2"

# Model to load on _ACTIVATION_GPU for hidden-state extraction.
# Must be the same weights the activation_tracker expects (8B by default).
_ACTIVATION_MODEL = DEFAULT_MODEL


def format_messages(messages: list[dict], variables: dict = {}) -> list[dict]:
    """Replace [[KEY]] placeholders in the last user message."""
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)
    
    # 只匹配 [[UPPER_CASE]] 或 [[snake_case]] 风格的 key
    keys_still_in_prompt = re.findall(r"\[\[([A-Za-z_][A-Za-z0-9_]*)\]\]", last_user_msg["content"])
    # keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if keys_still_in_prompt:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages


# ---------------------------------------------------------------------------
# HF model cache — only used for hidden-state extraction, pinned to GPU 3
# ---------------------------------------------------------------------------
_activation_model_cache: dict[str, tuple] = {}


def _get_activation_model(model_name: str = _ACTIVATION_MODEL):
    """
    Load (and cache) the HF model used for hidden-state extraction.
    Pinned to _ACTIVATION_GPU (cuda:3) — never touches the vLLM GPUs.
    """
    if model_name not in _activation_model_cache:
        print(f"[HF-activation] Loading '{model_name}' onto {_ACTIVATION_GPU} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": _ACTIVATION_GPU},   # pin every layer to GPU 3
        )
        model.eval()
        _activation_model_cache[model_name] = (model, tokenizer)
        print(f"[HF-activation] '{model_name}' ready on {_ACTIVATION_GPU}.")
    return _activation_model_cache[model_name]


# ---------------------------------------------------------------------------
# vLLM client — one instance per model, cached
# ---------------------------------------------------------------------------
_vllm_cache: dict[str, "vLLM"] = {}


class vLLM:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ):
        self.model = model_name
        self.url = f"http://{host}:{port}/v1/completions"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def inference(self, messages: list[dict]) -> dict:
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        payload = {
            "model": self.model,
            "prompt": rendered_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": False,
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        return {
            "message": data["choices"][0]["text"],
            "finish_reason": data["choices"][0].get("finish_reason"),
            "total_usd": 0.0,
        }


def _get_vllm_client(
    model_name: str,
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> vLLM:
    """
    Return a cached vLLM client for the requested model.
    Looks up (host, port) from _VLLM_REGISTRY.
    Raises KeyError with a helpful message if the model isn't registered.
    """
    if model_name not in _vllm_cache:
        if model_name not in _VLLM_REGISTRY:
            registered = list(_VLLM_REGISTRY.keys())
            raise KeyError(
                f"[vLLM] No server registered for model '{model_name}'. "
                f"Add it to _VLLM_REGISTRY. Currently registered: {registered}"
            )
        host, port = _VLLM_REGISTRY[model_name]
        # print(f"[vLLM] Initialising client for '{model_name}' → {host}:{port}")
        _vllm_cache[model_name] = vLLM(
            host=host,
            port=port,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return _vllm_cache[model_name]


def _apply_chat_template(messages: list[dict], tokenizer) -> str:
    """Use tokenizer chat template if available, else fall back to ChatML."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Model class
# ---------------------------------------------------------------------------

class Model:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Hidden-state extraction (always runs on GPU 3 via HF)
    # ------------------------------------------------------------------
    def record_activations(
        self,
        messages: list[dict],
        model_name: str,           # which model weights to use for extraction
        is_first_turn: bool,
        activation_tracker: ActivationTracker,
    ) -> None:
        # Always use the dedicated activation model on GPU 3.
        # If you want the 20B hidden states you can pass model_name=_ACTIVATION_MODEL
        # and load the 20B weights on GPU 3 (needs ~40 GB — won't fit on one V100).
        # For now we default to the 8B model.
        model, tokenizer = _get_activation_model(model_name)

        prompt = _apply_chat_template(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            prefill_out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = list(prefill_out.hidden_states)  # list[layer] of (1, seq_len, hidden)

        if is_first_turn:
            activation_tracker.set_goal(hs)
        else:
            activation_tracker.record_activation(hs)

        elapsed = time.time() - t0
        # print(f"[record_activations] elapsed: {elapsed:.2f}s on {_ACTIVATION_GPU}")

    # ------------------------------------------------------------------
    # vLLM generation — routes to the correct server by model_name
    # ------------------------------------------------------------------
    def generate_vllm(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        client = _get_vllm_client(model_name, max_tokens=max_tokens, temperature=temperature)
        return client.inference(messages)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        variables: dict = {},
        is_first_turn: Optional[bool] = None,
        activation_tracker=None,
        return_metadata: bool = False,
    ) -> str:
        messages = format_messages(list(messages), variables)
        max_tokens = max_tokens or 1000

        last_exc: Optional[Exception] = None
        # print(f"\n[generate] model={model_name} | messages={messages}")
        t0 = time.time()

        for attempt in range(max_retries):
            try:
                if activation_tracker is not None:
                    self.record_activations(
                        messages,
                        _ACTIVATION_MODEL,   # always extract with the 8B on GPU 3
                        is_first_turn,
                        activation_tracker,
                    )
                    # print(f"[generate] activation_tracker updated in {time.time() - t0:.2f}s")

                response = self.generate_vllm(messages, model_name, temperature, max_tokens)
                # print(f"[generate] vLLM completed in {time.time() - t0:.2f}s")
                return response

            except Exception as exc:
                last_exc = exc
                print(f"[generate] attempt {attempt + 1}/{max_retries} failed: {exc}")

        raise RuntimeError(
            f"generate() failed after {max_retries} attempts"
        ) from last_exc

    def generate_json(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> dict:
        """Like generate() but parses the response as JSON."""
        kwargs["return_metadata"] = True
        result = self.generate(messages, model_name=model, **kwargs)

        raw = result["message"]
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"\s*```$", "", clean)

        result["message"] = json.loads(clean)
        return result


model = Model()
generate = model.generate
generate_json = model.generate_json