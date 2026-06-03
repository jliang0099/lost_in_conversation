from pyexpat.errors import messages
import re
import json
from typing import Optional
import torch
from activation_tracker import ActivationTracker
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# vLLM routing table
#   key   : model name (must match what callers pass as model_name)
#   value : (host, port) of the corresponding vLLM server
# ---------------------------------------------------------------------------
_VLLM_REGISTRY: dict[str, tuple[str, int]] = {
    "meta-llama/Llama-3.1-8B-Instruct":  ("127.0.0.1", 5001),
    "Qwen/Qwen2.5-14B-Instruct":         ("127.0.0.1", 5002),
}


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
# Hidden-state extraction
# ---------------------------------------------------------------------------
_activation_model_cache: dict[str, tuple] = {}
def _get_activation_model(model_name):
    """
    Load (and cache) the HF model used for hidden-state extraction.
    """
    if model_name not in _activation_model_cache:
        # print(f"[HF-activation] Loading '{model_name}' onto {_ACTIVATION_GPU} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 权重主体放GPU1，输入输出层放GPU0（中间激活跟着层走）
        device_map = {
            "model.embed_tokens": 2,      # 输入embedding在GPU0
            "model.norm": 2,
            "lm_head": 2,
            # 前半层放GPU0（中间激活留在GPU0） 
            **{f"model.layers.{i}": 0 for i in range(0, 16)},
            # 后半层放GPU1
            **{f"model.layers.{i}": 1 for i in range(16, 32)},
            # **{f"model.layers.{i}": 3 for i in range(23, 32)},
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,  # 自动分配到GPU0、GPU1、GPU2，根据实际显存情况调整
        )
        model.eval()
        
        # 关键：长序列下关闭不需要的功能
        model.config.use_cache = False  # prefill-only不需要KV cache
        
        _activation_model_cache[model_name] = (model, tokenizer)
        # print(f"[HF-activation] '{model_name}' ready on {_ACTIVATION_GPU}.")
    return _activation_model_cache[model_name]

def _apply_chat_template(messages: list[dict], tokenizer) -> str:
    """Use tokenizer chat template if available, else fall back to ChatML."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


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
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
        usage = data.get("usage", {})
        return {
            "message": data["choices"][0]["text"],
            "finish_reason": data["choices"][0].get("finish_reason"),
            "total_usd": 0.0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
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


# ---------------------------------------------------------------------------
# Main Model class
# ---------------------------------------------------------------------------

class Model:
    def __init__(self):
        pass
    
    def record_activations(
        self,
        messages: list[dict],
        model_name: str,
        is_first_turn: bool,
        activation_tracker: ActivationTracker,
    ) -> None:
        
        model, tokenizer = _get_activation_model(model_name)

        prompt = _apply_chat_template(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            prefill_out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = list(prefill_out.hidden_states)  # list[layer] of (1, seq_len, hidden)
        
        # TODO: 检查这个hidden token是否包含了特殊的思考标记（如果有的话），如果有的话可能需要剥离掉再传给tracker，或者让tracker自己处理

        if is_first_turn:
            activation_tracker.set_goal(hs)
        else:
            activation_tracker.record_activation(hs)
            
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
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_retries: int = 3,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        variables: dict = {},
        is_first_turn: Optional[bool] = None,
        activation_tracker=None,
        return_metadata: bool = False,
    ) -> dict:
        
        # Format the prompt with any provided variables.
        messages = format_messages(list(messages), variables)
        max_tokens = max_tokens or 1000

        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # Original path: optional activation extraction, then vLLM
                if activation_tracker is not None:
                    self.record_activations(
                        messages,
                        model_name,
                        is_first_turn,
                        activation_tracker,
                    )

                response = self.generate_vllm(
                    messages, 
                    model_name, 
                    temperature, 
                    max_tokens
                )

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
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs,
    ) -> dict:
        """Like generate() but parses the response as JSON."""
        kwargs["return_metadata"] = True
        result = self.generate(messages, model_name=model, **kwargs)

        raw = result["message"]
        # print(f"[DEBUG] raw repr: {repr(raw)}")  # 加这行
        clean = re.sub(r"^```(?:json)?\s*", "", raw)
        clean = re.sub(r"\s*```$", "", clean)
        # clean = clean.replace("\\$", "$")
        # print(f"[DEBUG] clean repr: {repr(clean)}")  # 加这行

        result["message"] = json.loads(clean)
        return result


model = Model()
generate = model.generate
generate_json = model.generate_json