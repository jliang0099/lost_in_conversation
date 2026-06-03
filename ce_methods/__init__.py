from .base import CEMethod
from .few_shot import FewShotCE
from .cot import CoTCE
from .self_refine import SelfRefineCE
from .context_compression import ContextCompressionCE

_REGISTRY = {
    "fewshot":             FewShotCE,
    "cot":                 CoTCE,
    "self_refine":         SelfRefineCE,
    "context_compression": ContextCompressionCE,
}

CHOICES = ["none"] + list(_REGISTRY)


def build_ce_method(name: str, **kwargs) -> "CEMethod | None":
    """
    Factory for CE methods.  Returns None for name='none'.

    Extra kwargs are forwarded to the method's constructor, so callers can do:
        build_ce_method("fewshot", log_path="...", k=3)
    """
    if name in ("none", "no") or name is None:
        return None
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CE method {name!r}. Choices: {CHOICES}")
    return _REGISTRY[name](**kwargs)
