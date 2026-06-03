"""
vLLM-based conversation quality discriminator.

Scores a conversation's current state on [0.0, 1.0].

Two modes
─────────
"prompt" (default)
    Makes a single vLLM call with a short scoring prompt.
    Returns (parsed_number / 10.0).  Falls back to 0.5 on parse failure.

"logprob"
    Uses the model's token log-probabilities to estimate quality as
    P(correct | compressed_ctx).  The reference token is set during
    the first call via `set_answer_token()`.
    Falls back to 0.5 if log-probs are unavailable.

"none"
    Always returns 0.5 (disables discriminator overhead; useful when
    only final task score matters).
"""

from __future__ import annotations

import re
from typing import Callable, Optional


_SCORE_PROMPT_TEMPLATE = """\
You are an impartial evaluator.  Below is a multi-turn problem-solving \
conversation.  Rate how well the assistant is progressing toward a correct \
final answer.

Scoring guide
  10 = clearly on the right track, meaningful progress every turn
   5 = mixed signals, some progress but also confusion or backtracking
   0 = going in circles, irrelevant, or clearly wrong

Conversation (last 8000 chars shown):
{conversation}

Reply with a single integer between 0 and 10, nothing else."""


class VLLMDiscriminator:
    """
    Quality scorer for in-progress conversations.

    Parameters
    ----------
    model_name  : HuggingFace / vLLM model ID used for scoring
    mode        : "prompt" | "logprob" | "none"
    max_ctx_chars : maximum conversation characters fed to the scorer
                   (truncates from the left to stay within context)
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "prompt",
        max_ctx_chars: int = 8_000,
    ):
        if mode not in ("prompt", "logprob", "none"):
            raise ValueError(f"Unknown discriminator mode: {mode!r}")
        self.model_name = model_name
        self.mode = mode
        self.max_ctx_chars = max_ctx_chars
        self._answer_token: Optional[str] = None  # used by logprob mode

    # ── public API ────────────────────────────────────────────────────────────

    def score(self, messages: list[dict], generate_fn: Callable) -> float:
        """
        Return a quality score ∈ [0.0, 1.0].

        Parameters
        ----------
        messages     : current conversation (roles: system / user / assistant)
        generate_fn  : the project's `generate()` callable
                       (imported from model_huggingface)
        """
        if self.mode == "none":
            return 0.5
        if self.mode == "prompt":
            return self._prompt_score(messages, generate_fn)
        if self.mode == "logprob":
            return self._logprob_score(messages, generate_fn)
        return 0.5

    def set_answer_token(self, token: str) -> None:
        """Set the reference answer token for logprob mode."""
        self._answer_token = token

    # ── internal ──────────────────────────────────────────────────────────────

    def _conversation_text(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            parts.append(f"{role.upper()}: {content}")
        full = "\n".join(parts)
        return full[-self.max_ctx_chars:]  # keep the most recent context

    def _prompt_score(self, messages: list[dict], generate_fn: Callable) -> float:
        conv_text = self._conversation_text(messages)
        scoring_messages = [
            {
                "role": "system",
                "content": "You are an impartial conversation quality evaluator.",
            },
            {
                "role": "user",
                "content": _SCORE_PROMPT_TEMPLATE.format(conversation=conv_text),
            },
        ]
        try:
            result = generate_fn(
                messages=scoring_messages,
                model_name=self.model_name,
                temperature=0,
                max_tokens=8,
                return_metadata=True,
            )
            text: str = (
                result.get("message", "5")
                if isinstance(result, dict)
                else str(result)
            )
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            if numbers:
                raw = float(numbers[0])
                return min(raw / 10.0, 1.0)
        except Exception as exc:
            print(f"[VLLMDiscriminator] scoring failed: {exc}")
        return 0.5

    def _logprob_score(self, messages: list[dict], generate_fn: Callable) -> float:
        """
        Estimate P(answer_token | context) from vLLM log-probs.

        Requires generate_fn to return a 'logprobs' key.
        Falls back to 0.5 when not available.
        """
        # Build a single-step prompt that ends just before the answer token
        if self._answer_token is None:
            return 0.5
        try:
            conv_text = self._conversation_text(messages)
            probe_messages = [
                {"role": "system", "content": "Continue the conversation naturally."},
                {"role": "user", "content": conv_text},
            ]
            result = generate_fn(
                messages=probe_messages,
                model_name=self.model_name,
                temperature=0,
                max_tokens=1,
                return_metadata=True,
            )
            logprobs = result.get("logprobs")
            if logprobs:
                # logprobs is expected to be a list of {token: logp} dicts
                first = logprobs[0] if logprobs else {}
                lp = first.get(self._answer_token, None)
                if lp is not None:
                    import math
                    return min(max(math.exp(lp), 0.0), 1.0)
        except Exception as exc:
            print(f"[VLLMDiscriminator] logprob scoring failed: {exc}")
        return 0.5
