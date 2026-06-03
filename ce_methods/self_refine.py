"""
SelfRefineCE — iterative self-critique and revision for sharded math problems.

Reference: Madaan et al. (2023) "Self-Refine: Iterative Refinement with
Self-Feedback" — adapted to the multi-turn sharded setting.

Distinct from FewShotCE and CoTCE:
  • FewShot / CoT operate on the SYSTEM MESSAGE before the conversation starts.
  • SelfRefine operates AFTER each assistant generation via post_process_response.
    It leaves the system message completely untouched.

Mechanism:
  After the assistant produces an initial response, SelfRefineCE appends a
  critique instruction and calls generate() once more.  The refined response
  replaces the original in the conversation trace.

The critique prompt handles both turn types:
  • Accumulation turns  — confirm the facts noted so far; keep it brief.
  • Answer turns        — verify each arithmetic step; correct any error;
                          restate the final answer as a single number.

Cost: one extra generate() call per assistant turn.
"""

from .base import CEMethod

_CRITIQUE_PROMPT = (
    "Review your previous response carefully.\n"
    "- If you were still waiting for information: confirm the key facts "
    "you have noted so far. Keep it brief.\n"
    "- If you gave a numeric answer: verify every arithmetic step. "
    "Correct any error and restate the final answer as a single number.\n\n"
    "Provide your revised response (or repeat it unchanged if already correct):"
)


class SelfRefineCE(CEMethod):

    name = "self_refine"

    # augment_system_message is intentionally NOT overridden —
    # SelfRefine is a generation-level intervention, not a prompt-level one.

    def post_process_response(
        self,
        assistant_response: str,
        generation_messages: list,
        generate_fn,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        critique_messages = generation_messages + [
            {"role": "assistant", "content": assistant_response},
            {"role": "user",      "content": _CRITIQUE_PROMPT},
        ]
        result = generate_fn(
            messages=critique_messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._last_ce_tokens = {
            "prompt_tokens":     result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
        }
        return result["message"]
