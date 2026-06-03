"""
CoTCE — Zero-shot Chain-of-Thought for sharded multi-turn math problems.

Distinct from FewShotCE:
  • FewShot gives worked EXAMPLES (what to produce);
    CoT gives a REASONING PROTOCOL (how to think) — no examples at all.
  • FewShot teaches via imitation; CoT teaches via explicit instruction.

The protocol has two phases that match the sharded turn structure:
  1. Accumulation phase (intermediate turns): track known facts explicitly.
  2. Solution phase (final turn): chain through the computation step by step.

Reference: Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models" — adapted here for incremental information disclosure.
"""

from .base import CEMethod

_PROTOCOL = """\
A math problem will be revealed to you one fact at a time. After each new fact, \
follow this decision rule:

Step 1 — Acknowledge: briefly note the new fact.
Step 2 — Self-check: ask yourself "Can I now fully and correctly answer the \
original question using only the facts I have been given?"
  • If NO: list the key facts collected so far. Do NOT guess or speculate.
  • If YES: solve the problem step by step —
      – Restate every known quantity.
      – Write each arithmetic step explicitly: operation, values, result.
      – State the final answer on the last line.\
"""


class CoTCE(CEMethod):

    name = "cot"

    def augment_system_message(self, system_message: str, sample: dict) -> str:
        return f"{system_message}\n\n{_PROTOCOL}"
