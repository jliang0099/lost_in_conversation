"""
FewShotCE — inject k worked sharded-dialogue examples as real conversation turns.

Demo source:  a previously-logged .jsonl file whose task_ids do NOT overlap
              with the evaluation set (e.g. the (260-428) log while evaluating
              on 40-103).

Selection strategy: stratified by conversation length (# of user turns),
capped at max_demo_turns, so the k demos span different reasoning chain lengths.

Injection strategy ("real turns"):
  Demo conversations are inserted as actual user/assistant messages in the
  trace, before the real problem starts.  This avoids the format-leakage bug
  that occurs when Human:/Assistant: labels are embedded in a system prompt —
  small models tend to continue writing those labels in their own responses.

Demo style ("clean"):
  Intermediate assistant turns are replaced with a one-line acknowledgment
  that echoes the newly-revealed fact.  Only the final assistant turn keeps
  the full (truncated) step-by-step computation.
"""

import json
from .base import CEMethod


class FewShotCE(CEMethod):

    name = "fewshot"

    def __init__(
        self,
        log_path: str,
        k: int = 3,
        seed: int = 42,
        max_demo_turns: int = 4,
        max_final_chars: int = 1000,
    ):
        self.max_final_chars = max_final_chars
        self.demos = _load_demos(log_path, k, seed, max_demo_turns)

    def augment_system_message(self, system_message: str, sample: dict) -> str:
        instruction = (
            "A math problem will be revealed to you one fact at a time. "
            "For each intermediate turn: briefly acknowledge the new fact. "
            "On the final turn: compute step by step and state the numeric answer."
        )
        return f"{system_message}\n\n{instruction}"

    def get_demo_messages(self, sample: dict) -> list:
        messages = []
        for demo in self.demos:
            if demo.get("task_id") == sample.get("task_id"):
                continue
            messages.extend(_demo_to_messages(demo, self.max_final_chars))
        return messages


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_demos(log_path: str, k: int, seed: int, max_demo_turns: int) -> list:
    with open(log_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    correct = [e for e in entries if e.get("is_correct")]

    def _user_turns(e):
        return sum(1 for t in e["trace"] if t["role"] == "user")

    correct = [e for e in correct if _user_turns(e) <= max_demo_turns]

    if len(correct) < k:
        raise ValueError(
            f"FewShotCE: only {len(correct)} correct entries with ≤{max_demo_turns} "
            f"turns in {log_path!r}, need k={k}. Lower max_demo_turns or k."
        )

    correct.sort(key=_user_turns)
    if k == 1:
        indices = [len(correct) // 2]
    else:
        indices = [round(i * (len(correct) - 1) / (k - 1)) for i in range(k)]

    seen, demos = set(), []
    for idx in indices:
        entry = correct[idx]
        tid = entry.get("task_id")
        if tid not in seen:
            seen.add(tid)
            demos.append(entry)

    return demos


def _demo_to_messages(entry: dict, max_final_chars: int) -> list:
    """Convert a demo log entry into a flat list of {role, content} message dicts."""
    pairs: list[dict] = []
    for msg in entry["trace"]:
        role = msg["role"]
        if role in ("system", "log"):
            continue
        content = msg.get("content", "")
        if role == "user":
            pairs.append({"user": content, "assistant": None})
        elif role == "assistant" and pairs:
            pairs[-1]["assistant"] = content

    messages = []
    for i, pair in enumerate(pairs):
        is_last = (i == len(pairs) - 1)
        messages.append({"role": "user", "content": pair["user"]})

        if is_last:
            resp = _truncate_at_sentence(pair["assistant"] or "", max_final_chars)
        else:
            fact = pair["user"][:100].rstrip()
            if len(pair["user"]) > 100:
                fact += "..."
            resp = f'Noted — "{fact}". I\'ll incorporate this once I have all the information.'

        messages.append({"role": "assistant", "content": resp})

    return messages


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    import re
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = window.rfind(".\n")
    if cut != -1:
        return text[:cut + 1]
    matches = list(re.finditer(r'\. [A-Z]', window))
    if matches:
        return text[:matches[-1].start() + 1]
    return window.rsplit(" ", 1)[0] + "..."
