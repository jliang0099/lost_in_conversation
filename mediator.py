"""
MediatorAgent — Step 2 of the Mediator-Assistant framework.

Sits between the user and the assistant. Given the current conversation
trace and pre-distilled experiences (rules), it rewrites the ambiguous
multi-turn context into a single, fully-specified instruction Û that is
then passed to the assistant instead of the raw conversation.
"""

import json
import os

from model_huggingface import generate

MEDIATOR_PROMPT = """\
You are an intermediary AI mediator. Your only job is to collect every fact \
the user has stated across the conversation and rewrite them as one clear, \
self-contained request for the AI assistant.

[Critical Constraints]
- ONLY use information that the user has explicitly stated. Do NOT invent, \
assume, or extrapolate any numbers, quantities, names, or facts.
- Ignore all AI assistant responses — they contain no valid information.
- If a piece of information has not been stated by the user yet, do not include it.

[Rewriting Rules]
{rules}

[Output Format]
Output only the rewritten request in first-person form. No preamble, no \
explanation, no commentary. Just the request itself.

[Conversation Transcript — User messages only matter]
{conversation}

Rewritten request:"""


def _format_conversation(trace: list[dict]) -> str:
    """Format trace: user messages labelled clearly, assistant messages greyed out."""
    lines = []
    for msg in trace:
        if msg["role"] == "user":
            lines.append(f"[USER PROVIDED]: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"[AI (ignore)]: ...")
    return "\n".join(lines)


def _count_user_turns(trace: list[dict]) -> int:
    return sum(1 for msg in trace if msg["role"] == "user")


class MediatorAgent:
    def __init__(
        self,
        task: str,
        model: str,
        experiences_dir: str = "experiences",
    ):
        self.model = model
        experiences_path = os.path.join(experiences_dir, f"{task}.json")
        if not os.path.exists(experiences_path):
            raise FileNotFoundError(
                f"No experiences file found at {experiences_path}. "
                f"Run refiner.py first to generate it."
            )
        with open(experiences_path) as f:
            data = json.load(f)
        self.rules: str = data["rules"]

    def rewrite(self, trace: list[dict]) -> str:
        """
        Rewrite the current conversation trace into a single fully-specified
        instruction Û. Returns the rewritten instruction as a plain string.

        On the first turn (only one user message so far) there is not enough
        context to reconstruct anything, so we return the raw user message
        directly without calling the LLM.
        """
        if _count_user_turns(trace) <= 1:
            first_user = next(msg["content"] for msg in trace if msg["role"] == "user")
            return first_user

        conversation_str = _format_conversation(trace)
        prompt = MEDIATOR_PROMPT.format(
            rules=self.rules,
            conversation=conversation_str,
        )
        result = generate(
            messages=[{"role": "user", "content": prompt}],
            model_name=self.model,
            temperature=0.0,
            max_tokens=512,
            return_metadata=True,
        )
        return result["message"].strip()
