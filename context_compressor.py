"""
ContextCompressor — LangChain SummaryBufferMemory pattern.

Mechanism:
  • Keep the most recent `keep_last_n_turns` user/assistant pairs verbatim.
  • Extract facts from older history using an LLM call.
  • Always preserve the original question (first user message) separately in
    the system message — it is the goal and must never be summarised away.
  • The fact list + original question are injected into the system message
    so the chat format stays valid for the vLLM backend.

Key design choices vs. naive summarisation:
  • Extract facts only from [user] messages — assistant messages contain
    reasoning that may be wrong or incomplete, and should not pollute the
    fact store.
  • State the original question explicitly — without it the model loses track
    of what it is trying to solve.
  • No progressive-update path: generation_messages are reconstructed from
    the raw trace each turn, so the system message never contains a previous
    summary block. Fresh extraction each turn is the correct behaviour.
"""

_QUESTION_HEADER  = "\n\nOriginal question: "
_FACTS_START      = "\n\n[Known facts from earlier conversation]\n"
_FACTS_END        = "\n[End of known facts]"

_FACT_EXTRACTION_SYSTEM = """\
Extract every concrete fact stated by the [user] in the conversation below.
Output as a numbered list. Each item must be a complete sentence that includes the subject and what it refers to — never output a bare number.
Rules:
  1. Write each fact as a complete, self-contained statement (e.g. "Each middle shelf holds 10 books." not "10").
  2. Copy every number and quantity EXACTLY as stated — do not compute or infer new values.
  3. Include only facts from [user] messages — ignore [assistant] messages entirely.
  4. Do NOT include questions, filler phrases, or statements with no number or relationship.
  5. If no concrete facts have been stated yet, output: [No facts stated yet]

Example input:
[user]: each of the two middle shelves can hold 10 books.
[user]: the bottom shelf holds twice as many books as one middle shelf.
Example output:
1. Each of the two middle shelves can hold 10 books.
2. The bottom shelf holds twice as many books as one middle shelf.\
"""


class ContextCompressor:
    """
    LangChain-style SummaryBufferMemory for vLLM-backed conversations.

    Args:
        generate_fn:        The project's generate() callable
                            (signature: messages, model_name, temperature, max_tokens → dict).
        model_name:         Model used for fact extraction (can be the same as
                            the assistant model or a cheaper one).
        keep_last_n_turns:  User/assistant pairs kept verbatim (uncompressed).
        min_history_turns:  Skip compression when history is shorter than this.
        summary_max_tokens: Token budget for the generated fact list.
    """

    def __init__(
        self,
        generate_fn,
        model_name: str,
        keep_last_n_turns: int = 2,
        min_history_turns: int = 2,
        summary_max_tokens: int = 256,
    ):
        self.generate_fn      = generate_fn
        self.model_name       = model_name
        self.keep_last_n_turns  = keep_last_n_turns
        self.min_history_turns  = min_history_turns
        self.summary_max_tokens = summary_max_tokens

    # ── public API ─────────────────────────────────────────────────────────────

    def compress(self, messages: list[dict]) -> tuple[list[dict], dict]:
        """
        Compress old history into the system message as a fact list.

        Returns:
            (new_messages, metadata)
            metadata["compressed"] is False when nothing changed.
        """
        sys_msgs  = [m for m in messages if m["role"] == "system"]
        conv_msgs = [m for m in messages if m["role"] != "system"]

        user_idxs = [i for i, m in enumerate(conv_msgs) if m["role"] == "user"]
        if len(user_idxs) <= self.keep_last_n_turns:
            return messages, {"compressed": False, "reason": "too_short"}

        split_at = user_idxs[-self.keep_last_n_turns]
        history  = conv_msgs[:split_at]   # turns to compress
        recent   = conv_msgs[split_at:]   # verbatim tail, always starts with user

        if len(history) < self.min_history_turns:
            return messages, {"compressed": False, "reason": "history_too_short"}

        # The first user message is the goal question — always keep it verbatim
        # in the system message so the model never loses track of what to solve.
        original_question = next(
            (m["content"] for m in conv_msgs if m["role"] == "user"), ""
        )

        # Extract facts only from [user] messages in history.
        # Skip the original question itself (it's a question, not a fact).
        history_for_facts = [
            m for m in history
            if m["role"] == "user" and m["content"] != original_question
        ]

        if history_for_facts:
            facts = self._extract_facts(history_for_facts)
            ce_tokens = self._last_usage
        else:
            facts = "[No facts stated yet]"
            ce_tokens = {}

        system_base = sys_msgs[0]["content"] if sys_msgs else ""
        new_system = (
            system_base
            + _QUESTION_HEADER + original_question
            + _FACTS_START + facts + _FACTS_END
        )

        new_messages = [{"role": "system", "content": new_system}] + recent

        metadata = {
            "compressed":     True,
            "method":         "llm_fact_extraction",
            "original_turns": len(history),
            "facts_chars":    len(facts),
            "ce_tokens":      ce_tokens,
        }
        print(
            f"[ContextCompressor] compressed {metadata['original_turns']} turns "
            f"→ fact list ({metadata['facts_chars']} chars)"
        )
        return new_messages, metadata

    # ── private ────────────────────────────────────────────────────────────────

    def _extract_facts(self, user_msgs: list[dict]) -> str:
        conversation_text = "\n".join(
            f"[user]: {m['content']}" for m in user_msgs
        )
        result = self.generate_fn(
            messages=[
                {"role": "system", "content": _FACT_EXTRACTION_SYSTEM},
                {"role": "user",   "content": conversation_text},
            ],
            model_name=self.model_name,
            temperature=0,
            max_tokens=self.summary_max_tokens,
        )
        self._last_usage = {
            "prompt_tokens":     result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
        }
        return result["message"].strip()
