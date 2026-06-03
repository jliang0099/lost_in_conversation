"""
ContextCompressionCE — per-turn context compression via LLM summarisation.

Follows the LangChain ConversationSummaryBufferMemory pattern:
  • Keep the most recent `keep_last_n_turns` user/assistant pairs verbatim.
  • Summarise everything older with an LLM call that explicitly preserves
    all numbers and mathematical relationships.
  • Progressive update: if the system message already contains a summary from
    a previous turn, only the new history delta is merged in — cheaper than
    re-summarising from scratch each turn.

Distinct from FewShot / CoT / SelfRefine:
  • FewShot / CoT operate on the system message once at conversation start.
  • SelfRefine operates after each generation.
  • ContextCompressionCE transforms generation_messages each turn, BEFORE
    the model sees them — it trades verbatim history for a dense LLM summary.

Demo-message safety:
  When FewShot demos are also active the caller passes `n_demo_messages` so
  those injected turns are never included in the compressed history.
"""

from .base import CEMethod


class ContextCompressionCE(CEMethod):

    name = "context_compression"

    def __init__(
        self,
        model_name: str,
        keep_last_n_turns: int = 2,
        min_history_turns: int = 2,
        summary_max_tokens: int = 256,
    ):
        from model_huggingface import generate
        from context_compressor import ContextCompressor

        self.compressor = ContextCompressor(
            generate_fn=generate,
            model_name=model_name,
            keep_last_n_turns=keep_last_n_turns,
            min_history_turns=min_history_turns,
            summary_max_tokens=summary_max_tokens,
        )

    def transform_generation_messages(
        self, messages: list, n_demo_messages: int = 0
    ) -> list:
        if n_demo_messages == 0:
            compressed, meta = self.compressor.compress(messages)
            self._last_ce_tokens = meta.get("ce_tokens", {})
            return compressed

        # Separate system, demo, and real conversation messages so that
        # demos are never folded into the compressed history.
        sys_msgs  = [m for m in messages if m["role"] == "system"]
        non_sys   = [m for m in messages if m["role"] != "system"]
        demo_msgs = non_sys[:n_demo_messages]
        real_msgs = non_sys[n_demo_messages:]

        compressed_temp, meta = self.compressor.compress(sys_msgs + real_msgs)
        self._last_ce_tokens = meta.get("ce_tokens", {})
        if not meta.get("compressed"):
            return messages

        compressed_sys  = [m for m in compressed_temp if m["role"] == "system"]
        compressed_real = [m for m in compressed_temp if m["role"] != "system"]
        return compressed_sys + demo_msgs + compressed_real
