from typing import Optional


class ContextCompressor:
    """
    Compresses old conversation turns using LLMLingua-2 before the full
    generation pipeline (activation tracking, inertia check, vLLM).

    The compressed history is appended to the system message so the chat
    format (alternating user/assistant) stays valid for the vLLM backend.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "cpu",
        rate: float = 0.5,
        keep_last_n_turns: int = 1,
        min_history_turns: int = 1,
    ):
        """
        Args:
            model_name:         LLMLingua-2 scorer model.
            device_map:         Device for the scorer ("cpu" avoids GPU conflicts).
            rate:               Target compression rate (0.5 = keep 50% of tokens).
            keep_last_n_turns:  Number of recent user/assistant pairs kept verbatim.
            min_history_turns:  Skip compression when history is shorter than this.
        """
        from llmlingua import PromptCompressor

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )
        self.rate = rate
        self.keep_last_n_turns = keep_last_n_turns
        self.min_history_turns = min_history_turns

    def compress(self, messages: list[dict]) -> tuple[list[dict], dict]:
        """
        Compress old history into the system message.

        Returns:
            (compressed_messages, metadata)
            metadata["compressed"] is False when nothing was changed.
        """
        system_msgs = [m for m in messages if m["role"] == "system"]
        conv_msgs = [m for m in messages if m["role"] != "system"]

        # Split at user-turn boundaries so `recent` always starts with a user
        # message (required for valid chat format after the system message).
        # keep_last_n_turns=1 means: keep only the current (last) user turn.
        user_idxs = [i for i, m in enumerate(conv_msgs) if m["role"] == "user"]
        if len(user_idxs) <= self.keep_last_n_turns:
            return messages, {"compressed": False, "reason": "too_short"}

        split_at = user_idxs[-self.keep_last_n_turns]
        history = conv_msgs[:split_at]   # everything before the N-th-last user turn
        recent = conv_msgs[split_at:]    # always starts with a user message

        if len(history) < self.min_history_turns:
            return messages, {"compressed": False, "reason": "history_too_short"}

        # Use the original question (first user message) for question-aware
        # compression, not the latest shard. The latest shard is new input,
        # not the goal — using it would cause LLMLingua to deprioritize the
        # earlier shard tokens we actually want to keep.
        first_user = next(
            (m["content"] for m in conv_msgs if m["role"] == "user"), ""
        )

        context_segments = [f"[{m['role']}]: {m['content']}" for m in history]

        result = self.compressor.compress_prompt(
            context_segments,
            question=first_user,
            rate=self.rate,
            concate_question=False,
            force_tokens=["\n"],
            drop_consecutive=True,
            # Disable context-level filter: it scores each segment as a whole,
            # which breaks when a long assistant turn exceeds xlm-roberta's
            # 512-token limit. Token-level compression uses iterative_size=200
            # chunks and is not affected.
            use_context_level_filter=False,
        )

        compressed_text = result["compressed_prompt"]

        system_base = system_msgs[0]["content"] if system_msgs else ""
        new_system = (
            system_base
            + "\n\n[Earlier conversation - compressed]\n"
            + compressed_text
            + "\n[End of earlier conversation]"
        )

        new_messages = [{"role": "system", "content": new_system}] + recent

        metadata = {
            "compressed": True,
            "original_turns": len(history),
            "origin_tokens": result.get("origin_tokens", -1),
            "compressed_tokens": result.get("compressed_tokens", -1),
        }
        print(f"[ContextCompressor] compressed {metadata['original_turns']} turns "
              f"from {metadata['origin_tokens']} tokens to {metadata['compressed_tokens']} tokens (rate={self.rate})")
        # print(f"[ContextCompressor] messages: {messages}")
        # print(f"[ContextCompressor] new_messages: {new_messages}")
        return new_messages, metadata
