class CEMethod:
    """
    Base class for Context Engineering methods.

    Subclasses override one or more hooks:
      • augment_system_message      — prompt-level intervention at conversation start (FewShot, CoT)
      • transform_generation_messages — per-turn message-list transformation before generation
                                        (ContextCompression)
      • post_process_response       — generation-level intervention after each turn (SelfRefinement)
    """

    name: str = "base"

    def augment_system_message(self, system_message: str, sample: dict) -> str:
        """Called once per conversation to modify the system message before any turn."""
        return system_message

    def get_demo_messages(self, sample: dict) -> list:
        """Return {role, content} dicts to prepend (after system) into generation_messages."""
        return []

    def transform_generation_messages(
        self, messages: list, n_demo_messages: int = 0
    ) -> list:
        """
        Called each turn, after demo injection and before generate().

        Args:
            messages:         Full generation_messages list (system + demos + real conv).
            n_demo_messages:  Number of demo messages injected after the system message.
                              CE methods that modify history should skip these.

        Returns the (possibly modified) messages list.
        """
        return messages

    def post_process_response(
        self,
        assistant_response: str,
        generation_messages: list,
        generate_fn,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Called after each assistant generation.  Return a (possibly refined)
        response string.  The default is a no-op pass-through.

        Args:
            assistant_response:  the raw assistant output just generated.
            generation_messages: the messages list that was fed to generate_fn
                                 for this turn (role/content dicts, no log entries).
            generate_fn:         the project's generate() callable.
            model_name / temperature / max_tokens: forwarded from the simulator.
        """
        return assistant_response

    @property
    def last_turn_ce_tokens(self) -> dict:
        """Token usage from CE overhead in the last turn (0 if no LLM call was made)."""
        return getattr(self, "_last_ce_tokens", {})
