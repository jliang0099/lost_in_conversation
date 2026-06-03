"""
LUBandit — the top-level orchestrator.

Wraps LinUCBBandit + state extractor + reward computer + discriminator
into a single object that the simulator calls per turn.

Actions: ["neutral", "explore", "commit", "challenge"]
  neutral   — no injection; let the model decide naturally
  explore   — hint to gather more information before answering
  commit    — hint to provide a definitive answer now
  challenge — hint to reconsider current reasoning

Simulator call pattern
──────────────────────
bandit = LUBandit(...)

for each turn t:
    behavior, gen_messages, meta = bandit.decide(
        tracker, inertia_checker, messages, n_total_turns, task_type
    )
    # ... generate assistant response ...
    kappa_t = bandit.current_kappa(tracker, inertia_checker)
    bandit.observe(messages_after_response, generate_fn, kappa_t)
    # ... system verification → response_type, extracted_answer ...
    bandit.note_verification(response_type, extracted_answer)

bandit.end_episode(final_task_score, kappa_final)
bandit.save_if_configured()
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from lu_bandit.bandit import LinUCBBandit, DEFAULT_ACTIONS
from lu_bandit.state import extract_state, STATE_DIM
from lu_bandit.reward import RewardComputer
from lu_bandit.discriminator import VLLMDiscriminator

if TYPE_CHECKING:
    from activation_tracker import ActivationTracker
    from inertia_checker import InertiaChecker

# Text injected at the end of the last user message per behavior mode
BEHAVIOR_INJECTIONS: dict[str, str] = {
    "neutral":   "",
    "explore":   "\n\n[Hint: Take time to analyze all available information before committing to a final answer.]",
    "commit":    "\n\n[Hint: Based on the information gathered so far, provide your definitive best answer now.]",
    "challenge": "\n\n[Hint: Reconsider your previous reasoning carefully. If your current approach may be incorrect, try a different angle.]",
}

# Alignment bonus: reward when the model's response type matched the hint
_ALIGNMENT_TABLE: dict[str, dict[str, float]] = {
    "explore":   {"discussion": 0.1, "clarification": 0.1, "interrogation": 0.05},
    "commit":    {"answer_attempt": 0.1},
    "challenge": {"answer_attempt": 0.05},
    "neutral":   {},
}
_MISALIGNMENT_PENALTY: float = -0.05


class LUBandit:
    """
    Contextual bandit that adaptively selects per-turn behavioral hints.

    Parameters
    ----------
    bandit          : LinUCBBandit instance (or None → auto-create)
    discriminator   : VLLMDiscriminator for intermediate quality scoring
    reward_computer : RewardComputer controlling λ_drift, λ_final
    checkpoint_path : if set, bandit weights are saved here after each episode
    epsilon         : ε-greedy exploration override (0 = pure UCB)
    """

    def __init__(
        self,
        bandit: Optional[LinUCBBandit] = None,
        discriminator: Optional[VLLMDiscriminator] = None,
        reward_computer: Optional[RewardComputer] = None,
        checkpoint_path: Optional[str] = None,
        epsilon: float = 0.0,
        actions: Optional[list[str]] = None,
        alpha: float = 1.0,
    ):
        self.actions: list[str] = actions if actions is not None else DEFAULT_ACTIONS
        self.bandit: LinUCBBandit = bandit or LinUCBBandit(
            d=STATE_DIM, alpha=alpha, actions=self.actions
        )
        self.discriminator: VLLMDiscriminator = discriminator or VLLMDiscriminator(
            model_name="", mode="none"
        )
        self.reward_computer: RewardComputer = reward_computer or RewardComputer()
        self.checkpoint_path = checkpoint_path
        self.epsilon = epsilon

        # Per-episode state
        self._response_type_history: list[str] = []
        self._answer_history: list[str] = []
        self._last_action_idx: Optional[int] = None
        self._last_state: Optional[np.ndarray] = None
        self._last_behavior: Optional[str] = None
        self._episode_log: list[dict] = []

    # ── per-turn API ──────────────────────────────────────────────────────────

    def decide(
        self,
        tracker: Optional["ActivationTracker"],
        inertia_checker: Optional["InertiaChecker"],
        messages: list[dict],
        n_total_turns: int,
        task_type: str,
    ) -> tuple[str, list[dict], dict]:
        """
        Select behavioral mode, inject hint into messages, return modified messages.

        Returns
        -------
        (behavior, generation_messages, metadata_dict)
        """
        state = extract_state(
            tracker=tracker,
            inertia_checker=inertia_checker,
            messages=messages,
            n_total_turns=n_total_turns,
            response_type_history=self._response_type_history,
            answer_history=self._answer_history,
            task_type=task_type,
        )

        action_idx, ucb_score = self.bandit.select_action(state, epsilon=self.epsilon)
        behavior = self.actions[action_idx]

        injected_messages = _inject_hint(messages, behavior)

        self._last_action_idx = action_idx
        self._last_state = state
        self._last_behavior = behavior

        meta = {
            "action_idx": action_idx,
            "behavior":   behavior,
            "ucb_score":  ucb_score,
            "state":      state.tolist(),
        }
        self._episode_log.append({"turn": len(self._response_type_history) + 1, **meta})
        return behavior, injected_messages, meta

    def observe(
        self,
        messages_after: list[dict],
        generate_fn: Callable,
        kappa_after: Optional[float],
    ) -> dict:
        """
        Called after the assistant responds.

        1. Flush the deferred reward for the previous action (drift now known).
        2. Score quality and stage the current action's immediate reward.
        """
        if self.reward_computer.has_pending():
            pending = self.reward_computer.flush_deferred(kappa_curr=kappa_after)
            if pending is not None:
                self.bandit.update(pending.action_idx, pending.state, pending.total_reward)
                if len(self._episode_log) >= 2:
                    self._episode_log[-2]["bandit_reward"] = pending.total_reward

        quality = self.discriminator.score(messages_after, generate_fn)

        if self._last_action_idx is None or self._last_state is None:
            return {"quality": quality}

        self.reward_computer.stage(
            action_idx=self._last_action_idx,
            state=self._last_state,
            quality=quality,
            behavior=self._last_behavior or "neutral",
            kappa=kappa_after,
        )
        if self._episode_log:
            self._episode_log[-1]["quality"] = quality

        return {"quality": quality, "kappa": kappa_after}

    def note_verification(
        self,
        response_type: str,
        extracted_answer: Optional[str] = None,
    ) -> None:
        """
        Called after system verification for the current turn.

        Records the model's response type, updates answer history, and
        adds the alignment bonus to the currently staged reward.
        """
        self._response_type_history.append(response_type)
        if response_type == "answer_attempt" and extracted_answer:
            self._answer_history.append(extracted_answer)

        if self._last_behavior is not None:
            bonus = _alignment_bonus(self._last_behavior, response_type)
            self.reward_computer.add_alignment_bonus(bonus)

    def end_episode(
        self,
        final_task_score: Optional[float],
        kappa_final: Optional[float] = None,
    ) -> None:
        """
        Flush the last action's reward and reset per-episode state.
        """
        bonus = self.reward_computer.final_bonus_reward(final_task_score)
        pending = self.reward_computer.flush_deferred(
            kappa_curr=None,
            extra_bonus=bonus,
        )
        if pending is not None:
            self.bandit.update(pending.action_idx, pending.state, pending.total_reward)

        self._response_type_history = []
        self._answer_history = []
        self._last_action_idx = None
        self._last_state = None
        self._last_behavior = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def current_kappa(
        self,
        tracker: Optional["ActivationTracker"],
        inertia_checker: Optional["InertiaChecker"],
    ) -> Optional[float]:
        if tracker is None or inertia_checker is None:
            return None
        turn_states = inertia_checker._turn_states(tracker)
        kappa = inertia_checker.compute_curvature(turn_states)
        return float(kappa) if kappa is not None else None

    def episode_log(self) -> list[dict]:
        return list(self._episode_log)

    def save_if_configured(self) -> None:
        if self.checkpoint_path:
            self.bandit.save(self.checkpoint_path)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        alpha: float = 1.0,
        epsilon: float = 0.0,
        lambda_drift: float = 0.5,
        final_bonus: float = 1.0,
        discriminator_model: str = "",
        discriminator_mode: str = "none",
        checkpoint_path: Optional[str] = None,
    ) -> "LUBandit":
        discriminator = VLLMDiscriminator(
            model_name=discriminator_model,
            mode=discriminator_mode,
        )
        reward_computer = RewardComputer(
            lambda_drift=lambda_drift,
            final_bonus=final_bonus,
        )
        bandit = LinUCBBandit.load_or_new(
            path=checkpoint_path,
            d=STATE_DIM,
            alpha=alpha,
            actions=DEFAULT_ACTIONS,
        )
        return cls(
            bandit=bandit,
            discriminator=discriminator,
            reward_computer=reward_computer,
            checkpoint_path=checkpoint_path,
            epsilon=epsilon,
            actions=DEFAULT_ACTIONS,
            alpha=alpha,
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _inject_hint(messages: list[dict], behavior: str) -> list[dict]:
    """Append the behavior hint sentence to the last user message."""
    hint = BEHAVIOR_INJECTIONS.get(behavior, "")
    if not hint:
        return messages

    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return messages

    modified = list(messages)
    msg = dict(modified[last_user_idx])
    content = msg.get("content", "")
    if isinstance(content, str):
        msg["content"] = content + hint
    modified[last_user_idx] = msg
    return modified


def _alignment_bonus(behavior: str, response_type: str) -> float:
    """Return the alignment bonus for a (behavior, response_type) pair."""
    bonuses = _ALIGNMENT_TABLE.get(behavior, {})
    if response_type in bonuses:
        return bonuses[response_type]
    if behavior != "neutral":
        return _MISALIGNMENT_PENALTY
    return 0.0
