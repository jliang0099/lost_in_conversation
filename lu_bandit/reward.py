"""
Reward computation for LUBandit.

Reward decomposition
────────────────────
R_total(a_t) = R_quality(t)                              ← discriminator / task score
             + R_alignment(t)                            ← did model follow the hint? (immediate)
             − λ_drift × max(0, κ_t − κ_{t+1})          ← curvature-worsening penalty (deferred)

The drift penalty is deferred: computed at turn t+1 when κ_{t+1} becomes available,
then retroactively added to the reward for action a_t.

The alignment bonus is added via add_alignment_bonus() immediately after system
verification, before the reward is flushed.

Usage pattern in the simulator loop
────────────────────────────────────
reward_computer = RewardComputer(...)

for each turn t:
    behavior = bandit.decide(state_t)
    ... generate response ...
    quality_t = discriminator.score(...)

    # Flush deferred reward from previous turn (uses κ_t as κ_{t+1})
    if t > 0:
        pending = reward_computer.flush_deferred(kappa_curr=kappa_t)
        if pending is not None:
            bandit.update(pending.action_idx, pending.state, pending.total_reward)

    # Stage current turn's data
    reward_computer.stage(action_idx, state_t, quality_t, behavior, kappa_t)

    # After system verification — add alignment bonus
    reward_computer.add_alignment_bonus(alignment_score)

# After the last turn
pending = reward_computer.flush_deferred(kappa_curr=None)
if pending is not None:
    bandit.update(pending.action_idx, pending.state, pending.total_reward)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PendingUpdate:
    """Holds all data needed to call bandit.update() once drift is known."""
    action_idx: int
    state: "np.ndarray"  # type: ignore[name-defined]
    quality: float
    behavior: str
    kappa: Optional[float]           # κ at time of action
    immediate_reward: float = 0.0
    drift_penalty: float = 0.0

    @property
    def total_reward(self) -> float:
        return self.immediate_reward + self.drift_penalty


class RewardComputer:
    """
    Manages the two-stage reward: immediate (known now) + drift (known next turn).

    Parameters
    ----------
    lambda_drift : weight on curvature-worsening penalty
    final_bonus  : weight on final task score applied to the last action
    """

    def __init__(
        self,
        lambda_drift: float = 0.5,
        final_bonus: float = 1.0,
    ):
        self.lambda_drift = lambda_drift
        self.final_bonus = final_bonus
        self._pending: Optional[PendingUpdate] = None

    # ── staging ───────────────────────────────────────────────────────────────

    def stage(
        self,
        action_idx: int,
        state,
        quality: float,
        behavior: str,
        kappa: Optional[float],
    ) -> None:
        """
        Record the current turn's data; drift and alignment will be added later.
        """
        self._pending = PendingUpdate(
            action_idx=action_idx,
            state=state,
            quality=quality,
            behavior=behavior,
            kappa=kappa,
            immediate_reward=quality,
        )

    def add_alignment_bonus(self, bonus: float) -> None:
        """Add the alignment signal to the currently staged reward."""
        if self._pending is not None:
            self._pending.immediate_reward += bonus

    # ── flushing ──────────────────────────────────────────────────────────────

    def flush_deferred(
        self,
        kappa_curr: Optional[float],
        extra_bonus: float = 0.0,
    ) -> Optional[PendingUpdate]:
        """
        Complete the pending update with the drift penalty.

        Parameters
        ----------
        kappa_curr   : κ observed at the *next* turn (or None at end of episode)
        extra_bonus  : additional scalar reward (e.g. final task score bonus)
        """
        if self._pending is None:
            return None

        pending = self._pending
        self._pending = None

        if kappa_curr is not None and pending.kappa is not None:
            worsening = max(0.0, pending.kappa - kappa_curr)
            pending.drift_penalty = -self.lambda_drift * worsening

        pending.immediate_reward += extra_bonus
        return pending

    # ── helpers ───────────────────────────────────────────────────────────────

    def has_pending(self) -> bool:
        return self._pending is not None

    def final_bonus_reward(self, task_score: Optional[float]) -> float:
        """Scale the task score into a reward bonus."""
        if task_score is None:
            return 0.0
        return self.final_bonus * float(task_score)
