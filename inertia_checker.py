"""
Temporal curvature extractor for hidden-state trajectories.

Metrics
───────
  κ_t          cos(Δh_t, Δh_{t-1})  — local direction change (used by LUBandit)
  Δκ           κ_t − κ_{t-1}        — rate of direction change
  var_slope    slope of trace(Cov)   — overall spreading rate
  goal_drift   1 − cos(h_t, h_goal) — how far current state drifted from goal

goal_drift is the primary signal for CE validation:
  • low and stable  → model still "remembers" the original problem
  • rising sharply  → model is losing context (compression may be too aggressive)
  • drop after compression fires → CE successfully restored context alignment

activation_history layout (from ActivationTracker):
  index 0  : goal state   (set_goal, is_first_turn=True)
  index 1+ : turn states  (record_activation)
"""

import math
import numpy as np
import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from activation_tracker import ActivationTracker


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _to_numpy(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy().astype(np.float32)
    return np.array(v, dtype=np.float32)


class InertiaChecker:
    """
    Extracts per-turn hidden-state metrics from an ActivationTracker.

    Works independently of LUBandit — instantiate whenever track_activation=True
    to log metrics that reveal whether CE is preserving goal alignment.
    """

    def __init__(self, focus_layer_idx: int = 3):
        """
        Args:
            focus_layer_idx: Index into ActivationTracker.layers list.
                             Default 3 → layer 24 for layers=[12,16,20,24,28].
        """
        self.focus_layer_idx = focus_layer_idx

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _turn_states(self, tracker: "ActivationTracker") -> list[np.ndarray]:
        """Per-turn hidden states (skip goal at index 0)."""
        return [_to_numpy(h[self.focus_layer_idx]) for h in tracker.activation_history[1:]]

    def _goal_vec(self, tracker: "ActivationTracker") -> Optional[np.ndarray]:
        if not tracker.activation_history:
            return None
        return _to_numpy(tracker.activation_history[0][self.focus_layer_idx])

    # ── Existing metrics (used by LUBandit) ───────────────────────────────────

    def compute_curvature(self, turn_states: list[np.ndarray]) -> Optional[float]:
        """
        κ = cos(Δh_t, Δh_{t-1}).
        Returns None when fewer than 3 turn states are available.
        """
        if len(turn_states) < 3:
            return None
        delta_prev = turn_states[-2] - turn_states[-3]
        delta_curr = turn_states[-1] - turn_states[-2]
        kappa = _cosine(delta_curr, delta_prev)
        return None if math.isnan(kappa) else kappa

    def compute_var_slope(self, turn_states: list[np.ndarray]) -> Optional[float]:
        """
        Slope of trace(Cov(h_1, …, h_t)) over prefix length t.
        Returns None when fewer than 3 turn states are available.
        """
        if len(turn_states) < 3:
            return None
        hs = np.stack(turn_states)
        trace_covs = [float(hs[:t].var(axis=0).sum()) for t in range(2, len(hs) + 1)]
        if len(trace_covs) < 2:
            return None
        return float((trace_covs[-1] - trace_covs[0]) / max(len(trace_covs) - 1, 1))

    # ── CE-validation metrics ─────────────────────────────────────────────────

    def goal_drift(self, tracker: "ActivationTracker") -> Optional[float]:
        """
        Cosine distance between the current (latest) turn state and the goal state.

        Interpretation:
          ~0.0   model's representation is still aligned with the initial goal
          > 0.2  noticeable drift — model may be losing context
          rising across turns with CE off vs. stable with CE on → CE is helping
        """
        goal = self._goal_vec(tracker)
        if goal is None or len(tracker.activation_history) < 2:
            return None
        latest = _to_numpy(tracker.activation_history[-1][self.focus_layer_idx])
        cos_sim = _cosine(latest, goal)
        return None if math.isnan(cos_sim) else 1.0 - cos_sim

    def goal_drift_history(self, tracker: "ActivationTracker") -> list[float]:
        """
        Cosine distance from goal state for every recorded turn (index 0 = turn 1).
        Useful for plotting CE effect across a full conversation.
        """
        goal = self._goal_vec(tracker)
        if goal is None:
            return []
        out = []
        for h in tracker.activation_history[1:]:
            v = _to_numpy(h[self.focus_layer_idx])
            cos_sim = _cosine(v, goal)
            out.append(float("nan") if math.isnan(cos_sim) else 1.0 - cos_sim)
        return out

    def compression_delta(
        self, tracker: "ActivationTracker", compression_turn: int
    ) -> Optional[float]:
        """
        Change in goal_drift immediately after a compression event.

        compression_turn: 1-based turn index when compression fired.
        Negative value → compression brought the model closer to the goal (good).
        Positive value → compression pushed the model further away (bad).
        """
        history = self.goal_drift_history(tracker)
        idx = compression_turn - 1  # convert to 0-based
        if idx < 1 or idx >= len(history):
            return None
        before = history[idx - 1]
        after  = history[idx]
        if math.isnan(before) or math.isnan(after):
            return None
        return after - before

    # ── Aggregate summary (for trace logging) ────────────────────────────────

    def summary(self, tracker: "ActivationTracker") -> dict:
        """
        All metrics in one call — attach to each assistant turn in the trace.

        Keys:
          kappa              current temporal curvature (None if < 3 turns)
          var_slope          variance slope (None if < 3 turns)
          goal_drift         distance from goal state this turn
          goal_drift_history list of per-turn goal drifts (full trajectory)
          n_turns            number of recorded turns (excl. goal)
        """
        turn_states = self._turn_states(tracker)
        return {
            "kappa":              self.compute_curvature(turn_states),
            "var_slope":          self.compute_var_slope(turn_states),
            # "goal_drift":         self.goal_drift(tracker),
            # "goal_drift_history": self.goal_drift_history(tracker),
            "n_turns":            len(turn_states),
        }

