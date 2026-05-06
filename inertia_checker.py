"""
Online inertia checker for hidden-state trajectory quality.

Translates two offline metrics from conversation_analysis_v4.ipynb into an
online pass/fail test that can trigger prompt interventions during simulation:

Metric A — Temporal Curvature (κ):
    κ_t = cos(Δh_t, Δh_{t-1})
    Correct paths maintain directional momentum (κ closer to 1).
    Low/negative κ signals a sharp reversal — premature lock-in.
    Available from the 3rd turn onward (needs 2 consecutive deltas).

Metric D — Trajectory Variance Slope:
    Slope of trace(Cov(h_1, …, h_t)) over prefix length t.
    Correct paths expand state-space exploration (positive slope).
    Negative/near-zero slope signals premature convergence.
    Available from the 3rd turn onward (needs prefix lengths 2 and 3).

Thresholds are set to the empirically observed correct/incorrect boundary
from the notebook violin plots. Calibrate on your data if needed.
"""

import math
import numpy as np
import torch
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from activation_tracker import ActivationTracker


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ── Intervention prompts ──────────────────────────────────────────────────────

_PROMPT_CURVATURE = (
    "[System guidance] Your reasoning has made a sharp directional shift. "
    "Before answering, step back and ensure your logic is progressing "
    "consistently—avoid reversing your approach mid-way through."
)

_PROMPT_VARIANCE = (
    "[System guidance] Your reasoning may be converging too quickly. "
    "Before answering, explore at least one alternative interpretation or "
    "approach to the problem, then commit to the best path."
)

_PROMPT_BOTH = (
    "[System guidance] Your reasoning shows both a sharp directional shift "
    "and premature convergence. Before answering, step back entirely: "
    "reconsider the problem from scratch, explore multiple approaches, "
    "and then reason systematically toward your answer."
)


class InertiaChecker:
    """
    Checks whether the current activation trajectory has 'correct mechanical
    inertia' and returns an intervention prompt if not.

    activation_history layout (from ActivationTracker):
      index 0  : goal state   (set_goal, is_first_turn=True)
      index 1+ : turn states  (record_activation)

    The checker uses only the turn states (index 1+) to match the notebook
    analysis, requiring at least 3 turn states before any check is applied.
    """

    def __init__(
        self,
        focus_layer_idx: int = 2,
        curvature_threshold: float = -0.2,
        var_slope_threshold: float = 0.0,
    ):
        """
        Args:
            focus_layer_idx: Index into ActivationTracker.layers list.
                             Default 2 → layer 20 for layers=[12,16,20,24,28].
            curvature_threshold: Fail if κ < this value.
                                 Empirical: correct mean ≈ +0.05, incorrect ≈ -0.15.
            var_slope_threshold: Fail if var_slope < this value.
                                 Empirical: correct slope > 0, incorrect slope ≈ 0 or negative.
        """
        self.focus_layer_idx = focus_layer_idx
        self.curvature_threshold = curvature_threshold
        self.var_slope_threshold = var_slope_threshold

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _turn_states(self, tracker: "ActivationTracker") -> list[np.ndarray]:
        """Return per-turn hidden states (skip goal at index 0)."""
        out = []
        for h in tracker.activation_history[1:]:
            layer_vec = h[self.focus_layer_idx]
            if isinstance(layer_vec, torch.Tensor):
                layer_vec = layer_vec.cpu().numpy()
            else:
                layer_vec = np.array(layer_vec)
            out.append(layer_vec.astype(np.float32))
        return out

    def compute_curvature(self, turn_states: list[np.ndarray]) -> Optional[float]:
        """
        κ = cos(Δh_t, Δh_{t-1}) using the two most recent consecutive deltas.
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
        Slope of trace(Cov(h_1, …, h_t)) over prefix length t (t ∈ [2, T]).
        Uses simple rise/run rather than linregress for speed.
        Returns None when fewer than 3 turn states are available.
        """
        if len(turn_states) < 3:
            return None
        hs = np.stack(turn_states)           # (T, D)
        T = len(hs)
        trace_covs = [float(hs[:t].var(axis=0).sum()) for t in range(2, T + 1)]
        if len(trace_covs) < 2:
            return None
        # slope over the full available window
        slope = (trace_covs[-1] - trace_covs[0]) / max(len(trace_covs) - 1, 1)
        return float(slope)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, tracker: "ActivationTracker") -> Tuple[bool, dict]:
        """
        Run the inertia check against the current tracker state.

        Returns:
            passed  : True if the trajectory is within acceptable inertia bounds.
            info    : Dict with metrics and per-criterion pass/fail flags.
        """
        turn_states = self._turn_states(tracker)
        n_turns = len(turn_states)

        info: dict = {
            "n_turns": n_turns,
            "curvature": None,
            "var_slope": None,
            "curvature_fail": False,
            "var_slope_fail": False,
            "reason": "skip",
        }

        if n_turns < 3:
            info["reason"] = "skip_too_few_turns"
            return True, info

        kappa = self.compute_curvature(turn_states)
        vslope = self.compute_var_slope(turn_states)

        info["curvature"] = kappa
        info["var_slope"] = vslope

        curv_fail = kappa is not None and kappa < self.curvature_threshold
        var_fail  = vslope is not None and vslope < self.var_slope_threshold

        info["curvature_fail"] = curv_fail
        info["var_slope_fail"] = var_fail

        if curv_fail and var_fail:
            info["reason"] = "both_fail"
            return False, info
        elif curv_fail:
            info["reason"] = "curvature_fail"
            return False, info
        elif var_fail:
            info["reason"] = "var_slope_fail"
            return False, info

        info["reason"] = "pass"
        return True, info

    def get_intervention_prompt(self, reason: str) -> str:
        if reason == "both_fail":
            return _PROMPT_BOTH
        if reason == "curvature_fail":
            return _PROMPT_CURVATURE
        if reason == "var_slope_fail":
            return _PROMPT_VARIANCE
        return ""

    def inject_into_messages(
        self, messages: list[dict], reason: str
    ) -> list[dict]:
        """
        Return a copy of `messages` with the intervention appended to the last
        user turn's content.  Does not mutate the original list.
        """
        prompt = self.get_intervention_prompt(reason)
        if not prompt:
            return messages
        modified = [m.copy() for m in messages]
        for i in range(len(modified) - 1, -1, -1):
            if modified[i]["role"] == "user":
                modified[i]["content"] = modified[i]["content"] + "\n\n" + prompt
                return modified
        # fallback: no user message found, append as a new user turn
        modified.append({"role": "user", "content": prompt})
        return modified
