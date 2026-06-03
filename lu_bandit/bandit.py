"""
LinUCB contextual bandit.

For each action a ∈ {0%, 30%, 50%, 70%} compression, maintains:
  A_a  (d×d)  — feature covariance accumulator
  b_a  (d,)   — weighted reward accumulator

Decision rule:
  θ_a  = A_a^{-1} b_a
  p_a(x) = x^T θ_a  +  α * sqrt(x^T A_a^{-1} x)   (UCB)

Update (ridge regression):
  A_a += x x^T
  b_a += r x

Persistence: save/load via pickle.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Behavioral mode action set
DEFAULT_ACTIONS: list[str] = ["neutral", "explore", "commit", "challenge"]


class LinUCBBandit:
    """
    Disjoint LinUCB — separate parameter vector per action.

    Parameters
    ----------
    d       : dimension of state feature vector
    alpha   : exploration coefficient (higher → more exploration)
    actions : list of compression rates, one per arm
    """

    def __init__(
        self,
        d: int,
        alpha: float = 1.0,
        actions: Optional[list[float]] = None,
    ):
        self.d = d
        self.alpha = alpha
        self.actions: list[float] = actions if actions is not None else DEFAULT_ACTIONS
        n = len(self.actions)
        # Initialise each arm with identity A (= L2 regularisation prior)
        self.A: list[np.ndarray] = [np.eye(d, dtype=np.float64) for _ in range(n)]
        self.b: list[np.ndarray] = [np.zeros(d, dtype=np.float64) for _ in range(n)]
        self._update_count: list[int] = [0] * n
        self._lock = threading.Lock()  # safe for N_workers > 1

    # ── core methods ──────────────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
    ) -> tuple[int, float]:
        """
        Select the action with the highest UCB score.

        Returns
        -------
        (action_idx, ucb_value)

        Parameters
        ----------
        state   : 1-D feature vector of shape (d,)
        epsilon : optional ε-greedy exploration override (0 = pure UCB)
        """
        x = np.array(state, dtype=np.float64)
        assert x.shape == (self.d,), f"State dim mismatch: expected {self.d}, got {x.shape}"

        if epsilon > 0.0 and np.random.random() < epsilon:
            idx = int(np.random.randint(len(self.actions)))
            return idx, 0.0

        with self._lock:
            A_snap = [a.copy() for a in self.A]
            b_snap = [b.copy() for b in self.b]

        ucb: list[float] = []
        for i in range(len(self.actions)):
            theta_i = np.linalg.solve(A_snap[i], b_snap[i])
            Ainv_x = np.linalg.solve(A_snap[i], x)
            confidence = float(np.sqrt(max(0.0, x @ Ainv_x)))
            ucb.append(float(x @ theta_i) + self.alpha * confidence)

        idx = int(np.argmax(ucb))
        return idx, ucb[idx]

    def update(self, action_idx: int, state: np.ndarray, reward: float) -> None:
        """
        Online ridge-regression update for arm `action_idx`.

        Parameters
        ----------
        action_idx : which arm was pulled
        state      : feature vector at the time of the pull
        reward     : observed (possibly deferred) reward
        """
        x = np.array(state, dtype=np.float64)
        with self._lock:
            self.A[action_idx] += np.outer(x, x)
            self.b[action_idx] += reward * x
            self._update_count[action_idx] += 1

    # ── diagnostics ───────────────────────────────────────────────────────────

    def estimated_values(self, state: np.ndarray) -> list[float]:
        """Return θ_a^T x for each arm (exploitation term only)."""
        x = np.array(state, dtype=np.float64)
        return [float(x @ np.linalg.solve(self.A[i], self.b[i])) for i in range(len(self.actions))]

    def update_counts(self) -> list[int]:
        return list(self._update_count)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "d": self.d,
                    "alpha": self.alpha,
                    "actions": self.actions,
                    "A": self.A,
                    "b": self.b,
                    "_update_count": self._update_count,
                },
                fh,
            )
        print(f"[LinUCBBandit] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LinUCBBandit":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        bandit = cls(d=data["d"], alpha=data["alpha"], actions=data["actions"])
        bandit.A = data["A"]
        bandit.b = data["b"]
        bandit._update_count = data["_update_count"]
        print(f"[LinUCBBandit] loaded ← {path}  (updates={sum(bandit._update_count)})")
        return bandit

    @classmethod
    def load_or_new(cls, path: Optional[str], d: int, alpha: float = 1.0, actions: Optional[list[float]] = None) -> "LinUCBBandit":
        """Load from file if it exists, else create fresh."""
        if path and Path(path).exists():
            return cls.load(path)
        return cls(d=d, alpha=alpha, actions=actions)
