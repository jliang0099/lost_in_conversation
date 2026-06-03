"""
State feature extractor for LUBandit.

State vector — 8 dimensions
───────────────────────────────────────────────────────────────────
 idx  name                   description
───────────────────────────────────────────────────────────────────
  0   kappa                  temporal curvature κ_t  ∈ [-1, 1]
  1   delta_kappa            Δκ = κ_t − κ_{t−1}     ∈ [-2, 2]
  2   relative_pos           turn_idx / total_turns  ∈ [0, 1]
  3   n_attempts_norm        # answer attempts / # assistant turns  ∈ [0, 1]
  4   n_same_answers_norm    # repeated answers / max(1, # attempts)  ∈ [0, 1]
  5   last_rt_code           response_type of last turn  ∈ {0..6}
  6   ctx_len_norm           context chars / 10000  (soft normalisation)
  7   task_type              integer in [0, N_tasks) cast to float
───────────────────────────────────────────────────────────────────

Diagnosis of which arm is needed:
  EVOLVES_WRONG   → high kappa drop, low n_attempts_norm  →  EXPLORE
  NEVER_ATTEMPTED → high relative_pos, near-zero n_attempts → COMMIT
  SAME_WRONG      → high n_same_answers_norm              →  CHALLENGE
  SUCCESS         → moderate attempt timing               →  NEUTRAL
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from activation_tracker import ActivationTracker
    from inertia_checker import InertiaChecker

TASK_TYPES: list[str] = [
    "math",
    "code",
    "database",
    "actions",
    "data2text",
    "summary",
    "translation",
]

# Numeric encoding for response_type strings
RESPONSE_TYPE_CODES: dict[str, float] = {
    "discussion":    1.0,
    "clarification": 2.0,
    "interrogation": 3.0,
    "answer_attempt": 4.0,
    "hedge":         5.0,
    "refuse":        6.0,
}

STATE_DIM: int = 8
_CTX_NORM_FACTOR: float = 10_000.0

STATE_NAMES: list[str] = [
    "kappa",
    "delta_kappa",
    "relative_pos",
    "n_attempts_norm",
    "n_same_answers_norm",
    "last_rt_code",
    "ctx_len_norm",
    "task_type",
]


def extract_state(
    tracker: Optional["ActivationTracker"],
    inertia_checker: Optional["InertiaChecker"],
    messages: list[dict],
    n_total_turns: int,
    response_type_history: list[str],
    answer_history: list[str],
    task_type: str,
) -> np.ndarray:
    """
    Build an 8-dimensional state vector from the current conversation state.

    Parameters
    ----------
    tracker               : ActivationTracker with ≥1 recorded activation
    inertia_checker       : InertiaChecker (used to extract turn states)
    messages              : current message list (roles: system / user / assistant)
    n_total_turns         : expected total number of assistant turns (len(shards))
    response_type_history : list of response_type strings seen so far this episode
    answer_history        : list of extracted answers from answer_attempt turns
    task_type             : task name string (e.g. "math")
    """

    # ── 0 & 1 : temporal curvature and its delta ──────────────────────────────
    kappa: float = 0.0
    delta_kappa: float = 0.0

    if tracker is not None and inertia_checker is not None:
        turn_states = inertia_checker._turn_states(tracker)
        kappa_curr = inertia_checker.compute_curvature(turn_states)

        if kappa_curr is not None:
            kappa = float(kappa_curr)
            if len(turn_states) >= 4:
                kappa_prev = inertia_checker.compute_curvature(turn_states[:-1])
                if kappa_prev is not None:
                    delta_kappa = kappa - float(kappa_prev)

    # ── 2 : relative conversation position ───────────────────────────────────
    n_assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    relative_pos: float = float(n_assistant_turns) / float(max(n_total_turns, 1))

    # ── 3 : normalised answer-attempt count ──────────────────────────────────
    n_attempts = sum(1 for rt in response_type_history if rt == "answer_attempt")
    n_attempts_norm: float = float(n_attempts) / float(max(n_assistant_turns, 1))

    # ── 4 : same-answer repetition rate ──────────────────────────────────────
    if answer_history:
        counts = Counter(answer_history)
        # 0 if all answers unique, approaches 1 if model keeps repeating same answer
        n_same = max(counts.values()) - 1
        n_same_answers_norm = float(n_same) / float(max(n_attempts, 1))
    else:
        n_same_answers_norm = 0.0

    # ── 5 : last response-type code ───────────────────────────────────────────
    last_rt_code: float = (
        RESPONSE_TYPE_CODES.get(response_type_history[-1], 0.0)
        if response_type_history else 0.0
    )

    # ── 6 : context length (soft-normalised) ─────────────────────────────────
    ctx_chars = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
        for m in messages
        if m.get("role") not in ("log",)
    )
    ctx_len_norm: float = ctx_chars / _CTX_NORM_FACTOR

    # ── 7 : task type ─────────────────────────────────────────────────────────
    task_id: float = float(
        TASK_TYPES.index(task_type) if task_type in TASK_TYPES else 0
    )

    return np.array(
        [kappa, delta_kappa, relative_pos, n_attempts_norm,
         n_same_answers_norm, last_rt_code, ctx_len_norm, task_id],
        dtype=np.float32,
    )
