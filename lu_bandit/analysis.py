"""
LUBandit post-hoc analysis utilities.

Three entry points
──────────────────
1. inspect_checkpoint(path)
   Loads a .pkl and prints what the bandit learned:
   - arm exploration counts
   - learned θ per arm  (which state features drive each behavior mode)
   - policy map: what action the bandit would take across a 2-D slice of state space

2. parse_bandit_logs(jsonl_paths)
   Reads JSONL conversation logs that contain "bandit_meta" fields.
   Returns a list of per-turn bandit decisions + outcomes.

3. compare_runs(baseline_paths, bandit_paths)
   Compares task score distributions between a static baseline and the
   bandit run.  Prints/returns effect-size and per-task breakdown.

Quick start
───────────
from lu_bandit.analysis import inspect_checkpoint, parse_bandit_logs, compare_runs

# 1. Was anything learned?
inspect_checkpoint("checkpoints/lu_bandit_math.pkl")

# 2. What did the bandit actually do?
rows = parse_bandit_logs(["logs/math/bandit-run/*.jsonl"])

# 3. Did it help?
compare_runs(
    baseline_paths=["logs/math/sharded-at0-ut0/(40-103)*.jsonl"],
    bandit_paths=["logs_bandit_eval/math/**/*.jsonl"],
)
"""

from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# State feature names — must match lu_bandit/state.py
STATE_NAMES = [
    "kappa",
    "delta_kappa",
    "relative_pos",
    "n_attempts_norm",
    "n_same_answers_norm",
    "last_rt_code",
    "ctx_len_norm",
    "task_type",
]

# Behavior mode names — must match lu_bandit/bandit.py DEFAULT_ACTIONS
BEHAVIOR_MODES = ["neutral", "explore", "commit", "challenge"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_checkpoint(path: str, verbose: bool = True) -> dict:
    """
    Load a LinUCB checkpoint and report what the bandit learned.

    Returns a dict with:
      - actions        : list of behavior modes
      - update_counts  : how many times each arm was updated
      - theta          : learned parameter vector per arm  (shape [n_arms, d])
      - feature_importance : |θ_a| per feature per arm
      - dominant_arm   : which arm has highest estimated value at neutral state
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    actions = data["actions"]
    A = data["A"]
    b = data["b"]
    counts = data["_update_count"]

    thetas = [np.linalg.solve(A[i], b[i]) for i in range(len(actions))]

    result = {
        "actions": actions,
        "update_counts": counts,
        "theta": np.stack(thetas),
        "feature_names": STATE_NAMES,
        "feature_importance": np.abs(np.stack(thetas)),
    }

    # Neutral state: no curvature, mid-conversation, no attempts yet,
    # no repeated answers, last_rt=discussion (1.0), moderate ctx, math task
    neutral = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.2, 0.0])
    values = [float(neutral @ thetas[i]) for i in range(len(actions))]
    result["neutral_values"] = values
    result["dominant_arm"] = int(np.argmax(values))

    if verbose:
        _print_checkpoint_report(result)
    return result


def _print_checkpoint_report(r: dict) -> None:
    actions = r["actions"]
    counts  = r["update_counts"]
    total   = max(sum(counts), 1)
    thetas  = r["theta"]      # (n_arms, d)
    names   = r["feature_names"]

    print("=" * 60)
    print("  LinUCB Checkpoint Analysis")
    print("=" * 60)

    print("\n[Arm exploration]")
    for i, (a, c) in enumerate(zip(actions, counts)):
        bar = "█" * int(c / total * 40)
        print(f"  {a:<10}  {c:4d} updates ({c/total*100:4.1f}%)  {bar}")

    print("\n[Learned θ per arm  (each row = one arm, cols = state features)]")
    col_w = 12
    header = "behavior".ljust(10) + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    print("  " + header)
    print("  " + "-" * len(header))
    for i, a in enumerate(actions):
        row = f"{a:<10}" + "".join(f"{v:>{col_w}.4f}" for v in thetas[i])
        print("  " + row)

    print("\n[Feature importance  |θ_a| — which features drive each arm]")
    fi = r["feature_importance"]   # (n_arms, d)
    for j, name in enumerate(names):
        col = fi[:, j]
        dominant = int(np.argmax(col))
        bar_vals = "  ".join(f"{actions[i]}:{col[i]:.3f}" for i in range(len(actions)))
        print(f"  {name:<22}  {bar_vals}  ← {actions[dominant]} most sensitive")

    print(f"\n[Policy at neutral state]")
    for i, (a, v) in enumerate(zip(actions, r["neutral_values"])):
        marker = " ← chosen" if i == r["dominant_arm"] else ""
        print(f"  {a:<10}  estimated_value={v:+.4f}{marker}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parse bandit logs
# ─────────────────────────────────────────────────────────────────────────────

def parse_bandit_logs(jsonl_paths: list[str]) -> list[dict]:
    """
    Parse JSONL conversation logs for bandit decision data.

    Each returned dict represents one assistant turn and contains:
      conv_id, task, task_id, turn_idx, n_turns_total,
      behavior, action_idx, ucb_score,
      state (8-dim list), quality (if recorded),
      is_correct, score  (conversation-level outcome)
    """
    expanded = []
    for pattern in jsonl_paths:
        expanded.extend(glob.glob(pattern, recursive=True))

    rows = []
    for path in expanded:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rows.extend(_extract_turns(rec))
                except Exception:
                    pass
    return rows


def _extract_turns(record: dict) -> list[dict]:
    conv_id    = record.get("conv_id", "?")
    task       = record.get("task", "?")
    task_id    = record.get("task_id", "?")
    is_correct = record.get("is_correct", None)
    score      = record.get("score", None)

    trace = record.get("trace", [])
    assistant_turns = [m for m in trace if m.get("role") == "assistant"]
    n_total = len(assistant_turns)

    rows = []
    for turn_idx, msg in enumerate(assistant_turns):
        meta = msg.get("bandit_meta")
        if meta is None:
            continue
        row = {
            "conv_id":    conv_id,
            "task":       task,
            "task_id":    task_id,
            "turn_idx":   turn_idx,
            "n_turns":    n_total,
            "behavior":   meta.get("behavior", None),
            "action_idx": meta.get("action_idx", None),
            "ucb_score":  meta.get("ucb_score", None),
            "state":      meta.get("state", None),
            "quality":    meta.get("quality", None),
            "is_correct": is_correct,
            "score":      score,
        }
        if row["state"] and len(row["state"]) == len(STATE_NAMES):
            for name, val in zip(STATE_NAMES, row["state"]):
                row[f"s_{name}"] = val
        rows.append(row)
    return rows


def summarise_decisions(rows: list[dict]) -> None:
    if not rows:
        print("No bandit decisions found in logs.")
        return

    from collections import Counter
    behavior_counts = Counter(r["behavior"] for r in rows)
    total = sum(behavior_counts.values())

    print(f"\n[Decision summary]  {len(rows)} bandit turns across "
          f"{len({r['conv_id'] for r in rows})} conversations")
    for behavior, cnt in sorted(behavior_counts.items()):
        bar = "█" * int(cnt / total * 40)
        print(f"  {behavior:<10}  {cnt:4d} ({cnt/total*100:4.1f}%)  {bar}")

    print("\n[Task score by behavior mode]")
    behavior_scores: dict[str, list] = {}
    for r in rows:
        bh = r.get("behavior")
        sc = r.get("score")
        if bh is not None and sc is not None:
            behavior_scores.setdefault(bh, []).append(sc)
    for behavior in BEHAVIOR_MODES:
        vals = behavior_scores.get(behavior, [])
        if not vals:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        print(f"  {behavior:<10}  mean_score={mean:.3f} ± {std:.3f}  (n={len(vals)})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compare bandit vs. baseline
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(
    baseline_paths: list[str],
    bandit_paths: list[str],
    task_filter: Optional[str] = None,
) -> dict:
    """
    Load two sets of JSONL logs and compare task score distributions.

    Returns dict with keys:
      baseline_scores, bandit_scores,
      baseline_mean, bandit_mean,
      delta_mean, p_value (Mann-Whitney U)
    """
    def _load_scores(paths):
        expanded = []
        for p in paths:
            expanded.extend(glob.glob(p, recursive=True))
        scores = []
        for path in expanded:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if task_filter and rec.get("task") != task_filter:
                            continue
                        sc = rec.get("score")
                        if sc is not None:
                            scores.append(float(sc))
                    except Exception:
                        pass
        return scores

    base_scores   = _load_scores(baseline_paths)
    bandit_scores = _load_scores(bandit_paths)

    result: dict = {
        "baseline_scores": base_scores,
        "bandit_scores":   bandit_scores,
        "baseline_n":      len(base_scores),
        "bandit_n":        len(bandit_scores),
        "baseline_mean":   float(np.mean(base_scores)) if base_scores else float("nan"),
        "bandit_mean":     float(np.mean(bandit_scores)) if bandit_scores else float("nan"),
    }
    result["delta_mean"] = result["bandit_mean"] - result["baseline_mean"]

    if base_scores and bandit_scores:
        from scipy import stats as sp_stats
        u_stat, p_val = sp_stats.mannwhitneyu(
            bandit_scores, base_scores, alternative="greater"
        )
        result["u_stat"] = float(u_stat)
        result["p_value"] = float(p_val)
    else:
        result["p_value"] = float("nan")

    _print_comparison(result)
    return result


def _print_comparison(r: dict) -> None:
    print("\n" + "=" * 60)
    print("  Baseline vs. LUBandit  —  Task Score Comparison")
    print("=" * 60)
    print(f"  Baseline  : n={r['baseline_n']:4d}  mean={r['baseline_mean']:.4f}")
    print(f"  LUBandit  : n={r['bandit_n']:4d}  mean={r['bandit_mean']:.4f}")
    delta = r["delta_mean"]
    sign  = "+" if delta >= 0 else ""
    print(f"  Δ (bandit − baseline) = {sign}{delta:.4f}")
    pv = r.get("p_value", float("nan"))
    if pv == pv:   # not NaN
        sig = "✓ significant (p<0.05)" if pv < 0.05 else "✗ not significant"
        print(f"  Mann-Whitney p={pv:.4f}  {sig}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exploitation: use learned policy (no explore)
# ─────────────────────────────────────────────────────────────────────────────

def load_exploitation_bandit(checkpoint_path: str) -> "LUBandit":
    """
    Load a trained LUBandit in pure-exploitation mode (α=0, ε=0).

    Use this for offline evaluation after training is complete.
    """
    from lu_bandit.agent import LUBandit

    bandit = LUBandit.from_config(
        checkpoint_path=checkpoint_path,
        alpha=0.0,
        epsilon=0.0,
        discriminator_mode="none",
    )
    print(f"[load_exploitation_bandit] loaded from {checkpoint_path}, exploitation mode (α=0)")
    return bandit


# ─────────────────────────────────────────────────────────────────────────────
# 5. Policy map: visualise what the bandit would choose
# ─────────────────────────────────────────────────────────────────────────────

def policy_map(
    checkpoint_path: str,
    x_feature: str = "relative_pos",
    y_feature: str = "kappa",
    n_grid: int = 20,
    fixed_state: Optional[dict] = None,
) -> "np.ndarray":
    """
    Compute a 2-D grid of actions the bandit would choose as two state
    features vary, holding all others fixed at neutral values.

    Returns (n_grid × n_grid) int array of chosen action indices.

    Example — display with matplotlib
    ----------------------------------
    import matplotlib.pyplot as plt
    grid = policy_map("checkpoints/lu_bandit_math.pkl",
                      x_feature="relative_pos", y_feature="kappa")
    actions = ["neutral", "explore", "commit", "challenge"]
    plt.imshow(grid, origin="lower", cmap="viridis",
               extent=[0, 1, -1, 1], aspect="auto")
    plt.colorbar(label="arm index (0=neutral,1=explore,2=commit,3=challenge)")
    plt.xlabel("relative_pos"); plt.ylabel("kappa")
    plt.title("LUBandit policy map"); plt.show()
    """
    with open(checkpoint_path, "rb") as fh:
        data = pickle.load(fh)
    A, b, actions = data["A"], data["b"], data["actions"]
    thetas = [np.linalg.solve(A[i], b[i]) for i in range(len(actions))]

    # Neutral state defaults (index matches STATE_NAMES order)
    neutral = {
        "kappa":              0.0,
        "delta_kappa":        0.0,
        "relative_pos":       0.5,
        "n_attempts_norm":    0.0,
        "n_same_answers_norm": 0.0,
        "last_rt_code":       1.0,   # discussion
        "ctx_len_norm":       0.2,
        "task_type":          0.0,   # math
    }
    if fixed_state:
        neutral.update(fixed_state)

    xi = STATE_NAMES.index(x_feature)
    yi = STATE_NAMES.index(y_feature)

    x_range = _feature_range(x_feature)
    y_range = _feature_range(y_feature)

    grid = np.zeros((n_grid, n_grid), dtype=int)
    for ix, xv in enumerate(np.linspace(*x_range, n_grid)):
        for iy, yv in enumerate(np.linspace(*y_range, n_grid)):
            state = np.array([neutral[n] for n in STATE_NAMES])
            state[xi] = xv
            state[yi] = yv
            vals = [float(state @ theta) for theta in thetas]
            grid[iy, ix] = int(np.argmax(vals))
    return grid


def _feature_range(name: str) -> tuple[float, float]:
    defaults = {
        "kappa":               (-1.0, 1.0),
        "delta_kappa":         (-1.0, 1.0),
        "relative_pos":        ( 0.0, 1.0),
        "n_attempts_norm":     ( 0.0, 1.0),
        "n_same_answers_norm": ( 0.0, 1.0),
        "last_rt_code":        ( 0.0, 6.0),
        "ctx_len_norm":        ( 0.0, 1.0),
        "task_type":           ( 0.0, 6.0),
    }
    return defaults.get(name, (-1.0, 1.0))
