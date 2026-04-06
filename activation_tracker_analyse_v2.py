"""
Temporal geometry analysis for hidden states (v2).

Changes from v1:
- Only analyze turns 3–8 (turn_idx <= 8)
- Bootstrap 95% CI shading on all line plots
- Min-sample guard: skip turn×layer cells with n < MIN_N per class

Metrics:
1) Temporal Curvature : cos(step_n, step_{n-1})
2) Goal Progress      : cos(step_n, goal - turn_{n-1})
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# ── 配置 ──────────────────────────────────────────────────────────────────────
PT_DIR = Path(
    "logs/hidden_states/math/sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
JSONL_PATH = Path(
    "logs/math/sharded-at0-ut0/sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)
LAYERS = [12, 16, 20, 24, 28]
OUTPUT_DIR = PT_DIR.parent / "temporal_cosine_analysis_v2"
EPS = 1e-12
MAX_TURN = 8       # only keep turn_3 … turn_8
MIN_N    = 5       # minimum samples per class to plot a point
N_BOOT   = 2000    # bootstrap resamples for CI
CI       = 95      # confidence interval %

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def turn_sort_key(label: str):
    if not label.startswith("turn_"):
        return (1, label)
    try:
        return (0, int(label.split("_", 1)[1]))
    except ValueError:
        return (0, label)


def turn_index(label: str) -> int:
    return int(label.split("_", 1)[1])


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, values))


def mean_for_class(values: np.ndarray, labels: np.ndarray, cls: int):
    mask = labels == cls
    if mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, ci: float = CI):
    """Return (lower, upper) bootstrap percentile CI for the mean."""
    if len(values) < 2:
        m = float(values.mean()) if len(values) == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


# ── Step 1: 读取 JSONL ────────────────────────────────────────────────────────
print(f"Reading scores from:\n  {JSONL_PATH}\n")
score_map = {}

with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        conv_id = rec.get("conv_id")
        score   = rec.get("score")
        if conv_id is None or score is None:
            continue
        score_map[conv_id] = int(float(score))

n_correct   = sum(v == 1 for v in score_map.values())
n_incorrect = sum(v == 0 for v in score_map.values())
print(f"  Total records : {len(score_map)}")
print(f"  Correct       : {n_correct}")
print(f"  Incorrect     : {n_incorrect}\n")

# ── Step 2: 收集指标（只保留 turn_idx <= MAX_TURN）────────────────────────────
print(f"Loading .pt files from:\n  {PT_DIR}\n")

layer_data = {i: {} for i in range(len(LAYERS))}
skipped_no_score  = 0
skipped_no_goal   = 0
skipped_short_turn = 0

for pt_file in sorted(PT_DIR.glob("*.pt")):
    conv_id = pt_file.stem
    if conv_id not in score_map:
        skipped_no_score += 1
        continue

    score = score_map[conv_id]
    data  = torch.load(pt_file, map_location="cpu", weights_only=False)
    hs_list    = data.get("hidden_states", [])
    hs_by_label = {entry["label"]: entry["hidden_states"] for entry in hs_list}

    if "goal" not in hs_by_label:
        skipped_no_goal += 1
        continue

    turn_labels = sorted(
        [k for k in hs_by_label if k.startswith("turn_")],
        key=turn_sort_key,
    )
    if len(turn_labels) < 2:
        skipped_short_turn += 1
        continue

    goal_hs = hs_by_label["goal"]

    for t_idx in range(2, len(turn_labels)):
        turn_now = turn_labels[t_idx]
        if turn_index(turn_now) > MAX_TURN:   # ← 关键过滤
            continue

        turn_prev2 = turn_labels[t_idx - 2]
        turn_prev1 = turn_labels[t_idx - 1]

        hs_prev2 = hs_by_label[turn_prev2]
        hs_prev1 = hs_by_label[turn_prev1]
        hs_now   = hs_by_label[turn_now]

        for i in range(len(LAYERS)):
            vec_prev2 = hs_prev2[i].numpy()
            vec_prev1 = hs_prev1[i].numpy()
            vec_now   = hs_now[i].numpy()
            vec_goal  = goal_hs[i].numpy()

            step_prev = vec_prev1 - vec_prev2
            step_now  = vec_now  - vec_prev1
            to_goal   = vec_goal - vec_prev1

            cos_step = cosine_similarity(step_now, step_prev)
            cos_goal = cosine_similarity(step_now, to_goal)

            store = layer_data[i].setdefault(
                turn_now,
                {"cos_step": [], "cos_goal": [], "label": []},
            )
            store["cos_step"].append(cos_step)
            store["cos_goal"].append(cos_goal)
            store["label"].append(score)

all_turns = sorted(layer_data[0].keys(), key=turn_sort_key)
assert all_turns, "No valid turns found."

print(f"  Turn range analyzed : {all_turns[0]} – {all_turns[-1]}")
if skipped_no_score:   print(f"  Skipped (no score)  : {skipped_no_score}")
if skipped_no_goal:    print(f"  Skipped (no goal)   : {skipped_no_goal}")
if skipped_short_turn: print(f"  Skipped (<3 turns)  : {skipped_short_turn}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 3: CSV 汇总 ──────────────────────────────────────────────────────────
summary_csv = OUTPUT_DIR / "temporal_cosine_summary.csv"
rows = []

for i, layer in enumerate(LAYERS):
    for turn_label in sorted(layer_data[i].keys(), key=turn_sort_key):
        entry  = layer_data[i][turn_label]
        labels   = np.array(entry["label"],    dtype=np.int32)
        cos_step = np.array(entry["cos_step"], dtype=np.float32)
        cos_goal = np.array(entry["cos_goal"], dtype=np.float32)

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        step_pos = mean_for_class(cos_step, labels, 1)
        step_neg = mean_for_class(cos_step, labels, 0)
        goal_pos = mean_for_class(cos_goal, labels, 1)
        goal_neg = mean_for_class(cos_goal, labels, 0)

        # bootstrap CI for cos_step
        step_pos_lo, step_pos_hi = bootstrap_ci(cos_step[labels == 1]) if n_pos >= 2 else (float("nan"), float("nan"))
        step_neg_lo, step_neg_hi = bootstrap_ci(cos_step[labels == 0]) if n_neg >= 2 else (float("nan"), float("nan"))

        rows.append({
            "layer": layer,
            "turn": turn_label,
            "turn_idx": turn_index(turn_label),
            "n_correct": n_pos,
            "n_incorrect": n_neg,
            "mean_cos_step_correct": step_pos,
            "ci95_lo_correct": step_pos_lo,
            "ci95_hi_correct": step_pos_hi,
            "mean_cos_step_incorrect": step_neg,
            "ci95_lo_incorrect": step_neg_lo,
            "ci95_hi_incorrect": step_neg_hi,
            "gap_cos_step": step_pos - step_neg,
            "auc_cos_step": safe_auc(labels, cos_step),
            "mean_cos_goal_correct": goal_pos,
            "mean_cos_goal_incorrect": goal_neg,
            "gap_cos_goal": goal_pos - goal_neg,
            "auc_cos_goal": safe_auc(labels, cos_goal),
        })

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved CSV -> {summary_csv}")

# ── Step 4: 画图（带 bootstrap CI 阴影）────────────────────────────────────────

COLOR_CORRECT   = "#2563EB"
COLOR_INCORRECT = "#DC2626"

def plot_metric(metric_key: str, title: str, out_name: str):
    fig, axes = plt.subplots(
        1, len(LAYERS),
        figsize=(5 * len(LAYERS), 4.8),
        sharey=True,
    )
    if len(LAYERS) == 1:
        axes = [axes]

    for i, layer in enumerate(LAYERS):
        ax = axes[i]

        x_vals = []
        y_c, y_i         = [], []
        ci_c_lo, ci_c_hi = [], []
        ci_i_lo, ci_i_hi = [], []

        for turn_label in sorted(layer_data[i].keys(), key=turn_sort_key):
            entry  = layer_data[i][turn_label]
            labels = np.array(entry["label"], dtype=np.int32)
            vals   = np.array(entry[metric_key], dtype=np.float32)

            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())

            # skip cells with too few samples in either class
            if n_pos < MIN_N or n_neg < MIN_N:
                continue

            vals_c = vals[labels == 1]
            vals_i = vals[labels == 0]

            x_vals.append(turn_index(turn_label))
            y_c.append(float(vals_c.mean()))
            y_i.append(float(vals_i.mean()))

            lo, hi = bootstrap_ci(vals_c)
            ci_c_lo.append(lo); ci_c_hi.append(hi)

            lo, hi = bootstrap_ci(vals_i)
            ci_i_lo.append(lo); ci_i_hi.append(hi)

        if not x_vals:
            ax.set_title(f"Layer {layer}\n(no data)", fontsize=11)
            continue

        x = np.array(x_vals)

        # correct
        ax.fill_between(x, ci_c_lo, ci_c_hi,
                        color=COLOR_CORRECT, alpha=0.15)
        ax.plot(x, y_c, "o-", color=COLOR_CORRECT,
                linewidth=1.8, markersize=5, label="correct")

        # incorrect
        ax.fill_between(x, ci_i_lo, ci_i_hi,
                        color=COLOR_INCORRECT, alpha=0.15)
        ax.plot(x, y_i, "^-", color=COLOR_INCORRECT,
                linewidth=1.8, markersize=5, label="incorrect")

        # annotate n per point
        for xi, nc, ni in zip(x_vals,
                               [int((np.array(layer_data[i][f"turn_{xv}"]["label"]) == 1).sum()) for xv in x_vals],
                               [int((np.array(layer_data[i][f"turn_{xv}"]["label"]) == 0).sum()) for xv in x_vals]):
            ax.annotate(f"n={nc}/{ni}", xy=(xi, ax.get_ylim()[0]),
                        fontsize=6, ha="center", va="bottom", color="gray")

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("turn_n")
        ax.set_xticks(x_vals)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("cosine")
        if i == len(LAYERS) - 1:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{title}\n(95% bootstrap CI shaded, min n={MIN_N}/class, turns 3–{MAX_TURN})",
        fontsize=12, y=1.03,
    )
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out_path}")


plot_metric(
    metric_key="cos_step",
    title="Temporal Curvature: cos(step_n, step_{n-1}) by correctness",
    out_name="temporal_cos_step_by_turn_v2.png",
)

plot_metric(
    metric_key="cos_goal",
    title="Goal Progress: cos(step_n, goal − turn_{n-1}) by correctness",
    out_name="temporal_cos_goal_by_turn_v2.png",
)

print(f"\nAll done. Outputs -> {OUTPUT_DIR}")