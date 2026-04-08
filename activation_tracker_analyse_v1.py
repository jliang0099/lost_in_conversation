"""
Temporal geometry analysis for hidden states.

Instead of relying on t-SNE overlap, this script compares trajectory dynamics:
1) turn curvature cosine:
   cos(turn_n - turn_{n-1}, turn_{n-1} - turn_{n-2})
2) goal-progress cosine:
   cos(turn_n - turn_{n-1}, goal - turn_{n-1})

It summarizes correct vs incorrect by turn and layer with:
- line plots (mean per turn)
- a CSV table (means, gaps, AUC)
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
OUTPUT_DIR = PT_DIR.parent / "temporal_cosine_analysis_v1"
EPS = 1e-12


def turn_sort_key(label: str):
    """Sort labels like turn_1, turn_2, ..., turn_10 in numeric order."""
    if not label.startswith("turn_"):
        return (1, label)
    try:
        return (0, int(label.split("_", 1)[1]))
    except ValueError:
        return (0, label)


def turn_index(label: str) -> int:
    """Extract numeric index from turn label."""
    return int(label.split("_", 1)[1])


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    """Cosine similarity with numerical safety."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """AUC for a single scalar feature, returns NaN if only one class exists."""
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, values))


def mean_for_class(values: np.ndarray, labels: np.ndarray, cls: int) -> float:
    mask = labels == cls
    if mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


# ── Step 1: 从 JSONL 读取 conv_id → score 映射 ───────────────────────────────
print(f"Reading scores from:\n  {JSONL_PATH}\n")
score_map = {}

with open(JSONL_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        conv_id = rec.get("conv_id")
        score = rec.get("score")
        if score is None:
            score = 0.0
        if conv_id is None or score is None:
            continue
        score_map[conv_id] = int(float(score))

n_correct = sum(v == 1 for v in score_map.values())
n_incorrect = sum(v == 0 for v in score_map.values())
print(f"  Total records in JSONL : {len(score_map)}")
print(f"  Correct   (score=1)    : {n_correct}")
print(f"  Incorrect (score=0)    : {n_incorrect}\n")

# ── Step 2: 收集 temporal cosine 指标 ────────────────────────────────────────
print(f"Loading .pt files from:\n  {PT_DIR}\n")

# layer_idx -> turn_label -> metrics
# metrics: {"cos_step": [...], "cos_goal": [...], "label": [...]}
layer_data = {i: {} for i in range(len(LAYERS))}
skipped_no_score = 0
skipped_no_goal = 0
skipped_short_turn = 0

for pt_file in sorted(PT_DIR.glob("*.pt")):
    conv_id = pt_file.stem
    if conv_id not in score_map:
        skipped_no_score += 1
        continue

    score = score_map[conv_id]
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    hs_list = data.get("hidden_states", [])
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
        turn_prev2 = turn_labels[t_idx - 2]
        turn_prev1 = turn_labels[t_idx - 1]
        turn_now = turn_labels[t_idx]

        hs_prev2 = hs_by_label[turn_prev2]
        hs_prev1 = hs_by_label[turn_prev1]
        hs_now = hs_by_label[turn_now]

        for i in range(len(LAYERS)):
            vec_prev2 = hs_prev2[i].numpy()
            vec_prev1 = hs_prev1[i].numpy()
            vec_now = hs_now[i].numpy()
            vec_goal = goal_hs[i].numpy()

            step_prev = vec_prev1 - vec_prev2
            step_now = vec_now - vec_prev1
            to_goal = vec_goal - vec_prev1

            cos_step = cosine_similarity(step_now, step_prev)
            cos_goal = cosine_similarity(step_now, to_goal)

            turn_store = layer_data[i].setdefault(
                turn_now,
                {"cos_step": [], "cos_goal": [], "label": []},
            )
            turn_store["cos_step"].append(cos_step)
            turn_store["cos_goal"].append(cos_goal)
            turn_store["label"].append(score)

all_turns = sorted(layer_data[0].keys(), key=turn_sort_key)
assert all_turns, "No valid turn_n (n>=3) found for temporal cosine analysis."

print(f"  Analyzed turn labels     : {', '.join(all_turns)}")
if skipped_no_score:
    print(f"  Skipped (no JSONL entry) : {skipped_no_score}")
if skipped_no_goal:
    print(f"  Skipped (missing goal)   : {skipped_no_goal}")
if skipped_short_turn:
    print(f"  Skipped (<3 turns)       : {skipped_short_turn}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 3: 导出统计 CSV ─────────────────────────────────────────────────────
summary_csv = OUTPUT_DIR / "temporal_cosine_summary.csv"
rows = []

for i, layer in enumerate(LAYERS):
    for turn_label in sorted(layer_data[i].keys(), key=turn_sort_key):
        entry = layer_data[i][turn_label]
        labels = np.array(entry["label"], dtype=np.int32)
        cos_step = np.array(entry["cos_step"], dtype=np.float32)
        cos_goal = np.array(entry["cos_goal"], dtype=np.float32)

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        step_pos = mean_for_class(cos_step, labels, 1)
        step_neg = mean_for_class(cos_step, labels, 0)
        goal_pos = mean_for_class(cos_goal, labels, 1)
        goal_neg = mean_for_class(cos_goal, labels, 0)

        rows.append(
            {
                "layer": layer,
                "turn": turn_label,
                "turn_idx": turn_index(turn_label),
                "n_correct": n_pos,
                "n_incorrect": n_neg,
                "mean_cos_step_correct": step_pos,
                "mean_cos_step_incorrect": step_neg,
                "gap_cos_step": step_pos - step_neg,
                "auc_cos_step": safe_auc(labels, cos_step),
                "mean_cos_goal_correct": goal_pos,
                "mean_cos_goal_incorrect": goal_neg,
                "gap_cos_goal": goal_pos - goal_neg,
                "auc_cos_goal": safe_auc(labels, cos_goal),
            }
        )

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved summary CSV -> {summary_csv}")


def plot_metric(metric_key: str, title: str, out_name: str):
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 4.6), sharey=True)
    if len(LAYERS) == 1:
        axes = [axes]

    for i, layer in enumerate(LAYERS):
        ax = axes[i]
        x_vals = []
        y_correct = []
        y_incorrect = []

        for turn_label in sorted(layer_data[i].keys(), key=turn_sort_key):
            entry = layer_data[i][turn_label]
            labels = np.array(entry["label"], dtype=np.int32)
            vals = np.array(entry[metric_key], dtype=np.float32)

            x_vals.append(turn_index(turn_label))
            y_correct.append(mean_for_class(vals, labels, 1))
            y_incorrect.append(mean_for_class(vals, labels, 0))

        ax.plot(x_vals, y_correct, "o-", color="#2563EB", label="correct")
        ax.plot(x_vals, y_incorrect, "^-", color="#DC2626", label="incorrect")
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("turn_n")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("cosine")
        if i == len(LAYERS) - 1:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved plot -> {out_path}")


# 图 1: 你提出的核心指标
plot_metric(
    metric_key="cos_step",
    title="Temporal Curvature: cos(step_n, step_{n-1}) by correctness",
    out_name="temporal_cos_step_by_turn.png",
)

# 图 2: 朝 goal 推进方向的对照指标
plot_metric(
    metric_key="cos_goal",
    title="Goal Progress: cos(step_n, goal - turn_{n-1}) by correctness",
    out_name="temporal_cos_goal_by_turn.png",
)

print(f"\nAll done. Outputs saved under: {OUTPUT_DIR}")