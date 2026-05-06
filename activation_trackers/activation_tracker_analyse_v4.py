"""
Temporal geometry analysis for hidden states (v4).

Key changes from v3:
- Balance correct/incorrect at the CONV level (not per layer×turn cell).
  For each turn_t, we find all convs that have data up to turn_t,
  split by correctness, take min(n_correct, n_incorrect), randomly
  subsample the majority class — same conv subset used for ALL layers.
- Removed the strict `len(turn_labels) != 6` filter; any conv with
  at least 3 turns (so step_n and step_{n-1} can be computed) is kept.
- Bootstrap 95% CI shading on all line plots (unchanged from v3).
- Min-sample guard: skip turn×layer cells with n < MIN_N per class.

Metrics:
1) Temporal Curvature : cos(step_n, step_{n-1})
2) Goal Progress      : cos(step_n, goal − turn_{n-1})
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
    "logs/hidden_states/math/(specific system prompt+no suffix)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
PT_DIR_ADD = Path(
    "logs/hidden_states/math/sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
JSONL_PATH = Path(
    "logs/math/sharded-at0-ut0/(specific system prompt+no suffix)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)
JSONL_PATH_ADD = Path(
    "logs/math/sharded-at0-ut0/sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)

PT_SOURCES    = [PT_DIR, PT_DIR_ADD]
JSONL_SOURCES = [JSONL_PATH, JSONL_PATH_ADD]

LAYERS  = [12, 16, 20, 24, 28]
OUTPUT_DIR = PT_DIR.parent / "temporal_cosine_analysis_v4_balanced"
EPS     = 1e-12
MAX_TURN = 8    # only keep turn_3 … turn_8
MIN_N   = 3     # minimum samples per class to plot a point
N_BOOT  = 2000  # bootstrap resamples for CI
CI      = 90    # confidence interval %
SEED    = 44
EXACT_TOTAL_TURN_ONLY = False  # True: analyze turn_t with only convs whose total turns == t

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
    rng = np.random.default_rng(SEED)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


# ── Step 1: 读取 JSONL ────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Reading scores")
print("=" * 60)

score_map = {}
for jsonl_path in JSONL_SOURCES:
    loaded = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec     = json.loads(line)
            conv_id = rec.get("conv_id")
            score   = rec.get("score")
            if conv_id is None or score is None:
                continue
            score_map[conv_id] = int(float(score))
            loaded += 1
    print(f"  Loaded {loaded:4d} records from {jsonl_path.name}")

n_correct   = sum(v == 1 for v in score_map.values())
n_incorrect = sum(v == 0 for v in score_map.values())
print(f"  Total  : {len(score_map)}  (correct={n_correct}, incorrect={n_incorrect})\n")


# ── Step 2: 加载所有 .pt，按 conv 存储原始 hidden states ─────────────────────
# Structure:
#   conv_records[conv_id] = {
#       "score"  : int,
#       "goal_hs": list of np.ndarray (one per layer),   # shape (hidden,)
#       "turns"  : {
#           turn_label: list of np.ndarray (one per layer)
#       }
#   }
print("=" * 60)
print("Step 2: Loading .pt files")
print("=" * 60)

conv_records = {}
stats = dict(no_score=0, no_goal=0, too_few_turns=0, duplicate=0)
seen_conv_ids = set()

for pt_dir in PT_SOURCES:
    pt_files = sorted(pt_dir.glob("*.pt"))
    print(f"  {pt_dir.name}: {len(pt_files)} files")
    for pt_file in pt_files:
        conv_id = pt_file.stem

        if conv_id in seen_conv_ids:
            stats["duplicate"] += 1
            continue
        seen_conv_ids.add(conv_id)

        if conv_id not in score_map:
            stats["no_score"] += 1
            continue

        data    = torch.load(pt_file, map_location="cpu", weights_only=False)
        hs_list = data.get("hidden_states", [])
        hs_by_label = {entry["label"]: entry["hidden_states"] for entry in hs_list}

        if "goal" not in hs_by_label:
            stats["no_goal"] += 1
            continue

        turn_labels = sorted(
            [k for k in hs_by_label if k.startswith("turn_")],
            key=turn_sort_key,
        )
        # Need at least 3 turns so we can compute step_prev and step_now at turn_3
        if len(turn_labels) < 3:
            stats["too_few_turns"] += 1
            continue

        # Extract arrays: goal_hs[layer_i] = np.ndarray
        goal_hs = [hs_by_label["goal"][i].numpy() for i in range(len(LAYERS))]

        turns_data = {}
        for tlabel in turn_labels:
            turns_data[tlabel] = [
                hs_by_label[tlabel][i].numpy() for i in range(len(LAYERS))
            ]

        max_turn_in_conv = max(turn_index(tlabel) for tlabel in turn_labels)

        conv_records[conv_id] = {
            "score"  : score_map[conv_id],
            "goal_hs": goal_hs,
            "turns"  : turns_data,
            "max_turn": max_turn_in_conv,
        }

print(f"\n  Loaded conv records : {len(conv_records)}")
for k, v in stats.items():
    if v:
        print(f"  Skipped ({k:16s}): {v}")


# ── Step 3: 按 turn 做 conv-level balance ─────────────────────────────────────
# For each turn_t (3..MAX_TURN), collect convs that have data for
# turn_{t-2}, turn_{t-1}, turn_t.  Split by score, balance, record
# which conv_ids are kept for that turn.
print("\n" + "=" * 60)
print("Step 3: Conv-level balancing per turn")
print("=" * 60)

rng = np.random.default_rng(SEED)

# turn_conv_ids[turn_label] = {"correct": [...], "incorrect": [...]}
turn_conv_ids = {}

for t in range(3, MAX_TURN + 1):
    turn_now   = f"turn_{t}"
    turn_prev1 = f"turn_{t-1}"
    turn_prev2 = f"turn_{t-2}"

    correct_ids   = []
    incorrect_ids = []

    for conv_id, rec in conv_records.items():
        turns = rec["turns"]
        # Conv must have all three turns needed to compute metrics
        if turn_now not in turns or turn_prev1 not in turns or turn_prev2 not in turns:
            continue
        # Optional strict mode: for turn_t, keep only convs with total turns == t
        if EXACT_TOTAL_TURN_ONLY and rec["max_turn"] != t:
            continue
        if rec["score"] == 1:
            correct_ids.append(conv_id)
        else:
            incorrect_ids.append(conv_id)

    n_c = len(correct_ids)
    n_i = len(incorrect_ids)
    n_min = min(n_c, n_i)

    if n_min == 0:
        print(f"  {turn_now}: correct={n_c}, incorrect={n_i}  → SKIPPED (one class empty)")
        continue

    # Subsample majority class (deterministic via seed)
    if n_c > n_min:
        correct_ids = list(rng.choice(correct_ids, size=n_min, replace=False))
    if n_i > n_min:
        incorrect_ids = list(rng.choice(incorrect_ids, size=n_min, replace=False))

    turn_conv_ids[turn_now] = {
        "correct"  : correct_ids,
        "incorrect": incorrect_ids,
    }
    print(f"  {turn_now}: original correct={n_c}, incorrect={n_i}  "
          f"→ balanced to {n_min} each")


# ── Step 4: 计算 metrics（使用 balanced conv 集合）────────────────────────────
# layer_data[layer_idx][turn_label] = {
#     "cos_step": list[float],
#     "cos_goal": list[float],
#     "label"   : list[int],    # 1=correct, 0=incorrect
# }
print("\n" + "=" * 60)
print("Step 4: Computing metrics on balanced conv sets")
print("=" * 60)

layer_data = {i: {} for i in range(len(LAYERS))}

for turn_label, id_dict in turn_conv_ids.items():
    t = turn_index(turn_label)
    turn_prev1 = f"turn_{t-1}"
    turn_prev2 = f"turn_{t-2}"

    for cls_name, cls_score, conv_ids in [
        ("correct",   1, id_dict["correct"]),
        ("incorrect", 0, id_dict["incorrect"]),
    ]:
        for conv_id in conv_ids:
            rec   = conv_records[conv_id]
            turns = rec["turns"]

            for li in range(len(LAYERS)):
                vec_prev2 = turns[turn_prev2][li]
                vec_prev1 = turns[turn_prev1][li]
                vec_now   = turns[turn_label][li]
                vec_goal  = rec["goal_hs"][li]

                step_prev = vec_prev1 - vec_prev2
                step_now  = vec_now  - vec_prev1
                to_goal   = vec_goal - vec_prev1

                cos_step = cosine_similarity(step_now, step_prev)
                cos_goal = cosine_similarity(step_now, to_goal)

                cell = layer_data[li].setdefault(
                    turn_label,
                    {"cos_step": [], "cos_goal": [], "label": []},
                )
                cell["cos_step"].append(cos_step)
                cell["cos_goal"].append(cos_goal)
                cell["label"].append(cls_score)

# Sanity check
print("\n  Sanity check (layer 12, all turns):")
for turn_label in sorted(layer_data[0].keys(), key=turn_sort_key):
    labels = np.array(layer_data[0][turn_label]["label"])
    nc = int((labels == 1).sum())
    ni = int((labels == 0).sum())
    print(f"    {turn_label}: n_correct={nc}, n_incorrect={ni}")


# ── Step 5: CSV 汇总 ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_csv = OUTPUT_DIR / "temporal_cosine_summary.csv"
rows = []

for li, layer in enumerate(LAYERS):
    for turn_label in sorted(layer_data[li].keys(), key=turn_sort_key):
        entry  = layer_data[li][turn_label]
        labels   = np.array(entry["label"],    dtype=np.int32)
        cos_step = np.array(entry["cos_step"], dtype=np.float32)
        cos_goal = np.array(entry["cos_goal"], dtype=np.float32)

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        step_pos = mean_for_class(cos_step, labels, 1)
        step_neg = mean_for_class(cos_step, labels, 0)
        goal_pos = mean_for_class(cos_goal, labels, 1)
        goal_neg = mean_for_class(cos_goal, labels, 0)

        step_pos_lo, step_pos_hi = bootstrap_ci(cos_step[labels == 1]) if n_pos >= 2 else (float("nan"), float("nan"))
        step_neg_lo, step_neg_hi = bootstrap_ci(cos_step[labels == 0]) if n_neg >= 2 else (float("nan"), float("nan"))

        rows.append({
            "layer"                  : layer,
            "turn"                   : turn_label,
            "turn_idx"               : turn_index(turn_label),
            "n_correct"              : n_pos,
            "n_incorrect"            : n_neg,
            "mean_cos_step_correct"  : step_pos,
            "ci95_lo_correct"        : step_pos_lo,
            "ci95_hi_correct"        : step_pos_hi,
            "mean_cos_step_incorrect": step_neg,
            "ci95_lo_incorrect"      : step_neg_lo,
            "ci95_hi_incorrect"      : step_neg_hi,
            "gap_cos_step"           : step_pos - step_neg,
            "auc_cos_step"           : safe_auc(labels, cos_step),
            "mean_cos_goal_correct"  : goal_pos,
            "mean_cos_goal_incorrect": goal_neg,
            "gap_cos_goal"           : goal_pos - goal_neg,
            "auc_cos_goal"           : safe_auc(labels, cos_goal),
        })

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved CSV → {summary_csv}")


# ── Step 6: 画图 ──────────────────────────────────────────────────────────────
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

    for li, layer in enumerate(LAYERS):
        ax = axes[li]

        x_vals = []
        y_c, y_i         = [], []
        ci_c_lo, ci_c_hi = [], []
        ci_i_lo, ci_i_hi = [], []
        n_per_point      = []   # (n_correct, n_incorrect) per x

        for turn_label in sorted(layer_data[li].keys(), key=turn_sort_key):
            entry  = layer_data[li][turn_label]
            labels = np.array(entry["label"], dtype=np.int32)
            vals   = np.array(entry[metric_key], dtype=np.float32)

            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())

            if n_pos < MIN_N or n_neg < MIN_N:
                continue

            vals_c = vals[labels == 1]
            vals_i = vals[labels == 0]

            x_vals.append(turn_index(turn_label))
            y_c.append(float(vals_c.mean()))
            y_i.append(float(vals_i.mean()))
            n_per_point.append((n_pos, n_neg))

            lo, hi = bootstrap_ci(vals_c)
            ci_c_lo.append(lo); ci_c_hi.append(hi)

            lo, hi = bootstrap_ci(vals_i)
            ci_i_lo.append(lo); ci_i_hi.append(hi)

        if not x_vals:
            ax.set_title(f"Layer {layer}\n(no data)", fontsize=11)
            continue

        x = np.array(x_vals)

        ax.fill_between(x, ci_c_lo, ci_c_hi, color=COLOR_CORRECT,   alpha=0.15)
        ax.plot(x, y_c, "o-", color=COLOR_CORRECT,
                linewidth=1.8, markersize=5, label="correct")

        ax.fill_between(x, ci_i_lo, ci_i_hi, color=COLOR_INCORRECT, alpha=0.15)
        ax.plot(x, y_i, "^-", color=COLOR_INCORRECT,
                linewidth=1.8, markersize=5, label="incorrect")

        # Annotate n (should be equal after balancing)
        ylim = ax.get_ylim()
        for xi, (nc, ni) in zip(x_vals, n_per_point):
            ax.annotate(
                f"n={nc}",
                xy=(xi, ylim[0]),
                fontsize=6, ha="center", va="bottom", color="gray",
            )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("turn index")
        ax.set_xticks(x_vals)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.25)
        if li == 0:
            ax.set_ylabel("cosine similarity")
        if li == len(LAYERS) - 1:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{title}\n"
        f"({CI}% bootstrap CI shaded, balanced n/class, min n={MIN_N}, turns 3–{MAX_TURN})",
        fontsize=12, y=1.03,
    )
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {out_path}")


plot_metric(
    metric_key="cos_step",
    title="Temporal Curvature: cos(Δhidden_n, Δhidden_{n-1}) by correctness",
    out_name="temporal_cos_step_by_turn_v4.png",
)

plot_metric(
    metric_key="cos_goal",
    title="Goal Progress: cos(Δhidden_n, goal − hidden_{n-1}) by correctness",
    out_name="temporal_cos_goal_by_turn_v4.png",
)

print(f"\nAll done. Outputs → {OUTPUT_DIR}")