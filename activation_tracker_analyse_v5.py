"""
Temporal geometry analysis for hidden states (v5).

Key changes from v4:
- Conversations are grouped STRICTLY by total turn count (max_turn == K).
  Groups never share samples.
- Balancing is done within each total_turns group at the conv level.
- One figure per total_turns group per metric (cos_step / cos_goal).
  Each figure: 1 row × len(LAYERS) subplots, x-axis = turn index,
  two lines (correct / incorrect) with 90% bootstrap CI shading.
  Naming: cos_step_total{K}.png, cos_goal_total{K}.png
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
    "logs/hidden_states/code/sharded-at0-ut0_code_Qwen_Qwen2.5-14B-Instruct"
)
PT_DIR_ADD = Path(
    "logs/hidden_states/code/(add_v2)sharded-at0-ut0_code_Qwen_Qwen2.5-14B-Instruct"
)
JSONL_PATH = Path(
    "logs/code/sharded-at0-ut0/sharded-at0-ut0_code_Qwen_Qwen2.5-14B-Instruct.jsonl"
)
JSONL_PATH_ADD = Path(
    "logs/code/sharded-at0-ut0/(add_v2)sharded-at0-ut0_code_Qwen_Qwen2.5-14B-Instruct.jsonl"
)

PT_SOURCES    = [PT_DIR]
JSONL_SOURCES = [JSONL_PATH]

LAYERS         = [10, 20, 30, 40, 46]
OUTPUT_DIR     = PT_DIR.parent / "temporal_cosine_analysis_v5_per_total_turns"
EPS            = 1e-12
MIN_TOTAL_TURN = 3
MAX_TOTAL_TURN = 8
MIN_N          = 3
N_BOOT         = 2000
CI             = 90
SEED           = 47
BALANCE_CLASSES = False  # 是否在每个 total_turns 组内平衡正确/错误样本数量

COLOR_CORRECT   = "#2563EB"
COLOR_INCORRECT = "#DC2626"

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, values))


def bootstrap_ci(values: np.ndarray):
    if len(values) < 2:
        m = float(values.mean()) if len(values) == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(SEED)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    lo = np.percentile(boot_means, (100 - CI) / 2)
    hi = np.percentile(boot_means, 100 - (100 - CI) / 2)
    return float(lo), float(hi)


# ── Step 1: 读取 JSONL ────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Reading scores")
print("=" * 60)

score_map = {}
for jsonl_path in JSONL_SOURCES:
    loaded = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec     = json.loads(line)
            conv_id = rec.get("conv_id")
            score   = rec.get("score")
            
            # TODO
            if score is None:
                score = 0.0
            
            if conv_id is None or score is None:
                continue
            score_map[conv_id] = int(float(score))
            loaded += 1
    print(f"  Loaded {loaded:4d} records from {jsonl_path.name}")

n_correct   = sum(v == 1 for v in score_map.values())
n_incorrect = sum(v == 0 for v in score_map.values())
print(f"  Total: {len(score_map)}  (correct={n_correct}, incorrect={n_incorrect})\n")


# ── Step 2: 加载 .pt 文件 ─────────────────────────────────────────────────────
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

        data        = torch.load(pt_file, map_location="cpu", weights_only=False)
        hs_list     = data.get("hidden_states", [])
        hs_by_label = {entry["label"]: entry["hidden_states"] for entry in hs_list}

        if "goal" not in hs_by_label:
            stats["no_goal"] += 1
            continue

        turn_labels = sorted(
            [k for k in hs_by_label if k.startswith("turn_")],
            key=turn_sort_key,
        )
        if len(turn_labels) < 3:
            stats["too_few_turns"] += 1
            continue

        goal_hs    = [hs_by_label["goal"][i].numpy() for i in range(len(LAYERS))]
        turns_data = {
            tlabel: [hs_by_label[tlabel][i].numpy() for i in range(len(LAYERS))]
            for tlabel in turn_labels
        }
        max_turn = max(turn_index(tl) for tl in turn_labels)

        conv_records[conv_id] = {
            "score"   : score_map[conv_id],
            "goal_hs" : goal_hs,
            "turns"   : turns_data,
            "max_turn": max_turn,
        }

print(f"\n  Loaded conv records: {len(conv_records)}")
for k, v in stats.items():
    if v:
        print(f"  Skipped ({k:16s}): {v}")


# ── Step 3: 按 total_turns 分组 + 组内 balance + 计算 metrics ─────────────────
# group_data[total_t][li][turn_label] = {
#     "cos_step": list[float],
#     "cos_goal": list[float],
#     "label"   : list[int],
# }
print("\n" + "=" * 60)
print(
    "Step 3: Group by total_turns, "
    f"{'balance' if BALANCE_CLASSES else 'no-balance'}, compute metrics"
)
print("=" * 60)

rng = np.random.default_rng(SEED)
group_data = {}

for total_t in range(MIN_TOTAL_TURN, MAX_TOTAL_TURN + 1):
    correct_ids   = [cid for cid, r in conv_records.items()
                     if r["max_turn"] == total_t and r["score"] == 1]
    incorrect_ids = [cid for cid, r in conv_records.items()
                     if r["max_turn"] == total_t and r["score"] == 0]

    n_c, n_i = len(correct_ids), len(incorrect_ids)
    n_min    = min(n_c, n_i)

    if n_min < MIN_N:
        print(
            f"  total_turns={total_t}: correct={n_c}, incorrect={n_i} "
            f"→ SKIPPED (n_min={n_min} < {MIN_N})"
        )
        continue

    if BALANCE_CLASSES:
        if n_c > n_min:
            correct_ids = list(rng.choice(correct_ids, size=n_min, replace=False))
        if n_i > n_min:
            incorrect_ids = list(rng.choice(incorrect_ids, size=n_min, replace=False))
        print(
            f"  total_turns={total_t}: correct={n_c}, incorrect={n_i} "
            f"→ balanced to {n_min} each"
        )
    else:
        print(
            f"  total_turns={total_t}: correct={n_c}, incorrect={n_i} "
            "→ keep original counts (no balancing)"
        )

    # Computable turns: need vec at t, t-1, t-2 → earliest is turn_3
    available_turns = [f"turn_{t}" for t in range(3, total_t + 1)]

    layer_dict = {li: {} for li in range(len(LAYERS))}

    for cls_score, conv_ids in [(1, correct_ids), (0, incorrect_ids)]:
        for conv_id in conv_ids:
            rec   = conv_records[conv_id]
            turns = rec["turns"]

            for turn_label in available_turns:
                t          = turn_index(turn_label)
                turn_prev1 = f"turn_{t-1}"
                turn_prev2 = f"turn_{t-2}"

                if (turn_label not in turns
                        or turn_prev1 not in turns
                        or turn_prev2 not in turns):
                    continue

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

                    cell = layer_dict[li].setdefault(
                        turn_label,
                        {"cos_step": [], "cos_goal": [], "label": []},
                    )
                    cell["cos_step"].append(cos_step)
                    cell["cos_goal"].append(cos_goal)
                    cell["label"].append(cls_score)

    group_data[total_t] = layer_dict


# ── Step 4: CSV ───────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_csv = OUTPUT_DIR / "temporal_cosine_summary_v5.csv"
rows = []

for total_t, layer_dict in group_data.items():
    for li, layer in enumerate(LAYERS):
        for turn_label, entry in sorted(
            layer_dict[li].items(), key=lambda x: turn_sort_key(x[0])
        ):
            labels   = np.array(entry["label"],    dtype=np.int32)
            cos_step = np.array(entry["cos_step"], dtype=np.float32)
            cos_goal = np.array(entry["cos_goal"], dtype=np.float32)

            def _mean(arr, lbl, cls):
                m = arr[lbl == cls]
                return float(m.mean()) if len(m) else float("nan")

            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())

            step_pos_lo, step_pos_hi = (
                bootstrap_ci(cos_step[labels == 1]) if n_pos >= 2
                else (float("nan"), float("nan"))
            )
            step_neg_lo, step_neg_hi = (
                bootstrap_ci(cos_step[labels == 0]) if n_neg >= 2
                else (float("nan"), float("nan"))
            )

            rows.append({
                "total_turns"            : total_t,
                "layer"                  : layer,
                "turn"                   : turn_label,
                "turn_idx"               : turn_index(turn_label),
                "n_correct"              : n_pos,
                "n_incorrect"            : n_neg,
                "mean_cos_step_correct"  : _mean(cos_step, labels, 1),
                "ci_lo_correct"          : step_pos_lo,
                "ci_hi_correct"          : step_pos_hi,
                "mean_cos_step_incorrect": _mean(cos_step, labels, 0),
                "ci_lo_incorrect"        : step_neg_lo,
                "ci_hi_incorrect"        : step_neg_hi,
                "gap_cos_step"           : _mean(cos_step, labels, 1) - _mean(cos_step, labels, 0),
                "auc_cos_step"           : safe_auc(labels, cos_step),
                "mean_cos_goal_correct"  : _mean(cos_goal, labels, 1),
                "mean_cos_goal_incorrect": _mean(cos_goal, labels, 0),
                "gap_cos_goal"           : _mean(cos_goal, labels, 1) - _mean(cos_goal, labels, 0),
                "auc_cos_goal"           : safe_auc(labels, cos_goal),
            })

if rows:
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV → {summary_csv}")


# ── Step 5: One line-plot figure per total_turns group per metric ─────────────
print("\n" + "=" * 60)
print("Step 5: Generating per-group line plots")
print("=" * 60)


def plot_group(total_t: int, layer_dict: dict, metric_key: str,
               title_prefix: str, out_name: str):
    """
    One figure: 1 row × len(LAYERS) subplots.
    x-axis = turn index (turn_3 … turn_{total_t}).
    Two lines: correct (blue) / incorrect (red) with CI shading.
    """
    fig, axes = plt.subplots(
        1, len(LAYERS),
        figsize=(5 * len(LAYERS), 4.8),
        sharey=True,
    )
    if len(LAYERS) == 1:
        axes = [axes]

    for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
        x_vals           = []
        y_c, y_i         = [], []
        ci_c_lo, ci_c_hi = [], []
        ci_i_lo, ci_i_hi = [], []
        n_per_point      = []

        for turn_label in sorted(layer_dict[li].keys(), key=turn_sort_key):
            entry  = layer_dict[li][turn_label]
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
            ax.set_ylim(-1.0, 1.0)
            continue

        x = np.array(x_vals)

        ax.fill_between(x, ci_c_lo, ci_c_hi, color=COLOR_CORRECT,   alpha=0.15)
        ax.plot(x, y_c, "o-", color=COLOR_CORRECT,
                linewidth=1.8, markersize=5, label="correct")

        ax.fill_between(x, ci_i_lo, ci_i_hi, color=COLOR_INCORRECT, alpha=0.15)
        ax.plot(x, y_i, "^-", color=COLOR_INCORRECT,
                linewidth=1.8, markersize=5, label="incorrect")

        # n annotation at bottom of each x position
        for xi, (nc, ni) in zip(x_vals, n_per_point):
            n_text = f"n={nc}" if nc == ni else f"n={nc}/{ni}"
            ax.annotate(
                n_text,
                xy=(xi, -1.0),
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

    balance_note = "balanced n/class" if BALANCE_CLASSES else "no balancing"
    fig.suptitle(
        f"{title_prefix}  [total_turns = {total_t}]\n"
        f"({CI}% bootstrap CI shaded, {balance_note}, min n={MIN_N})",
        fontsize=12, y=1.03,
    )
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.name}")


for total_t, layer_dict in sorted(group_data.items()):
    plot_group(
        total_t      = total_t,
        layer_dict   = layer_dict,
        metric_key   = "cos_step",
        title_prefix = "Temporal Curvature: cos(Δhidden_n, Δhidden_{n-1})",
        out_name     = f"cos_step_total{total_t}.png",
    )
    plot_group(
        total_t      = total_t,
        layer_dict   = layer_dict,
        metric_key   = "cos_goal",
        title_prefix = "Goal Progress: cos(Δhidden_n, goal − hidden_{n-1})",
        out_name     = f"cos_goal_total{total_t}.png",
    )


# ── Step 6: v4-style overview (all convs merged, balance per turn) ────────────
# Mirrors v4 exactly: for each turn_t, pool ALL convs that have data up to
# turn_t (regardless of their total length), balance correct/incorrect at
# the conv level, then plot as a single line figure.
print("\n" + "=" * 60)
print(
    "Step 6: Generating v4-style overview plots (all turns merged, "
    f"{'balanced' if BALANCE_CLASSES else 'no-balance'})"
)
print("=" * 60)

# Re-run the v4 conv-level pipeline across all conv_records
overview_layer_data = {li: {} for li in range(len(LAYERS))}

for t in range(3, MAX_TOTAL_TURN + 1):
    turn_now   = f"turn_{t}"
    turn_prev1 = f"turn_{t-1}"
    turn_prev2 = f"turn_{t-2}"

    correct_ids   = []
    incorrect_ids = []

    for conv_id, rec in conv_records.items():
        turns = rec["turns"]
        if turn_now not in turns or turn_prev1 not in turns or turn_prev2 not in turns:
            continue
        if rec["score"] == 1:
            correct_ids.append(conv_id)
        else:
            incorrect_ids.append(conv_id)

    n_c, n_i = len(correct_ids), len(incorrect_ids)
    n_min    = min(n_c, n_i)

    if n_min < MIN_N:
        continue

    if BALANCE_CLASSES:
        if n_c > n_min:
            correct_ids = list(rng.choice(correct_ids, size=n_min, replace=False))
        if n_i > n_min:
            incorrect_ids = list(rng.choice(incorrect_ids, size=n_min, replace=False))
        print(f"  turn_{t}: correct={n_c}, incorrect={n_i} → balanced to {n_min} each")
    else:
        print(f"  turn_{t}: correct={n_c}, incorrect={n_i} → keep original counts (no balancing)")

    for cls_score, conv_ids in [(1, correct_ids), (0, incorrect_ids)]:
        for conv_id in conv_ids:
            rec   = conv_records[conv_id]
            turns = rec["turns"]
            for li in range(len(LAYERS)):
                vec_prev2 = turns[turn_prev2][li]
                vec_prev1 = turns[turn_prev1][li]
                vec_now   = turns[turn_now][li]
                vec_goal  = rec["goal_hs"][li]

                step_prev = vec_prev1 - vec_prev2
                step_now  = vec_now  - vec_prev1
                to_goal   = vec_goal - vec_prev1

                cos_step = cosine_similarity(step_now, step_prev)
                cos_goal = cosine_similarity(step_now, to_goal)

                cell = overview_layer_data[li].setdefault(
                    turn_now,
                    {"cos_step": [], "cos_goal": [], "label": []},
                )
                cell["cos_step"].append(cos_step)
                cell["cos_goal"].append(cos_goal)
                cell["label"].append(cls_score)


def plot_overview(metric_key: str, title: str, out_name: str):
    fig, axes = plt.subplots(
        1, len(LAYERS),
        figsize=(5 * len(LAYERS), 4.8),
        sharey=True,
    )
    if len(LAYERS) == 1:
        axes = [axes]

    for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
        x_vals           = []
        y_c, y_i         = [], []
        ci_c_lo, ci_c_hi = [], []
        ci_i_lo, ci_i_hi = [], []
        n_per_point      = []

        for turn_label in sorted(overview_layer_data[li].keys(), key=turn_sort_key):
            entry  = overview_layer_data[li][turn_label]
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
            ax.set_ylim(-1.0, 0.0)
            continue

        x = np.array(x_vals)

        ax.fill_between(x, ci_c_lo, ci_c_hi, color=COLOR_CORRECT,   alpha=0.15)
        ax.plot(x, y_c, "o-", color=COLOR_CORRECT,
                linewidth=1.8, markersize=5, label="correct")

        ax.fill_between(x, ci_i_lo, ci_i_hi, color=COLOR_INCORRECT, alpha=0.15)
        ax.plot(x, y_i, "^-", color=COLOR_INCORRECT,
                linewidth=1.8, markersize=5, label="incorrect")

        for xi, (nc, ni) in zip(x_vals, n_per_point):
            n_text = f"n={nc}" if nc == ni else f"n={nc}/{ni}"
            ax.annotate(
                n_text,
                xy=(xi, -1.0),
                fontsize=6, ha="center", va="bottom", color="gray",
            )

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("turn index")
        ax.set_xticks(x_vals)
        ax.set_ylim(-1.0, 0.0)
        ax.grid(alpha=0.25)
        if li == 0:
            ax.set_ylabel("cosine similarity")
        if li == len(LAYERS) - 1:
            ax.legend(loc="best", fontsize=9)

    overview_note = (
        "all convs balanced per turn"
        if BALANCE_CLASSES
        else "all eligible convs per turn (no balancing)"
    )
    fig.suptitle(
        f"{title}\n"
        f"({CI}% bootstrap CI shaded, {overview_note}, min n={MIN_N})",
        fontsize=12, y=1.03,
    )
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.name}")


plot_overview(
    metric_key = "cos_step",
    title      = "Temporal Curvature: cos(Δhidden_n, Δhidden_{n-1})  [all turns overview]",
    out_name   = "cos_step_overview_all.png",
)
plot_overview(
    metric_key = "cos_goal",
    title      = "Goal Progress: cos(Δhidden_n, goal − hidden_{n-1})  [all turns overview]",
    out_name   = "cos_goal_overview_all.png",
)

print(f"\nAll done. Outputs → {OUTPUT_DIR}")