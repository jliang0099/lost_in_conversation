"""
Activation Tracker Analysis v8 — Temporal Curvature by Failure Category (Math)
==============================================================================

Extends v6 to classify failures using the failure_classifier logic for math tasks.
Instead of binary correct/incorrect, we categorize into:
  1. SUCCESS                     - final answer correct (score=1)
  2. SAME_WRONG_DESPITE_HINTS    - ignores all hints
  3. EVOLVES_STAYS_WRONG         - changes but always wrong
  4. ATTEMPTED_BUT_UNRESOLVED    - tried to answer (1-4 evals) but ended in discussion
  5. NEVER_ATTEMPTED_ANSWER      - never entered answer submission (0 evals, score=None)
  6. REFUSES_TO_ANSWER           - refuses or silent

For each category, we compute temporal curvature metrics:
  - cos_step   (cosine between successive steps)
  - cos_goal   (cosine between step direction and goal direction)
  - step_mag   (magnitude of each step)

Figures produced
─────────────────
  fig_signature_by_category.png       Mean cos_step vs position-from-end, per category
  fig_distributions_by_category.png   Violin plots of trajectory features per category
    fig_auroc_by_total_turns.png        Direction-aware AUROC vs SUCCESS by total_turns
    auroc_by_total_turns.csv            AUROC of each error category vs SUCCESS by total_turns

CSVs produced
──────────────
  per_category_dynamics_v8.csv        Aggregated metrics per category/layer/turn
  per_conv_category_v8.csv            Category assignment + trajectory features per conv
    auroc_by_total_turns.csv            Per-turn-count AUROC summary vs SUCCESS
"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

# ── Configuration ──────────────────────────────────────────────────────────────

PT_DIR = Path(
    "logs/hidden_states/math/(400-698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
PT_DIR_ADD = Path(
    "logs/hidden_states/math/(40-103)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)

JSONL_PATH = Path(
    "logs/math/sharded-at0-ut0/(400-698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)
JSONL_PATH_ADD = Path(
    "logs/math/sharded-at0-ut0/(40-103)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)

PT_SOURCES    = [PT_DIR, PT_DIR_ADD]
JSONL_SOURCES = [JSONL_PATH, JSONL_PATH_ADD]

LAYERS      = [12, 16, 20, 24, 28]
OUTPUT_DIR  = PT_DIR.parent / "failure_category_analysis_v8_40-103_400-698"

EPS             = 1e-12
MIN_TOTAL_TURN  = 3
MAX_TOTAL_TURN  = 8
MIN_N           = 3
N_BOOT          = 2000
CI              = 90
SEED            = 44

COLORS = {
    "SUCCESS": "#10b981",
    "SAME_WRONG_DESPITE_HINTS": "#ef4444",
    "EVOLVES_STAYS_WRONG": "#f97316",
    "ATTEMPTED_BUT_UNRESOLVED": "#eab308",
    "NEVER_ATTEMPTED_ANSWER": "#a78bfa",
    "REFUSES_TO_ANSWER": "#8b5cf6",
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def turn_sort_key(label):
    if not label.startswith("turn_"):
        return (1, label)
    try:
        return (0, int(label.split("_", 1)[1]))
    except ValueError:
        return (0, label)


def turn_index(label):
    return int(label.split("_", 1)[1])


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_auc(labels, values):
    """AUROC; returns NaN if only one class is present."""
    labels = np.asarray(labels)
    values = np.asarray(values)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, values))


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        m = float(values.mean()) if len(values) == 1 else float("nan")
        return m, m
    rng  = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    lo = np.percentile(boot, (100 - CI) / 2)
    hi = np.percentile(boot, 100 - (100 - CI) / 2)
    return float(lo), float(hi)


# ── Failure Classification Logic (from failure_classifier.py) ────────────────

def _get_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content or "")


def _parse_log_content(msg: dict) -> Optional[dict]:
    """Return the parsed dict payload of a log message, or None."""
    if msg.get("role") != "log":
        return None
    c = msg.get("content", {})
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except json.JSONDecodeError:
            return None
    return c if isinstance(c, dict) else None


def _extract_trace_events(trace: list[dict]) -> dict:
    """Walk a trace and extract structured events into named lists."""
    verifications = []
    evaluations = []

    for msg in trace:
        c = _parse_log_content(msg)
        if c is not None:
            t = c.get("type", "")
            if t == "system-verification":
                verifications.append(c.get("response", {}))
            elif t == "answer-evaluation":
                evaluations.append(c)

    return dict(
        verifications=verifications,
        evaluations=evaluations,
    )


def classify_entry_math(entry: dict) -> str:
    """
    Classify a math task entry into one of:
      SUCCESS, SAME_WRONG_DESPITE_HINTS, EVOLVES_STAYS_WRONG,
      ATTEMPTED_BUT_UNRESOLVED, NEVER_ATTEMPTED_ANSWER, REFUSES_TO_ANSWER
    """
    score = entry.get("score")

    # If correct (score not in 0/0.0/None), it's success
    if score not in (0, 0.0, None):
        return "SUCCESS"

    trace = entry.get("trace", [])
    events = _extract_trace_events(trace)

    verifications = events["verifications"]
    evals = events["evaluations"]

    # No score field (score=None): never entered answer submission flow
    if score is None:
        return "NEVER_ATTEMPTED_ANSWER"

    # Score=0 cases
    if not verifications:
        return "ATTEMPTED_BUT_UNRESOLVED"

    last_resp_type = verifications[-1].get("response_type", "")

    if last_resp_type == "refuse":
        return "REFUSES_TO_ANSWER"

    # No evaluations (tried to discuss but never submitted)
    if not evals:
        return "ATTEMPTED_BUT_UNRESOLVED"

    # Ended in discussion state but has evaluations (tried but unresolved)
    if last_resp_type in ("discussion", "clarification", "interrogation"):
        return "ATTEMPTED_BUT_UNRESOLVED"

    # We have evaluations, check answer progression
    answers = [e.get("exact_answer", "") for e in evals]
    unique_answers = list(dict.fromkeys(answers))

    if len(unique_answers) == 1:
        return "SAME_WRONG_DESPITE_HINTS"

    return "EVOLVES_STAYS_WRONG"


# ── Step 1: Load and classify JSONL ────────────────────────────────────────────
print("=" * 70)
print("Step 1: Reading JSONL and classifying entries")
print("=" * 70)

category_map = {}  # conv_id -> category
score_map = {}     # conv_id -> score

for jp in JSONL_SOURCES:
    loaded = 0
    with open(jp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("conv_id")
            if cid is None:
                continue
            score_val = rec.get("score")
            # Include entries with score=0, score=0.0, or score=None (all failures)
            score_map[cid] = float(score_val) if score_val is not None else None
            category_map[cid] = classify_entry_math(rec)
            loaded += 1

    print(f"  {jp.name}: {loaded} records")

# Count categories
cat_counts = defaultdict(int)
for cat in category_map.values():
    cat_counts[cat] += 1

print(f"\n  Total: {len(category_map)} records")
for cat in sorted(cat_counts.keys()):
    print(f"    {cat}: {cat_counts[cat]}")
print()


# ── Step 2: Load .pt hidden-state files ────────────────────────────────────────
print("=" * 70)
print("Step 2: Loading .pt files")
print("=" * 70)

conv_records = {}
stats = dict(no_score=0, no_goal=0, too_few_turns=0, duplicate=0)
seen  = set()

for pt_dir in PT_SOURCES:
    pt_files = sorted(pt_dir.glob("*.pt"))
    print(f"  {pt_dir.name}: {len(pt_files)} files")
    for pt_file in pt_files:
        cid = pt_file.stem
        if cid in seen:
            stats["duplicate"] += 1
            continue
        seen.add(cid)
        if cid not in score_map:
            stats["no_score"] += 1
            continue

        data        = torch.load(pt_file, map_location="cpu", weights_only=False)
        hs_by_label = {e["label"]: e["hidden_states"] for e in data.get("hidden_states", [])}

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

        conv_records[cid] = {
            "score"   : score_map[cid] if score_map[cid] is not None else 0,  # Treat None as 0 for storage
            "category": category_map.get(cid, "UNKNOWN"),
            "goal_hs" : [hs_by_label["goal"][i].numpy() for i in range(len(LAYERS))],
            "turns"   : {tl: [hs_by_label[tl][i].numpy() for i in range(len(LAYERS))]
                         for tl in turn_labels},
            "max_turn": max(turn_index(tl) for tl in turn_labels),
        }

print(f"\n  Loaded: {len(conv_records)} conversations")
for k, v in stats.items():
    if v:
        print(f"  Skipped ({k}): {v}")


# ── Step 3: Group by category, compute metrics ────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Computing temporal metrics per category")
print("=" * 70)

# category_data[category][total_t][li][turn_label] -> {cos_step, cos_goal, step_mag}
category_data = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: {"cos_step": [], "cos_goal": [], "step_mag": []}
        )
    )
)

# rel_pool[category][rel_pos][li] -> {cos_step, cos_goal, step_mag, total_t}
rel_pool = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: {"cos_step": [], "cos_goal": [], "step_mag": [], "total_t": []}
        )
    )
)

for total_t in range(MIN_TOTAL_TURN, MAX_TOTAL_TURN + 1):
    # Group conversations by category at this total_t
    conv_by_cat = defaultdict(list)
    for cid, rec in conv_records.items():
        if rec["max_turn"] == total_t:
            conv_by_cat[rec["category"]].append(cid)
    
    for category, conv_ids in conv_by_cat.items():
        if len(conv_ids) < MIN_N:
            continue
        
        print(f"  total_turns={total_t}, category={category}: n={len(conv_ids)}")
        
        layer_dict = {li: {} for li in range(len(LAYERS))}
        available  = [f"turn_{t}" for t in range(3, total_t + 1)]
        
        for cid in conv_ids:
            rec   = conv_records[cid]
            turns = rec["turns"]
            
            for tl in available:
                t  = turn_index(tl)
                p1 = f"turn_{t-1}"
                p2 = f"turn_{t-2}"
                if tl not in turns or p1 not in turns or p2 not in turns:
                    continue
                
                for li in range(len(LAYERS)):
                    vp2   = turns[p2][li]
                    vp1   = turns[p1][li]
                    vn    = turns[tl][li]
                    vgoal = rec["goal_hs"][li]
                    
                    step_prev = vp1 - vp2   # Δh_{t-1}
                    step_now  = vn  - vp1   # Δh_t
                    to_goal   = vgoal - vp1
                    
                    cs = cosine_similarity(step_now, step_prev)
                    cg = cosine_similarity(step_now, to_goal)
                    sm = np.linalg.norm(step_now)
                    
                    # Per-group
                    cell = layer_dict[li].setdefault(
                        tl, {"cos_step": [], "cos_goal": [], "step_mag": []}
                    )
                    cell["cos_step"].append(cs)
                    cell["cos_goal"].append(cg)
                    cell["step_mag"].append(sm)
                    
                    # Relative-position pool
                    rel_pos = total_t - t
                    rp = rel_pool[category][rel_pos][li]
                    rp["cos_step"].append(cs)
                    rp["cos_goal"].append(cg)
                    rp["step_mag"].append(sm)
                    rp["total_t"].append(total_t)
        
        category_data[category][total_t] = layer_dict


# ── Step 4: Save per-category CSV ──────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
per_cat_csv = OUTPUT_DIR / "per_category_dynamics_v8.csv"

rows = []
for category in sorted(category_data.keys()):
    for total_t, layer_dict in sorted(category_data[category].items()):
        for li, layer_num in enumerate(LAYERS):
            for tl in sorted(layer_dict[li].keys(), key=turn_sort_key):
                entry = layer_dict[li][tl]
                cos_step = np.array(entry["cos_step"], dtype=np.float32)
                cos_goal = np.array(entry["cos_goal"], dtype=np.float32)
                step_mag = np.array(entry["step_mag"], dtype=np.float32)
                
                n = len(cos_step)
                m_cs, (ci_cs_lo, ci_cs_hi) = cos_step.mean(), bootstrap_ci(cos_step)
                m_cg, (ci_cg_lo, ci_cg_hi) = cos_goal.mean(), bootstrap_ci(cos_goal)
                m_sm, (ci_sm_lo, ci_sm_hi) = step_mag.mean(), bootstrap_ci(step_mag)
                
                rows.append({
                    "category": category,
                    "total_turns": total_t,
                    "layer": layer_num,
                    "turn_label": tl,
                    "n": n,
                    "cos_step_mean": f"{m_cs:.4f}",
                    "cos_step_ci_lo": f"{ci_cs_lo:.4f}",
                    "cos_step_ci_hi": f"{ci_cs_hi:.4f}",
                    "cos_goal_mean": f"{m_cg:.4f}",
                    "cos_goal_ci_lo": f"{ci_cg_lo:.4f}",
                    "cos_goal_ci_hi": f"{ci_cg_hi:.4f}",
                    "step_mag_mean": f"{m_sm:.4f}",
                    "step_mag_ci_lo": f"{ci_sm_lo:.4f}",
                    "step_mag_ci_hi": f"{ci_sm_hi:.4f}",
                })

with open(per_cat_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category", "total_turns", "layer", "turn_label", "n",
            "cos_step_mean", "cos_step_ci_lo", "cos_step_ci_hi",
            "cos_goal_mean", "cos_goal_ci_lo", "cos_goal_ci_hi",
            "step_mag_mean", "step_mag_ci_lo", "step_mag_ci_hi",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {per_cat_csv}")


# ── Step 5: Create visualization — cos_step signature by category ──────────────
print("\n" + "=" * 70)
print("Step 5: Creating visualizations")
print("=" * 70)

fig, axes = plt.subplots(len(LAYERS), 1, figsize=(12, 3 * len(LAYERS)))
if len(LAYERS) == 1:
    axes = [axes]

focus_layer_idx = 2  # layer 20

for ax_idx in range(len(LAYERS)):
    ax = axes[ax_idx]
    li_num = LAYERS[ax_idx]
    
    # Collect per-rel_pos statistics per category
    for category in sorted(rel_pool.keys()):
        rel_pos_list = []
        mean_cs_list = []
        ci_lo_list = []
        ci_hi_list = []
        
        for rel_pos in sorted(rel_pool[category].keys()):
            entry = rel_pool[category][rel_pos][ax_idx]
            if not entry["cos_step"]:
                continue
            cs_arr = np.array(entry["cos_step"], dtype=np.float32)
            m, (lo, hi) = cs_arr.mean(), bootstrap_ci(cs_arr)
            rel_pos_list.append(rel_pos)
            mean_cs_list.append(m)
            ci_lo_list.append(lo)
            ci_hi_list.append(hi)
        
        if not rel_pos_list:
            continue
        
        rel_pos_list = np.array(rel_pos_list)
        mean_cs_list = np.array(mean_cs_list)
        ci_lo_list = np.array(ci_lo_list)
        ci_hi_list = np.array(ci_hi_list)
        
        ax.plot(rel_pos_list, mean_cs_list, marker="o", label=category, 
                color=COLORS.get(category, "#000000"), linewidth=2)
        ax.fill_between(rel_pos_list, ci_lo_list, ci_hi_list, 
                        color=COLORS.get(category, "#000000"), alpha=0.15)
    
    ax.set_xlabel("Position from end (0=last turn)")
    ax.set_ylabel("cos_step")
    ax.set_title(f"Layer {li_num}")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

plt.tight_layout()
fig_path = OUTPUT_DIR / "fig_signature_by_category.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")


# ── Step 6: Violin plots of trajectory features per category ────────────────────

fig, axes = plt.subplots(3, len(LAYERS), figsize=(15, 10))

metrics = ["cos_step", "cos_goal", "step_mag"]
metric_idx_map = {m: i for i, m in enumerate(metrics)}

for metric_row, metric in enumerate(metrics):
    for layer_col in range(len(LAYERS)):
        ax = axes[metric_row, layer_col]
        li_num = LAYERS[layer_col]
        
        data_by_cat = {cat: [] for cat in sorted(rel_pool.keys())}
        
        for category in rel_pool.keys():
            all_values = []
            for rel_pos in rel_pool[category].keys():
                entry = rel_pool[category][rel_pos][layer_col]
                if metric in entry:
                    all_values.extend(entry[metric])
            data_by_cat[category] = all_values
        
        # Filter out empty categories
        categories_sorted = sorted(data_by_cat.keys())
        data_lists = [data_by_cat[cat] for cat in categories_sorted if data_by_cat[cat]]
        categories_filtered = [cat for cat in categories_sorted if data_by_cat[cat]]
        
        if not data_lists:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} @ Layer {li_num}")
            continue
        
        # Create violin plot
        positions = range(len(data_lists))
        colors_list = [COLORS.get(cat, "#000000") for cat in categories_filtered]
        
        parts = ax.violinplot(data_lists, positions=positions, widths=0.7, 
                              showmeans=True, showmedians=True)
        
        for pc, color in zip(parts["bodies"], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(categories_filtered, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} @ Layer {li_num}")
        ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig_path = OUTPUT_DIR / "fig_distributions_by_category.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")


# ── Step 7: Per-conversation summary CSV ───────────────────────────────────────

per_conv_csv = OUTPUT_DIR / "per_conv_category_v8.csv"
conv_rows = []

for cid, rec in sorted(conv_records.items()):
    category = rec["category"]
    score = rec["score"]
    max_turn = rec["max_turn"]
    
    # Compute per-conversation trajectory features
    turns_list = sorted(
        [k for k in rec["turns"].keys() if k.startswith("turn_")],
        key=turn_sort_key,
    )
    
    # Use layer 20 (index 2 for LAYERS=[12,16,20,24,28])
    li = 2
    cos_steps = []
    cos_goals = []
    step_mags = []
    
    for i in range(1, len(turns_list)):
        t_curr = turns_list[i]
        t_prev = turns_list[i-1]
        if i >= 2:
            t_prev2 = turns_list[i-2]
            vp2 = rec["turns"][t_prev2][li]
        else:
            vp2 = rec["turns"][t_prev][li]
        
        vp1 = rec["turns"][t_prev][li]
        vn = rec["turns"][t_curr][li]
        vgoal = rec["goal_hs"][li]
        
        step_prev = vp1 - vp2
        step_now = vn - vp1
        to_goal = vgoal - vp1
        
        cs = cosine_similarity(step_now, step_prev)
        cg = cosine_similarity(step_now, to_goal)
        sm = np.linalg.norm(step_now)
        
        cos_steps.append(cs)
        cos_goals.append(cg)
        step_mags.append(sm)
    
    mean_cs = np.mean(cos_steps) if cos_steps else float("nan")
    mean_cg = np.mean(cos_goals) if cos_goals else float("nan")
    mean_sm = np.mean(step_mags) if step_mags else float("nan")
    
    conv_rows.append({
        "conv_id": cid,
        "category": category,
        "score": score,
        "max_turn": max_turn,
        "mean_cos_step": f"{mean_cs:.4f}",
        "mean_cos_goal": f"{mean_cg:.4f}",
        "mean_step_mag": f"{mean_sm:.4f}",
    })

with open(per_conv_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["conv_id", "category", "score", "max_turn", "mean_cos_step", "mean_cos_goal", "mean_step_mag"],
    )
    writer.writeheader()
    writer.writerows(conv_rows)

print(f"Saved: {per_conv_csv}")


# ── Step 8: AUROC vs SUCCESS by total_turns ────────────────────────────────────

print("\n" + "=" * 70)
print("Step 8: Computing AUROC vs SUCCESS by total_turns")
print("=" * 70)

error_categories = [
    "SAME_WRONG_DESPITE_HINTS",
    "EVOLVES_STAYS_WRONG",
    "ATTEMPTED_BUT_UNRESOLVED",
    "NEVER_ATTEMPTED_ANSWER",
]
metric_specs = [
    ("mean_cos_step", "cos_step"),
    ("mean_cos_goal", "cos_goal"),
    ("mean_step_mag", "step_mag"),
]

auroc_rows = []
grouped_summary = defaultdict(lambda: defaultdict(dict))
total_turn_groups = sorted({int(row["max_turn"]) for row in conv_rows})

for total_turns in total_turn_groups:
    group_rows = [row for row in conv_rows if int(row["max_turn"]) == total_turns]
    success_rows = [row for row in group_rows if row["category"] == "SUCCESS"]

    print(f"\nTotal turns = {total_turns}")
    print(f"  SUCCESS samples: {len(success_rows)}")

    for error_category in error_categories:
        error_rows = [row for row in group_rows if row["category"] == error_category]
        print(f"  {error_category}: {len(error_rows)}")

        if len(success_rows) < MIN_N or len(error_rows) < MIN_N:
            print("    skipped: need at least MIN_N samples in both classes")
            continue

        for metric, display_name in metric_specs:
            labels = [1] * len(success_rows) + [0] * len(error_rows)
            values = [float(row[metric]) for row in success_rows + error_rows]
            auroc = safe_auc(labels, values)
            auroc_dir = max(auroc, 1.0 - auroc)
            direction = "SUCCESS_higher" if auroc >= 0.5 else "ERROR_higher"

            grouped_summary[total_turns][error_category][display_name] = {
                "auroc": auroc,
                "auroc_directional": auroc_dir,
                "direction": direction,
                "n_success": len(success_rows),
                "n_error": len(error_rows),
            }

            auroc_rows.append({
                "total_turns": total_turns,
                "error_category": error_category,
                "metric": display_name,
                "auroc": f"{auroc:.4f}",
                "auroc_directional": f"{auroc_dir:.4f}",
                "direction": direction,
                "n_success": len(success_rows),
                "n_error": len(error_rows),
                "n_total": len(success_rows) + len(error_rows),
            })

            print(
                f"    {display_name:15s} AUROC={auroc:.4f}  "
                f"dir_AUROC={auroc_dir:.4f}  {direction}"
            )

auroc_csv = OUTPUT_DIR / "auroc_by_total_turns.csv"
with open(auroc_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "total_turns",
            "error_category",
            "metric",
            "auroc",
            "auroc_directional",
            "direction",
            "n_success",
            "n_error",
            "n_total",
        ],
    )
    writer.writeheader()
    writer.writerows(auroc_rows)

print(f"\nSaved: {auroc_csv}")

print("\n" + "=" * 70)
print("AUROC summary by total_turns")
print("=" * 70)
for total_turns in total_turn_groups:
    print(f"\nTotal turns = {total_turns}")
    print(f"{'Error Category':<35} | {'cos_step':>8} | {'cos_goal':>8} | {'step_mag':>8}")
    print("-" * 80)
    for error_category in error_categories:
        row = grouped_summary[total_turns].get(error_category)
        if not row:
            continue
        print(
            f"{error_category:<35} | "
            f"{row['cos_step']['auroc_directional']:8.4f} | "
            f"{row['cos_goal']['auroc_directional']:8.4f} | "
            f"{row['step_mag']['auroc_directional']:8.4f}"
        )


# ── Step 9: Direction-aware AUROC figures by total_turns ──────────────────────

print("\n" + "=" * 70)
print("Step 9: Creating AUROC figures by total_turns")
print("=" * 70)

metric_order = ["cos_step", "cos_goal", "step_mag"]

for total_turns in total_turn_groups:
    group_rows = grouped_summary.get(total_turns, {})
    if not group_rows:
        continue

    fig, axes = plt.subplots(1, len(metric_order), figsize=(5 * len(metric_order), 4), sharey=True)
    if len(metric_order) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        metric_vals = []
        metric_labels = []
        metric_colors = []

        for error_category in error_categories:
            row = group_rows.get(error_category, {}).get(metric)
            if row is None:
                continue
            metric_labels.append(error_category)
            metric_vals.append(row["auroc_directional"])
            metric_colors.append(COLORS.get(error_category, "#666666"))

        if not metric_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric)
            ax.set_xlabel("Directional AUROC")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="x", alpha=0.25)
            continue

        y_pos = np.arange(len(metric_vals))
        ax.barh(y_pos, metric_vals, color=metric_colors, alpha=0.9)

        for y, value in zip(y_pos, metric_vals):
            ax.text(value + 0.01, y, f"{value:.3f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_labels)
        ax.set_xlim(0.45, 1.0)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Directional AUROC")
        ax.grid(True, axis="x", alpha=0.25)

    axes[0].set_ylabel("Error category")
    fig.suptitle(f"Direction-aware AUROC vs SUCCESS (total_turns={total_turns})", y=1.02)
    plt.tight_layout()
    auroc_fig = OUTPUT_DIR / f"fig_auroc_by_total_turns_{total_turns}.png"
    plt.savefig(auroc_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {auroc_fig}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
