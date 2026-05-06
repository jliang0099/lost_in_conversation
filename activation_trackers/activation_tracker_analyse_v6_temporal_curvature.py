"""
Temporal geometry analysis for hidden states (v6).

Key additions over v5
─────────────────────
1.  Relative-position reindexing
    rel_pos = total_t − turn_idx   (0 = last turn, 1 = penultimate, …)
    Pools identical positions across all total_turns groups so the
    "position-from-end" hypothesis can be tested directly.

    Background: v5 CSV showed that cos_step AUROC peaks at turn_4 for 5-turn
    conversations (0.745 at layer 46) but reverses to 0.198 for 6-turn
    conversations at the same absolute turn. This analysis tests whether
    the effect is driven by position-from-end or absolute turn number.

2.  AUROC heatmap  (total_turns × rel_pos per layer)
    One figure for cos_step, one for cos_goal.
    Makes the direction-reversal between short and long conversations explicit:
    green = AUROC > 0.5 (higher metric → correct), red = reversed.

3.  Pooled relative-position plots (two-row figure per metric)
    Row 0: pooled mean ± CI curves (correct vs incorrect)
    Row 1: AUROC vs rel_pos, one line per total_turns group + pooled line
    Reveals whether the penultimate-turn peak persists when groups are merged.

4.  Per-group line plots from v5 retained unchanged.

Two new CSVs:
  per_group_summary_v6.csv          — same schema as v5 + rel_pos column
  relative_position_summary_v6.csv  — pooled across groups at each rel_pos
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_auc_score

# ── Configuration ─────────────────────────────────────────────────────────────
# Swap the commented/uncommented blocks to switch between tasks.

PT_DIR = Path(
    "logs/hidden_states/math/(452-698)snowball-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
PT_DIR_ADD = Path(
    "logs/hidden_states/math/(698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)

JSONL_PATH = Path(
    "logs/math/snowball-at0-ut0/(452-698)snowball-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)
JSONL_PATH_ADD = Path(
    "logs/math/sharded-at0-ut0/(698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)

PT_SOURCES    = [PT_DIR]
JSONL_SOURCES = [JSONL_PATH]

LAYERS         = [12, 16, 20, 24, 28]
OUTPUT_DIR     = PT_DIR.parent / "temporal_cosine_analysis_v6"
EPS            = 1e-12
MIN_TOTAL_TURN = 3
MAX_TOTAL_TURN = 8
MIN_N          = 3       # minimum samples per class for a cell to be included
SMALL_N_WARN   = 15      # cells below this get a ⚠ annotation in the heatmap
N_BOOT         = 2000
CI             = 90
SEED           = 44

COLOR_CORRECT   = "#2563EB"
COLOR_INCORRECT = "#DC2626"
GROUP_COLORS    = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728", 7: "#9467bd"}


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


# ── Step 1: Load JSONL ────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Reading JSONL scores")
print("=" * 60)

score_map = {}
for jp in JSONL_SOURCES:
    loaded = 0
    with open(jp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec   = json.loads(line)
            cid   = rec.get("conv_id")
            score = rec.get("score")
            if cid is None:
                continue
            score_map[cid] = int(float(score if score is not None else 0))
            loaded += 1
    print(f"  {jp.name}: {loaded} records")

n_c = sum(v == 1 for v in score_map.values())
n_i = sum(v == 0 for v in score_map.values())
print(f"  Total: {len(score_map)}  (correct={n_c}, incorrect={n_i})\n")


# ── Step 2: Load .pt hidden-state files ───────────────────────────────────────
print("=" * 60)
print("Step 2: Loading .pt files")
print("=" * 60)

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
            "score"   : score_map[cid],
            "goal_hs" : [hs_by_label["goal"][i].numpy() for i in range(len(LAYERS))],
            "turns"   : {tl: [hs_by_label[tl][i].numpy() for i in range(len(LAYERS))]
                         for tl in turn_labels},
            "max_turn": max(turn_index(tl) for tl in turn_labels),
        }

print(f"\n  Loaded: {len(conv_records)} conversations")
for k, v in stats.items():
    if v:
        print(f"  Skipped ({k}): {v}")


# ── Step 3: Group by total_turns, compute metrics ─────────────────────────────
# Also accumulate rel_pool for the relative-position analysis.
#
# rel_pos = total_t - turn_idx
#   rel_pos 0 → last turn of the conversation
#   rel_pos 1 → penultimate
#   …
#
# group_data[total_t][li][turn_label] → {cos_step, cos_goal, label}
# rel_pool[rel_pos][li]               → {cos_step, cos_goal, label, total_t}
print("\n" + "=" * 60)
print("Step 3: Computing cos_step and cos_goal (per group + relative position)")
print("=" * 60)

group_data = {}
rel_pool   = defaultdict(
    lambda: defaultdict(lambda: {"cos_step": [], "cos_goal": [], "label": [], "total_t": []})
)

for total_t in range(MIN_TOTAL_TURN, MAX_TOTAL_TURN + 1):
    correct_ids   = [cid for cid, r in conv_records.items()
                     if r["max_turn"] == total_t and r["score"] == 1]
    incorrect_ids = [cid for cid, r in conv_records.items()
                     if r["max_turn"] == total_t and r["score"] == 0]
    n_c, n_i = len(correct_ids), len(incorrect_ids)
    n_min    = min(n_c, n_i)

    if n_min < MIN_N:
        print(f"  total_turns={total_t}: n_c={n_c}, n_i={n_i} → SKIPPED (n < {MIN_N})")
        continue

    warn = "  ⚠ small sample" if n_min < SMALL_N_WARN else ""
    print(f"  total_turns={total_t}: n_correct={n_c}, n_incorrect={n_i}{warn}")

    layer_dict = {li: {} for li in range(len(LAYERS))}
    available  = [f"turn_{t}" for t in range(3, total_t + 1)]

    for cls_score, conv_ids in [(1, correct_ids), (0, incorrect_ids)]:
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
                    to_goal   = vgoal - vp1 # direction from h_{t-1} toward goal

                    cs = cosine_similarity(step_now, step_prev)
                    cg = cosine_similarity(step_now, to_goal)

                    # Per-group
                    cell = layer_dict[li].setdefault(
                        tl, {"cos_step": [], "cos_goal": [], "label": []}
                    )
                    cell["cos_step"].append(cs)
                    cell["cos_goal"].append(cg)
                    cell["label"].append(cls_score)

                    # Relative-position pool
                    rel_pos = total_t - t
                    rp = rel_pool[rel_pos][li]
                    rp["cos_step"].append(cs)
                    rp["cos_goal"].append(cg)
                    rp["label"].append(cls_score)
                    rp["total_t"].append(total_t)

    group_data[total_t] = layer_dict


# ── Step 4: Save per-group CSV (v5 schema + rel_pos column) ───────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
per_group_csv = OUTPUT_DIR / "per_group_summary_v6.csv"
rows = []

for total_t, layer_dict in sorted(group_data.items()):
    for li, layer in enumerate(LAYERS):
        for tl, entry in sorted(layer_dict[li].items(), key=lambda x: turn_sort_key(x[0])):
            labels   = np.array(entry["label"],    dtype=np.int32)
            cos_step = np.array(entry["cos_step"], dtype=np.float32)
            cos_goal = np.array(entry["cos_goal"], dtype=np.float32)
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            rel_pos = total_t - turn_index(tl)

            def _m(arr, lbl, cls):
                v = arr[lbl == cls]
                return float(v.mean()) if len(v) else float("nan")

            lo_c, hi_c = (bootstrap_ci(cos_step[labels == 1]) if n_pos >= 2
                          else (float("nan"), float("nan")))
            lo_i, hi_i = (bootstrap_ci(cos_step[labels == 0]) if n_neg >= 2
                          else (float("nan"), float("nan")))

            rows.append({
                "total_turns"            : total_t,
                "rel_pos"                : rel_pos,
                "layer"                  : layer,
                "turn"                   : tl,
                "turn_idx"               : turn_index(tl),
                "n_correct"              : n_pos,
                "n_incorrect"            : n_neg,
                "mean_cos_step_correct"  : _m(cos_step, labels, 1),
                "ci_lo_correct"          : lo_c,
                "ci_hi_correct"          : hi_c,
                "mean_cos_step_incorrect": _m(cos_step, labels, 0),
                "ci_lo_incorrect"        : lo_i,
                "ci_hi_incorrect"        : hi_i,
                "gap_cos_step"           : _m(cos_step, labels, 1) - _m(cos_step, labels, 0),
                "auc_cos_step"           : safe_auc(labels, cos_step),
                "mean_cos_goal_correct"  : _m(cos_goal, labels, 1),
                "mean_cos_goal_incorrect": _m(cos_goal, labels, 0),
                "gap_cos_goal"           : _m(cos_goal, labels, 1) - _m(cos_goal, labels, 0),
                "auc_cos_goal"           : safe_auc(labels, cos_goal),
            })

with open(per_group_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved per-group CSV → {per_group_csv}")


# ── Step 5: Save relative-position CSV ────────────────────────────────────────
rel_csv  = OUTPUT_DIR / "relative_position_summary_v6.csv"
rel_rows = []

for rel_pos in sorted(rel_pool.keys()):
    for li, layer in enumerate(LAYERS):
        cell     = rel_pool[rel_pos][li]
        labels   = np.array(cell["label"],    dtype=np.int32)
        cos_step = np.array(cell["cos_step"], dtype=np.float32)
        cos_goal = np.array(cell["cos_goal"], dtype=np.float32)
        total_ts = np.array(cell["total_t"],  dtype=np.int32)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if min(n_pos, n_neg) < MIN_N:
            continue

        def _m(arr, lbl, cls):
            v = arr[lbl == cls]
            return float(v.mean()) if len(v) else float("nan")

        rel_rows.append({
            "rel_pos"                : rel_pos,
            "layer"                  : layer,
            "n_correct"              : n_pos,
            "n_incorrect"            : n_neg,
            "groups_included"        : ",".join(str(t) for t in sorted(set(total_ts.tolist()))),
            "mean_cos_step_correct"  : _m(cos_step, labels, 1),
            "mean_cos_step_incorrect": _m(cos_step, labels, 0),
            "gap_cos_step"           : _m(cos_step, labels, 1) - _m(cos_step, labels, 0),
            "auc_cos_step"           : safe_auc(labels, cos_step),
            "mean_cos_goal_correct"  : _m(cos_goal, labels, 1),
            "mean_cos_goal_incorrect": _m(cos_goal, labels, 0),
            "gap_cos_goal"           : _m(cos_goal, labels, 1) - _m(cos_goal, labels, 0),
            "auc_cos_goal"           : safe_auc(labels, cos_goal),
        })

with open(rel_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rel_rows[0].keys()))
    writer.writeheader()
    writer.writerows(rel_rows)
print(f"Saved relative-position CSV → {rel_csv}")

# Print summary at a focus layer
FOCUS_LI = len(LAYERS) // 2
print(f"\nRelative-position AUROC summary (layer={LAYERS[FOCUS_LI]}, pooled across groups):")
print(f"  rel_pos=0 is the last turn;  higher = earlier in the conversation")
print(f"{'rel_pos':>8}  {'n_c':>5}  {'n_i':>5}  {'groups':>12}  {'auc_step':>9}  {'auc_goal':>9}")
print("  " + "─" * 58)
for r in [r for r in rel_rows if r["layer"] == LAYERS[FOCUS_LI]]:
    flag = "  ⚠ small" if min(r["n_correct"], r["n_incorrect"]) < SMALL_N_WARN else ""
    print(f"  {r['rel_pos']:>6}  {r['n_correct']:>5}  {r['n_incorrect']:>5}  "
          f"  {r['groups_included']:>10}  {r['auc_cos_step']:>9.4f}  "
          f"{r['auc_cos_goal']:>9.4f}{flag}")


# ── Step 6: Per-group line plots (identical to v5) ────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Per-group line plots (v5-style)")
print("=" * 60)


def plot_group(total_t, layer_dict, metric_key, title_prefix, out_name, ylim=(-1., 0.)):
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 4.8), sharey=True)
    if len(LAYERS) == 1:
        axes = [axes]

    for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
        x_vals, y_c, y_i = [], [], []
        clo, chi, ilo, ihi, npts = [], [], [], [], []

        for tl in sorted(layer_dict[li].keys(), key=turn_sort_key):
            entry  = layer_dict[li][tl]
            labels = np.array(entry["label"],      dtype=np.int32)
            vals   = np.array(entry[metric_key],   dtype=np.float32)
            n_pos  = int((labels == 1).sum())
            n_neg  = int((labels == 0).sum())
            if n_pos < MIN_N or n_neg < MIN_N:
                continue
            x_vals.append(turn_index(tl))
            y_c.append(float(vals[labels == 1].mean()))
            y_i.append(float(vals[labels == 0].mean()))
            npts.append((n_pos, n_neg))
            lo, hi = bootstrap_ci(vals[labels == 1]); clo.append(lo); chi.append(hi)
            lo, hi = bootstrap_ci(vals[labels == 0]); ilo.append(lo); ihi.append(hi)

        if not x_vals:
            ax.set_title(f"Layer {layer}\n(no data)", fontsize=11)
            ax.set_ylim(ylim)
            continue

        x = np.array(x_vals)
        ax.fill_between(x, clo, chi, color=COLOR_CORRECT,   alpha=0.15)
        ax.plot(x, y_c, "o-", color=COLOR_CORRECT,   lw=1.8, ms=5, label="correct")
        ax.fill_between(x, ilo, ihi, color=COLOR_INCORRECT, alpha=0.15)
        ax.plot(x, y_i, "^-", color=COLOR_INCORRECT, lw=1.8, ms=5, label="incorrect")

        for xi, (nc, ni) in zip(x_vals, npts):
            ax.annotate(
                f"n={nc}" if nc == ni else f"n={nc}/{ni}",
                xy=(xi, ylim[0]), fontsize=6, ha="center", va="bottom", color="gray",
            )

        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("turn index")
        ax.set_xticks(x_vals)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.25)
        if li == 0:
            ax.set_ylabel("cosine similarity")
        if li == len(LAYERS) - 1:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"{title_prefix}  [total_turns={total_t}]\n({CI}% bootstrap CI, min n={MIN_N})",
        fontsize=12, y=1.03,
    )
    plt.savefig(OUTPUT_DIR / out_name, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_name}")


for total_t, layer_dict in sorted(group_data.items()):
    plot_group(total_t, layer_dict, "cos_step",
               "Temporal Curvature: cos(Δhₜ, Δhₜ₋₁)",
               f"cos_step_total{total_t}.png", ylim=(-1., 0.))
    plot_group(total_t, layer_dict, "cos_goal",
               "Goal Alignment: cos(Δhₜ, goal − hₜ₋₁)",
               f"cos_goal_total{total_t}.png", ylim=(-1., 0.))


# ── Step 7: AUROC heatmap (total_turns × rel_pos per layer) ───────────────────
# Green  = AUROC > 0.5  → higher metric value predicts correct
# Red    = AUROC < 0.5  → higher metric value predicts incorrect (reversal)
# Centre = 0.5          → no discrimination
print("\n" + "=" * 60)
print("Step 7: AUROC heatmaps (total_turns × rel_pos)")
print("=" * 60)

all_total_t = sorted(group_data.keys())
max_rel_pos = max(total_t - 3 for total_t in all_total_t)
all_rel_pos = list(range(0, max_rel_pos + 1))

for raw_key, metric_label, fname in [
    ("cos_step", "cos(Δhₜ, Δhₜ₋₁)  [temporal curvature]",  "heatmap_auc_cos_step.png"),
    ("cos_goal", "cos(Δhₜ, goal−hₜ₋₁)  [goal alignment]",   "heatmap_auc_cos_goal.png"),
]:
    fig, axes = plt.subplots(
        1, len(LAYERS),
        figsize=(4.5 * len(LAYERS), 2.0 + 0.6 * len(all_total_t)),
    )
    if len(LAYERS) == 1:
        axes = [axes]

    norm = TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.9)

    for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
        mat   = np.full((len(all_total_t), len(all_rel_pos)), np.nan)
        annot = [["" for _ in all_rel_pos] for _ in all_total_t]

        for ti, total_t in enumerate(all_total_t):
            if total_t not in group_data:
                continue
            for tl, entry in group_data[total_t][li].items():
                labels = np.array(entry["label"],    dtype=np.int32)
                vals   = np.array(entry[raw_key],    dtype=np.float32)
                n_pos  = int((labels == 1).sum())
                n_neg  = int((labels == 0).sum())
                if min(n_pos, n_neg) < MIN_N:
                    continue
                auc     = safe_auc(labels, vals)
                rel_pos = total_t - turn_index(tl)
                if rel_pos not in all_rel_pos:
                    continue
                ri = all_rel_pos.index(rel_pos)
                mat[ti, ri] = auc
                n_min = min(n_pos, n_neg)
                warn  = f"\n⚠n={n_min}" if n_min < SMALL_N_WARN else f"\nn={n_min}"
                annot[ti][ri] = f"{auc:.2f}{warn}"

        im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

        for ti in range(len(all_total_t)):
            for ri in range(len(all_rel_pos)):
                if annot[ti][ri]:
                    ax.text(ri, ti, annot[ti][ri], ha="center", va="center",
                            fontsize=7, color="black")

        # x-axis: rel_pos; label as "−N" for penultimate etc., "last" for 0
        ax.set_xticks(range(len(all_rel_pos)))
        ax.set_xticklabels(
            [f"last" if rp == 0 else f"−{rp}" for rp in all_rel_pos],
            fontsize=8,
        )
        ax.set_xlabel("position from end", fontsize=8)
        ax.set_yticks(range(len(all_total_t)))
        ax.set_yticklabels([f"T={t}" for t in all_total_t], fontsize=8)
        ax.set_title(f"Layer {layer}", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle(
        f"AUROC heatmap: {metric_label}\n"
        f"green > 0.5 (higher→correct)   red < 0.5 (reversed)   ⚠ = n < {SMALL_N_WARN}/class",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


# ── Step 8: Pooled relative-position plots ────────────────────────────────────
# Two-row figure per metric:
#   Row 0 — pooled mean ± CI (correct vs incorrect), x = rel_pos
#   Row 1 — AUROC vs rel_pos; one coloured line per total_turns group + black pooled line
#
# Interpretation key:
#   If all group lines in Row 1 peak at the same rel_pos → position-from-end drives the signal.
#   If lines diverge (e.g. peak at rel_pos=1 for T=5 but at rel_pos=2 for T=6) → absolute
#   turn number matters more than relative position.
print("\n" + "=" * 60)
print("Step 8: Pooled relative-position plots")
print("=" * 60)

for raw_key, metric_label, fname_prefix in [
    ("cos_step", "cos(Δhₜ, Δhₜ₋₁)",    "relpos_cos_step"),
    ("cos_goal", "cos(Δhₜ, goal−hₜ₋₁)", "relpos_cos_goal"),
]:
    fig, axes = plt.subplots(
        2, len(LAYERS),
        figsize=(5 * len(LAYERS), 9),
        gridspec_kw={"hspace": 0.40},
    )

    for li, layer in enumerate(LAYERS):
        ax_mean = axes[0][li]
        ax_auc  = axes[1][li]

        # ── Row 0: pooled mean ± CI ───────────────────────────────────────────
        x_vals, y_c, y_i = [], [], []
        clo, chi, ilo, ihi = [], [], [], []

        for rp in sorted(rel_pool.keys()):
            cell   = rel_pool[rp][li]
            labels = np.array(cell["label"],   dtype=np.int32)
            vals   = np.array(cell[raw_key],   dtype=np.float32)
            n_pos  = int((labels == 1).sum())
            n_neg  = int((labels == 0).sum())
            if min(n_pos, n_neg) < MIN_N:
                continue
            x_vals.append(rp)
            y_c.append(float(vals[labels == 1].mean()))
            y_i.append(float(vals[labels == 0].mean()))
            lo, hi = bootstrap_ci(vals[labels == 1]); clo.append(lo); chi.append(hi)
            lo, hi = bootstrap_ci(vals[labels == 0]); ilo.append(lo); ihi.append(hi)

        if x_vals:
            x = np.array(x_vals)
            ax_mean.fill_between(x, clo, chi, color=COLOR_CORRECT,   alpha=0.18)
            ax_mean.plot(x, y_c, "o-", color=COLOR_CORRECT,   lw=2, ms=5, label="correct")
            ax_mean.fill_between(x, ilo, ihi, color=COLOR_INCORRECT, alpha=0.18)
            ax_mean.plot(x, y_i, "^-", color=COLOR_INCORRECT, lw=2, ms=5, label="incorrect")

        ax_mean.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_mean.set_xlabel("turns from end  (0 = last turn)", fontsize=8)
        ax_mean.set_ylabel(metric_label, fontsize=8)
        ax_mean.set_title(f"Layer {layer}\nMean ± {CI}% CI (pooled)", fontsize=9)
        ax_mean.legend(fontsize=8)
        ax_mean.grid(alpha=0.25)

        # ── Row 1: per-group AUROC lines + pooled ─────────────────────────────
        for total_t, grp_color in sorted(GROUP_COLORS.items()):
            if total_t not in group_data:
                continue
            rp_vals, auc_vals = [], []
            for tl, entry in sorted(
                group_data[total_t][li].items(), key=lambda x: turn_sort_key(x[0])
            ):
                labels = np.array(entry["label"],  dtype=np.int32)
                vals   = np.array(entry[raw_key],  dtype=np.float32)
                n_pos  = int((labels == 1).sum())
                n_neg  = int((labels == 0).sum())
                if min(n_pos, n_neg) < MIN_N:
                    continue
                auc = safe_auc(labels, vals)
                if not np.isnan(auc):
                    rp_vals.append(total_t - turn_index(tl))
                    auc_vals.append(auc)
            if rp_vals:
                ax_auc.plot(rp_vals, auc_vals, "o-",
                            color=grp_color, lw=1.6, ms=5,
                            label=f"T={total_t}", alpha=0.85)

        # Pooled AUROC line
        rp_pool, auc_pool = [], []
        for rp in sorted(rel_pool.keys()):
            cell   = rel_pool[rp][li]
            labels = np.array(cell["label"],   dtype=np.int32)
            vals   = np.array(cell[raw_key],   dtype=np.float32)
            n_pos  = int((labels == 1).sum())
            n_neg  = int((labels == 0).sum())
            if min(n_pos, n_neg) < MIN_N:
                continue
            auc = safe_auc(labels, vals)
            if not np.isnan(auc):
                rp_pool.append(rp)
                auc_pool.append(auc)
        if rp_pool:
            ax_auc.plot(rp_pool, auc_pool, "D--",
                        color="black", lw=2, ms=6, label="pooled", alpha=0.9, zorder=5)

        ax_auc.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5, label="chance (0.5)")
        ax_auc.set_xlabel("turns from end  (0 = last turn)", fontsize=8)
        ax_auc.set_ylabel("AUROC", fontsize=8)
        ax_auc.set_title(f"Layer {layer}\nAUROC per group + pooled", fontsize=9)
        ax_auc.legend(fontsize=7, ncol=2)
        ax_auc.set_ylim(0.1, 1.0)
        ax_auc.grid(alpha=0.25)

        # Annotate: if group lines align at the same rel_pos peak → position drives signal
        #           if they diverge → absolute turn number matters more
        ax_auc.text(
            0.01, 0.02,
            "Lines align at same peak → rel_pos drives signal\n"
            "Lines diverge → absolute turn number matters more",
            transform=ax_auc.transAxes, fontsize=5.5, alpha=0.7,
            verticalalignment="bottom",
        )

    fig.suptitle(
        f"Relative-position analysis: {metric_label}\n"
        f"x = 0 is always the last turn of the conversation",
        fontsize=12, fontweight="bold",
    )
    out_path = OUTPUT_DIR / f"{fname_prefix}_relpos.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.name}")


print(f"\nAll done.  Outputs → {OUTPUT_DIR}")
print(f"Key files:")
print(f"  {per_group_csv.name}        (per-group, same schema as v5 + rel_pos)")
print(f"  {rel_csv.name}  (pooled by rel_pos)")
print(f"  heatmap_auc_cos_step.png / heatmap_auc_cos_goal.png  (direction-reversal map)")
print(f"  relpos_cos_step_relpos.png / relpos_cos_goal_relpos.png  (position-from-end test)")
