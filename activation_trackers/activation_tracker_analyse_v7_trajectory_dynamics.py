"""
Activation Tracker Analysis v7 — Hidden-State Trajectory Dynamics
===================================================================

Core question: what does a "smooth" trajectory look like vs a "frequently
reversing" one — and how does that dynamic distinguish correct from incorrect
model answers in a sharded, multi-turn reasoning task?

New over v6
───────────────────────────────────────────────────────────────────────────────
1. Per-turn step magnitude  ||Δhₜ||              (how large each step is)
2. Per-turn goal distance   ||hₜ − hgoal||       (closeness to goal representation)
3. Displacement efficiency  = net / total_path   (1=straight, 0=pure zigzag)
4. Per-conversation trajectory features
     smoothness          = mean cos_step over all turns
     smoothness_std      = std dev of cos_step  (high → oscillating)
     reversal_count      = # turns with cos_step < REVERSAL_THR
     last_reversal_relpos= rel_pos of last reversal (0=last turn; NaN if none)
     magnitude_slope     = linear trend of ||Δhₜ|| (pos=diverging, neg=converging)
     late_curvature_excess = mean cos_step[-2:] − mean cos_step[:-2]  (neg=worse late)
     peak_curvature      = min(cos_step) per conversation (worst single reversal)

Figures produced
───────────────────────────────────────────────────────────────────────────────
  fig_signature.png       Mean ± CI of cos_step vs position-from-end, per group
  fig_exemplars.png       8 raw trajectory curves (4 smooth correct, 4 reversing)
  fig_reversal_profile.png P(reversal event) at each rel_pos, correct vs incorrect
  fig_magnitude.png       Mean ||Δhₜ|| vs position-from-end + mean goal distance
  fig_distributions.png   Violin plots of per-conversation trajectory features
  fig_auroc_features.png  AUROC of every trajectory feature across layers
  fig_scatter.png         Smoothness vs efficiency scatter coloured by correctness

CSVs produced
───────────────────────────────────────────────────────────────────────────────
  per_turn_dynamics_v7.csv    cos_step / cos_goal / step_mag / goal_dist per turn
  per_conv_features_v7.csv    trajectory-level features per conversation × layer
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

# ── Configuration ──────────────────────────────────────────────────────────────

PT_DIR = Path(
    "logs/hidden_states/math/(57-104)snowball-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
)
# PT_DIR_ADD = Path(
#     "logs/hidden_states/math/(698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"
# )
JSONL_PATH = Path(
    "logs/math/snowball-at0-ut0/(57-104)snowball-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
)
# JSONL_PATH_ADD = Path(
#     "logs/math/sharded-at0-ut0/(698)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"
# )

PT_SOURCES    = [PT_DIR]
JSONL_SOURCES = [JSONL_PATH]

LAYERS      = [12, 16, 20, 24, 28]


FOCUS_LI    = 3                   # layer index with strongest v6 AUROC (layer=20)
FOCUS_LAYER = LAYERS[FOCUS_LI]

OUTPUT_DIR      = PT_DIR.parent / "trajectory_dynamics_v7_57-104"
EPS             = 1e-12
MIN_TOTAL_TURN  = 3
MAX_TOTAL_TURN  = 7               # T=8,11 have too few samples
MIN_N           = 5               # minimum samples per class for any pooled cell
N_BOOT          = 2000
CI              = 90
SEED            = 44
REVERSAL_THR    = -0.30           # cos_step below this → "reversal event"
N_EXEMPLARS     = 4               # exemplars per class in exemplar plot
MIN_T_EXEMPLAR  = 4               # min turns to include as an exemplar (need ≥2 cos_step points)

COLOR_CORRECT   = "#2563EB"
COLOR_INCORRECT = "#DC2626"
COLOR_NEUTRAL   = "#6B7280"
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


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_auc(labels, values):
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0 or labels[mask].min() == labels[mask].max():
        return float("nan")
    return float(roc_auc_score(labels[mask], values[mask]))


def bootstrap_mean_ci(values, n_boot=N_BOOT, ci=CI, seed=SEED):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng  = np.random.default_rng(seed)
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo   = np.percentile(boot, (100 - ci) / 2)
    hi   = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(v.mean()), float(lo), float(hi)


# ── Step 1: Load JSONL scores ──────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading JSONL scores")
print("=" * 60)

score_map = {}
for jp in JSONL_SOURCES:
    loaded = 0
    with open(jp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("conv_id")
            s   = rec.get("score")
            if cid is None:
                continue
            score_map[cid] = int(float(s if s is not None else 0))
            loaded += 1
    print(f"  {jp.name}: {loaded} records")

n_c = sum(v == 1 for v in score_map.values())
n_i = sum(v == 0 for v in score_map.values())
print(f"  Total: {len(score_map)}  (correct={n_c}, incorrect={n_i})\n")


# ── Step 2: Load hidden states, compute per-turn metrics ──────────────────────
print("=" * 60)
print("Step 2: Loading .pt files and computing per-turn metrics")
print("=" * 60)

# conv_records[cid] stores raw hidden-state sequences for later feature computation
conv_records = {}
per_turn_rows = []       # for per_turn_dynamics_v7.csv
skip_stats = dict(no_score=0, no_goal=0, too_few=0, dup=0, too_many=0)
seen = set()

for pt_dir in PT_SOURCES:
    pt_files = sorted(pt_dir.glob("*.pt"))
    print(f"  {pt_dir.name}: {len(pt_files)} files")

    for pt_file in pt_files:
        cid = pt_file.stem
        if cid in seen:
            skip_stats["dup"] += 1
            continue
        seen.add(cid)

        if cid not in score_map:
            skip_stats["no_score"] += 1
            continue

        data        = torch.load(pt_file, map_location="cpu", weights_only=False)
        hs_by_label = {e["label"]: e["hidden_states"] for e in data.get("hidden_states", [])}

        if "goal" not in hs_by_label:
            skip_stats["no_goal"] += 1
            continue

        turn_labels = sorted(
            [k for k in hs_by_label if k.startswith("turn_")],
            key=turn_sort_key,
        )
        T = len(turn_labels)

        if T < MIN_TOTAL_TURN:
            skip_stats["too_few"] += 1
            continue
        if T > MAX_TOTAL_TURN:
            skip_stats["too_many"] += 1
            continue

        total_t = max(turn_index(tl) for tl in turn_labels)
        score   = score_map[cid]

        # Per-layer raw hidden-state sequences
        goal_hs_per_layer = [hs_by_label["goal"][li].numpy() for li in range(len(LAYERS))]
        turns_hs = {
            tl: [hs_by_label[tl][li].numpy() for li in range(len(LAYERS))]
            for tl in turn_labels
        }

        # Per-turn metric dicts: [li] → {turn_label: value}
        pt_cos_step  = {li: {} for li in range(len(LAYERS))}
        pt_cos_goal  = {li: {} for li in range(len(LAYERS))}
        pt_step_mag  = {li: {} for li in range(len(LAYERS))}
        pt_goal_dist = {li: {} for li in range(len(LAYERS))}

        for li in range(len(LAYERS)):
            goal_h  = goal_hs_per_layer[li]
            hs_seq  = [turns_hs[tl][li] for tl in turn_labels]

            for i, tl in enumerate(turn_labels):
                t       = turn_index(tl)
                rel_pos = total_t - t

                h_curr = hs_seq[i]

                # Goal distance (all turns)
                gd = float(np.linalg.norm(h_curr - goal_h))
                pt_goal_dist[li][tl] = gd

                if i >= 1:
                    h_prev = hs_seq[i - 1]
                    delta  = h_curr - h_prev

                    # Step magnitude
                    sm = float(np.linalg.norm(delta))
                    pt_step_mag[li][tl] = sm

                    # cos_goal: how much current step points toward goal
                    cg = cosine_sim(delta, goal_h - h_prev)
                    pt_cos_goal[li][tl] = cg

                    if i >= 2:
                        h_prev2    = hs_seq[i - 2]
                        delta_prev = h_prev - h_prev2

                        # cos_step: temporal curvature (same as v6)
                        cs = cosine_sim(delta, delta_prev)
                        pt_cos_step[li][tl] = cs

                # CSV row
                per_turn_rows.append({
                    "conv_id"    : cid,
                    "score"      : score,
                    "total_turns": total_t,
                    "turn"       : tl,
                    "turn_idx"   : t,
                    "rel_pos"    : total_t - t,
                    "layer"      : LAYERS[li],
                    "cos_step"   : pt_cos_step[li].get(tl, float("nan")),
                    "cos_goal"   : pt_cos_goal[li].get(tl, float("nan")),
                    "step_mag"   : pt_step_mag[li].get(tl, float("nan")),
                    "goal_dist"  : pt_goal_dist[li][tl],
                })

        conv_records[cid] = {
            "score"      : score,
            "total_t"    : total_t,
            "turn_labels": turn_labels,
            "cos_step"   : pt_cos_step,
            "cos_goal"   : pt_cos_goal,
            "step_mag"   : pt_step_mag,
            "goal_dist"  : pt_goal_dist,
            "goal_hs"    : goal_hs_per_layer,   # kept for efficiency calculation
            "turns_hs"   : turns_hs,
        }

print(f"\n  Loaded: {len(conv_records)} conversations")
for k, v in skip_stats.items():
    if v:
        print(f"  Skipped ({k}): {v}")


# ── Step 3: Compute per-conversation trajectory features ──────────────────────
print("\n" + "=" * 60)
print("Step 3: Computing per-conversation trajectory features")
print("=" * 60)

conv_feature_rows = []

for cid, rec in conv_records.items():
    turn_labels = rec["turn_labels"]
    T           = rec["total_t"]
    score       = rec["score"]

    for li in range(len(LAYERS)):
        hs_seq  = [rec["turns_hs"][tl][li] for tl in turn_labels]
        goal_h  = rec["goal_hs"][li]

        # ── Displacement efficiency ──────────────────────────────────────────
        # net displacement / total path length (1=straight, 0=zigzag)
        deltas = [hs_seq[i + 1] - hs_seq[i] for i in range(len(hs_seq) - 1)]
        step_mags = [float(np.linalg.norm(d)) for d in deltas]
        total_path = sum(step_mags)
        net_disp   = float(np.linalg.norm(hs_seq[-1] - hs_seq[0]))
        efficiency = net_disp / (total_path + EPS)

        # ── cos_step series ──────────────────────────────────────────────────
        cs_series = [rec["cos_step"][li].get(tl, float("nan")) for tl in turn_labels]
        cs_series = [v for v in cs_series if not np.isnan(v)]   # drop NaN (turns <3)

        if len(cs_series) == 0:
            continue   # shouldn't happen given T≥3 filter

        smoothness     = float(np.mean(cs_series))
        smoothness_std = float(np.std(cs_series))
        peak_curvature = float(np.min(cs_series))   # most negative = worst reversal

        reversal_mask  = [cs < REVERSAL_THR for cs in cs_series]
        reversal_count = int(sum(reversal_mask))

        # Position of last reversal (rel_pos from end: 0=last turn)
        # cs_series[i] corresponds to turn_labels[i+2] (0-indexed) → rel_pos = T-(i+3)
        # More concisely: go through the cs_series in reverse to find last reversal
        last_reversal_relpos = float("nan")
        for i in range(len(reversal_mask) - 1, -1, -1):
            if reversal_mask[i]:
                # turn index = i + 3 (1-indexed, since cos_step starts at turn_3)
                last_reversal_relpos = float(T - (i + 3))
                break

        # ── Magnitude trend ──────────────────────────────────────────────────
        if len(step_mags) >= 2:
            xs = np.arange(len(step_mags), dtype=float)
            slope, *_ = scipy_stats.linregress(xs, step_mags)
            magnitude_slope = float(slope)
        else:
            magnitude_slope = float("nan")

        # ── Late vs early curvature ──────────────────────────────────────────
        # negative value = curvature got worse toward the end
        n_late = min(2, len(cs_series))
        n_early = len(cs_series) - n_late
        if n_early > 0:
            late_curv_excess = float(np.mean(cs_series[-n_late:]) - np.mean(cs_series[:n_early]))
        else:
            late_curv_excess = float("nan")

        # ── Net-displacement direction toward goal ───────────────────────────
        net_vec      = hs_seq[-1] - hs_seq[0]
        goal_vec     = goal_h - hs_seq[0]
        cos_net_goal = cosine_sim(net_vec, goal_vec)

        conv_feature_rows.append({
            "conv_id"              : cid,
            "score"                : score,
            "total_turns"          : T,
            "layer"                : LAYERS[li],
            "efficiency"           : efficiency,
            "smoothness"           : smoothness,
            "smoothness_std"       : smoothness_std,
            "reversal_count"       : reversal_count,
            "last_reversal_relpos" : last_reversal_relpos,
            "magnitude_slope"      : magnitude_slope,
            "late_curvature_excess": late_curv_excess,
            "peak_curvature"       : peak_curvature,
            "cos_net_goal"         : cos_net_goal,
            "total_path"           : total_path,
            "net_displacement"     : net_disp,
        })

print(f"  Computed features for {len(conv_feature_rows)} (conv × layer) records")


# ── Step 4: Save CSVs ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_turn = OUTPUT_DIR / "per_turn_dynamics_v7.csv"
with open(csv_turn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_turn_rows[0].keys()))
    w.writeheader(); w.writerows(per_turn_rows)
print(f"\nSaved → {csv_turn.name}  ({len(per_turn_rows)} rows)")

csv_conv = OUTPUT_DIR / "per_conv_features_v7.csv"
with open(csv_conv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(conv_feature_rows[0].keys()))
    w.writeheader(); w.writerows(conv_feature_rows)
print(f"Saved → {csv_conv.name}  ({len(conv_feature_rows)} rows)")


# ── Helper: pool per-turn data by rel_pos for signature plots ─────────────────

def build_rel_pool(metric_key):
    """Return rel_pool[rel_pos][li] → {vals_correct, vals_incorrect}."""
    pool = defaultdict(lambda: defaultdict(lambda: {0: [], 1: []}))
    for row in per_turn_rows:
        v = row[metric_key]
        if np.isnan(v):
            continue
        pool[row["rel_pos"]][LAYERS.index(row["layer"])][row["score"]].append(v)
    return pool


def pool_mean_ci(pool, max_relpos, metric_key):
    """Build arrays: x, mean_c, lo_c, hi_c, mean_i, lo_i, hi_i for the focus layer."""
    xs, m_c, lo_c, hi_c, m_i, lo_i, hi_i = [], [], [], [], [], [], []
    for rp in range(max_relpos, -1, -1):  # iterate from early to last
        d = pool[rp][FOCUS_LI]
        if len(d[1]) < MIN_N or len(d[0]) < MIN_N:
            continue
        mc, lc, hc = bootstrap_mean_ci(d[1])
        mi, li_, hi_ = bootstrap_mean_ci(d[0])
        xs.append(rp)
        m_c.append(mc);  lo_c.append(lc); hi_c.append(hc)
        m_i.append(mi);  lo_i.append(li_); hi_i.append(hi_)
    return np.array(xs), np.array(m_c), np.array(lo_c), np.array(hi_c), \
                         np.array(m_i), np.array(lo_i), np.array(hi_i)


# ── Step 5: Figure 1 — Trajectory Signature ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Figure 1 — Trajectory Signature")
print("=" * 60)

pool_cs = build_rel_pool("cos_step")
pool_sm = build_rel_pool("step_mag")
MAX_RP  = 5   # show rel_pos 0..5

fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 5), sharey=True)
for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
    xs, m_c, lo_c, hi_c, m_i, lo_i, hi_i = [], [], [], [], [], [], []
    for rp in range(MAX_RP, -1, -1):
        d = pool_cs[rp][li]
        if len(d[1]) < MIN_N or len(d[0]) < MIN_N:
            continue
        mc, lc, hc = bootstrap_mean_ci(d[1])
        mi, li_, hi_ = bootstrap_mean_ci(d[0])
        xs.append(rp)
        m_c.append(mc);  lo_c.append(lc); hi_c.append(hc)
        m_i.append(mi);  lo_i.append(li_); hi_i.append(hi_)

    if not xs:
        ax.set_title(f"Layer {layer}\n(no data)")
        continue

    x = np.array(xs[::-1])       # chronological: high relpos → small relpos
    def _rev(lst): return np.array(lst[::-1])

    ax.fill_between(x, _rev(lo_c), _rev(hi_c), color=COLOR_CORRECT,   alpha=0.18)
    ax.plot(x, _rev(m_c), "o-", color=COLOR_CORRECT,   lw=2, ms=5, label="correct")
    ax.fill_between(x, _rev(lo_i), _rev(hi_i), color=COLOR_INCORRECT, alpha=0.18)
    ax.plot(x, _rev(m_i), "^-", color=COLOR_INCORRECT, lw=2, ms=5, label="incorrect")

    ax.axhline(REVERSAL_THR, color="orange", lw=1.0, ls="--", alpha=0.8,
               label=f"reversal thr ({REVERSAL_THR})")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

    ax.set_xlabel("Steps before end  (0 = last turn)", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([str(int(xi)) for xi in x])
    ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    if li == 0:
        ax.set_ylabel("cos(Δhₜ, Δhₜ₋₁)  [temporal curvature]", fontsize=9)
    if li == len(LAYERS) - 1:
        ax.legend(fontsize=8)

fig.suptitle(
    f"Trajectory Signature: mean ± {CI}% CI of cos_step vs position from end\n"
    "Smooth trajectories stay near 0; reversing trajectories dip below orange line",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_signature.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 6: Figure 2 — Exemplar Trajectories ──────────────────────────────────
print("Step 6: Figure 2 — Exemplar Trajectories")

# For each conversation at focus layer, compute mean cos_step to rank smoothness
ranked_correct   = []
ranked_incorrect = []

for cid, rec in conv_records.items():
    if rec["total_t"] < MIN_T_EXEMPLAR:
        continue
    cs_vals = [v for v in rec["cos_step"][FOCUS_LI].values() if not np.isnan(v)]
    if len(cs_vals) < 2:
        continue
    mean_cs = float(np.mean(cs_vals))
    entry   = (mean_cs, cid)
    if rec["score"] == 1:
        ranked_correct.append(entry)
    else:
        ranked_incorrect.append(entry)

ranked_correct.sort(reverse=True)     # highest (smoothest) first
ranked_incorrect.sort()               # lowest (most reversing) first

sel_correct   = [cid for _, cid in ranked_correct[:N_EXEMPLARS]]
sel_incorrect = [cid for _, cid in ranked_incorrect[:N_EXEMPLARS]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, cids, label, color, title in [
    (axes[0], sel_correct,   "correct",   COLOR_CORRECT,   f"Smooth (correct, top-{N_EXEMPLARS})"),
    (axes[1], sel_incorrect, "incorrect", COLOR_INCORRECT, f"Frequently Reversing (incorrect, top-{N_EXEMPLARS})"),
]:
    for rank, cid in enumerate(cids):
        rec = conv_records[cid]
        T   = rec["total_t"]
        tls = rec["turn_labels"]
        # Build (x=rel_pos, y=cos_step) pairs in chronological order
        pts = []
        for tl in tls:
            cs = rec["cos_step"][FOCUS_LI].get(tl, float("nan"))
            if not np.isnan(cs):
                rp = T - turn_index(tl)
                pts.append((rp, cs))
        if not pts:
            continue
        pts.sort(key=lambda p: -p[0])  # high relpos first (chronological)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        alpha = 0.6 + 0.4 * rank / max(1, N_EXEMPLARS - 1)
        ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=6,
                alpha=alpha, label=f"conv {rank+1} (T={T})")
        # Mark each reversal event
        for x_, y_ in zip(xs, ys):
            if y_ < REVERSAL_THR:
                ax.scatter([x_], [y_], s=60, color="red", zorder=5,
                           marker="v", linewidths=0)

    ax.axhline(REVERSAL_THR, color="orange", lw=1.2, ls="--", alpha=0.9,
               label=f"reversal thr ({REVERSAL_THR})")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4, label="cos=0")
    ax.set_xlim(-0.3, MAX_RP + 0.3)
    ax.set_xlabel("Steps before end  (0 = last turn)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()  # 0 on the right = last turn on the right

axes[0].set_ylabel(f"cos(Δhₜ, Δhₜ₋₁)  [layer {FOCUS_LAYER}]", fontsize=9)

fig.suptitle(
    "Individual Trajectory Exemplars\n"
    "▼ marks reversal events (cos_step below orange threshold)   "
    "x-axis: time flows left→right, last turn at 0",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_exemplars.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 7: Figure 3 — Reversal Probability Profile ───────────────────────────
print("Step 7: Figure 3 — Reversal Probability Profile")

fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 4.5), sharey=True)
for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
    rp_vals = sorted(set(r["rel_pos"] for r in per_turn_rows
                         if r["layer"] == layer and not np.isnan(r["cos_step"])))
    rp_vals = [rp for rp in rp_vals if rp <= MAX_RP]

    xs, p_c, p_i, n_c_arr, n_i_arr = [], [], [], [], []
    for rp in sorted(rp_vals, reverse=True):  # chronological
        vals_c = [r["cos_step"] for r in per_turn_rows
                  if r["layer"] == layer and r["rel_pos"] == rp
                  and r["score"] == 1 and not np.isnan(r["cos_step"])]
        vals_i = [r["cos_step"] for r in per_turn_rows
                  if r["layer"] == layer and r["rel_pos"] == rp
                  and r["score"] == 0 and not np.isnan(r["cos_step"])]
        if len(vals_c) < MIN_N or len(vals_i) < MIN_N:
            continue
        rev_c = np.mean(np.array(vals_c) < REVERSAL_THR)
        rev_i = np.mean(np.array(vals_i) < REVERSAL_THR)
        xs.append(rp)
        p_c.append(rev_c); n_c_arr.append(len(vals_c))
        p_i.append(rev_i); n_i_arr.append(len(vals_i))

    if not xs:
        ax.set_title(f"Layer {layer}\n(no data)")
        continue

    x     = np.array(xs)
    width = 0.35
    ax.bar(x - width/2, p_i, width=width, color=COLOR_INCORRECT, alpha=0.75, label="incorrect")
    ax.bar(x + width/2, p_c, width=width, color=COLOR_CORRECT,   alpha=0.75, label="correct")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(xi)) for xi in x])
    ax.set_xlabel("Steps before end  (0 = last turn)", fontsize=9)
    ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.0)
    if li == 0:
        ax.set_ylabel(f"P(reversal)  [cos_step < {REVERSAL_THR}]", fontsize=9)
    if li == len(LAYERS) - 1:
        ax.legend(fontsize=8)

fig.suptitle(
    "Reversal Probability at Each Position\n"
    "Incorrect trajectories hit the reversal threshold more often, especially near the end",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_reversal_profile.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 8: Figure 4 — Step Magnitude + Goal Distance Evolution ───────────────
print("Step 8: Figure 4 — Step Magnitude + Goal Distance")

pool_sm  = build_rel_pool("step_mag")
pool_gd  = build_rel_pool("goal_dist")

fig, axes = plt.subplots(2, len(LAYERS), figsize=(5 * len(LAYERS), 9))

for li, layer in enumerate(LAYERS):
    for row_idx, (pool, ylabel, title_pref) in enumerate([
        (pool_sm, "||Δhₜ||  step magnitude",    "Step Magnitude"),
        (pool_gd, "||hₜ − hgoal||  goal distance", "Goal Distance"),
    ]):
        ax = axes[row_idx][li]
        xs, m_c, lo_c, hi_c, m_i, lo_i, hi_i = [], [], [], [], [], [], []

        for rp in range(MAX_RP, -1, -1):
            d = pool[rp][li]
            if len(d[1]) < MIN_N or len(d[0]) < MIN_N:
                continue
            mc, lc, hc = bootstrap_mean_ci(d[1])
            mi, li_, hi_ = bootstrap_mean_ci(d[0])
            xs.append(rp)
            m_c.append(mc); lo_c.append(lc); hi_c.append(hc)
            m_i.append(mi); lo_i.append(li_); hi_i.append(hi_)

        if not xs:
            ax.set_title(f"Layer {layer}\n(no data)")
            continue

        x = np.array(xs[::-1])
        def _rev(lst): return np.array(lst[::-1])

        ax.fill_between(x, _rev(lo_c), _rev(hi_c), color=COLOR_CORRECT,   alpha=0.18)
        ax.plot(x, _rev(m_c), "o-", color=COLOR_CORRECT,   lw=2, ms=5, label="correct")
        ax.fill_between(x, _rev(lo_i), _rev(hi_i), color=COLOR_INCORRECT, alpha=0.18)
        ax.plot(x, _rev(m_i), "^-", color=COLOR_INCORRECT, lw=2, ms=5, label="incorrect")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(xi)) for xi in x])
        ax.set_xlabel("Steps before end", fontsize=8)
        ax.set_title(f"{title_pref}  Layer {layer}", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.25)
        if li == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if li == len(LAYERS) - 1:
            ax.legend(fontsize=7)

fig.suptitle(
    f"Step Magnitude (top) and Goal Distance (bottom)  [{CI}% bootstrap CI]\n"
    "Step magnitude growing toward the end = divergence  |  "
    "Goal distance falling = convergence toward goal representation",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_magnitude.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 9: Figure 5 — Per-Conversation Feature Distributions ─────────────────
print("Step 9: Figure 5 — Feature Distributions")

# Focus on focus layer only
focus_rows = [r for r in conv_feature_rows if r["layer"] == FOCUS_LAYER]
correct_rows   = [r for r in focus_rows if r["score"] == 1]
incorrect_rows = [r for r in focus_rows if r["score"] == 0]

FEATURES = [
    ("smoothness",           "Mean cos_step  (↑ = smoother)",    True),
    ("smoothness_std",       "Std cos_step  (↓ = more stable)",  False),
    ("reversal_count",       "# Reversal events  (↓ = better)",  False),
    ("efficiency",           "Displacement efficiency  (↑ = straighter path)",  True),
    ("magnitude_slope",      "Step-mag trend  (↓ = converging)", False),
    ("late_curvature_excess","Late−early curvature  (↓ = worse late)", False),
    ("peak_curvature",       "Peak curvature = min cos_step  (↑ = milder worst)", True),
    ("cos_net_goal",         "Net-displacement toward goal  (↑ = more aligned)", True),
]

nf  = len(FEATURES)
fig, axes = plt.subplots(2, (nf + 1) // 2, figsize=(5 * ((nf + 1) // 2), 9))
axes = axes.flatten()

for ax_idx, (feat, ylabel, _) in enumerate(FEATURES):
    ax  = axes[ax_idx]
    vc  = [r[feat] for r in correct_rows   if not np.isnan(r[feat])]
    vi  = [r[feat] for r in incorrect_rows if not np.isnan(r[feat])]

    # Side-by-side violins
    parts = ax.violinplot([vi, vc], positions=[0, 1], showmedians=True,
                          showextrema=False)
    parts["bodies"][0].set_facecolor(COLOR_INCORRECT); parts["bodies"][0].set_alpha(0.6)
    parts["bodies"][1].set_facecolor(COLOR_CORRECT);   parts["bodies"][1].set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Overlay means
    ax.scatter([0, 1], [np.mean(vi), np.mean(vc)], color="black", s=40, zorder=5,
               marker="D", label=["mean"] if ax_idx == 0 else [])

    # AUROC annotation
    labels_all = [0] * len(vi) + [1] * len(vc)
    vals_all   = vi + vc
    auc = safe_auc(labels_all, vals_all)
    ax.set_title(f"{feat}\nAUROC={auc:.3f}", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["incorrect", "correct"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(axis="y", alpha=0.2)

# Hide unused axes
for i in range(len(FEATURES), len(axes)):
    axes[i].set_visible(False)

fig.suptitle(
    f"Per-Conversation Trajectory Feature Distributions  [layer {FOCUS_LAYER}]\n"
    "AUROC > 0.5 means higher value → more likely correct",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_distributions.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 10: Figure 6 — Multi-Feature AUROC Bar Chart ─────────────────────────
print("Step 10: Figure 6 — Multi-Feature AUROC")

# Compute AUROC for every feature × layer combination
auc_table = {feat: {layer: float("nan") for layer in LAYERS} for feat, *_ in FEATURES}

for layer in LAYERS:
    rows_l = [r for r in conv_feature_rows if r["layer"] == layer]
    c_rows = [r for r in rows_l if r["score"] == 1]
    i_rows = [r for r in rows_l if r["score"] == 0]
    for feat, *_ in FEATURES:
        vc  = [r[feat] for r in c_rows if not np.isnan(r[feat])]
        vi  = [r[feat] for r in i_rows if not np.isnan(r[feat])]
        if len(vc) < MIN_N or len(vi) < MIN_N:
            continue
        auc_table[feat][layer] = safe_auc([0]*len(vi) + [1]*len(vc), vi + vc)

fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 5.5), sharey=True)
x_positions = np.arange(len(FEATURES))
bar_width   = 0.6

for li, (ax, layer) in enumerate(zip(axes, LAYERS)):
    aucs   = [auc_table[feat][layer] for feat, *_ in FEATURES]
    colors = [
        COLOR_CORRECT if (not np.isnan(a) and a > 0.5) else
        (COLOR_INCORRECT if not np.isnan(a) else COLOR_NEUTRAL)
        for a in aucs
    ]
    bars = ax.bar(x_positions, aucs, color=colors, alpha=0.80, width=bar_width)
    ax.axhline(0.5, color="k", lw=1.0, ls="--", alpha=0.7, label="chance")

    for xp, a in zip(x_positions, aucs):
        if not np.isnan(a):
            ax.text(xp, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f[0] for f in FEATURES], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.3, 0.9)
    ax.set_title(f"Layer {layer}", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    if li == 0:
        ax.set_ylabel("AUROC  (correct vs incorrect)", fontsize=8)
    if li == len(LAYERS) - 1:
        ax.legend(fontsize=8)

fig.suptitle(
    "Per-Conversation Feature AUROC across Layers\n"
    "Blue > 0.5 = higher value → correct  |  Red < 0.5 = reversed",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = OUTPUT_DIR / "fig_auroc_features.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 11: Figure 7 — Feature Scatter (smoothness × efficiency) ─────────────
print("Step 11: Figure 7 — Feature Scatter")

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)

for panel, (xfeat, yfeat, xtitle, ytitle) in enumerate([
    ("smoothness",  "efficiency",
     "Smoothness  mean cos_step  (↑ = smoother)",
     "Displacement efficiency  (↑ = straighter path)"),
    ("smoothness",  "reversal_count",
     "Smoothness  mean cos_step  (↑ = smoother)",
     "Reversal count  (↓ = better)"),
]):
    ax = fig.add_subplot(gs[panel])

    xc = np.array([r[xfeat] for r in correct_rows   if not np.isnan(r[xfeat]) and not np.isnan(r[yfeat])])
    yc = np.array([r[yfeat] for r in correct_rows   if not np.isnan(r[xfeat]) and not np.isnan(r[yfeat])])
    xi = np.array([r[xfeat] for r in incorrect_rows if not np.isnan(r[xfeat]) and not np.isnan(r[yfeat])])
    yi = np.array([r[yfeat] for r in incorrect_rows if not np.isnan(r[xfeat]) and not np.isnan(r[yfeat])])

    ax.scatter(xi, yi, c=COLOR_INCORRECT, s=12, alpha=0.35, label=f"incorrect (n={len(xi)})", rasterized=True)
    ax.scatter(xc, yc, c=COLOR_CORRECT,   s=12, alpha=0.35, label=f"correct   (n={len(xc)})", rasterized=True)

    # Mean markers
    ax.scatter([np.mean(xi)], [np.mean(yi)], c=COLOR_INCORRECT, s=120,
               marker="X", edgecolors="black", linewidths=0.8, zorder=6)
    ax.scatter([np.mean(xc)], [np.mean(yc)], c=COLOR_CORRECT,   s=120,
               marker="X", edgecolors="black", linewidths=0.8, zorder=6, label="group mean (×)")

    ax.set_xlabel(xtitle, fontsize=9)
    ax.set_ylabel(ytitle, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.20)

    # Annotate separation line
    if len(xc) > 1 and len(xi) > 1:
        r_c, p_c = scipy_stats.pearsonr(xc, yc)
        r_i, p_i = scipy_stats.pearsonr(xi, yi)
        ax.set_title(
            f"Layer {FOCUS_LAYER}\n"
            f"corr(correct)={r_c:+.2f}  corr(incorrect)={r_i:+.2f}",
            fontsize=9, fontweight="bold",
        )

fig.suptitle(
    "Trajectory Feature Space  [per-conversation, layer 20]\n"
    "✕ marks group mean  |  smooth+straight = correct; rough+zigzag = incorrect",
    fontsize=11, fontweight="bold",
)
out = OUTPUT_DIR / "fig_scatter.png"
plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
print(f"  Saved → {out.name}")


# ── Step 12: Summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 12: Summary — key AUROCs at layer {FOCUS_LAYER}")
print("=" * 60)
print(f"  {'Feature':<26}  {'AUROC':>8}  {'mean_correct':>14}  {'mean_incorrect':>14}")
print("  " + "─" * 70)
focus_correct   = [r for r in conv_feature_rows if r["layer"] == FOCUS_LAYER and r["score"] == 1]
focus_incorrect = [r for r in conv_feature_rows if r["layer"] == FOCUS_LAYER and r["score"] == 0]
for feat, desc, _ in FEATURES:
    vc  = [r[feat] for r in focus_correct   if not np.isnan(r[feat])]
    vi  = [r[feat] for r in focus_incorrect if not np.isnan(r[feat])]
    auc = safe_auc([0]*len(vi) + [1]*len(vc), vi + vc)
    mc  = np.mean(vc)  if vc else float("nan")
    mi  = np.mean(vi)  if vi else float("nan")
    flag = "←" if abs(auc - 0.5) > 0.10 else ""
    print(f"  {feat:<26}  {auc:>8.3f}  {mc:>14.4f}  {mi:>14.4f}  {flag}")

print(f"\nAll outputs → {OUTPUT_DIR}")
print("Key files:")
for fname in [
    "fig_signature.png", "fig_exemplars.png", "fig_reversal_profile.png",
    "fig_magnitude.png", "fig_distributions.png",
    "fig_auroc_features.png", "fig_scatter.png",
    "per_turn_dynamics_v7.csv", "per_conv_features_v7.csv",
]:
    print(f"  {fname}")
