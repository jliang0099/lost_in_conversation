#!/usr/bin/env python3
"""
Prepare per-layer activation-steering artifacts from existing hidden-state .pt files.

One file is written per layer:  {output_dir}/layer_{L}.pt
Each file contains:
  {
    "layer":         int,
    "strategy_axis": Tensor(D,),   # normalize(mean[answer_attempt] − mean[discussion])
    "global_mean":   Tensor(D,),   # mean over all hidden states at this layer
    "metadata":      {...}
  }

SteeringController loads the entire output_dir at once.

Usage:
  python prepare_steering_artifacts.py \
      --steer_layers 20 21 22 23 24 25 26 27 28 29 30 31 \
      --output_dir   logs/steering_artifacts/math_llama8b/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ── Config: mirrors conversation_analysis_v4.ipynb ───────────────────────────
PT_SOURCES = [
    Path("logs/hidden_states/math/(l20-31)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"),
    Path("logs/hidden_states/math/(add_l20-l31)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct"),
]
JSONL_SOURCES = [
    Path("logs/math/sharded-at0-ut0/(l20-l31)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"),
    Path("logs/math/sharded-at0-ut0/(add_l20-l31)sharded-at0-ut0_math_meta-llama_Llama-3.1-8B-Instruct.jsonl"),
]

# Layers tracked during baseline runs (ActivationTracker default)
LAYERS       = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
STRATEGY_POS = "answer_attempt"
STRATEGY_NEG = "discussion"
RT_NORM      = {"hedge": "hedging", "refuse": "refusal"}


def _turn_sort_key(label: str):
    if not label.startswith("turn_"):
        return (1, label)
    try:
        return (0, int(label.split("_", 1)[1]))
    except ValueError:
        return (0, label)


def _turn_index(label: str) -> int:
    return int(label.split("_", 1)[1])


def _build_axis(rt_accum: dict, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute strategy_axis and global_mean for one layer from rt_accum[layer]."""
    accum = rt_accum[layer]

    if STRATEGY_POS not in accum:
        raise RuntimeError(f"Layer {layer}: RT '{STRATEGY_POS}' not found.")
    if STRATEGY_NEG not in accum:
        raise RuntimeError(f"Layer {layer}: RT '{STRATEGY_NEG}' not found.")

    v_pos = np.mean(accum[STRATEGY_POS], axis=0)
    v_neg = np.mean(accum[STRATEGY_NEG], axis=0)
    axis_raw = v_pos - v_neg
    strategy_axis = axis_raw / (np.linalg.norm(axis_raw) + 1e-9)

    all_vecs = np.stack([v for vecs in accum.values() for v in vecs])
    global_mean = all_vecs.mean(axis=0)

    # Sanity check
    proj_pos = float(np.dot(v_pos - global_mean, strategy_axis))
    proj_neg = float(np.dot(v_neg - global_mean, strategy_axis))
    if proj_pos <= proj_neg:
        raise RuntimeError(
            f"Layer {layer}: strategy axis direction wrong "
            f"(proj_pos={proj_pos:.4f} ≤ proj_neg={proj_neg:.4f})."
        )

    return strategy_axis, global_mean


def main(steer_layers: list[int], output_dir: Path) -> None:
    # Validate requested layers
    invalid = [l for l in steer_layers if l not in LAYERS]
    if invalid:
        raise ValueError(
            f"Requested layers {invalid} are not in tracked LAYERS {LAYERS}. "
            f"Re-run baseline with those layers added to ActivationTracker."
        )

    layer_to_idx = {l: LAYERS.index(l) for l in steer_layers}

    # ── 1. Load .pt hidden-state files ───────────────────────────────────────
    conv_records: dict = {}
    seen: set = set()

    for pt_dir in PT_SOURCES:
        if not pt_dir.exists():
            print(f"[warn] PT source not found, skipping: {pt_dir}")
            continue
        pt_files = sorted(pt_dir.glob("*.pt"))
        print(f"  {str(pt_dir)[-70:]}: {len(pt_files)} files")

        for pt_file in pt_files:
            conv_id = pt_file.stem
            if conv_id in seen:
                continue
            seen.add(conv_id)

            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            hs_by_label = {
                e["label"]: e["hidden_states"]
                for e in data.get("hidden_states", [])
            }
            if "goal" not in hs_by_label:
                continue

            turn_labels = sorted(
                [k for k in hs_by_label if k.startswith("turn_")],
                key=_turn_sort_key,
            )
            if len(turn_labels) < 3:
                continue

            # Store only the layers we need
            conv_records[conv_id] = {
                "turns_hs": {
                    tl: {
                        layer: hs_by_label[tl][idx].numpy()
                        for layer, idx in layer_to_idx.items()
                    }
                    for tl in turn_labels
                }
            }

    if not conv_records:
        raise RuntimeError("No conversations loaded — check PT_SOURCES paths.")
    print(f"\nLoaded: {len(conv_records)} conversations")

    # ── 2. Load JSONL → turn_rt_map ──────────────────────────────────────────
    turn_rt_map: dict = {}

    for jsonl_path in JSONL_SOURCES:
        if not jsonl_path.exists():
            print(f"[warn] JSONL source not found, skipping: {jsonl_path}")
            continue
        loaded = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                conv_id = rec.get("conv_id")
                if conv_id is None:
                    continue
                trace, turn_rt, user_turn_idx = rec.get("trace", []), {}, 0
                for entry in trace:
                    role = entry.get("role", "")
                    if role == "user":
                        user_turn_idx += 1
                    elif role == "log":
                        content = entry.get("content", {})
                        if (
                            isinstance(content, dict)
                            and content.get("type") == "system-verification"
                        ):
                            rt = content.get("response", {}).get(
                                "response_type", "missing"
                            )
                            rt = RT_NORM.get(rt, rt)
                            turn_rt[user_turn_idx] = rt
                turn_rt_map[conv_id] = turn_rt
                loaded += 1
        print(f"  {jsonl_path.name[:70]}: {loaded} records")

    # ── 3. Accumulate per-RT hidden states, separately per layer ─────────────
    # rt_accum[layer][rt] = list of (D,) arrays
    rt_accum: dict[int, dict] = {layer: defaultdict(list) for layer in steer_layers}

    for conv_id, rec in conv_records.items():
        turn_rt = turn_rt_map.get(conv_id, {})
        for tl, hs_per_layer in rec["turns_hs"].items():
            t_idx = _turn_index(tl)
            rt = turn_rt.get(t_idx, "missing")
            for layer in steer_layers:
                rt_accum[layer][rt].append(hs_per_layer[layer])

    # ── 4. Build strategy axis and global mean per layer ─────────────────────
    per_layer: dict[int, dict] = {}

    for layer in steer_layers:
        print(f"\n── Layer {layer} ──────────────────────────────────────")
        print(f"  Sample counts per RT:")
        for rt, vecs in sorted(rt_accum[layer].items()):
            print(f"    {rt:<20} n={len(vecs):>5}")

        strategy_axis, global_mean = _build_axis(rt_accum, layer)

        # Report projections
        v_pos = np.mean(rt_accum[layer][STRATEGY_POS], axis=0)
        v_neg = np.mean(rt_accum[layer][STRATEGY_NEG], axis=0)
        proj_pos = float(np.dot(v_pos - global_mean, strategy_axis))
        proj_neg = float(np.dot(v_neg - global_mean, strategy_axis))
        print(
            f"  Axis norm (pre-normalize): {np.linalg.norm(v_pos - v_neg):.4f}"
        )
        print(
            f"  Projection — {STRATEGY_POS}: {proj_pos:+.4f}  "
            f"{STRATEGY_NEG}: {proj_neg:+.4f}"
        )

        # ── 5. Save one file per layer ────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"layer_{layer}.pt"
        torch.save(
            {
                "layer":         layer,
                "strategy_axis": torch.from_numpy(strategy_axis).float(),
                "global_mean":   torch.from_numpy(global_mean).float(),
                "metadata": {
                    "strategy_pos":  STRATEGY_POS,
                    "strategy_neg":  STRATEGY_NEG,
                    "n_samples":     {rt: len(vecs) for rt, vecs in rt_accum[layer].items()},
                    "n_conversations": len(conv_records),
                },
            },
            out_file,
        )
        print(f"  → saved {out_file}")

    print(f"\nAll artifacts saved to {output_dir}/")
    print(f"  Files: {[f'layer_{l}.pt' for l in steer_layers]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steer_layers",
        nargs="+",
        type=int,
        default=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        help=f"Layers to extract artifacts for. Must be subset of {LAYERS}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/steering_artifacts/math_llama8b/",
        help="Directory to write one layer_L.pt file per layer.",
    )
    args = parser.parse_args()
    main(
        steer_layers=sorted(set(args.steer_layers)),
        output_dir=Path(args.output_dir),
    )
