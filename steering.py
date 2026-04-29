"""
Activation-steering primitives for Goal Imprint Gap (Metric C) intervention.

Many-to-many design: each steered layer has its own strategy_axis and
global_mean. Gap measurement, alpha computation, and the hook vector are
all computed independently per layer.

Artifacts are prepared by prepare_steering_artifacts.py (one file per layer).

Typical lifecycle per experiment run:

    # Shared across all conversations (load artifacts once):
    ctrl = SteeringController(
        "logs/steering_artifacts/math_llama8b/",  # directory with layer_L.pt files
        base_alpha=0.5,
    )
    # ctrl.steer_layers → [20, 21, ..., 31]  (all layers found in the directory)

    # Per conversation — the simulator owns goal_coords (set on turn 1):
    goal_coords: Optional[dict[int, float]] = None

See model_huggingface.py (_steered_hf_generate) for the hook integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch


class SteeringController:
    """
    Stateless container for per-layer steering artifacts derived from Metric C.

    Each layer in steer_layers has its own strategy_axis and global_mean,
    computed independently from that layer's hidden-state geometry.

    Proportional steering per layer:
        alpha_l = base_alpha * max(0, current_coord_l - goal_coord_l)

    where goal_coord_l is the projection of the turn-1 (goal) hidden state
    onto layer l's strategy axis.
    """

    def __init__(self, artifacts_dir: str | Path, base_alpha: float = 0.5):
        """
        Load all layer_L.pt files from artifacts_dir (written by
        prepare_steering_artifacts.py). steer_layers is derived from the
        files found — no need to specify it separately.
        """
        artifacts_dir = Path(artifacts_dir)
        layer_files = sorted(artifacts_dir.glob("layer_*.pt"))
        if not layer_files:
            raise FileNotFoundError(
                f"No layer_*.pt files found in {artifacts_dir}. "
                "Run prepare_steering_artifacts.py first."
            )

        self.base_alpha: float = base_alpha
        # per_layer[layer] = {"strategy_axis": Tensor(D,), "global_mean": Tensor(D,)}
        self.per_layer: dict[int, dict[str, torch.Tensor]] = {}

        for path in layer_files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            layer = int(data["layer"])
            self.per_layer[layer] = {
                "strategy_axis": data["strategy_axis"].float(),
                "global_mean":   data["global_mean"].float(),
            }

        self.steer_layers: list[int] = sorted(self.per_layer.keys())
        print(
            f"[SteeringController] loaded {len(self.steer_layers)} layers "
            f"from {artifacts_dir}: {self.steer_layers}"
        )

    # ------------------------------------------------------------------
    # Coordinate extraction  (one per layer)
    # ------------------------------------------------------------------

    def extract_coord(self, hidden_states: list, layer_idx: int) -> float:
        """
        Project the final input-token hidden state of layer_idx onto that
        layer's own strategy axis (centered by its own global_mean).

        hidden_states: list from model(..., output_hidden_states=True).
            Index convention: hidden_states[0] = embeddings,
            hidden_states[layer_idx + 1] = output of decoder layer layer_idx.
        """
        layer_data = self.per_layer[layer_idx]
        h = hidden_states[layer_idx + 1][0, -1, :].cpu().float()  # (D,)
        return float(
            torch.dot(h - layer_data["global_mean"], layer_data["strategy_axis"])
        )

    def extract_coords(self, hidden_states: list) -> dict[int, float]:
        """Extract coord for every layer in steer_layers."""
        return {l: self.extract_coord(hidden_states, l) for l in self.steer_layers}

    # ------------------------------------------------------------------
    # Proportional alpha  (called independently per layer)
    # ------------------------------------------------------------------

    def compute_steer_alpha(self, goal_coord: float, current_coord: float) -> float:
        """
        Proportional steering strength for one layer:
            base_alpha * max(0, current_coord - goal_coord)
        Zero when the model is already at or below the goal commitment level.
        """
        gap = current_coord - goal_coord
        return self.base_alpha * max(0.0, gap)

    # ------------------------------------------------------------------
    # Per-layer hook factory
    # ------------------------------------------------------------------

    def make_hook(self, alpha: float, layer_idx: int) -> Optional[Callable]:
        """
        Returns a forward hook for layer_idx that subtracts
        alpha * strategy_axis[layer_idx] from its hidden states, or None
        if alpha == 0.

        Handles both transformers output styles:
          - Plain Tensor  (newer transformers 4.45+)
          - Tuple[Tensor, ...] (older transformers)
        """
        if alpha == 0.0:
            return None

        layer_data = self.per_layer[layer_idx]
        sv = (-alpha * layer_data["strategy_axis"]).clone()  # (D,)

        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                sv_dev = sv.to(device=output.device, dtype=output.dtype)
                return output + sv_dev[None, None, :]
            else:
                hs = output[0]                                     # (batch, seq, D)
                sv_dev = sv.to(device=hs.device, dtype=hs.dtype)
                steered = hs + sv_dev[None, None, :]
                return (steered,) + output[1:]

        return _hook
