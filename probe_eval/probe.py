# ABOUTME: The game-state probe (3-layer MLP, hidden 1024 — MIRA §6.2) plus the NaN-masked
# ABOUTME: L2 loss and denormalized state-error metrics (uu / uu·s⁻¹ / radians).
from __future__ import annotations

import torch
from torch import nn

from .targets import BALL_VEL_SCALE, CAR_VEL_SCALE, ENTITY_DIM, N_ENTITIES, POS, POS_SCALE, QUAT, TARGET_DIM, VEL


class GameStateProbe(nn.Module):
    """Three-layer MLP with hidden dim 1024 mapping a per-latent-frame feature vector to the
    50-dim game state (MIRA §6.2). The input is whatever representation the caller reads —
    flattened codec latents (shared instrument across models) or pooled pre-head activations
    (paper-faithful, per-model)."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024, target_dim: int = TARGET_DIM,
                 arch: str = "mlp"):
        super().__init__()
        self.input_dim = input_dim
        self.arch = arch
        if arch == "linear":
            # Apples-to-apples with the paper's §6.7 scaling probe (linear readout).
            self.net = nn.Linear(input_dim, target_dim)
        elif arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, target_dim),
            )
        else:
            raise ValueError(f"unknown probe arch {arch!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def for_latents(cls, h: int = 9, w: int = 16, c: int = 32, **kw) -> "GameStateProbe":
        """Probe over flattened single-view codec latents (released codec: 9×16×32 at 288×512)."""
        return cls(input_dim=h * w * c, **kw)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the non-NaN target entries (invalid fields and frozen frames contribute nothing)."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    diff = pred[mask] - target[mask]
    return (diff * diff).mean()


def _masked_mean(x: torch.Tensor) -> float:
    valid = ~torch.isnan(x)
    return float(x[valid].mean()) if valid.any() else float("nan")


def state_errors(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Denormalized errors between (N, 50) prediction and target batches.

    Returns ball/car position L2 (uu), ball/car velocity L2 (uu/s), and quaternion geodesic
    error (radians, double-cover-invariant), each averaged over valid (non-NaN) entries.
    """
    p = pred.detach().reshape(-1, N_ENTITIES, ENTITY_DIM)
    t = target.detach().reshape(-1, N_ENTITIES, ENTITY_DIM)
    pos_scale = torch.as_tensor(POS_SCALE, device=p.device)

    pos_err = torch.linalg.norm((p[..., POS] - t[..., POS]) * pos_scale, dim=-1)  # (N, 5) uu
    vel_scale = torch.tensor([BALL_VEL_SCALE] + [CAR_VEL_SCALE] * 4, device=p.device)
    vel_err = torch.linalg.norm(p[..., VEL] - t[..., VEL], dim=-1) * vel_scale  # (N, 5) uu/s

    pq = torch.nn.functional.normalize(p[..., QUAT], dim=-1, eps=1e-6)
    dot = (pq * t[..., QUAT]).sum(-1).abs().clamp(max=1.0)
    quat_err = 2.0 * torch.arccos(dot)  # (N, 5) radians; NaN target quats propagate

    return {
        "ball_pos_uu": _masked_mean(pos_err[:, 0]),
        "car_pos_uu": _masked_mean(pos_err[:, 1:]),
        "ball_vel_uu_s": _masked_mean(vel_err[:, 0]),
        "car_vel_uu_s": _masked_mean(vel_err[:, 1:]),
        "quat_geodesic_rad": _masked_mean(quat_err),
        "n_valid_frames": int((~torch.isnan(t).all(dim=(1, 2))).sum()),
    }
