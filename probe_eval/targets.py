# ABOUTME: Privileged-physics → probe-target extraction: (T, 50) tensors of normalized
# ABOUTME: ball+car state (pos/quat/linvel × 5 entities) with NaN masking, aligned to latent frames.
"""Numpy-only target extraction, mirroring the style of :mod:`mira.data.physics`.

Target layout per frame (MIRA §6.2: "position, quaternion, and linear velocity for each of the
four players and the ball, 50 dimensions in total"): entity order is ball first, then the four
cars sorted by ``player_id``; each entity contributes [pos(3), quat(4), linvel(3)].

Normalization (documented deviation — the paper is silent): positions are divided by the arena
half-extents, velocities by the engine speed caps (Appendix C, Table 19), quaternions are
unit-normalized with the double-cover collapsed to w >= 0. Invalid fields are NaN, and the loss
masks NaNs — this covers demolished cars (stale frozen state), missing optional rotation, and
whole frames whose live physics is frozen (goal pauses / replays) while the video keeps rolling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mira.data.physics import KICKOFF, LIVE, step_badges
from mira.data.state import CarState, FrameState

if TYPE_CHECKING:
    from mira.data.dataset import MatchClip

# Arena half-extents (uu) and engine speed caps (uu/s) — MIRA Appendix C, Table 19.
POS_SCALE = np.array([4096.0, 5120.0, 2044.0], dtype=np.float32)
BALL_VEL_SCALE = 6000.0
CAR_VEL_SCALE = 2300.0

N_ENTITIES = 5  # ball + 4 cars
ENTITY_DIM = 10  # pos(3) + quat(4) + linvel(3)
TARGET_DIM = N_ENTITIES * ENTITY_DIM  # 50

# Slices into one entity's 10-dim block.
POS = slice(0, 3)
QUAT = slice(3, 7)
VEL = slice(7, 10)


def _vec3(d) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float32)


def canonical_quat(d) -> np.ndarray:
    """Unit quaternion as (x, y, z, w) with the double cover collapsed (w >= 0)."""
    q = np.array([d["x"], d["y"], d["z"], d["w"]], dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-6:
        return np.full(4, np.nan, dtype=np.float32)
    q = q / norm
    return -q if q[3] < 0 else q


def _entity_vec(location, rotation, velocity, vel_scale: float) -> np.ndarray:
    """One entity's normalized 10-dim block; ``rotation=None`` yields NaN quat dims."""
    out = np.empty(ENTITY_DIM, dtype=np.float32)
    out[POS] = _vec3(location) / POS_SCALE
    out[QUAT] = canonical_quat(rotation) if rotation is not None else np.nan
    out[VEL] = _vec3(velocity) / vel_scale
    return out


def _car_demolished(car: CarState) -> bool:
    # ``is_demolished`` is broken in the data (always false); the demolisher's id is the signal.
    return int(car.get("attacker_player_id", -1)) != -1


def frame_target(frame: FrameState) -> np.ndarray:
    """(50,) target for one physics frame: ball, then cars by player_id; NaN for invalid fields."""
    out = np.empty(TARGET_DIM, dtype=np.float32)
    ball = frame["ball"]
    out[0:ENTITY_DIM] = _entity_vec(ball["location"], ball.get("rotation"), ball["velocity"], BALL_VEL_SCALE)
    cars = sorted(frame["cars"], key=lambda c: int(c["player_id"]))
    for i, car in enumerate(cars):
        block = slice((i + 1) * ENTITY_DIM, (i + 2) * ENTITY_DIM)
        if _car_demolished(car):
            out[block] = np.nan  # demolished cars carry stale frozen state
        else:
            out[block] = _entity_vec(car["location"], car.get("rotation"), car["velocity"], CAR_VEL_SCALE)
    return out


def sequence_targets(persp_physics: list[FrameState], keep: list[bool] | None = None) -> np.ndarray:
    """(T, 50) targets for one perspective; frames with ``keep[t] == False`` are all-NaN."""
    out = np.stack([frame_target(fr) for fr in persp_physics])
    if keep is not None:
        if len(keep) != len(out):
            raise ValueError(f"keep has length {len(keep)}, physics has {len(out)} frames")
        out[~np.array(keep, dtype=bool)] = np.nan
    return out


def live_frames(clip: "MatchClip", perspective: int) -> list[bool]:
    """Per-frame flag: physics is live (LIVE/KICKOFF), not frozen by a goal pause or replay."""
    return [b.code in (LIVE, KICKOFF) for b in step_badges(clip, perspective)]


def clip_targets(clip: "MatchClip", perspective: int) -> np.ndarray:
    """(T, 50) probe targets for one clip perspective, frozen-physics frames masked to NaN."""
    if clip.physics is None:
        raise ValueError("clip carries no physics; load the dataset with physics members present")
    return sequence_targets(clip.physics[perspective], keep=live_frames(clip, perspective))


def align_to_latents(per_frame: np.ndarray, temporal_stride: int) -> np.ndarray:
    """Subsample per-video-frame rows to per-latent-frame rows.

    The codec aggregates ``temporal_stride`` consecutive frames into one latent; we label each
    latent with the LAST video frame of its window (the latest state it has seen). A trailing
    partial window is dropped, matching the encoder's floor division.
    """
    t = per_frame.shape[0] - (per_frame.shape[0] % temporal_stride)
    return per_frame[temporal_stride - 1 : t : temporal_stride]
