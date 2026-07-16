# ABOUTME: Clip → VideoActionBatch helpers for probe work: single-perspective batches paired
# ABOUTME: with latent-aligned probe targets (the training loader drops physics; this keeps it).
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionTensors

from .targets import align_to_latents, clip_targets

if TYPE_CHECKING:
    from mira.data.dataset import MatchClip
    from mira.world_model.actions_config import ActionConfig


def perspective_batch(
    clip: "MatchClip", perspective: int, action_config: "ActionConfig", frame_size: tuple[int, int] | None = None
) -> VideoActionBatch:
    """One perspective of a clip as a batch-of-1, built exactly like the training loader's
    ``_decode_sample`` (keyboard-only: zero mouse, NaN sensitivity)."""
    actions = ActionTensors(config=action_config, batch_size=1)
    actions.key_presses = torch.as_tensor(clip.actions[perspective]).unsqueeze(0).to(torch.int32)
    n_steps = actions.key_presses.shape[1]
    actions.mouse_movements = torch.zeros((1, n_steps, 2), dtype=torch.float32)
    actions.game_mouse_sensitivity = torch.full((1,), float("nan"), dtype=torch.float32)
    video = clip.decode_perspective(perspective, frame_size).unsqueeze(0)  # (1, T, C, H, W) uint8
    return VideoActionBatch(video=video, actions=actions)


@dataclass
class ProbeSample:
    """One perspective's model-ready batch plus its latent-aligned probe targets."""

    batch: VideoActionBatch
    targets: np.ndarray  # (T_latent, 50), NaN-masked
    match_id: str
    clip_id: str
    perspective: int


def iter_probe_samples(
    dataset,
    action_config: "ActionConfig",
    *,
    clip_len: int,
    target_fps: int,
    temporal_stride: int,
    frame_size: tuple[int, int] | None = None,
    exclude_replays: bool = True,
    max_clips: int | None = None,
    seed: int = 37,
) -> Iterator[ProbeSample]:
    """Stream (batch, aligned-targets) pairs over a ``RocketScienceDataset``.

    Frozen-physics frames are NaN-masked in the targets rather than skipped, so alignment with
    the latent stream is never disturbed; ``exclude_replays`` additionally drops clips that
    overlap goal replays entirely (same knob as the release eval loader).
    """
    n = 0
    for clip in dataset.iter_clips(
        clip_len=clip_len,
        target_fps=target_fps,
        exclude_replays=exclude_replays,
        perspective="all",
        frame_size=frame_size,
        seed=seed,
        carry_video=True,
        decode=False,
    ):
        if clip.physics is None:
            continue
        for p in range(len(clip.perspectives)):
            yield ProbeSample(
                batch=perspective_batch(clip, p, action_config, frame_size),
                targets=align_to_latents(clip_targets(clip, p), temporal_stride),
                match_id=clip.match_id,
                clip_id=str(clip.clip_id),
                perspective=p,
            )
        n += 1
        if max_clips is not None and n >= max_clips:
            return
