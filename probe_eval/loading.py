# ABOUTME: Memory-lean checkpoint loading — mmaps the .pth instead of materializing the fp32
# ABOUTME: state_dict in RAM, then casts to bf16; needed to coexist with other GPU/RAM tenants.
from __future__ import annotations

from unittest import mock

import torch

from mira.inference.loading import load_world_model
from mira.training.checkpoints import resolve_checkpoint

_ORIG_LOAD = torch.load


def _mmap_load(*args, **kwargs):
    kwargs.setdefault("mmap", True)
    kwargs.setdefault("map_location", "cpu")
    return _ORIG_LOAD(*args, **kwargs)


def load_world_model_lean(checkpoint: str, device: str, dtype: torch.dtype = torch.bfloat16):
    """``mira.inference.loading.load_world_model`` with mmap'd torch.load and a dtype cast.

    The release loader materializes the full fp32 state_dict in RAM on top of the fp32-built
    model (~20 GB transient for the 1.18B + codec checkpoint). mmap keeps the file on disk and
    load_state_dict copies tensor-by-tensor; the model then moves to ``device`` as ``dtype``.

    Call the model under ``autocast(device)`` (below): preprocessing emits fp32 video, which
    autocast reconciles with the bf16 weights per-op.
    """
    with mock.patch.object(torch, "load", _mmap_load):
        model, run_config = load_world_model(resolve_checkpoint(checkpoint), "cpu")
    return model.to(dtype).to(device).eval(), run_config


def autocast(device: str, dtype: torch.dtype = torch.bfloat16):
    """Autocast context for encode/rollout on a bf16-cast model."""
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def chunk_dino_forward(model, max_b: int = 1) -> None:
    """Make the codec ENCODER process ≤max_b batch rows per forward and drop the retained
    DINO features (the world-model path consumes only `.z`). Exact per-row; needed for
    4-view MP batches on a shared 24 GB card."""
    from mira.codec.rae_encoder import RAEEncoderOutputs

    encoder = model.codec.encoder
    orig = encoder.forward

    def chunked(video):
        if video.shape[0] <= max_b:
            return orig(video)
        zs = []
        for i in range(0, video.shape[0], max_b):
            out = orig(video[i:i + max_b])
            zs.append(out.z)
            del out
            torch.cuda.empty_cache()
        return RAEEncoderOutputs(z=torch.cat(zs, dim=0), dino_features=None)

    encoder.forward = chunked
