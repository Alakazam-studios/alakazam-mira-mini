# ABOUTME: One-pass sequence-level cache builder: per clip-perspective {latents, actions, targets}.
# ABOUTME: Decoding + codec encoding happen ONCE; per-model pre-head extraction then needs only DiT forwards.
"""Usage:
    .venv/bin/python -m probe_eval.extract_sequences \
        --checkpoint ckpts/wm1b_100k/checkpoint-45000 --index data/rocket-science/train \
        --out data/encoded/train_sequences.pt --max-clips 2400

The codec is frozen and shared by every model in the family, so the latent sequences cached here
feed pre-head extraction for ANY model (teacher or student) without re-decoding video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mira.data.dataset import RocketScienceDataset

from .batching import iter_probe_samples
from .loading import autocast, load_world_model_lean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="any model in the family (codec source)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clip-len", type=int, default=78)
    ap.add_argument("--target-fps", type=int, default=20)
    ap.add_argument("--max-clips", type=int, default=2400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, _ = load_world_model_lean(args.checkpoint, args.device)
    stride_t, _ = model.codec.encoder.get_downsampling_factors()

    records = []
    with torch.no_grad():
        for i, sample in enumerate(iter_probe_samples(
            RocketScienceDataset.from_local(args.index), model.config.actions,
            clip_len=args.clip_len, target_fps=args.target_fps,
            temporal_stride=stride_t, max_clips=args.max_clips,
        )):
            batch = sample.batch.to(args.device)
            with autocast(args.device):
                model.codec.preprocess_batch(batch)
                z = model.encode_video(batch)  # (1, t, h, w, c)
            records.append({
                "z": z[0].half().cpu(),                                # (t, h, w, c)
                "actions": batch.actions.key_presses[0].to(torch.int8).cpu(),  # (T, 9)
                "targets": torch.from_numpy(sample.targets).half(),    # (t, 50)
                "match_id": sample.match_id, "clip_id": sample.clip_id,
                "perspective": sample.perspective,
            })
            if i % 200 == 0:
                print(f"  cached {i} clip-perspectives ...", flush=True)

    torch.save({"records": records, "temporal_stride": stride_t,
                "args": {k: str(v) for k, v in vars(args).items()}}, args.out)
    print(f"wrote {len(records)} sequences to {args.out}")


if __name__ == "__main__":
    main()
