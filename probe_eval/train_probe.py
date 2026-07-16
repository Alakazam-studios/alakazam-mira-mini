# ABOUTME: Trains the latent-space game-state probe on REAL encoded latents (MIRA §6.2 protocol:
# ABOUTME: learn the readout on real data; rollout scoring happens in rollout_eval.py).
"""Train the shared latent-space probe.

Usage (needs the codec-bearing world-model checkpoint + a local dataset split):
    .venv/bin/python -m probe_eval.train_probe \
        --checkpoint /path/to/checkpoint-45000 --index data/rocket-science/train \
        --val-index data/rocket-science/test --out runs/probe_v1

The probe reads the world model's NORMALIZED latent space (``encode_video`` output), which is
byte-identical to what ``mira.inference.rollout`` emits — one instrument for real latents,
teacher rollouts, and student rollouts alike. Every run writes a metrics JSON artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mira.data.dataset import RocketScienceDataset

from .activations import prehead_features
from .batching import iter_probe_samples
from .loading import autocast, load_world_model_lean
from .probe import GameStateProbe, masked_mse, state_errors


@torch.no_grad()
def encode_split(model, dataset, action_config, device, *, clip_len, target_fps, stride_t,
                 max_clips, features="latents"):
    """Encode a split to (X, Y): X fp16 = flattened latents (h*w*c) or pre-head activations
    (hidden, paper §6.2/§6.7 input), Y (N, 50) fp32 targets."""
    xs, ys = [], []
    for i, sample in enumerate(iter_probe_samples(
        dataset, action_config, clip_len=clip_len, target_fps=target_fps,
        temporal_stride=stride_t, max_clips=max_clips,
    )):
        batch = sample.batch.to(device)
        with autocast(device):
            model.codec.preprocess_batch(batch)  # encode_video expects a preprocessed batch
            z = model.encode_video(batch)  # (1, t, h, w, c) normalized
            if features == "prehead":
                x = prehead_features(model, z, batch.actions.key_presses[0])  # (t, hidden)
            else:
                x = z[0].reshape(z.shape[1], -1)  # (t, h*w*c)
        t = min(x.shape[0], sample.targets.shape[0])
        xs.append(x[:t].half().cpu())
        ys.append(torch.from_numpy(sample.targets[:t]))
        if i % 200 == 0:
            print(f"  encoded {i} clip-perspectives ...", flush=True)
    return torch.cat(xs), torch.cat(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="world-model checkpoint (.pth / dir / run dir)")
    ap.add_argument("--index", required=True, help="train split dir (contains index.json)")
    ap.add_argument("--val-index", required=True, help="held-out split dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clip-len", type=int, default=78)
    ap.add_argument("--target-fps", type=int, default=20)
    ap.add_argument("--max-clips", type=int, default=2000)
    ap.add_argument("--max-val-clips", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--features", choices=["latents", "prehead"], default="latents",
                    help="probe input: codec latents, or pre-head WM activations (paper-faithful)")
    ap.add_argument("--encoded-cache", type=Path, default=Path("data/encoded"),
                    help="dir for cached encoded (X, Y) sets; re-used across probe archs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.encoded_cache.mkdir(parents=True, exist_ok=True)

    def encode_or_load(index_path: str, max_clips: int):
        feat_tag = "" if args.features == "latents" else f"_{args.features}"
        tag = (f"{Path(args.checkpoint).name}_{Path(index_path).name}_{max_clips}c"
               f"_{args.clip_len}f{args.target_fps}{feat_tag}")
        cache = args.encoded_cache / f"{tag}.pt"
        if cache.exists():
            d = torch.load(cache, weights_only=True)
            print(f"loaded encoded cache {cache} ({d['x'].shape[0]} frames)")
            return d["x"], d["y"]
        model, _ = load_world_model_lean(args.checkpoint, args.device)
        stride_t, _ = model.codec.encoder.get_downsampling_factors()
        print(f"encoding {index_path} (max {max_clips} clips, features={args.features}) ...")
        x, y = encode_split(model, RocketScienceDataset.from_local(index_path),
                            model.config.actions, args.device, clip_len=args.clip_len,
                            target_fps=args.target_fps, stride_t=stride_t, max_clips=max_clips,
                            features=args.features)
        del model
        torch.cuda.empty_cache()
        torch.save({"x": x, "y": y}, cache)
        return x, y

    x_train, y_train = encode_or_load(args.index, args.max_clips)
    x_val, y_val = encode_or_load(args.val_index, args.max_val_clips)

    probe = GameStateProbe(input_dim=x_train.shape[1], arch=args.arch).to(args.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"val_ball_pos_uu": float("inf")}
    history = []
    for epoch in range(args.epochs):
        probe.train()
        perm = torch.randperm(x_train.shape[0])
        losses = []
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = x_train[idx].float().to(args.device)
            yb = y_train[idx].to(args.device)
            loss = masked_mse(probe(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()

        probe.eval()
        with torch.no_grad():
            preds = torch.cat([
                probe(x_val[i : i + 1024].float().to(args.device)).cpu()
                for i in range(0, x_val.shape[0], 1024)
            ])
        errs = state_errors(preds, y_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in errs.items()}}
        history.append(row)
        print(row)
        if errs["ball_pos_uu"] < best["val_ball_pos_uu"]:
            best = {"epoch": epoch, "val_ball_pos_uu": errs["ball_pos_uu"]}
            torch.save({"state_dict": probe.state_dict(), "input_dim": probe.input_dim,
                        "arch": probe.arch, "checkpoint": str(args.checkpoint), "epoch": epoch},
                       args.out / "probe.pt")

    (args.out / "train_metrics.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "best": best, "history": history}, indent=2))
    print(f"best: {best}; artifacts in {args.out}")


if __name__ == "__main__":
    main()
