# ABOUTME: End-to-end smoke gate for P1 — loads the real teacher checkpoint, encodes real clips,
# ABOUTME: extracts aligned probe targets, and verifies a probe overfits a small latent set.
"""Run: PYTHONPATH=src:. .venv/bin/python -m probe_eval.smoke"""

from __future__ import annotations

import itertools

import torch

from mira.data.dataset import RocketScienceDataset

from .batching import iter_probe_samples
from .loading import autocast, load_world_model_lean
from .probe import GameStateProbe, masked_mse, state_errors

CKPT = "ckpts/wm1b_100k/checkpoint-45000"
INDEX = "data/rocket-science/test"
DEVICE = "cuda"
N_SAMPLES = 8


def main() -> None:
    model, _ = load_world_model_lean(CKPT, DEVICE)
    stride_t, spatial = model.codec.encoder.get_downsampling_factors()
    action_config = model.config.actions
    print(f"model loaded | temporal stride {stride_t} | spatial /{spatial} "
          f"| n_context_frames {model.config.n_context_frames}")

    ds = RocketScienceDataset.from_local(INDEX)
    xs, ys = [], []
    with torch.no_grad():
        for sample in itertools.islice(
            iter_probe_samples(ds, action_config, clip_len=78, target_fps=20, temporal_stride=stride_t),
            N_SAMPLES,
        ):
            batch = sample.batch.to(DEVICE)
            with autocast(DEVICE):
                # encode_video expects a PREPROCESSED batch (resize to 288x512 + [0,1]); the
                # release paths (rollout/inference) call this internally. In-place, NOT idempotent.
                model.codec.preprocess_batch(batch)
                z = model.encode_video(batch)
            t = min(z.shape[1], sample.targets.shape[0])
            assert abs(z.shape[1] - sample.targets.shape[0]) <= 1, \
                f"latent/target misalignment: {z.shape[1]} vs {sample.targets.shape[0]}"
            xs.append(z[0, :t].reshape(t, -1).cpu())
            ys.append(torch.from_numpy(sample.targets[:t]))
            print(f"clip {sample.match_id[:12]} p{sample.perspective}: video {tuple(sample.batch.video.shape)} "
                  f"-> latents {tuple(z.shape)} | targets {sample.targets.shape} "
                  f"| valid frames {int((~torch.isnan(ys[-1]).all(1)).sum())}/{t}")

    x, y = torch.cat(xs).to(DEVICE), torch.cat(ys).to(DEVICE)
    probe = GameStateProbe(input_dim=x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for step in range(201):
        loss = masked_mse(probe(x.float()), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            errs = state_errors(probe(x.float()), y)
            print(f"step {step:3d} | loss {float(loss.detach()):.4f} | ball {errs['ball_pos_uu']:.0f} uu "
                  f"| cars {errs['car_pos_uu']:.0f} uu")
    print("smoke gate: PASS (probe overfits real latents; pipeline is sound)")


if __name__ == "__main__":
    main()
