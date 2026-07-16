# ABOUTME: Rollout-time probe evaluation — instrument #1 (probe state error vs ground truth per
# ABOUTME: model) and instrument #2 (seed-matched teacher↔student state divergence). JSON artifacts.
"""Score physics retention of one or more checkpoints with a trained latent probe.

Usage:
    .venv/bin/python -m probe_eval.rollout_eval \
        --probe runs/probe_v1/probe.pt \
        --model teacher45k=ckpts/wm1b_100k/checkpoint-45000 \
        --model psd10k-2step=ckpts/wm1b_psd10k/checkpoint-10000@2 \
        --index data/rocket-science/test --out runs/rollout_eval_v1 --seeds 1234 1235

Protocol (MIRA §6.2): encode a real clip, roll out conditioned on the clip's REAL actions from a
real context, probe the generated latents, and compare to the real clip's privileged state.
Rollouts use the release's determinism anchor (fixed seed + noise_level=0.0 + linear schedule),
so with the same seed every model sees identical context and noise draws — any state divergence
between two models is attributable to the models.

Models are loaded ONE AT A TIME (a bf16 1.18B + codec is ~4.5 GB; several do not fit a shared
GPU), each making its own deterministic pass over the same clips; probe states are tiny and kept
in RAM, divergence is computed at the end. `--model name=ckpt[@steps]` (default 8 sampling steps).
Single-player checkpoints only (v1).
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mira.data.dataset import RocketScienceDataset
from mira.inference.rollout import rollout
from mira.world_model.config import WorldModelInferenceConfig

from .batching import iter_probe_samples
from .loading import autocast, load_world_model_lean
from .probe import GameStateProbe, state_errors

HORIZONS_S = (1.0, 2.0, 4.0)


def load_probe(path: str, device: str) -> GameStateProbe:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = GameStateProbe(input_dim=ckpt["input_dim"], arch=ckpt.get("arch", "mlp")).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    return probe.eval()


@torch.no_grad()
def probe_states(probe: GameStateProbe, z: torch.Tensor) -> torch.Tensor:
    """(1, t, h, w, c) latents → (t, 50) probe-decoded state."""
    return probe(z[0].reshape(z.shape[1], -1).float())


def horizon_slices(n_gen: int, latent_fps: float) -> dict[str, slice]:
    """Cumulative slices of the generated window at each horizon (1 s → first 10 latents @10 Hz)."""
    out = {}
    for h in HORIZONS_S:
        n = min(int(round(h * latent_fps)), n_gen)
        if n > 0:
            out[f"{h:g}s"] = slice(0, n)
    return out


def iter_clips(ds, action_config, args, stride_t):
    """The shared, deterministic clip stream every model pass sees identically."""
    return itertools.islice(
        iter_probe_samples(ds, action_config, clip_len=args.clip_len,
                           target_fps=args.target_fps, temporal_stride=stride_t),
        args.n_clips,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CKPT[@STEPS]",
                    help="repeatable; e.g. psd10k-2step=ckpts/wm1b_psd10k/checkpoint-10000@2")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-clips", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235])
    ap.add_argument("--clip-len", type=int, default=78)
    ap.add_argument("--target-fps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = []  # (name, ckpt, n_steps)
    for spec in args.model:
        name, _, rest = spec.partition("=")
        ckpt, _, steps = rest.partition("@")
        specs.append((name, ckpt, int(steps) if steps else 8))

    probe = load_probe(args.probe, args.device)
    ds = RocketScienceDataset.from_local(args.index)

    # Pass 0 with the first model: ground truth, real-latent ceiling, cached real latents
    # (fp16 CPU) for the per-model context-alignment check.
    first_name, first_ckpt, _ = specs[0]
    model, _ = load_world_model_lean(first_ckpt, args.device)
    stride_t, _ = model.codec.encoder.get_downsampling_factors()
    action_config = model.config.actions
    n_ctx_lat = model.config.n_context_frames // stride_t
    latent_fps = args.target_fps / stride_t

    clips, gts, z_reals, ceilings, actions = [], [], [], [], []
    for sample in iter_clips(ds, action_config, args, stride_t):
        actions.append(sample.batch.actions.key_presses[0].to(torch.int8).cpu().numpy())
        b = sample.batch.to(args.device).clone()
        with torch.no_grad(), autocast(args.device):
            model.codec.preprocess_batch(b)
            z_real = model.encode_video(b)
        states_real = probe_states(probe, z_real).cpu()
        gt = torch.from_numpy(sample.targets)
        t_common = min(states_real.shape[0], gt.shape[0])
        gen = slice(n_ctx_lat, t_common)
        clips.append({"match_id": sample.match_id, "clip_id": sample.clip_id,
                      "perspective": sample.perspective})
        gts.append(gt)
        z_reals.append(z_real.half().cpu())
        ceilings.append({h: state_errors(states_real[gen][s], gt[gen][s])
                         for h, s in horizon_slices(t_common - n_ctx_lat, latent_fps).items()})
    print(f"pass 0 ({first_name}): {len(clips)} clips prepared (gt + ceiling + real latents)")

    # Per-model passes: deterministic re-iteration of the same clips, one model resident at a time.
    all_states: dict[tuple[str, int, int], torch.Tensor] = {}  # (model, clip_idx, seed) -> states
    all_latents: dict[str, np.ndarray] = {}  # "model|clip|seed" -> (t, h, w, c) fp16
    model_errors: dict[str, list[dict]] = {}
    current = model
    del model  # `current` is the ONLY reference; a lingering alias here kept two models on the GPU
    for name, ckpt, n_steps in specs:
        if current is None:
            current, _ = load_world_model_lean(ckpt, args.device)
        inf_cfg = WorldModelInferenceConfig(
            n_diffusion_steps=n_steps, noise_level=0.0, schedule_type="linear")
        model_errors[name] = []
        for i, sample in enumerate(iter_clips(ds, action_config, args, stride_t)):
            raw = sample.batch.to(args.device)
            gt = gts[i]
            t_common = min(z_reals[i].shape[1], gt.shape[0])
            gen = slice(n_ctx_lat, t_common)
            per_seed = {}
            for seed in args.seeds:
                torch.manual_seed(seed)
                with torch.no_grad(), autocast(args.device):
                    z = rollout(current, raw.clone(), inf_cfg)  # preprocesses its clone in place
                ctx_gap = float((z[:, :n_ctx_lat].half().cpu() - z_reals[i][:, :n_ctx_lat])
                                .abs().max())
                if ctx_gap > 0.1:
                    raise RuntimeError(f"context misalignment: {name} clip {i} max |Δ| {ctx_gap}")
                states = probe_states(probe, z).cpu()[gen]
                all_states[(name, i, seed)] = states
                all_latents[f"{name}|{i}|{seed}"] = z[0].half().cpu().numpy()
                per_seed[str(seed)] = {h: state_errors(states[s], gt[gen][s])
                                       for h, s in horizon_slices(states.shape[0], latent_fps).items()}
            model_errors[name].append(per_seed)
            print(f"  {name} [{i + 1}/{len(clips)}]", flush=True)
        del current
        gc.collect()
        torch.cuda.empty_cache()
        current = None
        print(f"pass done: {name} ({n_steps} steps)")

    # Instrument #2: seed-matched cross-model state divergence on identical noise draws.
    divergence: dict[str, dict] = {}
    names = [s[0] for s in specs]
    for a, b in itertools.combinations(names, 2):
        pairs = {}
        for i in range(len(clips)):
            for seed in args.seeds:
                sa, sb = all_states[(a, i, seed)], all_states[(b, i, seed)]
                n = min(sa.shape[0], sb.shape[0])
                pairs[f"clip{i}@{seed}"] = {h: state_errors(sa[:n][s], sb[:n][s].clone())
                                            for h, s in horizon_slices(n, latent_fps).items()}
        divergence[f"{a}~{b}"] = pairs

    def agg(entries):
        vals: dict[str, list[float]] = {}
        for errs_by_h in entries:
            for h, errs in errs_by_h.items():
                for k, v in errs.items():
                    if isinstance(v, float) and not np.isnan(v):
                        vals.setdefault(f"{h}/{k}", []).append(v)
        return {k: float(np.mean(v)) for k, v in sorted(vals.items())}

    summary = {"ceiling": agg(ceilings)}
    for name in names:
        summary[name] = agg([s for clip in model_errors[name] for s in clip.values()])
    for pair, entries in divergence.items():
        summary[f"div:{pair}"] = agg(entries.values())

    # Cache everything a probe re-score needs: rollout + real latents (fp16) and GT targets.
    # A better probe can then re-score this run in seconds (see rescore.py) instead of
    # re-running hours of rollouts.
    np.savez_compressed(
        args.out / "latents.npz",
        **all_latents,
        **{f"real|{i}": z[0].numpy() for i, z in enumerate(z_reals)},
        **{f"gt|{i}": g.numpy() for i, g in enumerate(gts)},
        **{f"actions|{i}": a for i, a in enumerate(actions)},  # for pre-head re-forwarding
    )
    (args.out / "rollout_eval.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "models": {n: {"checkpoint": c, "n_steps": s} for n, c, s in specs},
        "n_ctx_lat": n_ctx_lat, "latent_fps": latent_fps,
        "summary": summary, "clips": clips,
        "ceiling": ceilings, "model_errors": model_errors, "divergence": divergence,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
