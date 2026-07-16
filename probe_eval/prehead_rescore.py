# ABOUTME: Scores cached ladder rollouts with a per-model PRE-HEAD probe (report §17.1 protocol):
# ABOUTME: model M re-processes real + its own rollout latents; probe reads M's activations.
"""Usage (one invocation per model whose activations the probe was trained on):
    .venv/bin/python -m probe_eval.prehead_rescore \
        --run runs/rollout_eval_v3 --checkpoint ckpts/wm1b_100k/checkpoint-45000 \
        --probe-dir runs/prehead_teacher45k --model-key teacher45k --out runs/prehead_scores

Emits floor (probe on real latents through M), generated (probe on M's rollout latents through
M), and the gen/floor ratio — the §17.1 decision variable. Pre-head feature spaces differ per
model, so cross-model divergence is NOT computed here (that stays with the shared latent probe).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activations import prehead_features
from .loading import autocast, load_world_model_lean
from .probe import GameStateProbe, state_errors
from .rollout_eval import horizon_slices
from .train_prehead_probe import ridge_predict


def load_heads(probe_dir: Path, device: str):
    heads = {}
    for p in sorted(probe_dir.glob("probe_*.pt")):
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        name = p.stem.replace("probe_", "")
        if ckpt.get("kind") == "ridge":
            heads[name] = ("ridge", ckpt["w"])
        else:
            probe = GameStateProbe(input_dim=ckpt["input_dim"], arch=ckpt["arch"])
            probe.load_state_dict(ckpt["state_dict"])
            heads[name] = ("torch", probe.eval())
    if not heads:
        raise SystemExit(f"no probe_*.pt in {probe_dir}")
    return heads


def predict(head, x: torch.Tensor) -> torch.Tensor:
    kind, obj = head
    if kind == "ridge":
        return ridge_predict(obj, x)
    with torch.no_grad():
        return obj(x.float())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="ladder dir with latents.npz (with actions)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--model-key", required=True, help="npz model key, e.g. teacher45k")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.run / "rollout_eval.json").read_text())
    n_ctx_lat, latent_fps = meta["n_ctx_lat"], meta["latent_fps"]
    n_clips = len(meta["clips"])
    seeds = [int(s) for s in meta["args"]["seeds"].strip("[]").split(",")]
    data = np.load(args.run / "latents.npz")
    if f"actions|0" not in data:
        raise SystemExit(f"{args.run}/latents.npz has no cached actions — re-run rollout_eval")

    heads = load_heads(args.probe_dir, args.device)
    model, _ = load_world_model_lean(args.checkpoint, args.device)

    def states(key: str, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.from_numpy(data[key]).unsqueeze(0).to(args.device).to(torch.bfloat16)
        with autocast(args.device):
            x = prehead_features(model, z, actions).cpu()
        return {name: predict(head, x) for name, head in heads.items()}

    floors, gens = {n: [] for n in heads}, {n: [] for n in heads}
    for i in range(n_clips):
        actions = torch.from_numpy(data[f"actions|{i}"]).to(torch.int32)
        gt = torch.from_numpy(data[f"gt|{i}"]).float()
        real_states = states(f"real|{i}", actions)
        t_common = min(next(iter(real_states.values())).shape[0], gt.shape[0])
        gen = slice(n_ctx_lat, t_common)
        hs = horizon_slices(t_common - n_ctx_lat, latent_fps)
        for name, st in real_states.items():
            floors[name].append({h: state_errors(st[gen][s], gt[gen][s]) for h, s in hs.items()})
        for seed in seeds:
            gen_states = states(f"{args.model_key}|{i}|{seed}", actions)
            for name, st in gen_states.items():
                gens[name].append({h: state_errors(st[gen][s], gt[gen][s]) for h, s in hs.items()})
        print(f"  clip {i + 1}/{n_clips}", flush=True)

    def agg(entries):
        vals: dict[str, list[float]] = {}
        for e in entries:
            for h, errs in e.items():
                for k, v in errs.items():
                    if isinstance(v, float) and not np.isnan(v):
                        vals.setdefault(f"{h}/{k}", []).append(v)
        return {k: float(np.mean(v)) for k, v in sorted(vals.items())}

    summary = {}
    for name in heads:
        f, g = agg(floors[name]), agg(gens[name])
        summary[name] = {"floor": f, "generated": g,
                         "ratio": {k: (g[k] / f[k] if f.get(k) else None)
                                   for k in g if k in f and "pos_uu" in k}}
    out_file = args.out / f"{args.model_key}.json"
    out_file.write_text(json.dumps({
        "model_key": args.model_key, "checkpoint": args.checkpoint,
        "probe_dir": str(args.probe_dir), "run": str(args.run), "summary": summary,
        "floors": floors, "generated": gens}, indent=2))
    print(json.dumps({k: v["ratio"] for k, v in summary.items()}, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
