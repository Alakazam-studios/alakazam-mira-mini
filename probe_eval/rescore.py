# ABOUTME: Re-scores a cached rollout_eval run (latents.npz) with a different probe —
# ABOUTME: probe iterations become seconds instead of hours of GPU rollouts.
"""Usage:
    .venv/bin/python -m probe_eval.rescore \
        --run runs/rollout_eval_v1 --probe runs/probe_v2/probe.pt --out runs/rollout_eval_v1_p2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .probe import state_errors
from .rollout_eval import horizon_slices, load_probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="a completed rollout_eval output dir")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not (args.run / "latents.npz").exists():
        raise SystemExit(f"{args.run} has no latents.npz — it predates latent caching; "
                         "re-run rollout_eval once to build the cache.")
    meta = json.loads((args.run / "rollout_eval.json").read_text())
    n_ctx_lat, latent_fps = meta["n_ctx_lat"], meta["latent_fps"]
    names = list(meta["models"])
    n_clips = len(meta["clips"])
    seeds = [int(s) for s in meta["args"]["seeds"].strip("[]").split(",")]
    probe = load_probe(args.probe, args.device)
    data = np.load(args.run / "latents.npz")

    @torch.no_grad()
    def states_of(key: str) -> torch.Tensor:
        z = torch.from_numpy(data[key]).to(args.device).float()  # (t, h, w, c)
        return probe(z.reshape(z.shape[0], -1)).cpu()

    gts = [torch.from_numpy(data[f"gt|{i}"]) for i in range(n_clips)]
    gens, ceilings = [], []
    for i in range(n_clips):
        real = states_of(f"real|{i}")
        gen = slice(n_ctx_lat, min(real.shape[0], gts[i].shape[0]))
        gens.append(gen)
        ceilings.append({h: state_errors(real[gen][s], gts[i][gen][s])
                         for h, s in horizon_slices(gen.stop - gen.start, latent_fps).items()})

    all_states = {(n, i, s): states_of(f"{n}|{i}|{s}")[gens[i]]
                  for n in names for i in range(n_clips) for s in seeds}
    model_errors = {n: [{str(s): {h: state_errors(all_states[(n, i, s)][sl], gts[i][gens[i]][sl])
                                  for h, sl in horizon_slices(all_states[(n, i, s)].shape[0], latent_fps).items()}
                         for s in seeds} for i in range(n_clips)] for n in names}
    divergence = {}
    for a, b in itertools.combinations(names, 2):
        divergence[f"{a}~{b}"] = {
            f"clip{i}@{s}": {h: state_errors(all_states[(a, i, s)][:n_min][sl],
                                             all_states[(b, i, s)][:n_min][sl].clone())
                             for h, sl in horizon_slices(n_min, latent_fps).items()}
            for i in range(n_clips) for s in seeds
            if (n_min := min(all_states[(a, i, s)].shape[0], all_states[(b, i, s)].shape[0]))}

    def agg(entries):
        vals: dict[str, list[float]] = {}
        for errs_by_h in entries:
            for h, errs in errs_by_h.items():
                for k, v in errs.items():
                    if isinstance(v, float) and not np.isnan(v):
                        vals.setdefault(f"{h}/{k}", []).append(v)
        return {k: float(np.mean(v)) for k, v in sorted(vals.items())}

    summary = {"ceiling": agg(ceilings)}
    for n in names:
        summary[n] = agg([s for clip in model_errors[n] for s in clip.values()])
    for pair, entries in divergence.items():
        summary[f"div:{pair}"] = agg(entries.values())

    (args.out / "rollout_eval.json").write_text(json.dumps({
        "rescored_from": str(args.run), "probe": args.probe,
        "models": meta["models"], "n_ctx_lat": n_ctx_lat, "latent_fps": latent_fps,
        "summary": summary, "clips": meta["clips"],
        "ceiling": ceilings, "model_errors": model_errors, "divergence": divergence,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
