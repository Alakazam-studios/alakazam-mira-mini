# ABOUTME: Per-model pre-head probes (paper §6.2/§6.7, Hugo's §17.1): linear (SGD), ridge
# ABOUTME: (closed-form), and MLP heads over pre-head activations from cached latent sequences.
"""Usage (one invocation per model — pre-head activations are model-specific):
    .venv/bin/python -m probe_eval.train_prehead_probe \
        --checkpoint ckpts/wm1b_100k/checkpoint-45000 \
        --train-cache data/encoded/train_sequences.pt \
        --val-cache data/encoded/test_sequences.pt \
        --out runs/prehead_teacher45k

Trains all three heads on the SAME features; ridge is the closed-form readout matching the
stricter instrument of the report's §17.1, linear matches the paper's §6.7 scaling probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activations import prehead_features
from .loading import autocast, load_world_model_lean
from .probe import GameStateProbe, masked_mse, state_errors


@torch.no_grad()
def features_for_cache(model, cache_path: Path, device: str, max_clips: int | None = None):
    """(X features fp32 kept on CPU, Y targets) from a sequence cache, via DiT-only forwards."""
    d = torch.load(cache_path, weights_only=True)
    xs, ys = [], []
    for i, rec in enumerate(d["records"][: max_clips]):
        z = rec["z"].unsqueeze(0).to(device).to(torch.bfloat16)
        with autocast(device):
            x = prehead_features(model, z, rec["actions"])  # (t, hidden)
        t = min(x.shape[0], rec["targets"].shape[0])
        xs.append(x[:t].half().cpu())
        ys.append(rec["targets"][:t].float())
        if i % 500 == 0:
            print(f"  features {i}/{len(d['records'])} ...", flush=True)
    return torch.cat(xs), torch.cat(ys)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge W (d+1, 50) with bias, on rows whose targets are fully valid.
    Gram matrices accumulate chunk-wise in float64 so RAM stays at O(d^2), not O(N*d)."""
    keep = ~torch.isnan(y).any(dim=1)
    xk, yk = x[keep], y[keep]
    d = xk.shape[1] + 1
    a = torch.zeros(d, d, dtype=torch.float64)
    b = torch.zeros(d, yk.shape[1], dtype=torch.float64)
    for i in range(0, xk.shape[0], 8192):
        xb = torch.cat([xk[i:i + 8192].double(), torch.ones(min(8192, xk.shape[0] - i), 1,
                                                            dtype=torch.float64)], dim=1)
        a += xb.T @ xb
        b += xb.T @ yk[i:i + 8192].double()
    a += lam * xk.shape[0] * torch.eye(d, dtype=torch.float64)
    return torch.linalg.solve(a, b).float()  # (d+1, 50)


def ridge_predict(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1) @ w


def train_sgd(x, y, x_val, y_val, arch: str, device: str, epochs=30, bs=256, lr=1e-3):
    probe = GameStateProbe(input_dim=x.shape[1], arch=arch).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        probe.train()
        for idx in torch.randperm(x.shape[0]).split(bs):
            loss = masked_mse(probe(x[idx].float().to(device)), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        probe.eval()
        with torch.no_grad():
            pred = torch.cat([probe(x_val[i:i + 2048].float().to(device)).cpu()
                              for i in range(0, x_val.shape[0], 2048)])
        e = state_errors(pred, y_val)["ball_pos_uu"]
        if e < best:
            best, best_state = e, {k: v.cpu().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe.cpu(), best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-cache", required=True, type=Path)
    ap.add_argument("--val-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-train-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, _ = load_world_model_lean(args.checkpoint, args.device)
    print("extracting train features (DiT forwards over cached latents) ...")
    x_train, y_train = features_for_cache(model, args.train_cache, args.device, args.max_train_clips)
    print(f"train features {tuple(x_train.shape)}; extracting val features ...")
    x_val, y_val = features_for_cache(model, args.val_cache, args.device)
    del model
    torch.cuda.empty_cache()

    results = {}
    w = fit_ridge(x_train, y_train)
    results["ridge"] = state_errors(ridge_predict(w, x_val), y_val)
    torch.save({"w": w, "kind": "ridge"}, args.out / "probe_ridge.pt")
    print("ridge:", results["ridge"])

    for arch in ("linear", "mlp"):
        probe, best = train_sgd(x_train, y_train, x_val, y_val, arch, args.device)
        with torch.no_grad():
            pred = probe(x_val.float())
        results[arch] = state_errors(pred, y_val)
        torch.save({"state_dict": probe.state_dict(), "input_dim": probe.input_dim,
                    "arch": arch, "checkpoint": str(args.checkpoint)},
                   args.out / f"probe_{arch}.pt")
        print(f"{arch}:", results[arch])

    (args.out / "metrics.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()},
         "real_floor": results}, indent=2, default=float))
    print(f"artifacts in {args.out}")


if __name__ == "__main__":
    main()
