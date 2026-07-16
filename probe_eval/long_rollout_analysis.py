# ABOUTME: Physics-vs-texture verdict on long rollouts: probe-decoded state drift over time
# ABOUTME: (windowed W1 on state marginals vs real reference + violation rates) per rung.
"""Usage:
    .venv/bin/python -m probe_eval.long_rollout_analysis \
        --latents runs/long_rollout/latents --probe runs/probe_v2/probe.pt \
        --real-cache "data/encoded/checkpoint-45000_test_100c_78f20.pt" \
        --out runs/long_rollout/analysis

The question (report §17.3): the student's windowed IMAGE-feature distance runs ~12x the
teacher's from the one-minute mark. Is that drift state-level physics or appearance? Here the
shared latent probe decodes state along the same-protocol rollouts; per 2 s window we score
(a) Wasserstein-1 of state marginals (ball speed / ball height / car speed) against the real
reference — small-sample-robust, unlike small-N Frechet — and (b) hard physicality violations
(engine caps + arena bounds + teleports, Appendix C constants). Flat curves at real-band level
=> texture drift; rising/elevated curves => physics drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .probe import GameStateProbe
from .targets import BALL_VEL_SCALE, CAR_VEL_SCALE, ENTITY_DIM, POS, POS_SCALE, VEL

LATENT_HZ = 10.0
WINDOW = 20  # latents (2 s), hop = WINDOW // 2

# violation thresholds (MIRA Appendix C engine caps, +5% slack for probe error)
BALL_SPEED_MAX = 6000.0 * 1.05
CAR_SPEED_MAX = 2300.0 * 1.05
ARENA = np.array([4096.0, 5120.0, 2044.0]) * 1.10  # uu, generous bound
TELEPORT_FACTOR = 1.5  # ||dpos||/dt beyond cap*factor counts as a teleport


def load_probe(path: str, device: str) -> GameStateProbe:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = GameStateProbe(input_dim=ckpt["input_dim"], arch=ckpt.get("arch", "mlp")).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    return probe.eval()


@torch.no_grad()
def decode_states(probe, z_flat: torch.Tensor, device: str, bs: int = 2048) -> np.ndarray:
    """(N, h*w*c) -> (N, 50) probe states (normalized target space)."""
    outs = [probe(z_flat[i:i + bs].float().to(device)).cpu() for i in range(0, len(z_flat), bs)]
    return torch.cat(outs).numpy()


def marginals(states: np.ndarray) -> dict[str, np.ndarray]:
    """Physical 1-D marginals from (N, 50) normalized states."""
    s = states.reshape(-1, 5, ENTITY_DIM)
    ball_pos = s[:, 0, POS] * POS_SCALE
    ball_vel = s[:, 0, VEL] * BALL_VEL_SCALE
    car_vel = s[:, 1:, VEL] * CAR_VEL_SCALE
    return {
        "ball_speed": np.linalg.norm(ball_vel, axis=-1),
        "ball_z": ball_pos[:, 2],
        "car_speed": np.linalg.norm(car_vel, axis=-1).reshape(-1),
    }


def w1(a: np.ndarray, b: np.ndarray, n_q: int = 64) -> float:
    """Wasserstein-1 via quantile matching."""
    q = np.linspace(0.01, 0.99, n_q)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def violations(states: np.ndarray) -> dict[str, float]:
    s = states.reshape(-1, 5, ENTITY_DIM)
    ball_pos = s[:, 0, POS] * POS_SCALE
    ball_speed = np.linalg.norm(s[:, 0, VEL] * BALL_VEL_SCALE, axis=-1)
    car_speed = np.linalg.norm(s[:, 1:, VEL] * CAR_VEL_SCALE, axis=-1)
    dpos = np.diff(ball_pos, axis=0)
    tele = np.linalg.norm(dpos, axis=-1) * LATENT_HZ > BALL_SPEED_MAX * TELEPORT_FACTOR
    return {
        "ball_overspeed": float((ball_speed > BALL_SPEED_MAX).mean()),
        "car_overspeed": float((car_speed > CAR_SPEED_MAX).mean()),
        "out_of_arena": float((np.abs(ball_pos) > ARENA).any(axis=-1).mean()),
        "ball_teleport": float(tele.mean()) if len(tele) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--real-cache", required=True,
                    help="encoded val cache (.pt with x=flattened latents) for the real reference")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    probe = load_probe(args.probe, args.device)

    real = torch.load(args.real_cache, weights_only=True)
    real_states = decode_states(probe, real["x"], args.device)
    real_marg = marginals(real_states)
    real_viol = violations(real_states)

    # Real-band: W1 of random real 20-frame windows against the full real reference.
    rng = np.random.default_rng(0)
    band = {k: [] for k in real_marg}
    n_real = len(real_states)
    for _ in range(200):
        i = rng.integers(0, n_real - WINDOW)
        wm = marginals(real_states[i:i + WINDOW])
        for k in band:
            band[k].append(w1(wm[k], real_marg[k]))
    real_band = {k: {"p50": float(np.median(v)), "p95": float(np.quantile(v, 0.95))}
                 for k, v in band.items()}

    out = {"real_band": real_band, "real_violations": real_viol, "rungs": {}}
    for f in sorted(args.latents.glob("*.npz")):
        d = np.load(f)  # self-generated dumps: plain arrays + unicode scalars, no pickle needed
        z = torch.from_numpy(d["latents"])  # (T, h, w, c) fp16
        states = decode_states(probe, z.reshape(z.shape[0], -1), args.device)
        rows = []
        for s0 in range(0, len(states) - WINDOW + 1, WINDOW // 2):
            wstates = states[s0:s0 + WINDOW]
            wm = marginals(wstates)
            rows.append({
                "t_s": s0 / LATENT_HZ,
                **{f"w1_{k}": w1(wm[k], real_marg[k]) for k in wm},
                **violations(wstates),
            })
        key = f.stem  # e.g. student_2s_s2_ctx0
        out["rungs"][key] = rows
        last = rows[-1]
        print(f"{key:<28} windows {len(rows):3d} | w1_ball_speed t0 {rows[0]['w1_ball_speed']:.0f} "
              f"-> end {last['w1_ball_speed']:.0f} | teleport end {last['ball_teleport']:.3f}")

    (args.out / "state_drift.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out / 'state_drift.json'}")


if __name__ == "__main__":
    main()
