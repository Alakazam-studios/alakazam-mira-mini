# ABOUTME: MP cross-view state agreement (report §9.3's planned instrument): probe each view's
# ABOUTME: latents for BALL position; pairwise disagreement across the 4 views vs the real floor.
"""Usage:
    PYTHONPATH=src:. python -m probe_eval.mp_crossview \
        --checkpoint ckpts/wm4p_100k/checkpoint-63000 --probe runs/probe_v3_full/probe.pt \
        --index data/rocket-science/test --out runs/mp_crossview --n-clips 12

The ball is shared world state: four views of one arena give four independent probe readouts
of the same ball. Pairwise readout distance = cross-view state binding. Calibration: the same
measurement on REAL 4-view clips is the probe-noise floor. The multiplayer tiling happens
AFTER the per-player codec ("(b p) t h w c -> b t (p h) w c"), so each horizontal band of the
rollout latents is that player's unmodified latent grid — the shared probe applies per band.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mira.data.batch import VideoActionBatch
from mira.data.dataset import RocketScienceDataset
from mira.inference.rollout import rollout
from mira.world_model.actions_config import ActionTensors
from mira.world_model.config import WorldModelInferenceConfig

from .batching import perspective_batch
from .loading import autocast, chunk_dino_forward, load_world_model_lean
from .probe import GameStateProbe
from .targets import ENTITY_DIM, POS, POS_SCALE

N_P = 4


def load_probe(path: str, device: str) -> GameStateProbe:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = GameStateProbe(input_dim=ckpt["input_dim"], arch=ckpt.get("arch", "mlp")).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    return probe.eval()


def mp_batch(clip, action_config) -> VideoActionBatch:
    """All four perspectives, player-id ordered, as a batch of 4 (the wrapper's layout).

    Views are decoded at 288x512 (decode_frames' antialiased bilinear — the same op the
    GPU preprocess would apply to native 720p): a 4-view 720p batch OOMs the shared card.
    """
    per = [perspective_batch(clip, p, action_config, frame_size=(288, 512)) for p in range(N_P)]
    actions = ActionTensors(config=action_config, batch_size=N_P)
    actions.key_presses = torch.cat([b.actions.key_presses for b in per])
    actions.mouse_movements = torch.cat([b.actions.mouse_movements for b in per])
    actions.game_mouse_sensitivity = torch.cat([b.actions.game_mouse_sensitivity for b in per])
    return VideoActionBatch(video=torch.cat([b.video for b in per]), actions=actions)


@torch.no_grad()
def ball_tracks(probe, z_views: torch.Tensor) -> np.ndarray:
    """(P, t, h, w, c) per-view latents -> (P, t, 3) probe ball positions in uu."""
    p, t = z_views.shape[:2]
    states = probe(z_views.reshape(p * t, -1).float().to(next(probe.parameters()).device))
    ball = states.reshape(p, t, -1)[:, :, :ENTITY_DIM][..., POS].cpu().numpy()
    return ball * POS_SCALE


def pairwise_disagreement(ball: np.ndarray) -> np.ndarray:
    """(P, t, 3) -> (t,) mean pairwise L2 between views."""
    dists = [np.linalg.norm(ball[a] - ball[b], axis=-1)
             for a, b in itertools.combinations(range(ball.shape[0]), 2)]
    return np.mean(dists, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--n-clips", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--n-diffusion-steps", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, _ = load_world_model_lean(args.checkpoint, args.device)
    chunk_dino_forward(model, max_b=1)  # 4-view DINO encode OOMs monolithically on 24 GB shared
    # The MP model trains with a 78-frame context (the full clip) — the release eval overrides
    # to 38 frames so 78-frame clips leave a 20-latent generated window; mirror that.
    model.set_inference_context(38)
    inner = model.single_world_model
    stride_t, _ = inner.codec.encoder.get_downsampling_factors()
    n_ctx_lat = 38 // stride_t
    probe = load_probe(args.probe, args.device)
    cfg = WorldModelInferenceConfig(n_diffusion_steps=args.n_diffusion_steps,
                                    noise_level=0.0, schedule_type="linear")

    ds = RocketScienceDataset.from_local(args.index)
    records = []
    n_done = 0
    for clip in ds.iter_clips(clip_len=78, target_fps=20, exclude_replays=True,
                              perspective="all", seed=37, carry_video=True, decode=False):
        if clip.physics is None or len(clip.perspectives) < N_P:
            continue
        batch = mp_batch(clip, inner.config.actions)  # stays on CPU; moved per use

        # real floor: encode each view independently (SP-style path through the shared codec;
        # encode uses only the video, so a single-row dummy ActionTensors suffices)
        z_views = []
        for p in range(N_P):
            dummy = ActionTensors(config=inner.config.actions, batch_size=1)
            dummy.key_presses = batch.actions.key_presses[p:p + 1].cpu()
            dummy.mouse_movements = batch.actions.mouse_movements[p:p + 1].cpu()
            dummy.game_mouse_sensitivity = batch.actions.game_mouse_sensitivity[p:p + 1].cpu()
            b1 = VideoActionBatch(video=batch.video[p:p + 1].clone(), actions=dummy).to(args.device)
            with autocast(args.device):
                inner.codec.preprocess_batch(b1)
                z_views.append(inner.encode_video(b1)[0].cpu())
            del b1
        torch.cuda.empty_cache()
        z_real = torch.stack(z_views)  # (P, t, h, w, c)
        ball_real = ball_tracks(probe, z_real)
        floor = pairwise_disagreement(ball_real)

        rec = {"match_id": clip.match_id, "clip_id": str(clip.clip_id),
               "floor_disagreement_uu": floor.tolist(), "rollout": {}}
        for seed in args.seeds:
            torch.manual_seed(seed)
            with torch.no_grad(), autocast(args.device):
                z = rollout(model, batch.clone().to(args.device), cfg)  # (1, t, P*h, w, c) tiled
            h = z.shape[2] // N_P
            z_split = torch.stack([z[0, :, i * h:(i + 1) * h] for i in range(N_P)]).cpu()
            del z
            torch.cuda.empty_cache()
            ball_gen = ball_tracks(probe, z_split)
            rec["rollout"][str(seed)] = pairwise_disagreement(ball_gen).tolist()
        rec["n_ctx_lat"] = n_ctx_lat
        records.append(rec)
        n_done += 1
        print(f"[mp-xview] clip {n_done}/{args.n_clips}", flush=True)
        if n_done >= args.n_clips:
            break

    # summary: mean disagreement, real floor vs rollout context vs rollout generated
    def agg(fn):
        vals = [fn(r) for r in records]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "real_floor_uu": agg(lambda r: np.mean(r["floor_disagreement_uu"])),
        "rollout_context_uu": agg(lambda r: np.mean([np.mean(v[: r["n_ctx_lat"]])
                                                     for v in r["rollout"].values()])),
        "rollout_generated_uu": agg(lambda r: np.mean([np.mean(v[r["n_ctx_lat"]:])
                                                       for v in r["rollout"].values()])),
        "rollout_gen_last5_uu": agg(lambda r: np.mean([np.mean(v[-5:])
                                                       for v in r["rollout"].values()])),
    }
    (args.out / "mp_crossview.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()},
         "summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
