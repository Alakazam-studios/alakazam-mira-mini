# ABOUTME: Pre-head activation extraction — the paper's probe input ("last layer of world-model
# ABOUTME: activations", §6.2/§6.7): teacher-forced pass at tau=1 with a hook on the velocity head.
"""The paper's probes read the world model's internal representations, not codec latents.

Protocol (documented choice): we run the DiT over a latent sequence teacher-forced at tau=1
(clean input, the boundary the final denoise step approaches), with the model's own clean-past
conditioning and action embeddings, and capture the input of the velocity head — "the last
layer of world-model activations" — then spatially mean-pool per frame. The same operation
serves probe training (real encoded latents) and rollout scoring (the model re-processes its
own generated latents), keeping the instrument identical across both, and making cached
rollout latents re-scorable without re-running rollouts.
"""

from __future__ import annotations

import torch

from mira.world_model.actions_config import ActionTensors


def _action_tensors(model, key_presses: torch.Tensor, n_steps: int, off: int) -> ActionTensors:
    """ActionTensors for frames [off, off + n_steps), padding by repeating the last held action
    (the model's own inference caveat: a live player holds their control across a latent chunk)."""
    kp = key_presses.to(torch.int32)
    need = off + n_steps
    if kp.shape[0] < need:
        kp = torch.cat([kp, kp[-1:].expand(need - kp.shape[0], -1)])
    kp = kp[off:need]
    at = ActionTensors(config=model.config.actions, batch_size=1)
    at.key_presses = kp.unsqueeze(0)
    at.mouse_movements = torch.zeros((1, n_steps, 2), dtype=torch.float32)
    at.game_mouse_sensitivity = torch.full((1,), float("nan"), dtype=torch.float32)
    return at


@torch.no_grad()
def prehead_features(model, z: torch.Tensor, key_presses: torch.Tensor) -> torch.Tensor:
    """(1, t, h, w, c) normalized latents + (T_frames, 9) key presses → (t, hidden) features.

    Mirrors ``LatentWorldModel.forward``'s input construction (action offset, clean-past shift,
    bos) but replaces the sampled tau with tau=1 and captures the pre-head activations.
    """
    device = z.device
    t = z.shape[1]
    atd = model.action_temporal_downsampling
    off = atd - 1
    # One action window per latent frame AFTER the first (the encoder prepends an
    # initial-action token for frame 0) — mirrors LatentWorldModel.n_action_steps.
    a = model.action_encoder(_action_tensors(model, key_presses, (t - 1) * atd, off).to(device))

    shifted_z = None
    if model.config.use_clean_past:
        bos = model.bos[None, None].to(z.dtype)
        shifted_z = torch.cat([bos, z[:, :-1]], dim=1)

    tau = torch.ones(1, t, 1, 1, 1, device=device, dtype=z.dtype)

    captured: dict[str, torch.Tensor] = {}
    hook = model.world_model.head.register_forward_pre_hook(
        lambda mod, inp: captured.__setitem__("x", inp[0]))
    try:
        model.world_model(z, a, tau, clean_past=shifted_z)
    finally:
        hook.remove()
    x = captured["x"]  # (1, t, h, w, hidden) — register tokens already stripped
    assert x.shape[1] == t, f"pre-head activations t={x.shape[1]} != latents t={t}"
    return x.mean(dim=(2, 3))[0].float()  # spatial mean pool → (t, hidden)
