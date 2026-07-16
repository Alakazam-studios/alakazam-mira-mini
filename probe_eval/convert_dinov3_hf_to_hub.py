# ABOUTME: One-off converter: HF transformers DINOv3-B safetensors -> facebookresearch/dinov3
# ABOUTME: hub-format .pth (Meta's gated original file is unavailable; needed for ARR + FDD).
"""Usage:
    .venv/bin/python -m probe_eval.convert_dinov3_hf_to_hub \
        --hf-dir <dir with model.safetensors> --out weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

Validates functionally: hub model with converted weights vs the HF transformers model on the
same random batch; refuses to write unless final-layer patch features match (cosine > 0.999).
The output filename mimics Meta's published name so mira's resolve_dino_weights finds it —
the content hash will NOT match Meta's file; provenance is this converter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

HUB_DIR = str(Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main")


def convert(hf_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    hf = {}
    with safe_open(hf_dir / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            hf[k] = f.get_tensor(k)

    out: dict[str, torch.Tensor] = {
        "cls_token": hf.pop("embeddings.cls_token"),
        "mask_token": hf.pop("embeddings.mask_token").reshape(1, -1),  # HF (1,1,768) -> hub (1,768)
        "storage_tokens": hf.pop("embeddings.register_tokens"),
        "patch_embed.proj.weight": hf.pop("embeddings.patch_embeddings.weight"),
        "patch_embed.proj.bias": hf.pop("embeddings.patch_embeddings.bias"),
        "norm.weight": hf.pop("norm.weight"),
        "norm.bias": hf.pop("norm.bias"),
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    for i in range(n_layers):
        p, b = f"layer.{i}", f"blocks.{i}"
        q_w, k_w, v_w = (hf.pop(f"{p}.attention.{x}_proj.weight") for x in "qkv")
        q_b, v_b = hf.pop(f"{p}.attention.q_proj.bias"), hf.pop(f"{p}.attention.v_proj.bias")
        out[f"{b}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w])
        out[f"{b}.attn.qkv.bias"] = torch.cat([q_b, torch.zeros_like(q_b), v_b])  # k-bias is 0
        out[f"{b}.attn.proj.weight"] = hf.pop(f"{p}.attention.o_proj.weight")
        out[f"{b}.attn.proj.bias"] = hf.pop(f"{p}.attention.o_proj.bias")
        out[f"{b}.ls1.gamma"] = hf.pop(f"{p}.layer_scale1.lambda1")
        out[f"{b}.ls2.gamma"] = hf.pop(f"{p}.layer_scale2.lambda1")
        out[f"{b}.mlp.fc1.weight"] = hf.pop(f"{p}.mlp.up_proj.weight")
        out[f"{b}.mlp.fc1.bias"] = hf.pop(f"{p}.mlp.up_proj.bias")
        out[f"{b}.mlp.fc2.weight"] = hf.pop(f"{p}.mlp.down_proj.weight")
        out[f"{b}.mlp.fc2.bias"] = hf.pop(f"{p}.mlp.down_proj.bias")
        out[f"{b}.norm1.weight"] = hf.pop(f"{p}.norm1.weight")
        out[f"{b}.norm1.bias"] = hf.pop(f"{p}.norm1.bias")
        out[f"{b}.norm2.weight"] = hf.pop(f"{p}.norm2.weight")
        out[f"{b}.norm2.bias"] = hf.pop(f"{p}.norm2.bias")
    assert not hf, f"unmapped HF keys: {sorted(hf)}"
    return out


def validate(hf_dir: Path, sd: dict[str, torch.Tensor]) -> float:
    from transformers import AutoModel

    hub = torch.hub.load(HUB_DIR, "dinov3_vitb16", source="local", pretrained=False)
    missing, unexpected = hub.load_state_dict(sd, strict=False)
    computed_ok = {"rope_embed.periods"} | {k for k in missing if k.endswith("bias_mask")}
    assert set(missing) <= computed_ok, f"missing non-buffer keys: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    hub.eval()

    ref = AutoModel.from_pretrained(hf_dir).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a = hub.get_intermediate_layers(x, n=1, reshape=False)[0]  # (B, patches, 768)
        h = ref(pixel_values=x).last_hidden_state  # (B, 1+regs+patches, 768)
        b = h[:, -a.shape[1]:]  # patch tokens
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).min()
    return float(cos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sd = convert(args.hf_dir)
    cos = validate(args.hf_dir, sd)
    print(f"functional validation: min cosine(hub, transformers) = {cos:.6f}")
    if cos < 0.999:
        raise SystemExit("REFUSING to write: converted weights do not reproduce reference features")
    # The hub's loader is strict=True: include the deterministic buffers (rope periods,
    # k-bias masks) from a freshly-built skeleton so the file is loadable standalone.
    skeleton = torch.hub.load(HUB_DIR, "dinov3_vitb16", source="local", pretrained=False)
    for k, v in skeleton.state_dict().items():
        if k not in sd:
            sd[k] = v
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    print(f"wrote {args.out} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
