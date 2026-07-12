"""Weight-bundle management for local play.

Layout (identical to the serving bundle produced by package_wm_for_modal.sh):
    <bundle>/world_model_config.yaml
    <bundle>/checkpoint-<step>/checkpoint.pth
    <bundle>/codec/codec_config.yaml
    <bundle>/codec/checkpoint-<cstep>/checkpoint.pth
    <bundle>/context/default.npz

Primary source: Hugging Face Hub (resumable, cached, mirrors). Fallback: a direct
tarball URL (MIRA_BUNDLE_URL) for pre-HF rehearsals and air-gapped installs.
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_REPO = os.environ.get("MIRA_HF_REPO", "alakazamworld/mira-mini-local")
CACHE = Path(os.environ.get("MIRA_HOME", Path.home() / ".cache" / "alakazam-mira"))


def _find_ckpt(root: Path) -> Path | None:
    hits = sorted(root.glob("checkpoint-*/checkpoint.pth"))
    return hits[-1] if hits else None


def bundle_ready(root: Path) -> bool:
    return (
        (root / "world_model_config.yaml").is_file()
        and _find_ckpt(root) is not None
        and (root / "context" / "default.npz").is_file()
    )


def ensure_weights() -> Path:
    """Return the local bundle dir, downloading it on first run."""
    root = CACHE / "bundle"
    if bundle_ready(root):
        _localize_config(root)
        return root
    root.mkdir(parents=True, exist_ok=True)

    tar_url = os.environ.get("MIRA_BUNDLE_URL")
    if tar_url:
        print("[mira] downloading weights bundle (direct URL) ...")
        tar_path = CACHE / "bundle.tar"
        _download(tar_url, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(CACHE, filter="data")
        # the tar contains a single top-level dir; normalize to CACHE/bundle
        for d in CACHE.iterdir():
            if d.is_dir() and d.name != "bundle" and bundle_ready(d):
                d.rename(root) if not root.exists() else None
        tar_path.unlink(missing_ok=True)
    else:
        print(f"[mira] downloading weights from Hugging Face ({DEFAULT_REPO}) ...")
        from huggingface_hub import snapshot_download

        local = snapshot_download(DEFAULT_REPO)
        root = Path(local)

    if not bundle_ready(root):
        raise SystemExit(
            f"[mira] weight bundle incomplete at {root} — delete it and retry, or set MIRA_BUNDLE_URL"
        )
    _localize_config(root)
    return root


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[mira] {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def checkpoint_path(root: Path) -> Path:
    ckpt = _find_ckpt(root)
    assert ckpt is not None
    return ckpt

def _localize_config(root: Path) -> None:
    """Rewrite codec_checkpoint to this machine's absolute bundle path (bundles ship with a
    BUNDLE/ placeholder or a foreign absolute path; the loader needs a real local path)."""
    import re
    cfgf = root / "world_model_config.yaml"
    s = cfgf.read_text()
    hits = sorted(root.glob("codec/checkpoint-*/checkpoint.pth"))
    if hits:
        s = re.sub(r"codec_checkpoint: \S+", f"codec_checkpoint: {hits[-1]}", s)
        cfgf.write_text(s)
