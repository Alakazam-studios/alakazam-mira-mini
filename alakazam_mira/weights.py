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

MODEL_REPOS = {
    "1b": "alakazamworld/mira-mini",        # 1B single-player; needs a discrete GPU
    "364m": "alakazamworld/mira-mini-364m",  # laptop tier; MLX/Core ML path
}
CACHE = Path(os.environ.get("MIRA_HOME", Path.home() / ".cache" / "alakazam-mira"))


def resolve_repo(model: str | None, device: str | None) -> str:
    """Pick the weight repo: explicit --model > MIRA_HF_REPO env > device-aware default
    (CUDA machines get the 1B; Apple silicon / CPU get the 364M laptop tier)."""
    if model and model != "auto":
        return MODEL_REPOS[model]
    env = os.environ.get("MIRA_HF_REPO")
    if env:
        return env
    return MODEL_REPOS["1b"] if device == "cuda" else MODEL_REPOS["364m"]


def _find_ckpt(root: Path) -> Path | None:
    hits = sorted(root.glob("checkpoint-*/checkpoint.pth"))
    return hits[-1] if hits else None


def bundle_ready(root: Path) -> bool:
    return (
        (root / "world_model_config.yaml").is_file()
        and _find_ckpt(root) is not None
        and (root / "context" / "default.npz").is_file()
    )


def ensure_weights(repo: str | None = None) -> Path:
    """Return the local bundle dir for `repo`, downloading it on first run."""
    repo = repo or resolve_repo(None, None)
    root = CACHE / "bundle"
    if bundle_ready(root):
        _localize_config(root)
        return root
    root.mkdir(parents=True, exist_ok=True)

    tar_url = os.environ.get("MIRA_BUNDLE_URL")
    if tar_url:
        print("[mira-mini] downloading weights bundle (direct URL) ...")
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
        print(f"[mira-mini] downloading weights from Hugging Face ({repo}) ...")
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        try:
            local = snapshot_download(repo)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise SystemExit(
                f"[mira-mini] the weights ({repo}) are not public yet; they unlock at launch.\n"
                "       Watch https://alakazam.gg/mira-mini for the date. If you DO have access, run\n"
                "       `hf auth login` first, then retry."
            ) from e
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                raise SystemExit(
                    f"[mira-mini] no access to {repo} (HTTP {code}); the weights unlock at launch.\n"
                    "       If you have access: `hf auth login`, then retry."
                ) from e
            raise
        root = Path(local)

    if not bundle_ready(root):
        raise SystemExit(
            f"[mira-mini] weight bundle incomplete at {root}; delete it and retry, or set MIRA_BUNDLE_URL"
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
                print(f"\r[mira-mini] {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
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
