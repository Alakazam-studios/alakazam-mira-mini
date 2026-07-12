"""`mira play` — run the MIRA world model locally, one command.

Wires three existing pieces into one process on one port:
  1. the GPU engine (mira_vm.app: FastAPI + WS, device auto-detected cuda -> mps -> cpu)
  2. the room relay (server.rocket_league router, pointed at the in-process engine WS)
  3. the built web UI (static files, same origin -> the app's dev-port heuristics no-op)

No cloud, no account, no telemetry: after the first weight download everything is local.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser


def _pick_device() -> str:
    import torch

    if os.environ.get("MIRA_DEVICE"):
        return os.environ["MIRA_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(prog="mira")
    sub = ap.add_subparsers(dest="cmd")
    play = sub.add_parser("play", help="download weights (first run) and play locally")
    play.add_argument("--port", type=int, default=8770)
    play.add_argument("--steps", type=int, default=None,
                      help="override default diffusion steps (else the lobby preset decides)")
    play.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.cmd != "play":
        ap.print_help()
        sys.exit(0)

    # torch.compile is a server-side optimization; on consumer machines (esp. MPS) the
    # Inductor/Metal path is slow-or-broken (measured: shader-compile crashes on M1).
    # Env kill-switches proved unreliable against explicit torch.compile() assignments,
    # so for local play we neuter the API itself before any model code imports.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch

    def _no_compile(model=None, **_kw):
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = _no_compile  # type: ignore[assignment]

    device = _pick_device()
    if device == "cpu":
        print("[mira] no GPU found (need CUDA or Apple Silicon). CPU is too slow to be fun — aborting.")
        print("       Override with MIRA_DEVICE=cpu if you really want to watch a slideshow.")
        sys.exit(2)
    print(f"[mira] device: {device}")

    from .weights import checkpoint_path, ensure_weights

    bundle = ensure_weights()
    ckpt = checkpoint_path(bundle)

    # Engine env (read by mira_vm.engine / mira_vm.app at import).
    os.environ.setdefault("MIRA_CKPT", str(ckpt))
    os.environ.setdefault("MIRA_DEVICE", device)
    os.environ.setdefault("MIRA_CONTEXT_PATH", str(bundle / "context" / "default.npz"))
    os.environ.setdefault("MIRA_WARMUP_STEPS", "0")  # no CUDA graphs locally; skip warmup session
    if args.steps:
        os.environ["MIRA_N_DIFFUSION_STEPS"] = str(args.steps)
    # Relay env: talk to the in-process engine over loopback.
    engine_port = args.port + 1
    os.environ["MIRA_MODAL_URL"] = f"ws://127.0.0.1:{engine_port}/ws"

    import uvicorn

    from .local_app import build_app

    # Engine server (its own port; the relay's broker dials it over loopback exactly like
    # it dials Modal in production — zero protocol drift between cloud and local).
    from mira_vm.app import app as engine_app

    eng_cfg = uvicorn.Config(engine_app, host="127.0.0.1", port=engine_port, log_level="warning")
    eng_server = uvicorn.Server(eng_cfg)
    threading.Thread(target=eng_server.run, daemon=True, name="mira-engine").start()

    app = build_app()
    url = f"http://127.0.0.1:{args.port}/?view=rocket"
    print(f"[mira] starting — {url}")
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True
        ).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
