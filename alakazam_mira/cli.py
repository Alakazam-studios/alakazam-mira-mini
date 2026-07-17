"""`mira-mini play`: run the MIRA Mini world model locally, one command.

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
    ap = argparse.ArgumentParser(prog="mira-mini")
    sub = ap.add_subparsers(dest="cmd")
    play = sub.add_parser("play", help="download weights (first run) and play locally")
    play.add_argument("--port", type=int, default=8770)
    play.add_argument("--model", choices=["auto", "1b", "364m"], default="auto",
                      help="which weights to run: 1b (needs a real GPU), 364m (laptop tier), "
                           "or auto (CUDA -> 1b, Apple/CPU -> 364m)")
    play.add_argument("--steps", type=int, default=None,
                      help="override default diffusion steps (else the lobby preset decides)")
    play.add_argument("--no-browser", action="store_true")
    play.add_argument("--no-fast", action="store_true",
                      help="disable the Apple fast stack (MLX + Core ML) and run plain torch")
    play.add_argument("--verbose", action="store_true",
                      help="show engine/runtime INFO logs (hidden by default)")
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
    if device == "cpu" and os.environ.get("MIRA_BACKEND") != "mlx":
        print("[mira-mini] no GPU found (need CUDA or Apple silicon); CPU generation is too slow to play. Aborting.")
        print("       Override with MIRA_DEVICE=cpu to run anyway.")
        sys.exit(2)

    from .weights import CACHE, bundle_ready, checkpoint_path, ensure_weights, resolve_repo

    repo = resolve_repo(args.model, device)
    size = "~5 GB" if repo.endswith("364m") else "~12 GB"
    first_run = not bundle_ready(CACHE / "bundle")

    DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
    ORANGE, GREEN = "\033[38;5;208m", "\033[32m"
    def line(icon, text):
        print(f"  {icon} {text}", flush=True)
    print()
    try:
        from importlib.metadata import version as _pkg_version
        _ver = "v" + _pkg_version("alakazam-mira-mini")
    except Exception:
        _ver = ""
    print(f"{ORANGE}{BOLD}  ▄▀█ MIRA MINI{RESET} {DIM}{_ver}{RESET}  {DIM}a world model you can drive{RESET}")
    print(f"{DIM}  every frame is generated · no game engine · CC BY-NC-SA weights · alakazam.gg/mira-mini{RESET}")
    print()
    dev_label = {"mps": "Apple silicon (Metal)", "cuda": "CUDA GPU", "cpu": "CPU"}.get(device, device)
    line("●", f"device   {BOLD}{dev_label}{RESET}")
    line("●", f"model    {BOLD}{repo.split('/')[-1]}{RESET}"
         + (f"  {DIM}first run downloads {size}, once{RESET}" if first_run else f"  {DIM}weights cached{RESET}"))
    bundle = ensure_weights(repo)
    ckpt = checkpoint_path(bundle)

    # ---- Apple fast stack: wired automatically when the pieces are present ----
    # (MLX carries the transformer on Metal, a Core ML package carries the
    # decoder, torch keeps the CPU-side glue. Disable with --no-fast.)
    fast = False
    if (sys.platform == "darwin" and device == "mps" and not args.no_fast
            and os.environ.get("MIRA_BACKEND") is None):
        mlpkg = bundle / "decoder60k.mlpackage"
        try:
            import mlx.core  # noqa: F401
            from coremltools.models import MLModel  # noqa: F401
            have_fast = mlpkg.exists()
        except Exception:
            have_fast = False
        if have_fast:
            os.environ.setdefault("MIRA_BACKEND", "mlx")
            os.environ["MIRA_DEVICE"] = "cpu"  # torch = glue only; MLX owns the GPU
            os.environ.setdefault("MIRA_DECODER", f"coreml:{mlpkg}")
            os.environ.setdefault("MIRA_DECODER_CU", "CPU_AND_GPU")
            os.environ.setdefault("MIRA_DECODE_PIPELINE", "1")
            os.environ.setdefault("MIRA_MLX_NO_RENOISE", "1")
            os.environ.setdefault("MIRA_FRAME_INTERP", "1")
            os.environ.setdefault("MIRA_WARMUP_STEPS", "8")
            device = "cpu"
            fast = True
            line("●", f"backend  {BOLD}fast stack{RESET}  {DIM}MLX transformer + Core ML decoder + 2x display interpolation{RESET}")
        else:
            line("●", f"backend  torch on {device}  {DIM}(fast stack unavailable: needs mlx + coremltools + the 364m bundle){RESET}")

    # Engine env (read by mira_vm.engine / mira_vm.app at import).
    os.environ.setdefault("MIRA_CKPT", str(ckpt))
    os.environ.setdefault("MIRA_DEVICE", device)
    os.environ.setdefault("MIRA_CONTEXT_PATH", str(bundle / "context" / "default.npz"))
    os.environ.setdefault("MIRA_WARMUP_STEPS", "0")  # no CUDA graphs locally; skip warmup session
    # Few-step distillates (364m student, psd) run at 2 steps; the lobby preset
    # sends 8, so use the hard override the engine honors over session rules.
    steps = args.steps or (2 if ("364m" in repo or "psd" in repo) else None)
    if steps:
        os.environ["MIRA_FORCE_STEPS"] = str(steps)
        os.environ["MIRA_N_DIFFUSION_STEPS"] = str(steps)
    # Relay env: talk to the in-process engine over loopback.
    engine_port = args.port + 1
    os.environ["MIRA_MODAL_URL"] = f"ws://127.0.0.1:{engine_port}/ws"

    import logging as _logging
    noisy = ["httpx", "uvicorn.access"]
    if not args.verbose:
        noisy += ["mira-engine", "mira-vm", "mira_vm.coreml_decoder",
                  "mira.codec.dino", "server.rocket_league.api",
                  "server.rocket_league.broker", "coremltools"]
    for name in noisy:
        _logging.getLogger(name).setLevel(_logging.WARNING)
    import uvicorn

    from .local_app import build_app

    # Engine server (its own port; the relay's broker dials it over loopback exactly like
    # it dials Modal in production; zero protocol drift between cloud and local).
    from mira_vm.app import app as engine_app

    eng_cfg = uvicorn.Config(engine_app, host="127.0.0.1", port=engine_port, log_level="warning")
    eng_server = uvicorn.Server(eng_cfg)
    threading.Thread(target=eng_server.run, daemon=True, name="mira-engine").start()

    app = build_app()
    url = f"http://127.0.0.1:{args.port}/?view=rocket&key=local"

    # Loading watcher: poll the engine's healthz, keep the human company while
    # 364M parameters wake up, open the browser only when the model is READY
    # (no more staring at a black canvas).
    def _watch_and_open():
        import json as _json
        import urllib.request as _rq
        quips = [
            "convincing 364 million parameters it's game day…",
            "teaching the ball object permanence…",
            "negotiating the laws of physics (they drive a hard bargain)…",
            "compiling dreams for your GPU…",
            "the referee is a neural net too; do not argue with it…",
        ]
        t0 = time.time(); qi = 0
        spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        tty = sys.stdout.isatty()
        si = 0
        last_h = 0.0
        h = {}
        while True:
            time.sleep(0.1 if tty else 0.5)
            waited = time.time() - t0
            if waited - last_h > 0.5:
                last_h = waited
                try:
                    with _rq.urlopen(f"http://127.0.0.1:{engine_port}/healthz", timeout=2) as r:
                        h = _json.load(r)
                except Exception:
                    h = {}
            if h.get("load_error"):
                if tty: print("\r\033[2K", end="")
                print(f"  ✗ engine failed: {h['load_error']}", flush=True)
                return
            if h.get("ready"):
                if tty: print("\r\033[2K", end="")
                print(f"  \033[32m●\033[0m engine   \033[1mready in {waited:.0f}s\033[0m")
                print()
                link = f"\033]8;;{url}\033\\{url}\033]8;;\033\\"
                print(f"  \033[1m▶ {link}\033[0m  \033[2m(click it)\033[0m")
                print("    drive with WASD · Space toggles ball-cam · Ctrl-C here to quit")
                print()
                if not args.no_browser:
                    webbrowser.open(url)
                return
            quip = quips[min(int(waited // 9), len(quips) - 1)]
            if tty:
                si = (si + 1) % len(spin)
                print(f"\r\033[2K  \033[38;5;208m{spin[si]}\033[0m engine   loading… "
                      f"\033[1m{waited:.0f}s\033[0m  \033[2m{quip}\033[0m", end="", flush=True)
            elif int(waited) % 10 == 0 and qi <= int(waited // 9):
                qi = int(waited // 9) + 1
                print(f"    …{waited:.0f}s  {quip}", flush=True)
            if waited > 300:
                if tty: print("\r\033[2K", end="")
                print("  ✗ engine took >5 min; something is wrong (check RAM, then rerun)")
                return

    if not args.verbose:
        print("  \033[2m(engine logs hidden; rerun with --verbose to see them)\033[0m", flush=True)
    threading.Thread(target=_watch_and_open, daemon=True, name="load-watcher").start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
