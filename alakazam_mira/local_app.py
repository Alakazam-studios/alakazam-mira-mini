"""The local FastAPI app: rocket relay router + static web UI, one origin."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).parent / "ui"


def build_app() -> FastAPI:
    app = FastAPI(title="alakazam-mira local", docs_url=None, redoc_url=None)

    # Room relay (same code that runs in production; MIRA_MODAL_URL points at the
    # in-process engine). ROCKET_ACCESS_KEY unset locally -> access gate no-ops.
    from server.rocket_league.api import router as rocket_router

    app.include_router(rocket_router)  # router already carries /api/rocket

    assert UI_DIR.is_dir(), f"UI bundle missing at {UI_DIR}; build with scripts/build_local_pkg.sh"
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
