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
    from fastapi.responses import HTMLResponse

    # The platform UI gates room creation on localStorage['rocket_access_key'].
    # The local relay never checks a key (_check_access is a no-op with
    # ROCKET_ACCESS_KEY unset), so pre-seed it and the prompt never appears.
    _inject = "<script>try{localStorage.setItem('rocket_access_key','local')}catch(e){}</script>"

    @app.get("/", include_in_schema=False)
    def _index():
        html = (UI_DIR / "index.html").read_text()
        return HTMLResponse(html.replace("<head>", "<head>" + _inject, 1))

    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
