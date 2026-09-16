"""The API process serves the frontend too, so the user needs no Node.

`StaticFiles(html=True)` mounted at "/" would swallow every API route if it
were added first. Starlette matches routes in registration order, so the mount
goes on last, after the application has declared everything it owns.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .paths import FrontendMissingError, frontend_dist


def mount_frontend(app: FastAPI, dist: Path) -> FastAPI:
    """Serve `dist` at "/" without shadowing routes already declared."""
    index = Path(dist) / "index.html"
    if not index.is_file():
        raise FrontendMissingError(
            f"unavailable: no index.html at {index}. The frontend has not "
            f"been built.")
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


def build_desktop_app() -> FastAPI:
    """The real Observatory API, with the built frontend mounted on it."""
    from backend.app.main import app  # imported late: heavy, and drags in JAX

    return mount_frontend(app, frontend_dist())
