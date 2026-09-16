"""Where the application finds its own files.

Two execution modes. From source, everything sits under the Observatory
directory. Frozen by PyInstaller, data files are unpacked into a temporary
directory whose location is `sys._MEIPASS`, and there is no repository at all.
Every path in the application resolves through here so that difference is
handled exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path


class FrontendMissingError(RuntimeError):
    """The built frontend is absent. Raised rather than served as a blank page."""


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """The directory containing `backend/` and `frontend/`."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def ensure_compiler_importable() -> None:
    """Put the compiler on the import path when running from source.

    Frozen, `tsu_compiler` is collected into the bundle and is already
    importable. From source it lives in the repository's `src/`, four levels
    up, and relying on the caller to set PYTHONPATH would mean the application
    could not simply be double-clicked.
    """
    if is_frozen():
        return
    src = resource_root().parents[1] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def frontend_dist() -> Path:
    """The directory holding the built `index.html`.

    Refuses loudly when the build is absent. A silent empty directory here
    produces a window opening onto nothing, with no message anywhere.
    """
    dist = resource_root() / "frontend" / "dist"
    index = dist / "index.html"
    if not index.is_file():
        raise FrontendMissingError(
            f"unavailable: no index.html at {index}. The frontend has not "
            f"been built. Run `npm run build` in frontend/, or rebuild the "
            f"application with desktop/build.ps1 which does it for you.")
    return dist
