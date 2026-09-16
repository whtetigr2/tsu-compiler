"""Entry point for the packaged application.

Starts the API in a background thread on a loopback port nobody else is using,
then opens a native window onto it. Loopback is internal plumbing: the user
never sees a URL, never starts a server, and nothing leaves the machine.

`--selftest` runs the headless checks and exits with a status code instead,
which is how the frozen binary is verified without a display.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
PREFERRED_PORT = 8088
WINDOW_TITLE = "Thermodynamic Workbench"


def pick_free_port(host: str = HOST, preferred: int = PREFERRED_PORT,
                   span: int = 40) -> int:
    """First free port at or after `preferred`. Never silently reuse one."""
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"unavailable: no free port in {preferred}..{preferred + span - 1}")


def _serve(app, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_health(url: str, timeout: float = 90.0) -> bool:
    """Poll until the API answers. JAX import is slow on a cold start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    from .paths import ensure_compiler_importable

    ensure_compiler_importable()

    if "--selftest" in argv:
        from .selftest import run_selftest

        code, lines = run_selftest()
        print("\n".join(lines))
        return code

    from .server import build_desktop_app

    app = build_desktop_app()
    port = pick_free_port()
    threading.Thread(target=_serve, args=(app, HOST, port),
                     daemon=True).start()

    if not _wait_for_health(f"http://{HOST}:{port}/api/health"):
        print(f"unavailable: the API did not answer on {HOST}:{port} within "
              f"90 seconds. The window was not opened.", file=sys.stderr)
        return 1

    import webview

    webview.create_window(WINDOW_TITLE, f"http://{HOST}:{port}/",
                          width=1600, height=1000, min_size=(1100, 700))
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
