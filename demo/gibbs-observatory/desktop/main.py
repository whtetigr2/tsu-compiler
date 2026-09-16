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


def hide_console_window() -> None:
    """Hide the console the frozen binary was built with.

    The executable is built as a console application on purpose: `--selftest`
    has to reach real stdout so a script can read its report and a build can
    gate on it. A windowed build would silently discard that output. The cost
    is a console window behind the UI, so it is hidden here when the
    application is being launched normally rather than self-tested.

    Best effort. If this fails the application still runs, with a spare
    console, which is a cosmetic problem and not a reason to refuse to start.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        window = ctypes.windll.kernel32.GetConsoleWindow()
        if window:
            ctypes.windll.user32.ShowWindow(window, 0)  # SW_HIDE
    except Exception:
        pass


def pick_free_port(host: str = HOST, preferred: int = PREFERRED_PORT,
                   span: int = 40) -> int:
    """First genuinely free port at or after `preferred`.

    The probe must NOT set SO_REUSEADDR. On Linux that option only bypasses
    TIME_WAIT, but on Windows it lets a bind succeed against a port another
    process is actively listening on -- provided that process also set it,
    which every Python server does, since `socketserver.TCPServer` sets
    `allow_reuse_address` and http.server and uvicorn inherit it.

    With the option set, this function reported an occupied port as free,
    uvicorn failed to bind with WinError 10048, and the application waited
    ninety seconds for a health check that could never pass before reporting
    anything. SO_EXCLUSIVEADDRUSE makes the probe stricter still on Windows.
    """
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                try:
                    s.setsockopt(socket.SOL_SOCKET, exclusive, 1)
                except OSError:
                    pass
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"unavailable: no free port in {preferred}..{preferred + span - 1}. "
        f"Close whatever is holding them, or the application cannot start.")


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

    from .paths import is_frozen

    if is_frozen():
        hide_console_window()

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
