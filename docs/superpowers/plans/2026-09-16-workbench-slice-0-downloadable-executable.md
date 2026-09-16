# Workbench Slice 0 — Downloadable Executable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Windows application someone can download and run with no Python, no Node, no terminal, and no network — which opens a native window and proves the compiler and THRML are alive inside it.

**Architecture:** The frontend is built once at package time and served as static files by the same FastAPI process, which removes Node from the end user's machine entirely. Uvicorn runs in a background thread inside the application process; a pywebview window points at it over loopback. PyInstaller freezes the whole thing — Python 3.14, JAX, THRML, the compiler and the built frontend — into a distributable folder.

**Tech Stack:** Python 3.14.2, FastAPI, uvicorn, pywebview (WebView2), PyInstaller 6.22.3, Vite 8 / React 19, JAX 0.11.1, THRML 0.1.4.

**Spec:** `docs/superpowers/specs/2026-09-16-thermodynamic-workbench-design.md`

## Global Constraints

- Work only in `C:\Users\whtet\Documents\tsu-compiler`. Never commit to another repository.
- Stage commits by explicit path. Never `git add -A`.
- Do not touch `receipts/EXP-G8-D2/`, `receipts/EXP-G8-D3/`, `figures/z1_lab_screenshot.png`, `demo/cascade.py`, `demo/cascade_runs/`.
- `specs/emergence_8x8.yaml` and `pyproject-review.toml` are leftovers from another agent. Leave untouched.
- No secret values in any file or commit message.
- Never weaken an assertion. Record failures rather than repairing them.
- No hardware claims. Everything is simulation until someone else runs it on silicon.
- A missing measurement reads `unavailable: <reason>`. Never a blank, a zero, a dash, or a guess.
- Interpreter for all commands: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"`
- Observatory venv for backend work: `demo/gibbs-observatory/.venv/Scripts/python.exe`
- The path-ownership split that reserved `demo/gibbs-observatory/**` for another agent is suspended at the owner's explicit request while that agent is unavailable.
- Ports are Observatory-owned: API `8088`, UI `5188`. Do not bind `8000` or `5173`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `demo/gibbs-observatory/desktop/__init__.py` | Package marker. Empty. |
| `demo/gibbs-observatory/desktop/paths.py` | Resolve resource locations under both source and frozen execution. Nothing else. |
| `demo/gibbs-observatory/desktop/server.py` | Mount the built frontend onto the FastAPI app; pick a free port; run uvicorn in a thread. |
| `demo/gibbs-observatory/desktop/selftest.py` | Headless proof that the compiler, THRML and the frontend bundle are present and functioning. Callable from a frozen binary. |
| `demo/gibbs-observatory/desktop/main.py` | Entry point. Parses `--selftest`, otherwise starts the server and opens the window. |
| `demo/gibbs-observatory/desktop/workbench.spec` | PyInstaller build definition. |
| `demo/gibbs-observatory/desktop/build.ps1` | One command that builds frontend then freezes the app. |
| `demo/gibbs-observatory/desktop/README.md` | How to build and what the result requires on a target machine. |
| `demo/gibbs-observatory/backend/tests/test_desktop_paths.py` | Tests for `paths.py`. |
| `demo/gibbs-observatory/backend/tests/test_desktop_server.py` | Tests for `server.py`. |
| `demo/gibbs-observatory/backend/tests/test_desktop_selftest.py` | Tests for `selftest.py`. |

`desktop/` is deliberately separate from `backend/`. The backend is the API; `desktop/` is the shell that packages it. Keeping them apart means the API stays runnable in development exactly as it is today, and nothing about packaging leaks into request handling.

---

## Task 1: Resource paths under source and frozen execution

A frozen PyInstaller application does not have a repository around it. `sys._MEIPASS` points at the unpacked bundle. Every other task depends on getting this right, so it comes first and it is the smallest possible unit.

**Files:**
- Create: `demo/gibbs-observatory/desktop/__init__.py`
- Create: `demo/gibbs-observatory/desktop/paths.py`
- Test: `demo/gibbs-observatory/backend/tests/test_desktop_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `resource_root() -> pathlib.Path` — bundle root when frozen, `demo/gibbs-observatory/` when running from source.
  - `frontend_dist() -> pathlib.Path` — the directory holding `index.html`.
  - `is_frozen() -> bool`
  - `FrontendMissingError(RuntimeError)` — raised by `frontend_dist()` when `index.html` is absent.

- [ ] **Step 1: Write the failing test**

Create `demo/gibbs-observatory/backend/tests/test_desktop_paths.py`:

```python
"""Where the application finds its own files, frozen and unfrozen.

A PyInstaller bundle has no repository around it: data files land in a
temporary directory named by `sys._MEIPASS`. Getting this wrong produces a
window that opens onto a blank page with no error, which is the single most
confusing failure this application can have. So `frontend_dist` refuses
loudly instead of returning a path that does not exist.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from desktop.paths import (  # noqa: E402
    FrontendMissingError,
    frontend_dist,
    is_frozen,
    resource_root,
)


def test_not_frozen_in_development():
    assert is_frozen() is False


def test_resource_root_is_the_observatory_directory_from_source():
    assert resource_root() == ROOT
    assert (resource_root() / "backend").is_dir(), (
        "resource_root must point at the directory holding backend/ and "
        "frontend/, or every other path resolution is wrong")


def test_resource_root_follows_meipass_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert is_frozen() is True
    assert resource_root() == tmp_path


def test_frontend_dist_refuses_when_index_html_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    (tmp_path / "frontend" / "dist").mkdir(parents=True)
    with pytest.raises(FrontendMissingError) as exc:
        frontend_dist()
    assert "index.html" in str(exc.value), (
        "the refusal must name what is missing, not just fail")


def test_frontend_dist_returns_the_directory_holding_index_html(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert frontend_dist() == dist
```

- [ ] **Step 2: Run the test to verify it fails**

```
cd demo/gibbs-observatory
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_paths.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'desktop'`.

- [ ] **Step 3: Write the implementation**

Create `demo/gibbs-observatory/desktop/__init__.py` as an empty file.

Create `demo/gibbs-observatory/desktop/paths.py`:

```python
"""Where the application finds its own files.

Two execution modes. From source, everything sits under the Observatory
directory. Frozen by PyInstaller, data files are unpacked into a temporary
directory whose location is `sys._MEIPASS`, and there is no repository at
all. Every path in the application resolves through here so that difference
is handled exactly once.
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
```

- [ ] **Step 4: Run the test to verify it passes**

```
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_paths.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add demo/gibbs-observatory/desktop/__init__.py demo/gibbs-observatory/desktop/paths.py demo/gibbs-observatory/backend/tests/test_desktop_paths.py
git commit -m "Resolve application resource paths under source and frozen execution

A frozen bundle has no repository around it. Every path resolves through one
module so that difference is handled once, and a missing frontend build is
refused by name rather than served as a blank window.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Serve the built frontend from the API process

This is what removes Node from the end user's machine. Today the launcher runs Vite as a second process; instead the built static files are mounted onto the existing FastAPI application.

**Files:**
- Create: `demo/gibbs-observatory/desktop/server.py`
- Test: `demo/gibbs-observatory/backend/tests/test_desktop_server.py`

**Interfaces:**
- Consumes: `desktop.paths.frontend_dist`, `desktop.paths.FrontendMissingError`.
- Produces:
  - `mount_frontend(app: FastAPI, dist: Path) -> FastAPI` — mounts static files at `/`, returns the same app.
  - `build_desktop_app() -> FastAPI` — the Observatory API with the frontend mounted.

- [ ] **Step 1: Write the failing test**

Create `demo/gibbs-observatory/backend/tests/test_desktop_server.py`:

```python
"""The API process serves the built frontend itself.

The existing launcher runs Vite as a second process, which means the end user
needs Node installed. Mounting the built output onto the same FastAPI app
removes that requirement entirely -- and it must do so without shadowing any
API route, which is what the ordering test below pins.
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from desktop.paths import FrontendMissingError  # noqa: E402
from desktop.server import build_desktop_app, mount_frontend  # noqa: E402


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>WB</title>",
                                  encoding="utf-8")
    (d / "app.js").write_text("console.log(1)", encoding="utf-8")
    return d


def test_index_is_served_at_root(dist):
    app = mount_frontend(FastAPI(), dist)
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "<title>WB</title>" in r.text


def test_assets_are_served(dist):
    app = mount_frontend(FastAPI(), dist)
    r = TestClient(app).get("/app.js")
    assert r.status_code == 200
    assert "console.log" in r.text


def test_api_routes_are_not_shadowed_by_the_static_mount(dist):
    """The mount is at '/', so ordering decides whether the API survives."""
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"service": "gibbs-observatory"}

    mount_frontend(app, dist)
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200, "the static mount swallowed an API route"
    assert r.json()["service"] == "gibbs-observatory"


def test_missing_build_is_refused_not_served_blank(tmp_path):
    with pytest.raises(FrontendMissingError):
        mount_frontend(FastAPI(), tmp_path / "does-not-exist")


def test_build_desktop_app_keeps_the_real_health_endpoint():
    """The packaged app is the real Observatory API, not a reduced copy."""
    try:
        app = build_desktop_app()
    except FrontendMissingError:
        pytest.skip("frontend not built in this checkout")
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["service"] == "gibbs-observatory"
```

- [ ] **Step 2: Run the test to verify it fails**

```
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_server.py -v
```

Expected: collection error, `ImportError: cannot import name 'build_desktop_app'`.

- [ ] **Step 3: Write the implementation**

Create `demo/gibbs-observatory/desktop/server.py`:

```python
"""The API process serves the frontend too, so the user needs no Node.

`StaticFiles(html=True)` mounted at "/" would swallow every API route if it
were added first. Starlette matches routes in registration order, so the
mount goes on last, after the application has declared everything it owns.
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
```

- [ ] **Step 4: Run the test to verify it passes**

```
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_server.py -v
```

Expected: 5 passed, or 4 passed and 1 skipped if the frontend has not been built in this checkout.

- [ ] **Step 5: Commit**

```bash
git add demo/gibbs-observatory/desktop/server.py demo/gibbs-observatory/backend/tests/test_desktop_server.py
git commit -m "Serve the built frontend from the API process

Removes Node from the end user's machine: the packaged application no longer
runs Vite as a second process. The static mount is registered last so it
cannot shadow an API route, which a test pins directly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Headless self-test

A frozen binary that opens a window is hard to verify automatically. A `--selftest` path makes the bundle testable without a display: it proves the compiler imports, THRML samples, and the frontend is present, then exits with a status code.

**Files:**
- Create: `demo/gibbs-observatory/desktop/selftest.py`
- Test: `demo/gibbs-observatory/backend/tests/test_desktop_selftest.py`

**Interfaces:**
- Consumes: `desktop.paths.frontend_dist`.
- Produces:
  - `run_selftest() -> tuple[int, list[str]]` — exit code and the report lines. `0` when every check passes.

- [ ] **Step 1: Write the failing test**

Create `demo/gibbs-observatory/backend/tests/test_desktop_selftest.py`:

```python
"""A frozen binary must be checkable without opening a window.

This is the acceptance test for the whole slice: if `--selftest` exits zero
inside the packaged application on a machine with no Python and no Node, the
bundle genuinely carries the compiler and THRML.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from desktop.selftest import run_selftest  # noqa: E402


def test_selftest_passes_in_a_working_checkout():
    code, lines = run_selftest()
    report = "\n".join(lines)
    assert code == 0, f"self-test failed in a working checkout:\n{report}"


def test_selftest_reports_every_check_by_name():
    _, lines = run_selftest()
    report = "\n".join(lines).lower()
    for probe in ("compiler", "thrml", "frontend", "preflight"):
        assert probe in report, f"{probe} check is not reported by name"


def test_selftest_reports_a_real_preflight_verdict():
    """The compiler must actually run, not merely import."""
    _, lines = run_selftest()
    report = "\n".join(lines)
    assert "verdict" in report.lower()
    assert "ok" in report.lower()


def test_failed_checks_produce_a_nonzero_exit(monkeypatch):
    """A self-test that cannot fail proves nothing."""
    import desktop.selftest as st

    monkeypatch.setattr(
        st, "_check_frontend",
        lambda: ("frontend", False, "unavailable: forced failure"))
    code, lines = run_selftest()
    assert code != 0, "a forced failure must not exit zero"
    assert "forced failure" in "\n".join(lines)
```

- [ ] **Step 2: Run the test to verify it fails**

```
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_selftest.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'desktop.selftest'`.

- [ ] **Step 3: Write the implementation**

Create `demo/gibbs-observatory/desktop/selftest.py`:

```python
"""Prove the packaged application carries a working compiler and sampler.

Run with `--selftest`, this exits zero only if the compiler imports AND
returns a real verdict, THRML imports AND draws samples, and the frontend
build is present. It opens no window, so it can be run by a script on a
machine that has neither Python nor Node installed.

Each check returns (name, ok, detail). A check that cannot fail proves
nothing, so `_check_frontend` is deliberately monkeypatchable and a test
forces it to fail.
"""
from __future__ import annotations

from .paths import frontend_dist


def _check_compiler() -> tuple[str, bool, str]:
    try:
        from tsu_compiler.preflight.check import preflight
        from tsu_compiler.preflight.model import load_model
    except Exception as exc:  # pragma: no cover - import failure path
        return ("compiler", False, f"unavailable: import failed: {exc!r}")

    import json

    # A 6x6 four-neighbour grid: small, bipartite, places immediately.
    w = h = 6
    idx = lambda x, y: y * w + x
    edges = []
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                edges.append([idx(x, y), idx(x + 1, y), 1.2])
            if y + 1 < h:
                edges.append([idx(x, y), idx(x, y + 1), 1.2])
    payload = json.dumps({"nodes": w * h, "edges": edges,
                          "biases": [0.3] * (w * h), "beta": 1.0})
    try:
        rep = preflight(load_model(edges_json=payload))
    except Exception as exc:
        return ("compiler", False, f"unavailable: preflight raised: {exc!r}")
    return ("compiler", rep.verdict == "ok",
            f"preflight verdict {rep.verdict} on a {w}x{h} grid")


def _check_thrml() -> tuple[str, bool, str]:
    try:
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram
    except Exception as exc:  # pragma: no cover - import failure path
        return ("thrml", False, f"unavailable: import failed: {exc!r}")

    try:
        a = [SpinNode() for _ in range(4)]
        b = [SpinNode() for _ in range(4)]
        nodes = a + b
        edges = [(u, v) for u in a for v in b]
        model = IsingEBM(nodes, edges,
                         jnp.zeros(len(nodes)), jnp.full(len(edges), 0.4),
                         jnp.array(1.0))
        program = IsingSamplingProgram(model, [Block(a), Block(b)], [])
        schedule = SamplingSchedule(n_warmup=5, n_samples=2, steps_per_sample=1)
        init = [jnp.zeros(len(a), dtype=bool), jnp.zeros(len(b), dtype=bool)]
        out = sample_states(jax.random.key(0), program, schedule, init, [],
                            [Block(nodes)])
    except Exception as exc:
        return ("thrml", False, f"unavailable: sampling raised: {exc!r}")
    shape = tuple(out[0].shape)
    return ("thrml", shape == (2, len(nodes)),
            f"sampled an 8-spin model, drew {shape}")


def _check_frontend() -> tuple[str, bool, str]:
    try:
        dist = frontend_dist()
    except Exception as exc:
        return ("frontend", False, str(exc))
    return ("frontend", True, f"built bundle at {dist}")


def run_selftest() -> tuple[int, list[str]]:
    """Run every check. Returns (exit_code, report_lines)."""
    checks = (_check_compiler(), _check_thrml(), _check_frontend())
    lines = ["Thermodynamic Workbench -- self-test", ""]
    failed = 0
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed += 1
    lines.append("")
    lines.append(f"{len(checks) - failed} of {len(checks)} checks passed.")
    lines.append("Simulation only. This is not Extropic silicon.")
    return (1 if failed else 0), lines
```

- [ ] **Step 4: Run the test to verify it passes**

```
.venv/Scripts/python.exe -m pytest backend/tests/test_desktop_selftest.py -v
```

Expected: 4 passed. If `_check_compiler` reports a verdict other than `ok`, stop and report it rather than relaxing the assertion — a 6x6 four-neighbour grid failing preflight is a real finding about the compiler, not a test problem.

- [ ] **Step 5: Commit**

```bash
git add demo/gibbs-observatory/desktop/selftest.py demo/gibbs-observatory/backend/tests/test_desktop_selftest.py
git commit -m "Add a headless self-test the frozen binary can run

Proves the packaged application carries a compiler that returns a real
verdict and a sampler that draws real samples, without opening a window, so
the bundle can be verified by a script on a machine with no Python or Node.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: The desktop entry point

**Files:**
- Create: `demo/gibbs-observatory/desktop/main.py`

**Interfaces:**
- Consumes: `desktop.server.build_desktop_app`, `desktop.selftest.run_selftest`.
- Produces: `main(argv: list[str] | None = None) -> int` — process exit code.

There is no unit test for window creation; pywebview needs a display. The entry point is kept deliberately thin so that everything worth testing already lives in Tasks 1 to 3, and the parts that remain are verified by Task 5's smoke test against the real binary.

- [ ] **Step 1: Install pywebview into both interpreters**

```
.venv/Scripts/python.exe -m pip install pywebview
```

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pip install pywebview
```

Expected: both succeed. If pywebview has no Python 3.14 wheel, stop and report it — that is a real finding and it changes the packaging approach, so do not silently substitute a different windowing library.

- [ ] **Step 2: Write the entry point**

Create `demo/gibbs-observatory/desktop/main.py`:

```python
"""Entry point for the packaged application.

Starts the API in a background thread on a loopback port nobody else is
using, then opens a native window onto it. Loopback is internal plumbing:
the user never sees a URL, never starts a server, and nothing leaves the
machine.
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

    uvicorn.Config(app, host=host, port=port, log_level="warning")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_health(url: str, timeout: float = 60.0) -> bool:
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
              f"60 seconds. The window was not opened.", file=sys.stderr)
        return 1

    import webview

    webview.create_window(WINDOW_TITLE, f"http://{HOST}:{port}/",
                          width=1600, height=1000, min_size=(1100, 700))
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run it from source**

```
cd demo/gibbs-observatory/frontend && npm install && npm run build && cd ..
.venv/Scripts/python.exe -m desktop.main --selftest
```

Expected: three PASS lines and exit code 0.

Then, without `--selftest`:

```
.venv/Scripts/python.exe -m desktop.main
```

Expected: a native window titled "Thermodynamic Workbench" showing the Observatory. No browser opens. Close the window; the process exits.

- [ ] **Step 4: Commit**

```bash
git add demo/gibbs-observatory/desktop/main.py
git commit -m "Add the desktop entry point

Runs the API in a background thread on a loopback port and opens a native
window onto it, so the user never starts a server or sees a URL. A
--selftest flag runs the headless checks and exits with a status code.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Freeze it, and prove the frozen thing works

The riskiest step. PyInstaller must carry JAX's native libraries and the built frontend, on Python 3.14, which is new enough that this may not work on the first attempt.

**Files:**
- Create: `demo/gibbs-observatory/desktop/workbench.spec`
- Create: `demo/gibbs-observatory/desktop/build.ps1`
- Create: `demo/gibbs-observatory/desktop/README.md`

- [ ] **Step 1: Write the PyInstaller spec**

Create `demo/gibbs-observatory/desktop/workbench.spec`:

```python
# PyInstaller build definition for the Thermodynamic Workbench.
#
# --onedir, not --onefile. A onefile bundle carrying JAX unpacks several
# hundred megabytes to a temporary directory on every launch, which costs
# tens of seconds of startup. Blender ships as a folder for the same reason.
# Distribution is a zip of the folder.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

for pkg in ("jax", "jaxlib", "thrml", "equinox"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("tsu_compiler")
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
]

datas += [
    ("../frontend/dist", "frontend/dist"),
    ("../programs", "programs"),
    ("../receipts", "receipts"),
]

a = Analysis(
    ["launch.py"],
    pathex=["..", "../../../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ThermodynamicWorkbench",
    console=False,
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ThermodynamicWorkbench",
)
```

Create `demo/gibbs-observatory/desktop/launch.py`, the frozen entry script, which must import the package absolutely because a frozen script has no parent package:

```python
"""Frozen entry script. PyInstaller runs this, not `python -m desktop.main`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the build script**

Create `demo/gibbs-observatory/desktop/build.ps1`:

```powershell
# Build the Thermodynamic Workbench into a distributable folder.
# Run from demo/gibbs-observatory/.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> building frontend"
Push-Location (Join-Path $Root "frontend")
npm install
npm run build
Pop-Location

Write-Host "==> freezing application"
$py = Join-Path $Root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pyinstaller pywebview
& $py -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build\workbench-work") `
    (Join-Path $PSScriptRoot "workbench.spec")

$exe = Join-Path $Root "dist\ThermodynamicWorkbench\ThermodynamicWorkbench.exe"
Write-Host "==> self-testing the frozen build"
& $exe --selftest
if ($LASTEXITCODE -ne 0) { throw "frozen build failed its self-test" }

$size = (Get-ChildItem (Split-Path $exe) -Recurse |
         Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("==> built: {0}  ({1:N0} MB)" -f $exe, $size)
```

- [ ] **Step 3: Build it**

```
powershell -ExecutionPolicy Bypass -File demo\gibbs-observatory\desktop\build.ps1
```

Expected: a folder at `demo/gibbs-observatory/dist/ThermodynamicWorkbench/`, a passing self-test, and a reported size.

This step is expected to fail at least once. Likely causes, in order of probability:

1. **JAX native libraries missing.** Symptom: `--selftest` reports the thrml check as `unavailable: import failed`. Fix: add the missing package to the `collect_all` loop.
2. **PyInstaller does not support Python 3.14.** Symptom: the analysis stage fails outright. Fix: report it and stop. The remedy is a 3.12 or 3.13 build environment, which is a change to the plan, not something to work around silently.
3. **pywebview has no 3.14 wheel.** Already surfaced in Task 4 Step 1 if so.

Record whichever occurred, and what fixed it, in `desktop/README.md`. Do not delete a failing check to make the build pass.

- [ ] **Step 4: Prove it runs without a development environment**

Copy `dist/ThermodynamicWorkbench/` to a directory outside the repository, then:

```
cd <that directory>\ThermodynamicWorkbench
.\ThermodynamicWorkbench.exe --selftest
```

Expected: three PASS lines, exit code 0. Then run it with no arguments and confirm the window opens.

The strongest available check is a machine with no Python and no Node on PATH. If one is not available, say so plainly in the README rather than claiming a verification that was not performed.

- [ ] **Step 5: Write the README**

Create `demo/gibbs-observatory/desktop/README.md` covering: what the build produces, its measured size, how to build it, what the target machine requires (WebView2, present by default on Windows 11), how to run the self-test, and whatever failed during Step 3 and what resolved it.

- [ ] **Step 6: Commit**

```bash
git add demo/gibbs-observatory/desktop/workbench.spec demo/gibbs-observatory/desktop/launch.py demo/gibbs-observatory/desktop/build.ps1 demo/gibbs-observatory/desktop/README.md
git commit -m "Freeze the application into a distributable folder

One build command produces a folder that runs on a machine with no Python,
no Node and no network. The build self-tests the frozen binary before
reporting success, so a bundle that cannot import its own compiler fails the
build rather than shipping.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Make the whole suite green

**Files:**
- Modify: `demo/gibbs-observatory/.gitignore` (add `dist/`, `build/`)

- [ ] **Step 1: Run every test**

```
cd demo/gibbs-observatory
.venv/Scripts/python.exe -m pytest backend/tests -v
```

Expected: everything passes. Any pre-existing failure is reported, not repaired inside this slice.

- [ ] **Step 2: Confirm build artefacts are not tracked**

```
git status --short demo/gibbs-observatory
```

Expected: no `dist/` or `build/` entries. Add them to `.gitignore` if present.

- [ ] **Step 3: Commit**

```bash
git add demo/gibbs-observatory/.gitignore
git commit -m "Ignore desktop build artefacts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** This plan implements section 16 (Packaging) and Slice 0 of section 17. It deliberately implements no part of sections 4 to 14 — those are later slices. Section 18's open questions on distribution and code signing are untouched and remain open; they are decisions for the owner, not implementation work.

**Placeholders.** None. Every step carries the content it needs. Task 5 Step 3 names three expected failure modes with their fixes rather than saying "handle errors".

**Type consistency.** `resource_root`, `frontend_dist`, `is_frozen`, `FrontendMissingError`, `mount_frontend`, `build_desktop_app`, `run_selftest`, `_check_frontend`, `pick_free_port` and `main` are spelled identically everywhere they appear.

**One known gap, stated rather than hidden.** Task 4 has no automated test, because window creation needs a display. It is mitigated by keeping the entry point thin and by Task 5 Step 4 exercising the real binary. If the entry point grows beyond its current size, it needs tests.
