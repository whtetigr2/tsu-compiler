#!/usr/bin/env python3
"""Gibbs Observatory Windows launcher — setup + run for clone-and-go.

Place GibbsObservatory.exe in the *Gibbs Observatory* repo root
(next to README.md / backend / frontend). Double-click: checks Python+Node,
creates .venv, pip/npm install, starts API+UI on Observatory-owned ports,
opens the browser only after /api/health says service=gibbs-observatory.

Dedicated default ports (avoid colliding with other TSU demos / ThermoLith):
  API  8088
  UI   5188
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "Gibbs Observatory"
# Observatory-owned defaults — do NOT reuse 5173/8000 (other labs often bind those).
API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8088
DEFAULT_UI_PORT = 5188


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, preferred: int, span: int = 20) -> int:
    for p in range(preferred, preferred + span):
        if _port_free(host, p):
            return p
    raise RuntimeError(f"No free port near {preferred}")


def looks_like_observatory(root: Path) -> bool:
    """Refuse to launch from unrelated trees (e.g. lithography demos)."""
    if not (root / "backend").is_dir() or not (root / "frontend").is_dir():
        return False
    main_py = root / "backend" / "app" / "main.py"
    if not main_py.is_file():
        return False
    try:
        blob = main_py.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if "gibbs-observatory" not in blob and "Gibbs Observatory" not in blob:
        return False
    # Hard no: lithography / ThermoLith product trees
    name = root.name.lower()
    banned = ("litho", "thermolith", "placement-lab", "placement_lab")
    if any(b in name for b in banned):
        return False
    return True


def repo_root() -> Path:
    """Exe must live in the Observatory repo root (or launcher/ under it)."""
    if getattr(sys, "frozen", False):
        cand = Path(sys.executable).resolve().parent
    else:
        here = Path(__file__).resolve().parent
        if (here.parent / "backend").is_dir() and (here.parent / "frontend").is_dir():
            cand = here.parent
        elif (here / "backend").is_dir():
            cand = here
        else:
            cand = here.parent

    if looks_like_observatory(cand):
        return cand

    # Walk up a few levels in case the exe was nested oddly
    for parent in list(cand.parents)[:4]:
        if looks_like_observatory(parent):
            return parent
        nested = parent / "gibbs-observatory"
        if looks_like_observatory(nested):
            return nested

    raise RuntimeError(
        "GibbsObservatory.exe must sit in the Gibbs Observatory repo root "
        "(folder with backend/, frontend/, and backend/app/main.py identifying "
        "gibbs-observatory).\n"
        f"Resolved path was: {cand}\n"
        "This launcher will not start lithography / ThermoLith / other TSU demos."
    )


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def find_python() -> str | None:
    if sys.platform.startswith("win"):
        py = which("py")
        if py:
            try:
                out = subprocess.check_output(
                    [py, "-3", "-c", "import sys; print(sys.executable)"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                ).strip()
                if out and Path(out).exists():
                    return out
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                pass
    for name in ("python", "python3"):
        p = which(name)
        if p:
            return p
    return None


def find_npm() -> str | None:
    return which("npm.cmd") or which("npm")


def find_node() -> str | None:
    return which("node")


class LauncherApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — launcher")
        self.root.geometry("740x500")
        self.root.minsize(560, 360)
        self.root.configure(bg="#0a0b0c")

        try:
            self.root_path = repo_root()
            self._root_error = None
        except RuntimeError as exc:
            self.root_path = Path(".")
            self._root_error = str(exc)

        self.procs: list[subprocess.Popen] = []
        self._stop = threading.Event()
        self.api_port = DEFAULT_API_PORT
        self.ui_port = DEFAULT_UI_PORT
        self.ui_url = f"http://{API_HOST}:{self.ui_port}"
        self.health_url = f"http://{API_HOST}:{self.api_port}/api/health"

        frm = tk.Frame(self.root, bg="#0a0b0c", padx=14, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(
            frm,
            text=APP_NAME,
            fg="#e8ecee",
            bg="#0a0b0c",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frm,
            text="Compiled-program inspector · JAX/THRML simulation — not Extropic silicon. "
            "Not ThermoLith / lithography.",
            fg="#8b939c",
            bg="#0a0b0c",
            font=("Segoe UI", 9),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        self.status = tk.StringVar(value="Ready.")
        tk.Label(
            frm, textvariable=self.status, fg="#7ec8c0", bg="#0a0b0c", font=("Consolas", 10)
        ).pack(anchor="w")

        self.bar = ttk.Progressbar(frm, mode="indeterminate", length=700)
        self.bar.pack(fill="x", pady=8)

        self.log = scrolledtext.ScrolledText(
            frm,
            height=18,
            bg="#121416",
            fg="#e8ecee",
            insertbackground="#e8ecee",
            font=("Consolas", 9),
            relief="flat",
        )
        self.log.pack(fill="both", expand=True, pady=(4, 8))

        btnrow = tk.Frame(frm, bg="#0a0b0c")
        btnrow.pack(fill="x")
        self.go_btn = tk.Button(
            btnrow,
            text="Install & Launch",
            command=self.start,
            bg="#1a1d21",
            fg="#e8ecee",
            activebackground="#2a2f35",
            relief="flat",
            padx=14,
            pady=6,
        )
        self.go_btn.pack(side="left")
        tk.Button(
            btnrow,
            text="Open UI",
            command=lambda: webbrowser.open(self.ui_url),
            bg="#121416",
            fg="#e8ecee",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=8)
        tk.Button(
            btnrow,
            text="Quit",
            command=self.shutdown,
            bg="#3a1b1b",
            fg="#e8ecee",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        if self._root_error:
            self._writeln(self._root_error)
            self._set_status("Wrong folder — see log.")
            self.go_btn.configure(state="disabled")
        else:
            self._writeln(f"Repo root: {self.root_path}")
            self._writeln(
                f"Will use API :{DEFAULT_API_PORT}+ and UI :{DEFAULT_UI_PORT}+ "
                "(not 8000/5173 — those often belong to other demos)."
            )

    def _writeln(self, msg: str) -> None:
        def _do() -> None:
            self.log.insert("end", msg + "\n")
            self.log.see("end")

        self.root.after(0, _do)

    def _set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.status.set(msg))

    def start(self) -> None:
        if self._root_error:
            return
        self.go_btn.configure(state="disabled")
        self.bar.start(12)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self._setup_and_launch()
        except Exception as exc:  # noqa: BLE001
            self._writeln(f"ERROR: {exc}")
            self._set_status("Failed — see log.")
            self.root.after(0, lambda: self.go_btn.configure(state="normal"))
            self.root.after(0, self.bar.stop)

    def _run_cmd(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict | None = None,
        check: bool = True,
    ) -> int:
        self._writeln(f"$ {' '.join(args)}")
        creation = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        p = subprocess.Popen(
            args,
            cwd=str(cwd or self.root_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creation,
        )
        assert p.stdout is not None
        for line in p.stdout:
            if self._stop.is_set():
                p.terminate()
                break
            line = line.rstrip()
            if line:
                self._writeln(line)
        rc = p.wait()
        if check and rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {' '.join(args)}")
        return rc

    def _setup_and_launch(self) -> None:
        root = self.root_path
        if not looks_like_observatory(root):
            raise RuntimeError(
                "Refusing to launch: this folder is not Gibbs Observatory "
                "(missing identity in backend/app/main.py)."
            )

        self._set_status("Checking Python & Node…")
        py = find_python()
        node = find_node()
        npm = find_npm()
        if not py:
            raise RuntimeError(
                "Python 3.10+ not found. Install from https://www.python.org/downloads/ "
                "(check “Add to PATH”) or `winget install Python.Python.3.12`."
            )
        if not node or not npm:
            raise RuntimeError(
                "Node.js / npm not found. Install LTS from https://nodejs.org/ "
                "or `winget install OpenJS.NodeJS.LTS`."
            )
        self._writeln(f"Python: {py}")
        self._writeln(f"Node:   {node}")
        self._writeln(f"npm:    {npm}")

        venv = root / ".venv"
        if sys.platform.startswith("win"):
            vpy = venv / "Scripts" / "python.exe"
        else:
            vpy = venv / "bin" / "python"

        if not vpy.exists():
            self._set_status("Creating virtualenv…")
            self._run_cmd([py, "-m", "venv", str(venv)])
        else:
            self._writeln("Using existing .venv")

        req = root / "requirements.txt"
        self._set_status("Installing Python deps (first run can take several minutes)…")
        self._run_cmd([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
        self._run_cmd([str(vpy), "-m", "pip", "install", "-r", str(req)])

        fe = root / "frontend"
        if not (fe / "node_modules").is_dir():
            self._set_status("Installing frontend (npm)…")
            self._run_cmd([npm, "install"], cwd=fe)
        else:
            self._writeln("frontend/node_modules present — skipping npm install")

        self._kill_children()

        self.api_port = pick_port(API_HOST, DEFAULT_API_PORT)
        self.ui_port = pick_port(API_HOST, DEFAULT_UI_PORT)
        self.ui_url = f"http://{API_HOST}:{self.ui_port}"
        self.health_url = f"http://{API_HOST}:{self.api_port}/api/health"
        self._writeln(f"Ports: API {self.api_port} · UI {self.ui_port}")

        env = os.environ.copy()
        # Strict PYTHONPATH: Observatory root + optional tsu for notepad only.
        # Never scan Documents or lithography trees.
        paths = [str(root)]
        tsu_root = os.environ.get("TSU_ROOT")
        candidates: list[Path] = []
        if tsu_root:
            candidates.append(Path(tsu_root))
        # Paul layout: .../tsu-compiler/demo/gibbs-observatory → .../tsu-compiler/src/tsu
        candidates.extend(
            [
                root.parent.parent / "src",
                root.parent.parent,
                root.parent / "src",
                root.parent,
                root.parent / "tsu-compiler" / "src",
                root.parent / "tsu-compiler",
                root / "vendor" / "tsu-compiler" / "src",
                root / "vendor" / "tsu-compiler",
            ]
        )
        for sib in candidates:
            if not sib:
                continue
            low = str(sib).lower()
            if "litho" in low or "thermolith" in low:
                continue
            if (sib / "tsu" / "__init__.py").is_file():
                paths.append(str(sib.resolve()))
                break
            if (sib / "src" / "tsu" / "__init__.py").is_file():
                paths.append(str((sib / "src").resolve()))
                break
        env["PYTHONPATH"] = os.pathsep.join(paths)
        self._writeln(f"PYTHONPATH={env['PYTHONPATH']}")

        creation = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0

        self._set_status("Starting API…")
        api = subprocess.Popen(
            [
                str(vpy),
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                API_HOST,
                "--port",
                str(self.api_port),
            ],
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation,
        )
        self.procs.append(api)
        threading.Thread(target=self._pump, args=(api, "api"), daemon=True).start()

        # Point Vite proxy at our API port via env (vite.config may read it)
        env_ui = env.copy()
        env_ui["VITE_API_PROXY_TARGET"] = f"http://{API_HOST}:{self.api_port}"
        env_ui["GIBBS_API_PORT"] = str(self.api_port)

        self._set_status("Starting UI…")
        ui = subprocess.Popen(
            [
                npm,
                "run",
                "dev",
                "--",
                "--host",
                API_HOST,
                "--port",
                str(self.ui_port),
                "--strictPort",
            ],
            cwd=str(fe),
            env=env_ui,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation,
        )
        self.procs.append(ui)
        threading.Thread(target=self._pump_ui, args=(ui,), daemon=True).start()

        self._set_status("Waiting for Gibbs Observatory health…")
        ok = self._wait_observatory_health(timeout=180)
        if not ok:
            raise RuntimeError(
                "API did not report service=gibbs-observatory in time. "
                "Refusing to open the browser (avoids landing on another demo)."
            )

        # Ensure UI answers on our port
        if not self._wait_http(self.ui_url, timeout=60):
            raise RuntimeError(f"UI did not respond at {self.ui_url}")

        self._set_status(f"Open → {self.ui_url}")
        self._writeln(f"Launching browser: {self.ui_url}")
        webbrowser.open(self.ui_url)
        self.root.after(0, self.bar.stop)
        self._writeln("Running Gibbs Observatory. Close this window or Quit to stop servers.")
        self.root.after(0, lambda: self.go_btn.configure(state="normal"))

    def _pump(self, proc: subprocess.Popen, tag: str) -> None:
        if not proc.stdout:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            line = line.rstrip()
            if line:
                self._writeln(f"[{tag}] {line}")

    def _pump_ui(self, proc: subprocess.Popen) -> None:
        if not proc.stdout:
            return
        local_re = re.compile(r"Local:\s+(https?://\S+)")
        for line in proc.stdout:
            if self._stop.is_set():
                break
            line = line.rstrip()
            if line:
                self._writeln(f"[ui] {line}")
            m = local_re.search(line)
            if m:
                self.ui_url = m.group(1).rstrip("/")
                self._writeln(f"Detected Vite URL: {self.ui_url}")

    def _wait_http(self, url: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if 200 <= r.status < 500:
                        return True
            except Exception:
                time.sleep(0.4)
        return False

    def _wait_observatory_health(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as r:
                    body = r.read().decode("utf-8", errors="replace")
                    self._writeln(f"health: {body[:240]}")
                    data = json.loads(body)
                    if data.get("service") == "gibbs-observatory" or data.get("ok") is True:
                        # Prefer explicit service name when present
                        if data.get("service") and data.get("service") != "gibbs-observatory":
                            self._writeln(
                                f"Refusing foreign service={data.get('service')!r} on {self.health_url}"
                            )
                            return False
                        return True
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                time.sleep(0.5)
        return False

    def _kill_children(self) -> None:
        for p in self.procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except OSError:
                pass
        self.procs.clear()

    def shutdown(self) -> None:
        self._stop.set()
        self._kill_children()
        self.root.destroy()

    def mainloop(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = LauncherApp()
    if not app._root_error:
        app.root.after(400, app.start)
    app.mainloop()


if __name__ == "__main__":
    main()
