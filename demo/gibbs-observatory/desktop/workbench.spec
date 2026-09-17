# PyInstaller build definition for the Thermodynamic Workbench.
#
# --onedir, not --onefile. A onefile bundle carrying JAX unpacks several
# hundred megabytes to a temporary directory on every launch, which costs tens
# of seconds of startup on each run. Blender ships as a folder for the same
# reason. Distribution is a zip of the folder, or an installer later.
#
# Paths here resolve against SPECPATH (this file's directory), which
# PyInstaller injects, rather than the working directory the build was
# launched from.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

HERE = SPECPATH                                   # desktop/
OBS = os.path.dirname(HERE)                       # gibbs-observatory/
REPO = os.path.dirname(os.path.dirname(OBS))      # tsu-compiler/
SRC = os.path.join(REPO, "src")

datas, binaries, hiddenimports = [], [], []

# JAX and THRML carry native libraries and data files that no automatic hook
# finds. pythonnet and clr_loader are pywebview's bridge to the Windows
# WebView2 runtime and are equally native.
for pkg in ("jax", "jaxlib", "thrml", "equinox", "webview", "clr_loader",
            "pythonnet"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("tsu_compiler")

# uvicorn resolves its protocol and loop implementations by string at runtime,
# so static analysis cannot see them.
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
]

# Application data. `frontend/dist` is the built UI; `programs` holds the
# Extropic workloads the self-test compiles; `receipts` are what the viewer
# loads. Destinations mirror the source layout because paths.resource_root()
# resolves against the bundle root exactly as it does against the repository.
datas += [
    (os.path.join(OBS, "frontend", "dist"), os.path.join("frontend", "dist")),
    (os.path.join(OBS, "programs"), "programs"),
    (os.path.join(OBS, "receipts"), "receipts"),
]

# The shipped programs import their physics from the audit reference rather
# than restating it, because retyping a constant is how two files stop
# describing the same model. That reference therefore has to travel with the
# bundle: without it the programs load fine in a checkout and raise
# ModuleNotFoundError in the frozen build, which is a failure only a frozen
# build exhibits. Placed beside programs/ so one sys.path entry finds both.
_AUDIT = os.path.join(os.path.dirname(os.path.dirname(OBS)), "audit")
for _shared in ("visibility_as_inference.py",):
    _src = os.path.join(_AUDIT, _shared)
    if not os.path.isfile(_src):
        raise SystemExit(
            f"workbench.spec: {_src} is missing and the shipped programs "
            f"import it; the build would produce a bundle whose programs "
            f"cannot load")
    datas.append((_src, "programs"))

a = Analysis(
    [os.path.join(HERE, "launch.py")],
    pathex=[OBS, SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # tkinter is LATTICE's toolkit and nothing here uses it; matplotlib is
    # pulled in by audit scripts that the application never calls.
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ThermodynamicWorkbench",
    debug=False,
    strip=False,
    upx=False,
    # A console application on purpose: `--selftest` must reach real stdout so
    # a script can read its report and the build can gate on it. A windowed
    # build discards that output silently. main.hide_console_window() hides the
    # window when the application is launched normally.
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ThermodynamicWorkbench",
)
