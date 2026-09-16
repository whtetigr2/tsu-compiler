"""Thermodynamic Program notepad: preflight / ideal-first compile / apply.

Wires the Observatory to an importable ``tsu`` package. Discovery order:
``TSU_ROOT`` env, then sibling dirs ``tsu-compiler`` / ``tsu-compiler-review``,
then a legacy box path fallback. Ideal-first compile is
``tsu_compiler.passes.search.compile_spec``, same path as ``tsuc compile``.

Standing prohibitions enforced here:
- no silent apply without a COMPILED receipt on disk
- no β slider / silicon claims in this module
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from .receipt_loader import DEFAULT_RECEIPTS_ROOT, load_receipt, receipts_root


def _tsu_root_candidates() -> list[Path]:
    """Candidate directories that may *contain* the ``tsu`` package.

    Paul's layout: ``Documents/tsu-compiler/demo/gibbs-observatory`` with the
    package at ``Documents/tsu-compiler/src/tsu``, so we must walk up to the
    compiler root and accept a ``src/`` layout, not only a top-level ``tsu/``.
    """
    repo = Path(__file__).resolve().parents[2]  # gibbs-observatory/
    parent = repo.parent  # often .../demo
    cands: list[Path] = []
    env = os.environ.get("TSU_ROOT", "").strip()
    if env:
        cands.append(Path(env).expanduser())

    # Walk upward: demo/ -> tsu-compiler/, plus named siblings
    for base in (repo, parent, *list(repo.parents)[:4]):
        cands.append(base)
        cands.append(base / "src")
        for name in ("tsu-compiler", "tsu-compiler-review"):
            cands.append(base / name)
            cands.append(base / name / "src")

    # Legacy box path
    cands.append(Path("/workspace/tsu-compiler-review"))
    cands.append(Path("/workspace/tsu-compiler-review/src"))

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for c in cands:
        try:
            key = str(c.resolve()) if c.exists() else str(c)
        except OSError:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# The compiler package was renamed tsu -> tsu_compiler (and the command to
# tsuc) in tsu-compiler commit 8a8e941, because the PyPI name "tsu" was taken
# and the thing compiles FOR a TSU rather than being one. This check was not
# updated with it -- note the docstring below already said tsu_compiler while
# the code underneath still looked for tsu/, so every discovery attempt failed,
# `tsu_root` read null whatever TSU_ROOT was set to, the notepad's preflight /
# compile / apply path was dead, and seven backend tests skipped themselves.
PACKAGE_DIR = "tsu_compiler"
DISTRIBUTION_NAME = "tsu-compiler"


def _is_tsu_package_root(root: Path) -> bool:
    """True if ``root`` belongs on PYTHONPATH (``import tsu_compiler`` works)."""
    init_py = root / PACKAGE_DIR / "__init__.py"
    cli_py = root / PACKAGE_DIR / "cli.py"
    return init_py.is_file() and cli_py.is_file()


# Placement effort tiers, tried cheapest first. The notepad is INTERACTIVE, so
# a flat high budget is the wrong answer -- it would make every trivial program
# wait on the worst case. Escalation only happens when placement actually fails.
#
# Measured on prog_seq_design_longer (16 spins, non-bipartite, needs mediators):
# does NOT place at 6/40k after ~2s, does NOT place at 12/100k after ~12s, and
# places cleanly at 24/250k in ~33s with 8 mediators. Before this the notepad
# reported that program as "fail" with every gate passing, because the default
# budget was the only one ever tried.
#
# 800k is a ceiling, not a target: something that cannot place inside it is
# telling you to change the encoding, which is what the compiler's own
# remediation already says.
PLACEMENT_TIERS: tuple[tuple[int, int], ...] = (
    (6, 40_000),
    (24, 250_000),
    (48, 800_000),
)

NOTEPAD_RECEIPT_ID = "notepad"
DEFAULT_SEED_SPEC = """\
name: observatory_toy
description: >
  Four-variable toy Thermodynamic Program for Gibbs Observatory notepad.
  Prefers YAML / tsu_compiler.spec style. Edit, Preflight, then Compile (ideal-first).
variables:
  a: {domain: binary}
  b: {domain: binary}
  c: {domain: categorical, k: 3}
terms:
  - {kind: product, a: {a: 1.0}, b: {b: 1.0}, weight: 2.0}
  - {kind: linear, form: {a: 1.0, b: 1.0}, weight: -0.5}
  - {kind: product, a: {"c=0": 1.0}, b: {a: 1.0}, weight: 1.5}
contract:
  validate:
    - {rule: forbid_both, vars: [a, b], message: "a and b must not both be occupied"}
    - {rule: forbid_value_with, var: c, value: 0, other: a,
       message: "c=0 forbidden while a is occupied"}
"""


class ProgramServiceError(Exception):
    """User-facing refusal (bad YAML, missing tsu, apply without receipt)."""

    def __init__(self, message: str, *, detail: Any | None = None):
        super().__init__(message)
        self.detail = detail


def discover_tsu_root() -> Path | None:
    for root in _tsu_root_candidates():
        if _is_tsu_package_root(root):
            return root.resolve()
        # Compiler checkout with src/ layout: .../tsu-compiler/src/tsu
        src = root / "src"
        if _is_tsu_package_root(src):
            return src.resolve()
    return None


def _already_importable_root() -> Path | None:
    """The directory holding `tsu_compiler/`, if the package already imports.

    Directory discovery cannot see two perfectly good installations: a frozen
    PyInstaller bundle, where the package lives inside the archive and no such
    directory exists anywhere on disk, and an ordinary `pip install`, where it
    sits in site-packages rather than beside the Observatory. In both the
    import works and the hunt returns nothing, so the import is tried first.

    Returns None rather than raising, so the caller still falls back to
    discovery when the package genuinely is not importable.
    """
    try:
        import tsu_compiler
    except Exception:
        return None
    spec_file = getattr(tsu_compiler, "__file__", None)
    if spec_file:
        return Path(spec_file).resolve().parent.parent
    # Namespace package or a frozen loader with no __file__: importable all
    # the same, and the path is not needed once it imports.
    return Path(sys.prefix)


def ensure_tsu_importable() -> Path:
    """Make `tsu_compiler` importable and return the root that holds it."""
    already = _already_importable_root()
    if already is not None:
        return already

    root = discover_tsu_root()
    if root is None:
        raise ProgramServiceError(
            "tsu_compiler package not found. Expected an importable tsu_compiler "
            "next to Observatory (e.g. Documents/tsu-compiler/src when "
            "Observatory lives under Documents/tsu-compiler/demo/"
            "gibbs-observatory). Set TSU_ROOT to the directory that "
            "*contains* the tsu_compiler/ folder (for src layout: "
            ".../tsu-compiler/src), or PYTHONPATH to that path."
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    try:
        import tsu_compiler  # noqa: F401
    except ImportError as exc:
        raise ProgramServiceError(
            f"tsu import failed after adding {root_s} to PYTHONPATH: {exc}. "
            "Install networkx + sympy into the Observatory .venv."
        ) from exc
    return root


def _gate_to_dict(g: Any) -> dict[str, Any]:
    limit = getattr(g, "limit", None)
    if isinstance(limit, float) and limit == float("inf"):
        limit_out: Any = None
        limit_infinite = True
    else:
        limit_out = float(limit) if limit is not None else None
        limit_infinite = False
    return {
        "name": str(getattr(g, "name", "?")),
        "value": float(getattr(g, "value", 0.0)),
        "limit": limit_out,
        "limit_infinite": limit_infinite,
        "status": str(getattr(g, "status", "?")),
        "note": str(getattr(g, "note", "") or ""),
        "assumed": bool(getattr(g, "assumed", False)),
        "downgraded": bool(getattr(g, "downgraded", False)),
        "passed": str(getattr(g, "status", "")) in ("ok", "warn", "downgraded"),
    }


def _write_temp_spec(yaml_text: str) -> Path:
    text = (yaml_text or "").strip()
    if not text:
        raise ProgramServiceError("Empty Thermodynamic Program buffer.")
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="gibbs_tp_",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(text)
        if not text.endswith("\n"):
            tmp.write("\n")
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def preflight_edges(edges_json: str, *, allow_assumed: bool = False) -> dict[str, Any]:
    """Preflight a model given directly as an edge list.

    `tsu_compiler.preflight.model.load_model` has always taken EITHER a spec or
    an edge list, and the Observatory only ever called the spec side. That left
    out every model that is data-driven rather than declarative: the playable
    level's visibility lattice is 11,200 spins whose biases come from the
    occupancy in front of the player and change every frame. Writing that as a
    workload spec would mean emitting 11,200 `linear` terms per frame, which is
    a data dump wearing a program's clothes.

    Schema is the compiler's own, unchanged:
        {"nodes": int, "edges": [[i, j, w], ...], "biases": [...], "beta": float}
    """
    ensure_tsu_importable()
    from tsu_compiler.preflight.model import load_model

    tmp = Path(tempfile.mkdtemp(prefix="obs_edges_")) / "model.edges.json"
    tmp.write_text(edges_json, encoding="utf-8")
    try:
        try:
            model = load_model(edges=str(tmp))
        except ValueError as exc:
            raise ProgramServiceError(f"edge list refused, {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProgramServiceError(
                f"edge list load failed: {type(exc).__name__}: {exc}") from exc
        return _preflight_model(model, allow_assumed=allow_assumed,
                                source="edges")
    finally:
        tmp.unlink(missing_ok=True)
        try:
            tmp.parent.rmdir()
        except OSError:
            pass


def _preflight_model(model, *, allow_assumed: bool = False,
                     source: str = "spec") -> dict[str, Any]:
    """Escalating placement + serialisation, shared by both input doors.

    Extracted so the spec path and the edge-list path cannot drift: a fix to
    the escalation ladder or the payload has to apply to both by construction.
    """
    from tsu_compiler.preflight.check import preflight as run_preflight

    t0 = time.time()
    rep = None
    attempts: list[dict[str, Any]] = []
    for restarts, iters in PLACEMENT_TIERS:
        try:
            rep = run_preflight(model, allow_assumed=allow_assumed,
                                restarts=restarts, iters=iters)
        except Exception as exc:  # noqa: BLE001
            raise ProgramServiceError(
                f"preflight failed: {type(exc).__name__}: {exc}") from exc
        attempts.append({"restarts": restarts, "iters": iters,
                         "placed": bool(rep.placed),
                         "seconds": round(float(rep.place_seconds), 3)})
        if rep.placed:
            break
        # A gate failure is a property of the model, not of how hard we looked
        # for an embedding. Spending the bigger budgets on it would burn a
        # minute to reach the same refusal.
        if any(g.status == "fail" for g in (rep.gates or ())):
            break

    elapsed = time.time() - t0
    return {
        "ok": True,
        "kind": "preflight",
        "source": source,
        "verdict": str(rep.verdict),
        "n_spins": int(rep.n_spins),
        "n_couplings": int(rep.n_couplings),
        "max_degree": int(rep.max_degree),
        "bipartite": bool(rep.bipartite),
        "embedding": str(rep.embedding),
        "mediators": int(rep.mediators),
        "place_seconds": float(rep.place_seconds),
        "placed": bool(rep.placed),
        "place_error": rep.place_error,
        "remediations": list(rep.remediations or ()),
        "fabric_note": rep.fabric_note,
        "gates": [_gate_to_dict(g) for g in getattr(rep, "gates", ()) or ()],
        "elapsed_seconds": round(elapsed, 4),
        "placement_attempts": attempts,
        "placement_effort": (
            {"restarts": attempts[-1]["restarts"], "iters": attempts[-1]["iters"]}
            if attempts else None),
        "placement_escalated": len(attempts) > 1,
        "label": "JAX/THRML simulation, not Extropic silicon",
        "notes": [
            "Preflight only, no receipt written; Apply is still forbidden.",
            "Ideal-first compile is a separate action (Compile).",
        ],
    }


def preflight_program(yaml_text: str, *, allow_assumed: bool = False) -> dict[str, Any]:
    """Run ``tsuc preflight``-equivalent on YAML text."""
    ensure_tsu_importable()
    from tsu_compiler.preflight.model import load_model

    path = _write_temp_spec(yaml_text)
    try:
        try:
            model = load_model(spec=str(path))
        except ValueError as exc:
            raise ProgramServiceError(f"preflight refused, {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProgramServiceError(
                f"spec load failed: {type(exc).__name__}: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    return _preflight_model(model, allow_assumed=allow_assumed, source="spec")


def compile_program(
    yaml_text: str,
    *,
    allow_assumed: bool = False,
    target: str = "z1",
    receipt_id: str = NOTEPAD_RECEIPT_ID,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Ideal-first compile; on success write receipt under receipts/{receipt_id}."""
    ensure_tsu_importable()
    from tsu_compiler.passes.search import compile_spec
    from tsu_compiler.receipt import write_receipt
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    if target not in PROFILES:
        raise ProgramServiceError(
            f"Unknown target {target!r}. Known: {sorted(PROFILES)}"
        )
    if not receipt_id or "/" in receipt_id or ".." in receipt_id or receipt_id.startswith("."):
        raise ProgramServiceError(f"Invalid receipt_id {receipt_id!r}")

    path = _write_temp_spec(yaml_text)
    t0 = time.time()
    errors: list[str] = []
    try:
        try:
            spec = load_spec(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ProgramServiceError(
                f"spec load failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            # compile_spec always runs ideal FIRST (tsu_compiler.passes.search docstring).
            comp = compile_spec(
                spec,
                PROFILES[target],
                allow_assumed,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=6)
            raise ProgramServiceError(
                f"compile failed: {type(exc).__name__}: {exc}",
                detail={"traceback": tb},
            ) from exc

        base = receipts_root(root)
        out_dir = base / receipt_id
        # Replace previous notepad receipt atomically-ish
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = write_receipt(comp, str(out_dir))
        verdict = str(comp.verdict)
        cand_rows = []
        repset = getattr(comp, "repset", None)
        if repset is not None:
            for cand in getattr(repset, "candidates", ()) or ():
                state = getattr(cand, "state", None)
                state_s = getattr(state, "value", None) or str(state)
                cand_rows.append(
                    {
                        "encoding": str(getattr(cand, "encoding", "")),
                        "state": str(state_s),
                        "reason": str(getattr(cand, "reason", "") or ""),
                    }
                )
                # VIABLE_NOT_SELECTED is a healthy runner-up encoding, not a compile error.
                # Only surface true failures (HARDWARE_INFEASIBLE, etc.) as errors.
                if str(state_s) in (
                    "HARDWARE_INFEASIBLE",
                    "IDEAL_INFEASIBLE",
                    "FAILED",
                    "ERROR",
                ) and getattr(cand, "reason", None):
                    errors.append(f"{getattr(cand, 'encoding', '?')}: {state_s}: {cand.reason}")

        # Prefer live inspect of written receipt when COMPILED
        inspect: dict[str, Any] | None = None
        if verdict == "COMPILED":
            try:
                inspect = load_receipt(receipt_id, root=base)
            except Exception:  # noqa: BLE001
                inspect = None

        gates = []
        if inspect and isinstance(inspect.get("gates"), list):
            gates = inspect["gates"]
        elif hasattr(comp, "gates") and comp.gates:
            # Fallback, structure varies; leave empty rather than invent
            gates = []

        med = None
        bipartite = None
        n_nodes = None
        if inspect:
            med = (inspect.get("spins") or {}).get("mediators")
            bipartite = (inspect.get("connectivity") or {}).get("physical", {}).get(
                "bipartite"
            )
            if bipartite is None:
                bipartite = (inspect.get("metrics") or {}).get("bipartite")
            n_nodes = (inspect.get("spins") or {}).get("n_nodes")

        elapsed = time.time() - t0
        return {
            "ok": verdict == "COMPILED",
            "kind": "compile",
            "verdict": verdict,
            "receipt_id": receipt_id if verdict == "COMPILED" else None,
            "receipt_path": str(written) if verdict == "COMPILED" else str(out_dir),
            "ideal_first": True,
            "target": target,
            "allow_assumed": allow_assumed,
            "gates": gates,
            "candidates": cand_rows,
            "encoding_note": (
                "Multiple encodings can be viable; one is SELECTED by ranking "
                "(physical p-bit count, then colour blocks, then |J|max; ties by "
                "declaration order). VIABLE_NOT_SELECTED means a runner-up that also "
                "fit, not a failure."
                if any(c.get("state") == "VIABLE_NOT_SELECTED" for c in cand_rows)
                else None
            ),
            "errors": errors,
            "mediator_count": med,
            "bipartite": bipartite,
            "n_nodes": n_nodes,
            "elapsed_seconds": round(elapsed, 4),
            "label": "JAX/THRML simulation, not Extropic silicon",
            "notes": [
                "Ideal-first compile (tsu_compiler.passes.search.compile_spec).",
                "Apply loads this receipt into the sampler only when verdict is COMPILED.",
                "Large / non-bipartite specs may be slow or fail placement, errors are shown, not invented.",
            ],
            "limits": {
                "heavy_specs": "8×8 mediated lattices can take tens of seconds; failures surface as errors.",
                "torx": "extro-torx optional; cross-check may be unavailable without it.",
            },
        }
    finally:
        path.unlink(missing_ok=True)


def apply_program(
    receipt_id: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Allow Apply only when a COMPILED receipt exists on disk."""
    if not receipt_id or "/" in receipt_id or ".." in receipt_id:
        raise ProgramServiceError(f"Invalid receipt_id {receipt_id!r}")
    base = receipts_root(root)
    path = base / receipt_id
    if not path.is_dir():
        raise ProgramServiceError(
            f"Apply refused, no receipt directory for {receipt_id!r}. "
            "Compile (ideal-first) must succeed first."
        )
    try:
        payload = load_receipt(receipt_id, root=base)
    except FileNotFoundError as exc:
        raise ProgramServiceError(
            f"Apply refused, receipt {receipt_id!r} incomplete: {exc}"
        ) from exc
    verdict = payload.get("verdict")
    if verdict != "COMPILED":
        raise ProgramServiceError(
            f"Apply refused, receipt verdict is {verdict!r}, not COMPILED. "
            "No silent apply without a successful ideal-first compile receipt."
        )
    if not payload.get("sampling", {}).get("thrml_ready"):
        raise ProgramServiceError(
            f"Apply refused, receipt {receipt_id!r} is not THRML-ready "
            f"({payload.get('sampling', {}).get('banner')})."
        )
    return {
        "ok": True,
        "kind": "apply",
        "receipt_id": receipt_id,
        "path": str(path),
        "verdict": verdict,
        "n_nodes": (payload.get("spins") or {}).get("n_nodes"),
        "beta": payload.get("beta"),
        "beta_fixed": payload.get("beta_fixed"),
        "label": "JAX/THRML simulation, not Extropic silicon",
        "notes": [
            "Load this receipt_id into the sampler the same way as curated 'small'.",
        ],
    }


def read_spec_yaml(receipt_id: str, *, root: Path | str | None = None) -> dict[str, Any]:
    """Return raw spec.yaml text for notepad seeding."""
    if not receipt_id or "/" in receipt_id or ".." in receipt_id:
        raise ProgramServiceError(f"Invalid receipt_id {receipt_id!r}")
    path = receipts_root(root) / receipt_id / "spec.yaml"
    if not path.is_file():
        return {
            "ok": False,
            "receipt_id": receipt_id,
            "spec_yaml": None,
            "message": f"No spec.yaml in receipt {receipt_id!r}",
            "default_seed": DEFAULT_SEED_SPEC,
        }
    return {
        "ok": True,
        "receipt_id": receipt_id,
        "spec_yaml": path.read_text(encoding="utf-8"),
        "message": None,
        "default_seed": DEFAULT_SEED_SPEC,
    }


def tsu_status() -> dict[str, Any]:
    # Do not gate this on `discover_tsu_root()` finding a directory. A frozen
    # bundle and a pip install both import cleanly with nothing to discover,
    # and gating here reported "importable: false" on a packaged application
    # whose compiler worked perfectly -- the UI said the product was dead
    # while it was alive.
    root: Path | None = None
    importable = False
    version = None
    err = None
    try:
        root = ensure_tsu_importable()
        import tsu_compiler

        importable = True
        version = getattr(tsu_compiler, "__version__", None)
        if version is None:
            try:
                import importlib.metadata as md

                version = md.version(DISTRIBUTION_NAME)
            except Exception:  # noqa: BLE001
                version = "0.1.0 (path)"
    except ProgramServiceError as exc:
        err = str(exc)
    return {
        "tsu_root": str(root) if root else None,
        "importable": importable,
        "version": version,
        "error": err,
        "notepad_receipt_id": NOTEPAD_RECEIPT_ID,
        "receipts_root": str(DEFAULT_RECEIPTS_ROOT),
    }


def dataclass_asdict_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: dataclass_asdict_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dataclass_asdict_safe(x) for x in obj]
    return obj
