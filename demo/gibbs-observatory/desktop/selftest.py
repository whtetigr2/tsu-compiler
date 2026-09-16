"""Prove the packaged application carries a working compiler and sampler.

Run with `--selftest`, this exits zero only if the compiler imports AND returns
a real verdict on a real shipped program, THRML imports AND draws real samples,
and the frontend build is present. It opens no window, so a script can run it on
a machine that has neither Python nor Node installed.

The compiler check deliberately uses one of Extropic's own published workloads
shipped in `programs/` rather than a model built here in memory. That way a
single check proves three things at once: the compiler is present, it runs, and
the bundled data files actually made it into the bundle.

Each check returns (name, ok, detail). A check that cannot fail proves nothing,
so `_check_frontend` is monkeypatchable and a test forces it to fail.
"""
from __future__ import annotations

from .paths import frontend_dist, resource_root

# An Extropic workload, not one of ours. If someone else's published program
# stops compiling, that is the strongest available signal of a regression.
ORACLE_PROGRAM = "ecology_lotka_lite.yaml"


def _check_compiler() -> tuple[str, bool, str]:
    try:
        from tsu_compiler.preflight.check import preflight
        from tsu_compiler.preflight.model import load_model
    except Exception as exc:  # pragma: no cover - import failure path
        return ("compiler", False, f"unavailable: import failed: {exc!r}")

    spec = resource_root() / "programs" / ORACLE_PROGRAM
    if not spec.is_file():
        return ("compiler", False,
                f"unavailable: {ORACLE_PROGRAM} not found at {spec}. The "
                f"bundle is missing its programs/ data files.")
    try:
        rep = preflight(load_model(spec=str(spec)))
    except Exception as exc:
        return ("compiler", False, f"unavailable: preflight raised: {exc!r}")
    return ("compiler", rep.verdict == "ok" and rep.placed,
            f"preflight verdict {rep.verdict}, placed={rep.placed}, "
            f"{rep.n_spins} spins on {ORACLE_PROGRAM}")


def _check_thrml() -> tuple[str, bool, str]:
    try:
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram
    except Exception as exc:  # pragma: no cover - import failure path
        return ("thrml", False, f"unavailable: import failed: {exc!r}")

    try:
        left = [SpinNode() for _ in range(4)]
        right = [SpinNode() for _ in range(4)]
        nodes = left + right
        edges = [(u, v) for u in left for v in right]
        model = IsingEBM(nodes, edges,
                         jnp.zeros(len(nodes)),
                         jnp.full(len(edges), 0.4),
                         jnp.array(1.0))
        program = IsingSamplingProgram(model, [Block(left), Block(right)], [])
        schedule = SamplingSchedule(n_warmup=5, n_samples=2, steps_per_sample=1)
        init = [jnp.zeros(len(left), dtype=bool),
                jnp.zeros(len(right), dtype=bool)]
        out = sample_states(jax.random.key(0), program, schedule, init, [],
                            [Block(nodes)])
    except Exception as exc:
        return ("thrml", False, f"unavailable: sampling raised: {exc!r}")

    shape = tuple(out[0].shape)
    return ("thrml", shape == (2, len(nodes)),
            f"sampled an 8-spin bipartite model, drew {shape}")


def _check_service() -> tuple[str, bool, str]:
    """Compile through the SERVICE the application uses, not a direct import.

    This check exists because of a real failure. `_check_compiler` imports the
    compiler itself and passed inside a frozen bundle whose application could
    not compile anything: the service discovers the compiler by hunting for a
    directory on disk, and a bundle has none. The binary self-tested clean,
    opened, served every asset, and reported the compiler unavailable.

    A check that does not travel the path the product travels can pass while
    the product is broken.
    """
    try:
        from backend.app.program_service import preflight_program, tsu_status
    except Exception as exc:  # pragma: no cover - import failure path
        return ("service", False, f"unavailable: import failed: {exc!r}")

    status = tsu_status()
    if not status.get("importable"):
        return ("service", False,
                f"unavailable: the service reports the compiler as not "
                f"importable: {status.get('error') or status}")

    spec = resource_root() / "programs" / ORACLE_PROGRAM
    try:
        res = preflight_program(spec.read_text(encoding="utf-8"))
    except Exception as exc:
        return ("service", False, f"unavailable: preflight_program raised: "
                                  f"{exc!r}")
    return ("service", res.get("verdict") == "ok",
            f"compiled {ORACLE_PROGRAM} through the API service, verdict "
            f"{res.get('verdict')}")


def _check_frontend() -> tuple[str, bool, str]:
    try:
        dist = frontend_dist()
    except Exception as exc:
        return ("frontend", False, str(exc))
    return ("frontend", True, f"built bundle at {dist}")


def run_selftest() -> tuple[int, list[str]]:
    """Run every check. Returns (exit_code, report_lines)."""
    checks = (_check_compiler(), _check_service(), _check_thrml(),
              _check_frontend())
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
