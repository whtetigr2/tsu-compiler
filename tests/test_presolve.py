"""Presolve must never remove an assignment that is actually valid.

R29. The compiler searches the encoding and analyses coefficient precision, but
has no pass that eliminates a variable which cannot take a value. Every serious
mixed-integer or SAT solver runs presolve first, because removing a variable
costs nothing downstream and shrinks every pass after it.

The whole value of such a pass rests on one property: SOUNDNESS. A presolve that
quietly discards a reachable configuration turns every later result into a
confident answer to the wrong question, and it would look like a win the entire
time. So the first tests here enumerate small specs exhaustively and require that
every assignment the contract accepts survives.
"""
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsu_compiler.passes.presolve import presolve  # noqa: E402
from tsu_compiler.spec import load_spec  # noqa: E402

PROGRAMS = ROOT / "demo" / "gibbs-observatory" / "programs"


def _domain(var) -> tuple[int, ...]:
    k = getattr(getattr(var, "domain", None), "k", None)
    return tuple(range(k)) if k else (0, 1)


def _all_valid(spec) -> list[dict[str, int]]:
    """Every assignment the contract accepts. Brute force, no presolve."""
    names = [v.name for v in spec.variables]
    doms = [_domain(v) for v in spec.variables]
    out = []
    for combo in itertools.product(*doms):
        a = dict(zip(names, combo))
        if spec.contract.validate(a).ok:
            out.append(a)
    return out


def _survives(report, assignment: dict[str, int]) -> bool:
    """Is this assignment still expressible after presolve?"""
    for name, value in assignment.items():
        if value not in report.domains.get(name, (value,)):
            return False
    for name, rep_name in report.groups.items():
        if assignment[name] != assignment[rep_name]:
            return False
    return True


# --------------------------------------------------------------------------
# Soundness. The property everything else depends on.
# --------------------------------------------------------------------------

def test_presolve_keeps_every_valid_assignment_of_the_ecology_program():
    spec = load_spec(str(PROGRAMS / "ecology_lotka_lite.yaml"))
    report = presolve(spec)
    valid = _all_valid(spec)
    assert valid, "fixture problem: this spec has no valid assignment at all"
    lost = [a for a in valid if not _survives(report, a)]
    assert not lost, (
        f"presolve discarded {len(lost)} of {len(valid)} assignments the "
        f"contract accepts. First: {lost[0]}")


@pytest.mark.parametrize("stem", ["ecology_lotka_lite", "jobshop_tiny",
                                  "knapsack_tiny", "maxcut_cycle7"])
def test_presolve_is_sound_on_every_small_shipped_program(stem):
    spec = load_spec(str(PROGRAMS / f"{stem}.yaml"))
    names = [v.name for v in spec.variables]
    doms = [_domain(v) for v in spec.variables]
    if len(list(itertools.product(*doms))) > 200_000:
        pytest.skip(f"{stem} is too large to enumerate here")
    report = presolve(spec)
    lost = [a for a in _all_valid(spec) if not _survives(report, a)]
    assert not lost, f"{stem}: presolve lost {len(lost)} valid assignments"


# --------------------------------------------------------------------------
# It has to actually do something, or it is a no-op wearing a pass's clothes.
# --------------------------------------------------------------------------

def test_the_ecology_contract_collapses_to_two_states():
    """The motivating case. Forbidding the two species from sharing an edge
    forces every cell to match on a connected grid, so 16 variables become 1."""
    spec = load_spec(str(PROGRAMS / "ecology_lotka_lite.yaml"))
    report = presolve(spec)
    assert report.n_merged == 15, (
        f"expected 15 of 16 cells absorbed into one representative, got "
        f"{report.n_merged}")
    assert report.upper_bound_states == 2, (
        f"expected an upper bound of 2 reachable states, got "
        f"{report.upper_bound_states}; the receipt's own verification pass "
        f"measures diversity_reachable = 2")


def test_a_program_with_no_declared_constraints_is_left_alone():
    """A control. Presolve must not invent reductions it cannot justify."""
    spec = load_spec(str(PROGRAMS / "maxcut_cycle7.yaml"))
    report = presolve(spec)
    assert report.n_fixed == 0
    assert report.n_merged == 0
    assert not report.infeasible


def test_an_impossible_contract_is_reported_as_infeasible():
    """A control that can fail: presolve must be able to say no."""
    from tsu_compiler.passes.presolve import presolve_domains

    # x and y binary, forced equal AND forced different: nothing satisfies it.
    report = presolve_domains(
        {"x": (0, 1), "y": (0, 1)},
        equal_pairs=[("x", "y")],
        forbidden={("x", "y"): {(0, 0), (1, 1)}})
    assert report.infeasible, "a contradictory contract must be reported"
    assert report.reason and "infeasible" in report.reason.lower()


def test_the_report_names_what_it_removed():
    """A reduction nobody can inspect is not usable evidence."""
    spec = load_spec(str(PROGRAMS / "ecology_lotka_lite.yaml"))
    report = presolve(spec)
    assert report.removals, "presolve reduced the model and logged nothing"
    assert any("equal" in r.lower() or "merge" in r.lower()
               for r in report.removals), report.removals


# --------------------------------------------------------------------------
# Cross-check against a measurement presolve never sees.
# --------------------------------------------------------------------------

RECEIPTS = ROOT / "demo" / "gibbs-observatory" / "receipts"


def _receipt_reachable(stem: str):
    import json

    for name in (f"prog_{stem}", stem):
        f = RECEIPTS / name / "verification.json"
        if f.is_file():
            v = json.loads(f.read_text(encoding="utf-8")).get("diversity_reachable")
            return v if isinstance(v, int) else None
    return None


@pytest.mark.parametrize("stem", [
    "codon_opt_tiny", "ecology_lotka_lite", "floorplan_4zone", "jobshop_tiny",
    "knapsack_tiny", "market_binary_factors", "maxcut_cycle7", "mimo_detect_4",
    "number_partition", "roster_shift_conflicts", "sat_3tiny",
    "ilt_mask_adjacency_2x2",
])
def test_the_bound_never_falls_below_what_enumeration_measured(stem):
    """The strongest soundness evidence available, and it is independent.

    Each receipt's verification pass enumerated the reachable space directly,
    knowing nothing about presolve. Presolve reports an UPPER BOUND from local
    propagation. If the bound ever came out below the measured count, presolve
    would have excluded a configuration that demonstrably exists.

    The bound is allowed to be loose. Propagation is local and cannot see every
    interaction, so jobshop_tiny bounds at 64 against a measured 22. Loose is
    fine. Below is a defect.
    """
    measured = _receipt_reachable(stem)
    if measured is None:
        pytest.skip(f"{stem} has no enumerated reachable count on disk")
    spec = load_spec(str(PROGRAMS / f"{stem}.yaml"))
    report = presolve(spec)
    if report.upper_bound_states is None:
        pytest.skip(f"{stem} bound not reported")
    assert report.upper_bound_states >= measured, (
        f"{stem}: presolve bounds the space at "
        f"{report.upper_bound_states} but enumeration measured {measured} "
        f"reachable configurations, so presolve excluded something real")
