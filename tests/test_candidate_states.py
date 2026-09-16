"""A search that ran out of time is not a hardware limit.

audit/findings/R27.md. `programs/seq_design_longer.yaml` produces a domain-wall
candidate that fails to place at the default budget and was recorded
HARDWARE_INFEASIBLE. Its own attached failure says otherwise:

    failure_class = 'placement_effort_exhausted'
    remediations  = ('increase placement effort', ...)

Given 24 restarts and 250,000 iterations the same candidate places and is then
SELECTED, at 24 physical spins against one-hot's 32. The hardware was never the
limit, and "HARDWARE_INFEASIBLE" is a claim about Extropic's silicon made on the
basis of a timer.

`states.py` already carries this exact argument for COMPILER_ERROR: "
HARDWARE_INFEASIBLE asserts 'your model does not fit the chip'; reporting our
own unexpected exception as that would tell a user to change a model that was
fine." A search budget is the same kind of thing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsu_compiler.states import CandidateState  # noqa: E402

SPEC = ROOT / "demo" / "gibbs-observatory" / "programs" / "seq_design_longer.yaml"


def test_there_is_a_state_meaning_not_found_within_budget():
    assert hasattr(CandidateState, "PLACEMENT_EFFORT_EXHAUSTED"), (
        "a candidate whose placement search ran out of budget has no state to "
        "be in, so it lands in HARDWARE_INFEASIBLE and reads as a claim about "
        "the silicon")


def test_effort_exhaustion_classifies_away_from_hardware_infeasible():
    from tsu_compiler.passes.search import _classify_placement_failure

    class _EffortFailure:
        failure_class = "placement_effort_exhausted"

    assert _classify_placement_failure(_EffortFailure()) is (
        CandidateState.PLACEMENT_EFFORT_EXHAUSTED)


def test_a_genuine_hardware_failure_is_still_hardware_infeasible():
    """A control that cannot fail is not a control."""
    from tsu_compiler.passes.search import _classify_placement_failure

    class _RealFailure:
        failure_class = "degree_exceeded"

    assert _classify_placement_failure(_RealFailure()) is (
        CandidateState.HARDWARE_INFEASIBLE)


def test_an_unclassifiable_failure_stays_hardware_infeasible():
    """Unknown means unchanged, not optimistically reclassified."""
    from tsu_compiler.passes.search import _classify_placement_failure

    assert _classify_placement_failure(None) is CandidateState.HARDWARE_INFEASIBLE


@pytest.mark.slow
def test_the_real_workload_is_no_longer_called_hardware_infeasible():
    """End to end, on the spec that exposed this."""
    from tsu_compiler.passes.search import compile_spec
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    comp = compile_spec(load_spec(str(SPEC)), PROFILES["z1"], allow_assumed=True)
    dw = {c.encoding: c for c in comp.repset.candidates}["domain_wall"]
    if dw.state is CandidateState.SELECTED:
        pytest.skip("domain_wall now places at the default budget")
    assert dw.state is not CandidateState.HARDWARE_INFEASIBLE, (
        f"domain_wall is recorded {dw.state.name} while its failure is "
        f"{getattr(dw.failure, 'failure_class', None)!r}. It places at 24 "
        f"restarts / 250k iterations, so the hardware is not the limit.")
    assert dw.state is CandidateState.PLACEMENT_EFFORT_EXHAUSTED


@pytest.mark.slow
def test_effort_exhausted_candidates_are_still_not_selected():
    """Relabelling must not make an unplaceable candidate look viable."""
    from tsu_compiler.passes.search import compile_spec
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    comp = compile_spec(load_spec(str(SPEC)), PROFILES["z1"], allow_assumed=True)
    selected = [c for c in comp.repset.candidates
                if c.state is CandidateState.SELECTED]
    assert len(selected) == 1, f"expected exactly one selection, got {selected}"
    assert selected[0].state is not CandidateState.PLACEMENT_EFFORT_EXHAUSTED
    assert comp.verdict == "COMPILED"


def test_all_candidates_out_of_budget_is_not_a_hardware_verdict(monkeypatch):
    """If nothing placed in time, the hardware question was never settled.

    search.py already makes this argument for COMPILER_ERROR, in a comment
    directly above the branch: "If every candidate died of a fault in THIS
    compiler, the hardware question was never actually evaluated, so 'HARDWARE'
    (which asserts the model does not fit) would be a claim we did not
    establish."

    An exhausted annealing budget is the same situation. The model may fit; no
    embedding was found within the time allowed, and saying "no candidate was
    hardware-feasible" claims more than was measured.
    """
    from tsu_compiler.passes import search as S
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    class _EffortFailure:
        failure_class = "placement_effort_exhausted"
        mediation = None

    real_try = S._try

    def fake_try(spec, target, encoding, *a, **k):
        # The ideal control runs through _try as well. Leave it alone, or the
        # compile aborts before the hardware branch under test is reached.
        if target is PROFILES["ideal"]:
            return real_try(spec, target, encoding, *a, **k)
        cand = S.Candidate(encoding, CandidateState.PLACEMENT_EFFORT_EXHAUSTED,
                           reason="placement effort exhausted",
                           failure=_EffortFailure(), report=None)
        return cand, {"report": None, "gate_checks": (), "durations": {}}

    monkeypatch.setattr(S, "_try", fake_try)
    comp = S.compile_spec(load_spec(str(SPEC)), PROFILES["z1"],
                          allow_assumed=True)

    assert comp.verdict != "HARDWARE", (
        "every candidate merely ran out of placement budget, so no hardware "
        "limit was established, but the verdict claims one")
    note = (comp.repset.ordering_rationale or "").lower()
    assert "effort" in note or "budget" in note, (
        f"the note should say the search ran out, not that the hardware "
        f"refused: {comp.repset.ordering_rationale!r}")
    assert comp.hardware_evaluated is False, (
        "the hardware question was not reached, so it was not evaluated")
