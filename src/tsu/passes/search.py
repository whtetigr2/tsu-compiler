"""The search stage, and the mandatory `ideal` control.

Every compilation runs `ideal` FIRST. A z1 failure reported without a passing ideal
control is a defect in this compiler, not a finding about the substrate
(spec section 5.1). The ideal control itself still probes with a single, fixed
encoding (SLICE_ENCODINGS[0]) -- it exists to prove the SPEC is logically sound
independent of hardware, not to search; that part of the pipeline is unchanged.

A5: SLICE_ENCODINGS now holds every encoding this compiler can produce, so a real
search happens once the ideal control passes -- each is tried against the actual
target and EVERY outcome is retained (rejected candidates included, as evidence).
Selection among the feasible candidates orders by physical p-bit count, then
colour blocks, then |J|max, ties broken by declaration order; `compare()` renders
the full measured table so that ordering is auditable, not just asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..failures import CompileError
from ..gates import check_gates, gate_checks
from ..regime import analyse_regime
from ..states import Candidate, CandidateState, RepresentationSet
from ..target import IDEAL, TargetProfile
from .analyse import analyse
from .encode import encode
from .lower import lower
from .place import place
from .program import build_program
from .route import route
from .verify import Verification, tv_noise_floor

SLICE_ENCODINGS = ("domain_wall", "one_hot")


@dataclass
class Compilation:
    spec: Any
    target: TargetProfile
    verdict: str                     # COMPILED | LOGICAL | HARDWARE
    ideal_passed: bool
    ideal_report: Any
    hardware_evaluated: bool
    repset: RepresentationSet
    allow_assumed: bool = False
    gate_checks: tuple = ()          # every gate evaluated against `target`,
                                      # pass or fail (empty when target was never
                                      # evaluated, i.e. verdict == LOGICAL)
    program: Any = None
    encoded: Any = None
    regime: Any = None
    placement: Any = None
    verification: Verification | None = None


def _try(spec, target, encoding, allow_assumed):
    """Run one candidate through the pipeline. Returns (Candidate, artefacts).

    `artefacts["gate_checks"]` is populated whenever the model reached gate
    evaluation at all (i.e. encode+lower succeeded), REGARDLESS of whether the
    candidate passed -- a receipt needs "every gate, passed/failed, with the
    measured value and threshold" (spec section 10), not only the gates a
    COMPILED run happened to pass.
    """
    try:
        enc = encode(spec, encoding)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"encode failed: {e}"), None

    try:
        ising = lower(enc.model)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"lower failed: {e}"), None

    report = analyse(ising)
    checks = gate_checks(ising, report, target, allow_assumed)
    gate_failures = check_gates(ising, report, target, allow_assumed)
    if gate_failures:
        return Candidate(encoding, CandidateState.HARDWARE_INFEASIBLE,
                         reason=gate_failures[0].cause, failure=gate_failures[0],
                         report=report), {"report": report, "gate_checks": checks}
    try:
        placement = place(ising, report, target)
        ising = route(ising, report, target)
    except CompileError as e:
        return Candidate(encoding, CandidateState.HARDWARE_INFEASIBLE,
                         reason=str(e), failure=e.failures[0], report=report), \
            {"report": report, "gate_checks": checks}

    prog = build_program(ising, report)
    regime = analyse_regime(report, target, weights=ising.weights)
    return (Candidate(encoding, CandidateState.HARDWARE_FEASIBLE, report=report,
                      regime=regime, placement=placement),
            {"encoded": enc, "ising": ising, "report": report,
             "gate_checks": checks, "program": prog,
             "regime": regime, "placement": placement})


ORDERING_RATIONALE = (
    "selection order: physical p-bit count, then colour blocks, then |J|max; "
    "ties broken by declaration order (" + ", ".join(SLICE_ENCODINGS) + ")")


def _physical_pbits(report) -> int:
    """The count of spins actually deployed to the substrate. In this vertical
    slice `route` never inserts a mediator spin (it is the identity for a graph
    that already satisfies the target's parity requirement, and raises rather
    than silently mediating one that does not -- see route.py), so physical
    p-bit count equals the logical spin count (`report.n_nodes`) for every
    candidate that reaches HARDWARE_FEASIBLE here. Kept as its own named
    quantity (not just an alias read as `n_nodes`) because that equality is a
    property of this slice's `route`, not a general truth the rest of the
    compiler should assume."""
    return report.n_nodes


def _selection_key(c: "Candidate"):
    r = c.report
    return (_physical_pbits(r), r.colour_blocks, r.max_abs_J)


def compare(repset: "RepresentationSet") -> tuple[dict, ...]:
    """The measured table over EVERY candidate in a RepresentationSet -- winner,
    runner-up, and rejected alike -- so a selection can be audited against the
    numbers that produced it, not just the ordering_rationale prose. Any field
    a candidate never reached (e.g. `analyse` never ran because `encode` itself
    failed) reads None here; callers must not treat that as zero."""
    rows = []
    for c in repset.candidates:
        r, p = c.report, c.placement
        rows.append({
            "encoding": c.encoding,
            "state": c.state.value,
            "reason": c.reason or "",
            "logical_spins": r.n_nodes if r else None,
            "logical_edges": r.n_edges if r else None,
            "bipartite": r.bipartite if r else None,
            "mediators": (r.mediators if r and r.mediators >= 0 else None),
            "physical_pbits": (_physical_pbits(r) if r and p else None),
            "colour_blocks": r.colour_blocks if r else None,
            "max_abs_J": r.max_abs_J if r else None,
            "max_abs_b": r.max_abs_b if r else None,
        })
    return tuple(rows)


def compile_spec(spec, target: TargetProfile, allow_assumed: bool = False) -> Compilation:
    # --- the mandatory ideal control, first and always -----------------------
    ideal_cand, ideal_art = _try(spec, IDEAL, SLICE_ENCODINGS[0], True)
    ideal_ok = ideal_cand.state == CandidateState.HARDWARE_FEASIBLE
    if not ideal_ok:
        return Compilation(
            spec=spec, target=target, verdict="LOGICAL", ideal_passed=False,
            ideal_report=ideal_cand, hardware_evaluated=False,
            repset=RepresentationSet((ideal_cand,), None,
                                     "ideal control failed; target not evaluated"),
            allow_assumed=allow_assumed)

    cands, arts = [], {}
    for enc_name in SLICE_ENCODINGS:
        c, a = _try(spec, target, enc_name, allow_assumed)
        cands.append(c)
        if a:
            arts[enc_name] = (c, a)

    feasible = [c for c in cands if c.state == CandidateState.HARDWARE_FEASIBLE]
    if not feasible:
        any_art = next(iter(arts.values()), None)
        checks = any_art[1]["gate_checks"] if any_art else ()
        return Compilation(
            spec=spec, target=target, verdict="HARDWARE", ideal_passed=True,
            ideal_report=ideal_cand, hardware_evaluated=True,
            repset=RepresentationSet(tuple(cands), None,
                                     "no candidate was hardware-feasible"),
            allow_assumed=allow_assumed, gate_checks=checks)

    chosen = min(feasible, key=_selection_key)
    art = arts[chosen.encoding][1]
    selected = Candidate(chosen.encoding, CandidateState.SELECTED,
                         report=chosen.report, regime=chosen.regime,
                         placement=chosen.placement)

    final = []
    for c in cands:
        if c is chosen:
            final.append(selected)
        elif c.state == CandidateState.HARDWARE_FEASIBLE:
            # feasible, but out-ranked by `chosen` under the selection order --
            # retained as evidence, not discarded (spec section 9: "search"
            # means every candidate stays visible, winner or not).
            final.append(Candidate(
                c.encoding, CandidateState.VIABLE_NOT_SELECTED,
                reason=(f"not selected: {chosen.encoding} ranked ahead of "
                       f"{c.encoding} under {ORDERING_RATIONALE}"),
                report=c.report, regime=c.regime, placement=c.placement))
        else:
            final.append(c)
    final = tuple(final)

    verification = _verify(spec, art)
    return Compilation(
        spec=spec, target=target, verdict="COMPILED", ideal_passed=True,
        ideal_report=ideal_cand, hardware_evaluated=True,
        repset=RepresentationSet(final, selected, ORDERING_RATIONALE),
        allow_assumed=allow_assumed, gate_checks=art["gate_checks"],
        program=art["program"], encoded=art["encoded"], regime=art["regime"],
        placement=art["placement"], verification=verification)


def _verify(spec, art) -> Verification:
    from ..backends.thrml_backend import EXACT_LIMIT, exact_distribution, sample
    from ..backends.torx_backend import torx_cross_check

    enc, prog, ising = art["encoded"], art["program"], art["ising"]
    n = len(ising.nodes)

    # Task layer FIRST, unconditionally: it is measured on DECODED SAMPLES
    # against the task contract and needs no exact reference at all (spec.py's
    # own docstring: "Workload correctness is measured on DECODED samples and
    # is never inferred from energy"). This must NOT be gated behind the same
    # `n > EXACT_LIMIT` early return the genuinely exact-reference-dependent
    # layers below need -- it previously was, which meant no spec large enough
    # to need this compiler's own exact-enumeration escape hatch could ever
    # get a real task_validity number, contradicting the very reason that
    # escape hatch exists. Found via the WFC stress test's 4x4 grid instance
    # (adjacency_4x4_k4.yaml), which exists specifically to exercise this path.
    #
    # `decode` is a PROJECTION, not an inverse (C5): handed a non-monotone
    # (invalid) domain-wall chain it still returns a legal-looking value with no
    # flag. A sample that is not a valid codeword is not a decoded sample at
    # all -- it must count as invalid, not be silently decoded and validated as
    # if the chain meant something. The codeword-violation rate is reported
    # separately so it is visible, not folded into (and hidden inside) task_validity.
    got = sample(prog, 32, 200, 400, 2, 0)
    ok = 0
    codeword_violations = 0
    for row in got:
        bits = dict(zip(ising.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            codeword_violations += 1
            continue
        ok += 1 if spec.contract.validate(enc.decode(bits)).ok else 0
    task_validity = ok / len(got)
    codeword_violation_rate = codeword_violations / len(got)

    if n > EXACT_LIMIT:
        return Verification(
            energy_tv=None, energy_note=f"unavailable: 2^{n} too large to enumerate",
            task_validity=task_validity, task_validity_note="",
            execution_tv=None, execution_note="unavailable: no exact reference",
            execution_noise_floor=None,
            cross_check_tv=None, cross_check_note="unavailable: model too large",
            codeword_violation_rate=codeword_violation_rate,
            codeword_violation_note="")

    states, probs = exact_distribution(prog)

    # energy layer: the lowered model must reproduce the encoded model's energies
    worst = 0.0
    for row, p in zip(states, probs):
        bits = dict(zip(ising.nodes, row.tolist()))
        worst = max(worst, abs(enc.model.energy(bits) - _from_ising(ising, row)))
    energy_tv = worst

    # execution layer: sampler vs its own model, reusing the SAME samples `got`
    # already drawn above for the task layer (one sampling run serves both).
    # The index into `probs` MUST be derived from `states` itself, never from
    # an independently-assumed bit order -- `exact_distribution`'s enumeration
    # order is an implementation
    # detail of itertools.product, and a hand-rolled (1 << arange(n)) index
    # silently assumes a specific one. A previous version assumed LSB-first
    # and got MSB-first, which inflated execution_tv from ~0.01 to ~0.52 with
    # no error raised anywhere -- a broken comparison that looked exactly like
    # a broken sampler. Building the lookup from `states` means the two can
    # never drift apart again, regardless of how either side is implemented.
    state_index = {tuple(int(x) for x in row): i for i, row in enumerate(states)}
    idx = np.array([state_index[tuple(int(x) for x in row)] for row in got])
    hist = np.bincount(idx, minlength=len(probs)).astype(float)
    hist /= hist.sum()
    execution_tv = float(0.5 * np.abs(hist - probs).sum())
    floor = tv_noise_floor(probs, len(got))

    tx, tx_note = torx_cross_check(ising)
    if tx is None:
        cross, cross_note = None, f"unavailable: {tx_note}"
    else:
        cross, cross_note = float(0.5 * np.abs(tx - probs).sum()), ""

    return Verification(
        energy_tv=energy_tv, energy_note="",
        task_validity=task_validity,
        execution_tv=execution_tv,
        execution_note=f"noise floor {floor:.6f}",
        execution_noise_floor=floor,
        cross_check_tv=cross, cross_check_note=cross_note,
        codeword_violation_rate=codeword_violation_rate, codeword_violation_note="")


def _from_ising(ising, row) -> float:
    s = 2 * row.astype(float) - 1
    total = ising.offset
    for i in range(len(ising.nodes)):
        total += -ising.biases[i] * s[i]
    for k, (u, v) in enumerate(ising.edges):
        total += -ising.weights[k] * s[u] * s[v]
    return float(total)
