"""The search stage, and the mandatory `ideal` control.

Every compilation runs `ideal` FIRST. A z1 failure reported without a passing ideal
control is a defect in this compiler, not a finding about the substrate
(spec section 5.1). In the vertical slice exactly one candidate is generated, so the
state machine is exercised end to end while no search occurs.
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

SLICE_ENCODINGS = ("domain_wall",)


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
    regime = analyse_regime(report, target)
    return (Candidate(encoding, CandidateState.HARDWARE_FEASIBLE, report=report,
                      regime=regime),
            {"encoded": enc, "ising": ising, "report": report,
             "gate_checks": checks, "program": prog,
             "regime": regime, "placement": placement})


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

    chosen = feasible[0]
    art = arts[chosen.encoding][1]
    selected = Candidate(chosen.encoding, CandidateState.SELECTED,
                         report=chosen.report, regime=chosen.regime)
    final = tuple(selected if c is chosen else c for c in cands)

    verification = _verify(spec, art)
    return Compilation(
        spec=spec, target=target, verdict="COMPILED", ideal_passed=True,
        ideal_report=ideal_cand, hardware_evaluated=True,
        repset=RepresentationSet(final, selected,
                                 "single candidate in the vertical slice"),
        allow_assumed=allow_assumed, gate_checks=art["gate_checks"],
        program=art["program"], encoded=art["encoded"], regime=art["regime"],
        placement=art["placement"], verification=verification)


def _verify(spec, art) -> Verification:
    from ..backends.thrml_backend import EXACT_LIMIT, exact_distribution, sample
    from ..backends.torx_backend import torx_cross_check

    enc, prog, ising = art["encoded"], art["program"], art["ising"]
    n = len(ising.nodes)

    if n > EXACT_LIMIT:
        return Verification(None, f"unavailable: 2^{n} too large to enumerate",
                            None, None, "unavailable: no exact reference", None,
                            None, "unavailable: model too large")

    states, probs = exact_distribution(prog)

    # energy layer: the lowered model must reproduce the encoded model's energies
    worst = 0.0
    for row, p in zip(states, probs):
        bits = dict(zip(ising.nodes, row.tolist()))
        worst = max(worst, abs(enc.model.energy(bits) - _from_ising(ising, row)))
    energy_tv = worst

    # task layer: measured on DECODED samples, never inferred from energy
    got = sample(prog, 32, 200, 400, 2, 0)
    ok = 0
    for row in got:
        bits = dict(zip(ising.nodes, row.tolist()))
        ok += 1 if spec.contract.validate(enc.decode(bits)).ok else 0
    task_validity = ok / len(got)

    # execution layer: sampler vs its own model. The index into `probs` MUST be
    # derived from `states` itself, never from an independently-assumed bit
    # order -- `exact_distribution`'s enumeration order is an implementation
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
        cross_check_tv=cross, cross_check_note=cross_note)


def _from_ising(ising, row) -> float:
    s = 2 * row.astype(float) - 1
    total = ising.offset
    for i in range(len(ising.nodes)):
        total += -ising.biases[i] * s[i]
    for k, (u, v) in enumerate(ising.edges):
        total += -ising.weights[k] * s[u] * s[v]
    return float(total)
