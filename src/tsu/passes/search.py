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

import dataclasses
import functools
import itertools
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ess import EssEstimate, effective_sample_size
from ..failures import CompileError
from ..gates import check_gates, gate_checks
from ..regime import analyse_regime, beta_recommendation_from_energy_scale
from ..states import Candidate, CandidateState, RepresentationSet
from ..target import IDEAL, TargetProfile
from .analyse import analyse
from .encode import encode, spec_beta, validate_coefficient_scale
from .lower import lower
from .place import place
from .program import build_program
from .route import insert_mediators, route
from .verify import DecodedSample, Verification, tv_noise_floor

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
    clamp: Any = None                # C1: the WORKLOAD-level clamp this compile
                                       # was run under, {} when none
    sample: DecodedSample | None = None   # G2: one concrete decoded sample from
                                            # this compile's own verification run,
                                            # None when verification never ran
    # C4: compilation cost. `pass_durations` is per-pass wall time (seconds)
    # for whichever candidate's pipeline this compile's own verdict is
    # sourced from -- the selected candidate's on COMPILED, else the best-
    # reached candidate's on LOGICAL/HARDWARE, so a rejected compile still
    # shows the cost of the passes it actually ran. `peak_memory_bytes` is
    # measured (via `tracemalloc`) around the WHOLE compile_spec call.
    pass_durations: dict = field(default_factory=dict)
    peak_memory_bytes: int | None = None
    sample_cost: dict = field(default_factory=dict)   # C4: sampler wall time/
                                                         # throughput/params and
                                                         # the CPU baseline, from
                                                         # _verify
    # Task 2: the uniform coefficient_scale this compile was run under (1.0 ==
    # unscaled, the pre-existing behaviour) and the beta it was COMPENSATED
    # to (spec_beta(spec) / coefficient_scale) so p(x) ~ exp(-beta*E(x)) is
    # unchanged by the scale. Always set, on every verdict (LOGICAL/HARDWARE/
    # COMPILED) -- both are deterministic given `spec` and `coefficient_scale`
    # alone, never a measurement that could be legitimately unavailable.
    coefficient_scale: float = 1.0
    scaled_beta: float = 1.0


def _physical_clamp(enc, ising, clamp):
    """C1: translate a WORKLOAD-level clamp ({logical name: value}) into the
    PHYSICAL clamp `build_program` needs ({node index: 0/1}) -- via
    `Encoded.encode_clamp` (a clamped categorical becomes its clamped chain
    spins) and `ising.nodes`' own index. None (not {}) when there is nothing
    to clamp, so `build_program`'s default (unclamped) path is untouched."""
    if not clamp:
        return None
    spin_clamp = enc.encode_clamp(clamp)
    idx = {n: i for i, n in enumerate(ising.nodes)}
    return {idx[name]: value for name, value in spin_clamp.items()}


def _try(spec, target, encoding, allow_assumed, clamp=None, coefficient_scale=1.0,
         placement_effort=None):
    """Run one candidate through the pipeline. Returns (Candidate, artefacts).

    `artefacts["gate_checks"]` is populated whenever the model reached gate
    evaluation at all (i.e. encode+lower succeeded), REGARDLESS of whether the
    candidate passed -- a receipt needs "every gate, passed/failed, with the
    measured value and threshold" (spec section 10), not only the gates a
    COMPILED run happened to pass.

    `clamp` (C1) is a WORKLOAD-level {name: value} map, threaded through to
    `build_program` as a physical node clamp -- it changes nothing about
    encode/lower/analyse/gates/placement (clamping is a SAMPLING-time
    concept: which nodes get resampled, not what the model means), only which
    nodes `build_program` puts in a free block.

    C4: `artefacts["durations"]` records wall time (seconds) for each pass
    this candidate actually reached -- an encode/lower failure returns None
    for artefacts (as it always has) and so carries no durations; every
    candidate that reaches `analyse` records at least that much, however it
    is ultimately rejected.

    Task 2: `coefficient_scale` is threaded straight into `encode()` (which
    scales every workload term weight AND the encoding's own structural
    penalty by it -- see encode.py), then, once `lower()` has produced the
    IsingModel, THIS is where beta gets compensated (beta -> spec_beta(spec)
    / coefficient_scale) -- exactly once, before `ising` reaches anything
    else (gates, `analyse_regime`, `build_program`'s SamplingProgram). Every
    downstream consumer of `ising` from this point on therefore already sees
    the physically-correct, scale-compensated beta: gate checks, the regime
    report's precision headroom, the sampling program actually built, and
    (via the receipt's program.json) a later `tsu simulate` replaying it.
    """
    durations: dict[str, float] = {}

    def _timed(name, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        durations[name] = time.perf_counter() - t0
        return out

    try:
        enc = _timed("encode", encode, spec, encoding, coefficient_scale)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"encode failed: {e}"), None

    try:
        ising = _timed("lower", lower, enc.model)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"lower failed: {e}"), None

    # E -> s*E leaves p(x) ~ exp(-beta*E(x)) unchanged only if beta -> beta/s
    # compensates it (spec section 4.9). `encode`/`lower` never touch beta
    # themselves (by design -- see encode()'s own docstring); this is the
    # ONE place that compensation happens, so it can never be applied twice
    # or forgotten on some downstream path.
    ising = dataclasses.replace(ising, beta=spec_beta(spec) / coefficient_scale)

    report = _timed("analyse", analyse, ising)
    t0 = time.perf_counter()
    checks = gate_checks(ising, report, target, allow_assumed)
    gate_failures = check_gates(ising, report, target, allow_assumed)
    durations["gate_checks"] = time.perf_counter() - t0
    if gate_failures:
        return Candidate(encoding, CandidateState.HARDWARE_INFEASIBLE,
                         reason=gate_failures[0].cause, failure=gate_failures[0],
                         report=report), {"report": report, "gate_checks": checks,
                                          "durations": durations}
    # C-5 (external review, 2026-09-10): which of place()/route() is running,
    # so an unexpected (non-CompileError) exception caught below can name it
    # -- there is no other way to tell the two apart once caught, since
    # Exception itself carries no notion of which call raised it.
    current_pass = "place"
    try:
        # `placement_effort` tunes ONLY the annealer fallback's budget
        # (restarts, iters). It changes how hard the geometric SEARCH tries,
        # never what counts as a valid embedding: a placement is accepted
        # only when every edge is realized, at any effort. Measured on
        # specs/lattice_small_8x8_k3.yaml, domain_wall: the default (6,
        # 40_000) leaves 98 edges unrealized, while (12, 200_000) places it
        # with 0 unrealized -- the default budget was the binding limit, not
        # the substrate.
        _pe = placement_effort or {}
        _place = functools.partial(
            place, **{k: v for k, v in _pe.items()
                      if k in ("restarts", "iters")})
        placement = _timed("place", _place, ising, report, target)
        # Task 6 (spec 5.3.6): `place` mediates a non-bipartite graph itself
        # rather than raising `parity_conflict` outright -- when it did,
        # `placement.mediation` is set and `placement.{mediated_ising,
        # mediated_report}` are the LARGER, bipartite model that was
        # actually embedded (placement.coords/realized are already indexed
        # against it). Everything from here on -- `route` (now a no-op,
        # since `mediated_report.bipartite` is True), `build_program`,
        # `regime`, and this candidate's own `report` -- must use that
        # model, not the pre-mediation one, so a receipt's numbers describe
        # what is ACTUALLY deployed to the substrate. Gate checks above
        # already ran against the pre-mediation model (spec 5.3.6 only asks
        # `place` to mediate; it does not re-run the hardware gates).
        if placement.mediation is not None:
            ising, report = placement.mediated_ising, placement.mediated_report
            # A-1 (external review, 2026-09-09): mediation's own gadget
            # coupling A = arccosh(exp(2*beta*|J|))/(2*beta) is STRICTLY
            # GREATER than |J| for every J != 0 (route.py's own docstring/
            # derivation: A > |J| for all J != 0), so a model that
            # legitimately cleared `coupling_cap`/`field_cap` PRE-mediation
            # (the only gate evaluation that has run so far, above) can
            # still carry a coupling the hardware cannot represent once
            # mediated -- solving A(|J|) <= 6.0 at beta=1 gives |J| <=
            # ln(cosh(12))/2 = 5.653426..., so any mediated edge with
            # 5.653426 < |J| <= 6.0 passes the pre-mediation cap and then
            # needs an unprogrammable coupling. Spec 5.3.6 only asks
            # `place` to mediate; it never asks anything to re-check the
            # hardware gates against the result, and nothing downstream
            # did either. Re-run the SAME evaluation against the model
            # that is now ACTUALLY going to be deployed (`ising`/`report`,
            # just swapped above) -- replacing `checks` so a receipt's own
            # gate_checks describe THIS model, not the stale pre-mediation
            # one, exactly the same rationale the comment above already
            # applies to `report`/`ising` themselves -- and raise through
            # the EXACT SAME `CompileError` path the placement-failure
            # `except` clause immediately below already handles, rather
            # than inventing a second, parallel failure-reporting shape.
            # `GateFailure` carries no `mediation` attribute, so that
            # except clause's own `getattr(failure, "mediation", None)`
            # correctly no-ops for this failure kind (it already IS the
            # mediated model; nothing needs recomputing).
            t_regate = time.perf_counter()
            checks = gate_checks(ising, report, target, allow_assumed)
            mediated_failures = check_gates(ising, report, target, allow_assumed)
            durations["post_mediation_gate_checks"] = time.perf_counter() - t_regate
            if mediated_failures:
                raise CompileError(
                    f"post-mediation gate failure: {mediated_failures[0].cause}",
                    mediated_failures)
        current_pass = "route"
        ising = _timed("route", route, ising, report, target)
    except CompileError as e:
        failure = e.failures[0]
        # C2 fix: the SUCCESS path above swaps to the post-mediation
        # report the moment `place()` had to mediate; this except branch is
        # the placement-FAILURE path (e.g. a mediated graph that then hits
        # `placement_effort_exhausted`) and used to leave `report` at its
        # PRE-mediation value here, because `place()` raises before ever
        # returning the `Placement` that carries `mediated_ising`/
        # `mediated_report` -- this is exactly what produced `bipartite:
        # False` in the committed L0/L1 receipts for a graph the mediation
        # pass itself had already proven bipartite (review finding C2).
        # `failure.mediation` (threaded through by `_embed_on_lattice`, see
        # place.py) is the one signal available here that `place()` DID
        # mediate before its separate geometric search then ran out of
        # budget; `insert_mediators` is a pure, deterministic function of
        # the pre-mediation ising's graph structure (route.py's own
        # docstring), so recomputing it from the SAME `ising`/`report` this
        # call already holds reproduces exactly the model `place` embedded
        # against -- not a second, independently-derived model that could
        # drift from it. A failure with no `mediation` (degree_exceeded,
        # budget_exceeded on the pre-mediation graph, or a target that
        # never needed mediation) leaves `ising`/`report` untouched, exactly
        # as before.
        med = getattr(failure, "mediation", None)
        if med is not None:
            ising, _ = insert_mediators(ising, report)
            report = analyse(ising)
        return Candidate(encoding, CandidateState.HARDWARE_INFEASIBLE,
                         reason=str(e), failure=failure, report=report), \
            {"report": report, "gate_checks": checks, "durations": durations}
    except Exception as e:
        # C-5 (external review, 2026-09-10): an exception that is NOT a
        # structured `CompileError` -- a genuine bug, not a documented
        # hardware-infeasibility cause `place`/`route` chose to raise -- used
        # to propagate straight out of `_try`/`compile_spec` as a raw
        # traceback, the one place left in this pipeline that broke the
        # clean-refusal standard the preflight/regime CLI paths hold to (and
        # that the encode/lower `except Exception` clauses above this try
        # block already hold to for THOSE two passes). Reported the same
        # shape those two already use -- a rejected Candidate, never a raised
        # exception -- so `compile_spec`'s search can still try the other
        # candidate encodings, and the CLI's existing `if comp.verdict !=
        # "COMPILED": print(cand.reason)` (cli.py) surfaces it without any
        # new try/except needed there. NOT swallowed: `current_pass` (set
        # just above each of the two calls this try block makes) and
        # `type(e).__name__` are folded into `reason` together with `e`
        # itself, specifically so a reader gets "which pass, which exception
        # type, what it said" from the printed reason alone -- enough to
        # debug from, not just "something went wrong". A genuine
        # `CompileError` never reaches this clause at all: the `except
        # CompileError` above it always matches first and keeps its own
        # existing, structured handling completely unchanged.
        return Candidate(encoding, CandidateState.COMPILER_ERROR,
                         reason=f"unexpected {type(e).__name__} in "
                                f"{current_pass}: {e}",
                         report=report), \
            {"report": report, "gate_checks": checks, "durations": durations}

    prog = _timed("build_program", build_program, ising, report,
                 _physical_clamp(enc, ising, clamp))
    regime = _timed("regime", analyse_regime, report, target, ising.weights)
    return (Candidate(encoding, CandidateState.HARDWARE_FEASIBLE, report=report,
                      regime=regime, placement=placement),
            {"encoded": enc, "ising": ising, "report": report,
             "gate_checks": checks, "program": prog,
             "regime": regime, "placement": placement, "durations": durations})


ORDERING_RATIONALE = (
    "selection order: physical p-bit count, then colour blocks, then |J|max; "
    "ties broken by declaration order (" + ", ".join(SLICE_ENCODINGS) + ")")


def _physical_pbits(report) -> int:
    """The count of spins actually deployed to the substrate. Task 6:
    `_try` swaps `report` to the MEDIATED graph's own report (`analyse` of
    `placement.mediated_ising`) the moment `place` had to mediate, so
    `report.n_nodes` here already includes every inserted mediator spin --
    this stays a plain alias for `n_nodes` (not a separately-tracked
    quantity) precisely because that swap is what keeps the equality true,
    not an assumption this function makes on its own."""
    return report.n_nodes


def _selection_key(c: "Candidate"):
    r = c.report
    return (_physical_pbits(r), r.colour_blocks, r.max_abs_J)


def compare(repset: "RepresentationSet") -> tuple[dict, ...]:
    """The measured table over EVERY candidate in a RepresentationSet -- winner,
    runner-up, and rejected alike -- so a selection can be audited against the
    numbers that produced it, not just the ordering_rationale prose. Any field
    a candidate never reached (e.g. `analyse` never ran because `encode` itself
    failed) reads None here; callers must not treat that as zero.

    Task 6: for a candidate `place` had to mediate, `r`/`c.report` is already
    the POST-mediation report (see `_try`'s own comment) -- "logical_spins"/
    "logical_edges"/"bipartite"/"mediators" below describe the graph that was
    actually placed and deployed, not the pre-mediation workload-derived one.
    "mediators_inserted"/"mediation_method"/"mediation_beta" are the ACTUAL
    mediation-pass record (`Placement.mediation`, spec 5.3), distinct from
    "mediators" (analyse()'s theoretical max-cut FLOOR for whatever graph `r`
    describes -- 0 once mediation has already made it bipartite): None for a
    candidate that was never mediated at all, never a fabricated 0."""
    rows = []
    for c in repset.candidates:
        r, p = c.report, c.placement
        # Task 6: prefer the SUCCEEDED placement's own MediationReport;
        # fall back to a rejected candidate's failure (e.g. mediation made
        # the graph bipartite but the geometric search that followed it
        # then hit placement_effort_exhausted) so real, already-computed
        # mediation evidence is never dropped just because placement
        # ultimately failed for a different reason.
        med = (p.mediation if p and p.mediation is not None else
              getattr(c.failure, "mediation", None))
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
            "mediators_inserted": med.mediator_count if med else None,
            "mediation_method": med.partition_method if med else None,
            "mediation_beta": med.beta_used if med else None,
        })
    return tuple(rows)


def compile_spec(spec, target: TargetProfile, allow_assumed: bool = False,
                 clamp=None, measure_memory: bool = False,
                 coefficient_scale: float = 1.0,
                 placement_effort=None) -> Compilation:
    """The public entry point.

    `measure_memory` is OPT-IN and defaults to False. Peak-memory measurement
    uses `tracemalloc`, which instruments every allocation in the interpreter
    and was measured to cost roughly 1.8x wall time on this workload -- the
    full test suite went from 278 s to 683 s with it always on. That is an
    unacceptable default for a caller that compiles repeatedly (an interactive
    application recompiling on each user action pays it every time) and it is
    diagnostic data most callers never read. Pass `measure_memory=True` when
    you actually want the number; `peak_memory_bytes` is then set, and is
    None otherwise so a receipt can honestly report it as unmeasured rather
    than as zero.

    `coefficient_scale` (Task 2, spec section 4.9/5.2): a uniform multiplier
    on every energy coefficient this compile produces, including the
    representation penalty -- see `encode()`'s own docstring for the full
    rationale. Validated HERE, before anything else runs, via the SAME
    `validate_coefficient_scale` helper `encode()` itself calls (never a
    second, independently-written check that could drift out of sync and
    re-open the gap one of them closes) -- so an invalid scale raises
    immediately and visibly rather than being caught by `_try`'s broad
    `except Exception` and buried inside an opaque SEMANTICALLY_INVALID
    ideal-control rejection three layers down. This closes a real gap found
    in review: a bare `coefficient_scale <= 0` guard lets `float('nan')`
    through silently (`nan <= 0` is `False` in Python), which previously
    produced a "COMPILED" receipt full of NaN coefficients -- see
    `validate_coefficient_scale`'s own docstring in encode.py for the full
    reasoning, including why +inf is rejected too.

    It wraps the whole implementation rather than threading a measurement
    through `_compile_spec_impl`'s several return points, so LOGICAL,
    HARDWARE and COMPILED paths are all covered identically. `Compilation`
    is a plain (non-frozen) dataclass specifically so the measurement can be
    attached after the fact.
    """
    validate_coefficient_scale(coefficient_scale)

    if not measure_memory:
        result = _compile_spec_impl(spec, target, allow_assumed, clamp,
                                    coefficient_scale, placement_effort)
        result.peak_memory_bytes = None
        return result

    tracemalloc.start()
    try:
        result = _compile_spec_impl(spec, target, allow_assumed, clamp,
                                    coefficient_scale, placement_effort)
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    result.peak_memory_bytes = peak
    return result


def _compile_spec_impl(spec, target: TargetProfile, allow_assumed: bool = False,
                       clamp=None, coefficient_scale: float = 1.0,
                       placement_effort=None) -> Compilation:
    """`clamp` (C1): an optional {workload variable name: value} map, pinning
    those variables for the SAMPLING stage only. The preferred entry point
    for clamping (over a spec-file `clamp:` block) precisely so an
    application can clamp at run time -- e.g. a player edits one cell --
    without rewriting or regenerating a spec file. The ideal control below is
    deliberately run UNCLAMPED regardless: its only job is proving the spec
    is logically sound on its own terms, independent of any one run's clamp
    choice.

    `coefficient_scale` (Task 2): run through the SAME `_try` call for the
    ideal control as for every real candidate -- IDEAL's own thresholds are
    all `inf` (target.py), so the scale changes nothing about whether the
    control passes; keeping it consistent across every `_try` call (rather
    than special-casing the control to always run unscaled) means there is
    only one code path to reason about, not two that could silently drift.
    `scaled_beta` is deterministic given `spec` and `coefficient_scale` alone
    (spec_beta(spec) / coefficient_scale) and is therefore always set below,
    on every verdict -- never a measurement that could be legitimately
    unavailable.
    """
    scaled_beta = spec_beta(spec) / coefficient_scale

    # --- the mandatory ideal control, first and always -----------------------
    ideal_cand, ideal_art = _try(spec, IDEAL, SLICE_ENCODINGS[0], True,
                                 coefficient_scale=coefficient_scale)
    ideal_ok = ideal_cand.state == CandidateState.HARDWARE_FEASIBLE
    if not ideal_ok:
        return Compilation(
            spec=spec, target=target, verdict="LOGICAL", ideal_passed=False,
            ideal_report=ideal_cand, hardware_evaluated=False,
            repset=RepresentationSet((ideal_cand,), None,
                                     "ideal control failed; target not evaluated"),
            allow_assumed=allow_assumed, clamp=dict(clamp or {}),
            pass_durations=(ideal_art or {}).get("durations", {}),
            coefficient_scale=coefficient_scale, scaled_beta=scaled_beta)

    cands, arts = [], {}
    for enc_name in SLICE_ENCODINGS:
        c, a = _try(spec, target, enc_name, allow_assumed, clamp,
                    coefficient_scale, placement_effort)
        cands.append(c)
        if a:
            arts[enc_name] = (c, a)

    feasible = [c for c in cands if c.state == CandidateState.HARDWARE_FEASIBLE]
    # If every candidate died of a fault in THIS compiler, the hardware
    # question was never actually evaluated, so "HARDWARE" (which asserts
    # the model does not fit) would be a claim we did not establish. Only
    # when some candidate genuinely reached and failed a gate does that
    # verdict describe what happened.
    errored = [c for c in cands if c.state == CandidateState.COMPILER_ERROR]
    if not feasible and errored and len(errored) == len(cands):
        any_art = next(iter(arts.values()), None)
        checks = any_art[1]["gate_checks"] if any_art else ()
        durations = any_art[1].get("durations", {}) if any_art else {}
        return Compilation(
            spec=spec, target=target, verdict="ERROR", ideal_passed=True,
            ideal_report=ideal_cand, hardware_evaluated=False,
            repset=RepresentationSet(
                tuple(cands), None,
                "every candidate hit an unexpected fault in this compiler; "
                "the hardware question was never evaluated"),
            allow_assumed=allow_assumed, gate_checks=checks,
            clamp=dict(clamp or {}), pass_durations=durations,
            coefficient_scale=coefficient_scale, scaled_beta=scaled_beta)
    if not feasible:
        any_art = next(iter(arts.values()), None)
        checks = any_art[1]["gate_checks"] if any_art else ()
        durations = any_art[1].get("durations", {}) if any_art else {}
        return Compilation(
            spec=spec, target=target, verdict="HARDWARE", ideal_passed=True,
            ideal_report=ideal_cand, hardware_evaluated=True,
            repset=RepresentationSet(tuple(cands), None,
                                     "no candidate was hardware-feasible"),
            allow_assumed=allow_assumed, gate_checks=checks, clamp=dict(clamp or {}),
            pass_durations=durations,
            coefficient_scale=coefficient_scale, scaled_beta=scaled_beta)

    chosen = min(feasible, key=_selection_key)
    art = arts[chosen.encoding][1]

    verification, ess_result, sample_cost, decoded_sample = _verify(
        spec, art, clamp or {})
    durations = dict(art.get("durations", {}))
    durations["verify"] = sample_cost.pop("verify_wall_time_s", 0.0)
    # RegimeReport.mixing_indicator can only be populated AFTER `_verify` has
    # actually sampled the chosen candidate's program -- `chosen.regime` (built
    # in `_try`, before any sampling happened) still reads None/"unmeasured"
    # for it. Patch it in here, once, from the SAME EssEstimate that fed
    # `verification.ess`, so both fields are honest about the same measurement
    # (or the same reason neither could be measured) rather than being
    # computed twice and risking drifting apart.
    regime = chosen.regime
    if regime is not None:
        if ess_result.reliable:
            regime = dataclasses.replace(regime, mixing_indicator=ess_result.iat)
        else:
            regime = dataclasses.replace(
                regime, mixing_indicator_note=ess_result.reason)
        # Item 2: energy_scale/beta_recommendation, same lazy-patch pattern as
        # mixing_indicator just above -- `_energy_scale` needs the ENCODED
        # model and the spec's own task contract, neither of which
        # `analyse_regime` (called from `_try`, before `encode`'s output was
        # attached to anything the regime pass sees) has access to.
        energy_scale, energy_scale_note = _energy_scale(
            spec, art["encoded"], art["ising"])
        regime = dataclasses.replace(
            regime, energy_scale=energy_scale, energy_scale_note=energy_scale_note,
            beta_recommendation=beta_recommendation_from_energy_scale(energy_scale))

    selected = Candidate(chosen.encoding, CandidateState.SELECTED,
                         report=chosen.report, regime=regime,
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

    return Compilation(
        spec=spec, target=target, verdict="COMPILED", ideal_passed=True,
        ideal_report=ideal_cand, hardware_evaluated=True,
        repset=RepresentationSet(final, selected, ORDERING_RATIONALE),
        allow_assumed=allow_assumed, gate_checks=art["gate_checks"],
        program=art["program"], encoded=art["encoded"], regime=regime,
        placement=art["placement"], verification=verification,
        clamp=dict(clamp or {}), pass_durations=durations, sample_cost=sample_cost,
        sample=decoded_sample,
        coefficient_scale=coefficient_scale, scaled_beta=scaled_beta)


def _measure_mixing(ising, chains: np.ndarray) -> EssEstimate:
    """chains: (n_chains, n_samples, n_spins) from ONE sampling run, kept
    UNFLATTENED -- autocorrelation across a chain boundary is meaningless
    (thrml_backend.sample_chains' own docstring). The scalar functional fed to
    tsu.ess is the sample's own energy under `ising`, the standard physics-
    MCMC mixing diagnostic (Sokal 1989, "Monte Carlo Methods in Statistical
    Mechanics", sec. 1: the autocorrelation time of the energy/magnetisation
    is the quantity actually tracked), rather than any single spin coordinate
    -- so the estimate reflects mixing of the WHOLE joint state, not one axis
    of it."""
    n_chains, n_samples, _ = chains.shape
    energy = np.array([[_from_ising(ising, chains[ci, ti]) for ti in range(n_samples)]
                       for ci in range(n_chains)])
    return effective_sample_size(energy)


# C4: the fixed sampling parameters `_verify` draws with, named once so the
# receipt's own record of them (cost.json's `sampler.params`) can never drift
# from what was actually run.
#
# C-6 (external review, 2026-09-10): this budget sits near, not comfortably
# above, `tsu.ess.RELIABILITY_MIN_N_OVER_TAU`'s reliability floor -- recorded
# here so a future reader does not mistake a refused `ess` at this default for
# a defect. The arithmetic: n_chains=32 * n_samples=200 = 6,400 total draws;
# `effective_sample_size` (ess.py) only calls its own estimate reliable when
# N/tau >= RELIABILITY_MIN_N_OVER_TAU = 5000, i.e. when tau <= 6400/5000 =
# 1.28 -- a chain that mixes only slightly slower than i.i.d. (tau ~ 1) still
# clears it, but there is little margin: any real autocorrelation pushes tau
# past 1.28 and `ess_result.reliable` goes False. Unlike `sweep()`
# (preflight/sweep.py), which escalates `n_samples` (doubling, up to
# `max_samples`) when a candidate beta's ESS comes back unreliable, `_verify`
# has no such escalation loop -- this is its one, fixed, non-adaptive draw.
# A slow-mixing model refusing `ess` here at the default budget is therefore
# EXPECTED behaviour (the budget genuinely was not enough to certify mixing),
# not a bug in this function or in `tsu.ess`. Deliberately NOT changed here:
# raising the default is a performance decision (it costs every compile, not
# just a slow-mixing one) and out of scope for this fix.
_VERIFY_SAMPLE_PARAMS = dict(n_chains=32, n_samples=200, n_warmup=400,
                             steps_per_sample=2, seed=0)


def _transport_note(spec) -> str:
    """Does this workload declare a DIRECTED quantity that TV cannot certify?

    `conserve_over_edges` generates one binary spin per edge, read as "the flow
    from u to v" in the direction the edge was declared (`spec._flow_var_name`).
    That is a transport model, and total-variation agreement against the exact
    Boltzmann distribution does not certify transport: SPR N-026/N-027 measured
    a flat energy-based model matching a ratchet's stationary distribution to
    TV ~1e-15 while its net current went to zero. Both facts can hold at once
    because a distribution and a current are independent properties.

    Detected from the spec's own source text rather than its `terms`, which are
    already expanded into Product/Linear by the time this runs -- the template
    kind is gone from them. A comment mentioning the kind would over-disclose,
    which is the safe direction to be wrong in; missing a real flow workload is
    not.
    """
    text = getattr(spec, "source_text", "") or ""
    if "conserve_over_edges" not in text:
        return ""
    return ("this workload declares conserve_over_edges (directed flow "
            "variables), and the TV figures above DO NOT CERTIFY TRANSPORT: "
            "they show the sampler reproduced the distribution that was "
            "compiled, not that the distribution carries the intended current. "
            "A flat model can match a ratchet's stationary distribution to "
            "TV ~1e-15 with zero net current (SPR N-026/N-027). Certifying "
            "directed behaviour needs a transition observable, which this "
            "compiler does not measure.")


def _verify(spec, art, clamp) -> tuple[Verification, EssEstimate, dict, DecodedSample]:
    from ..backends.thrml_backend import (EXACT_LIMIT, exact_conditional_distribution,
                                          exact_distribution, sample_chains)
    from ..backends.torx_backend import torx_cross_check

    verify_t0 = time.perf_counter()
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
    # escape hatch exists. Found via a large generated-shape spec exercising
    # exactly this path (a spec whose logical spin count exceeds EXACT_LIMIT).
    #
    # `decode` is a PROJECTION, not an inverse (C5): handed a non-monotone
    # (invalid) domain-wall chain it still returns a legal-looking value with no
    # flag. A sample that is not a valid codeword is not a decoded sample at
    # all -- it must count as invalid, not be silently decoded and validated as
    # if the chain meant something. The codeword-violation rate is reported
    # separately so it is visible, not folded into (and hidden inside) task_validity.
    #
    # Kept UNFLATTENED (`chains`) so the ESS/mixing estimate below can see
    # chain boundaries; `got` is the exact same draws flattened, unchanged
    # from before, for the task/execution layers that only need i.i.d.-
    # looking samples against a stationary reference.
    sampler_t0 = time.perf_counter()
    chains = sample_chains(prog, **_VERIFY_SAMPLE_PARAMS)
    sampler_wall_time = time.perf_counter() - sampler_t0
    n_drawn = _VERIFY_SAMPLE_PARAMS["n_chains"] * _VERIFY_SAMPLE_PARAMS["n_samples"]
    sample_cost = {"sampler": {
        "wall_time_s": sampler_wall_time,
        "samples_per_second": (n_drawn / sampler_wall_time
                               if sampler_wall_time > 0 else None),
        "params": dict(_VERIFY_SAMPLE_PARAMS),
    }}

    got = chains.reshape(-1, chains.shape[-1])
    ess_result = _measure_mixing(ising, chains)
    ok = 0
    codeword_violations = 0
    # C4: diversity. Needs no exact reference (same as task_validity above),
    # computed unconditionally over the SAME samples -- distinct decoded,
    # TASK-VALID configurations, over how many valid samples that count is
    # drawn from. The KNOWN TRAP the brief names: on a small state space a
    # high distinct/valid ratio is not evidence of much, so the size of the
    # whole reachable VALID set is reported alongside it below (when the
    # logical state space is enumerable), so the number can be read
    # correctly rather than mistaken for "how random" the sampler is.
    distinct_valid = set()
    # G2: track the ONE decoded sample this receipt will persist, at the same
    # point every other per-sample quantity above is computed -- never a
    # second pass over `got`. Preferred: the first codeword that also passes
    # the task contract (a real solution). If none ever does, the first
    # codeword that FAILED it instead, violations attached -- a workload
    # whose sampler never produces a valid state is exactly the case a
    # developer needs to see, not an empty field.
    best_valid_sample = None
    best_failing_sample = None    # (decoded, violations) of the first
                                    # codeword that failed the contract
    for row in got:
        bits = dict(zip(ising.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            codeword_violations += 1
            continue
        decoded = enc.decode(bits)
        result = spec.contract.validate(decoded)
        if result.ok:
            ok += 1
            distinct_valid.add(tuple(sorted(decoded.items())))
            if best_valid_sample is None:
                best_valid_sample = decoded
        elif best_failing_sample is None:
            best_failing_sample = (decoded, result.violations)
    task_validity = ok / len(got)
    codeword_violation_rate = codeword_violations / len(got)
    diversity_distinct = len(distinct_valid)
    diversity_valid_samples = ok

    seed = _VERIFY_SAMPLE_PARAMS["seed"]
    if best_valid_sample is not None:
        decoded_sample = DecodedSample(
            decoded=best_valid_sample, is_codeword=True, task_valid=True,
            violations=(), seed=seed, clamp=dict(clamp))
    elif best_failing_sample is not None:
        f_decoded, f_violations = best_failing_sample
        decoded_sample = DecodedSample(
            decoded=f_decoded, is_codeword=True, task_valid=False,
            violations=tuple(f_violations), seed=seed, clamp=dict(clamp))
    else:
        # not even one codeword was drawn in the whole run -- say so, rather
        # than leaving the field empty or fabricating a decode of a pattern
        # `Encoded.decode` itself would refuse to trust (see its docstring).
        decoded_sample = DecodedSample(
            decoded=None, is_codeword=False, task_valid=False, violations=(),
            seed=seed, clamp=dict(clamp),
            note=f"no codeword was drawn in this run ({len(got)} draws, "
                 f"{codeword_violations} codeword violation(s))")

    # The reachable valid set's size is a WORKLOAD-level enumeration (over
    # spec.assignments(), not Ising states) and needs no encoding/backend at
    # all -- gated on the SAME n (encoded spin count) the rest of this
    # function already uses as its "small enough to enumerate exactly"
    # threshold, since a spec whose spin count is this large also has a
    # combinatorially large logical assignment space.
    if n <= EXACT_LIMIT:
        diversity_reachable = sum(
            1 for a in spec.assignments() if spec.contract.validate(a).ok)
        diversity_reachable_note = ""
    else:
        diversity_reachable = None
        diversity_reachable_note = f"unavailable: 2^{n} too large to enumerate"

    diversity_kwargs = dict(
        diversity_distinct=diversity_distinct, diversity_note="",
        diversity_valid_samples=diversity_valid_samples,
        diversity_reachable=diversity_reachable,
        diversity_reachable_note=diversity_reachable_note)

    ess_kwargs = dict(
        ess=ess_result.ess,
        ess_note="" if ess_result.reliable else ess_result.reason)

    if n > EXACT_LIMIT:
        sample_cost["baseline"] = {
            "available": False,
            "note": f"unavailable: 2^{n} too large to enumerate",
            "exact_wall_time_s": None,
            "sampler_wall_time_s": sampler_wall_time,
            "disclaimer": "a correctness-and-cost REFERENCE only when "
                          "available; no speed or hardware claim",
        }
        sample_cost["verify_wall_time_s"] = time.perf_counter() - verify_t0
        return Verification(
            energy_tv=None, energy_note=f"unavailable: 2^{n} too large to enumerate",
            task_validity=task_validity, task_validity_note="",
            execution_tv=None, execution_note="unavailable: no exact reference",
            execution_noise_floor=None,
            cross_check_tv=None, cross_check_note="unavailable: model too large",
            transport_note=_transport_note(spec),
            codeword_violation_rate=codeword_violation_rate,
            codeword_violation_note="", **ess_kwargs, **diversity_kwargs
        ), ess_result, sample_cost, decoded_sample

    # C4: CPU reference baseline -- a correctness-and-cost REFERENCE, never a
    # speed/hardware claim. Both wall times are recorded honestly; nothing
    # here states or implies either approach is faster than the other.
    exact_t0 = time.perf_counter()
    states, probs = exact_distribution(prog)
    exact_wall_time = time.perf_counter() - exact_t0
    sample_cost["baseline"] = {
        "available": True,
        "note": "",
        "exact_wall_time_s": exact_wall_time,
        "sampler_wall_time_s": sampler_wall_time,
        "disclaimer": "a correctness-and-cost REFERENCE only; no speed or "
                      "hardware claim -- neither wall time is evidence that "
                      "either approach is faster than the other",
    }

    # energy layer: the lowered model must reproduce the encoded model's energies
    worst = 0.0
    for row, p in zip(states, probs):
        bits = dict(zip(ising.nodes, row.tolist()))
        worst = max(worst, abs(enc.model.energy(bits) - _from_ising(ising, row)))
    energy_tv = worst

    # execution layer: sampler vs its own model, reusing the SAME samples `got`
    # already drawn above for the task layer (one sampling run serves both).
    #
    # Clamp-aware reference (the fix this comment used to warn was missing):
    # every draw in `got` honours `prog`'s clamp (thrml_backend's own
    # `state_clamp`/`clamped_blocks` mechanism -- see test_clamp.py), so `got`
    # is drawn from the CONDITIONAL distribution whenever a clamp is active,
    # never the unconditional one. Comparing it against the unconditional
    # `probs` above made execution_tv spuriously large under a clamp --
    # measured on specs/adjacency_2x2_k3.yaml with clamp={'g0_0': 0}:
    # execution_tv=0.741946 against noise floor 0.032023 (23.2x), while the
    # unclamped run on the same spec sits at 0.045659 against the same floor
    # (1.4x) -- the sampler was never wrong, only the reference it was being
    # checked against. `exact_conditional_distribution` is exactly
    # `exact_distribution` restricted to the clamp-consistent states and
    # renormalised over them (and IS `exact_distribution` when there is no
    # clamp), so this works unconditionally without a clamped/unclamped
    # branch here.
    #
    # The index into `cond_probs` MUST be derived from `cond_states` itself,
    # never from an independently-assumed bit order -- `exact_distribution`'s
    # enumeration order is an implementation detail of itertools.product, and
    # a hand-rolled (1 << arange(n)) index silently assumes a specific one. A
    # previous version assumed LSB-first and got MSB-first, which inflated
    # execution_tv from ~0.01 to ~0.52 with no error raised anywhere -- a
    # broken comparison that looked exactly like a broken sampler. Building
    # the lookup from `cond_states` means the two can never drift apart
    # again, regardless of how either side is implemented.
    cond_states, cond_probs = exact_conditional_distribution(prog)
    state_index = {tuple(int(x) for x in row): i for i, row in enumerate(cond_states)}
    idx = np.array([state_index[tuple(int(x) for x in row)] for row in got])
    hist = np.bincount(idx, minlength=len(cond_probs)).astype(float)
    hist /= hist.sum()
    execution_tv = float(0.5 * np.abs(hist - cond_probs).sum())
    floor = tv_noise_floor(cond_probs, len(got))

    tx, tx_note = torx_cross_check(ising)
    if tx is None:
        cross, cross_note = None, f"unavailable: {tx_note}"
    else:
        cross, cross_note = float(0.5 * np.abs(tx - probs).sum()), ""

    sample_cost["verify_wall_time_s"] = time.perf_counter() - verify_t0
    return Verification(
        energy_tv=energy_tv, energy_note="",
        task_validity=task_validity,
        execution_tv=execution_tv,
        execution_note=f"noise floor {floor:.6f}",
        execution_noise_floor=floor,
        cross_check_tv=cross, cross_check_note=cross_note,
        transport_note=_transport_note(spec),
        codeword_violation_rate=codeword_violation_rate, codeword_violation_note="",
        **ess_kwargs, **diversity_kwargs
    ), ess_result, sample_cost, decoded_sample


def _from_ising(ising, row) -> float:
    s = 2 * row.astype(float) - 1
    total = ising.offset
    for i in range(len(ising.nodes)):
        total += -ising.biases[i] * s[i]
    for k, (u, v) in enumerate(ising.edges):
        total += -ising.weights[k] * s[u] * s[v]
    return float(total)


def _energy_scale(spec, enc, ising) -> tuple[float | None, str]:
    """Item 2: the physical energy gap between the best (lowest-energy)
    PHYSICAL state that decodes to a task-contract-satisfying answer and the
    best physical state that does not -- "does not" covering both an invalid
    codeword (`enc.is_codeword` false, e.g. a non-monotone domain-wall chain,
    per C5) and a valid codeword whose decoded assignment fails
    `spec.contract`. This is the gap the sampler must actually resolve to
    prefer a correct answer over an incorrect one; `RegimeReport.
    beta_recommendation` is derived from it (see regime.py's own docstring
    for why that is a WINDOW, not a point value).

    Exact by enumeration over every physical state (2**n) -- never
    estimated: a model too large to enumerate exhaustively (`n > EXACT_LIMIT`,
    the same bound thrml_backend's own exact reference uses -- imported
    locally here, same lazy-import pattern `_verify` already uses to keep
    thrml out of this module's top-level imports per the structural rule
    that only `src/tsu/backends/` may import it) reports None with a reason,
    and so does a model where every state landed on the same side of the
    contract -- there is then no gap to measure, not a zero one.

    Per-state energies are computed with the exact same formula `_from_ising`
    already uses just above (and that `_verify`'s energy_tv layer already
    cross-checks against thrml's own `IsingEBM.energy`), vectorised here over
    every state at once for speed -- not a second, independently-written
    formula that could quietly drift from the one already trusted.
    """
    from ..backends.thrml_backend import EXACT_LIMIT
    n = len(ising.nodes)
    if n > EXACT_LIMIT:
        return None, f"unavailable: 2^{n} too large to enumerate"

    states = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int8)
    spins = 2.0 * states.astype(float) - 1.0
    energies = np.full(states.shape[0], ising.offset, dtype=float) - spins @ ising.biases
    if ising.edges:
        u_idx = np.array([u for u, _ in ising.edges])
        v_idx = np.array([v for _, v in ising.edges])
        energies -= (spins[:, u_idx] * spins[:, v_idx]) @ ising.weights

    best_valid = None
    best_invalid = None
    for row, e in zip(states, energies):
        bits = dict(zip(ising.nodes, (int(x) for x in row)))
        ok = enc.is_codeword(bits) and spec.contract.validate(enc.decode(bits)).ok
        e = float(e)
        if ok:
            if best_valid is None or e < best_valid:
                best_valid = e
        else:
            if best_invalid is None or e < best_invalid:
                best_invalid = e

    if best_valid is None:
        return None, ("unavailable: no physical state decodes to a "
                      "task-contract-satisfying answer")
    if best_invalid is None:
        return None, "unavailable: no physical state violates the task contract"
    return best_invalid - best_valid, ""
