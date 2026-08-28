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
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ess import EssEstimate, effective_sample_size
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


def _try(spec, target, encoding, allow_assumed, clamp=None):
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
    """
    durations: dict[str, float] = {}

    def _timed(name, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        durations[name] = time.perf_counter() - t0
        return out

    try:
        enc = _timed("encode", encode, spec, encoding)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"encode failed: {e}"), None

    try:
        ising = _timed("lower", lower, enc.model)
    except Exception as e:
        return Candidate(encoding, CandidateState.SEMANTICALLY_INVALID,
                         reason=f"lower failed: {e}"), None

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
    try:
        placement = _timed("place", place, ising, report, target)
        ising = _timed("route", route, ising, report, target)
    except CompileError as e:
        return Candidate(encoding, CandidateState.HARDWARE_INFEASIBLE,
                         reason=str(e), failure=e.failures[0], report=report), \
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


def compile_spec(spec, target: TargetProfile, allow_assumed: bool = False,
                 clamp=None, measure_memory: bool = False) -> Compilation:
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

    It wraps the whole implementation rather than threading a measurement
    through `_compile_spec_impl`'s several return points, so LOGICAL,
    HARDWARE and COMPILED paths are all covered identically. `Compilation`
    is a plain (non-frozen) dataclass specifically so the measurement can be
    attached after the fact.
    """
    if not measure_memory:
        result = _compile_spec_impl(spec, target, allow_assumed, clamp)
        result.peak_memory_bytes = None
        return result

    tracemalloc.start()
    try:
        result = _compile_spec_impl(spec, target, allow_assumed, clamp)
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    result.peak_memory_bytes = peak
    return result


def _compile_spec_impl(spec, target: TargetProfile, allow_assumed: bool = False,
                       clamp=None) -> Compilation:
    """`clamp` (C1): an optional {workload variable name: value} map, pinning
    those variables for the SAMPLING stage only. The preferred entry point
    for clamping (over a spec-file `clamp:` block) precisely so an
    application can clamp at run time -- e.g. a player edits one cell --
    without rewriting or regenerating a spec file. The ideal control below is
    deliberately run UNCLAMPED regardless: its only job is proving the spec
    is logically sound on its own terms, independent of any one run's clamp
    choice.
    """
    # --- the mandatory ideal control, first and always -----------------------
    ideal_cand, ideal_art = _try(spec, IDEAL, SLICE_ENCODINGS[0], True)
    ideal_ok = ideal_cand.state == CandidateState.HARDWARE_FEASIBLE
    if not ideal_ok:
        return Compilation(
            spec=spec, target=target, verdict="LOGICAL", ideal_passed=False,
            ideal_report=ideal_cand, hardware_evaluated=False,
            repset=RepresentationSet((ideal_cand,), None,
                                     "ideal control failed; target not evaluated"),
            allow_assumed=allow_assumed, clamp=dict(clamp or {}),
            pass_durations=(ideal_art or {}).get("durations", {}))

    cands, arts = [], {}
    for enc_name in SLICE_ENCODINGS:
        c, a = _try(spec, target, enc_name, allow_assumed, clamp)
        cands.append(c)
        if a:
            arts[enc_name] = (c, a)

    feasible = [c for c in cands if c.state == CandidateState.HARDWARE_FEASIBLE]
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
            pass_durations=durations)

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
        sample=decoded_sample)


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
_VERIFY_SAMPLE_PARAMS = dict(n_chains=32, n_samples=200, n_warmup=400,
                             steps_per_sample=2, seed=0)


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
