"""Item 3: `tsu simulate` -- compile once, sample many.

A receipt's program.json already carries the FULL physical sampling program
(nodes, edges, weights, biases, beta, offset, blocks, clamp) -- `reconstruct_
program` reads it back VERBATIM. Sampling that program with fresh parameters
therefore never needs to re-run `encode`/`lower`/`gate_checks`/`place`/
`route`, and never touches `compile_spec`'s search over encodings: the model
is exactly the one the original compile selected, not a fresh derivation of
it that might (however unlikely) drift from it.

`analyse`/`build_program` DO run again here, but only when a caller's own
`clamp` differs from the receipt's -- clamping changes which nodes may share
a chromatic block, and that repartition is graph-theoretic bookkeeping over
the ALREADY-fixed edge set, not a re-derivation of the energy model itself
(no target, no gates, no placement, no search are involved). `encode` also
runs again, to translate a workload-level clamp into physical spin
indices and to decode raw samples back into the workload's own variable
names -- again a pure, deterministic re-derivation of a NAME MAPPING from
the receipt's own stored spec.yaml, not of the compiled program.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .passes.analyse import analyse
from .passes.encode import encode
from .passes.lower import IsingModel
from .passes.program import SamplingProgram, build_program
from .passes.verify import DecodedSample
from .spec import load_spec


def _selected_encoding(receipt_dir: Path) -> str:
    passes = json.loads((receipt_dir / "passes.json").read_text())
    for c in passes["candidates"]:
        if c["state"] == "SELECTED":
            return c["encoding"]
    raise ValueError(
        f"receipt at {receipt_dir} has no SELECTED candidate to simulate "
        f"(verdict was {passes.get('verdict')!r}, not COMPILED)")


def reconstruct_program(receipt_dir) -> SamplingProgram:
    """The exact SamplingProgram a compile produced, read straight back from
    program.json -- nodes/edges/weights/biases/beta/offset/blocks/schedule/
    form/clamped/clamp_values all come verbatim from the receipt, never
    recomputed. Raises if the receipt never reached a COMPILED verdict (no
    program to reconstruct)."""
    d = Path(receipt_dir)
    p = json.loads((d / "program.json").read_text())
    # A LOGICAL/HARDWARE receipt's program.json is not literally empty --
    # write_receipt always writes a bare {"clamp": {...}} even when
    # `c.program` was None -- so "nodes" (present on every COMPILED receipt)
    # is the real presence check, not truthiness of the whole dict.
    if "nodes" not in p:
        raise ValueError(
            f"receipt at {d} has no program (verdict was not COMPILED)")
    im = IsingModel(
        nodes=tuple(p["nodes"]),
        edges=tuple(tuple(e) for e in p["edges"]),
        weights=np.asarray(p["weights"], dtype=float),
        biases=np.asarray(p["biases"], dtype=float),
        beta=float(p["beta"]), offset=float(p["offset"]),
        # Task 6: round-trip which physical nodes are mediator spins so a
        # replayed/resampled program still knows it was mediated -- this is
        # what lets `simulate()` refuse a `--beta` override that would
        # silently reproduce the wrong mediator couplings (spec 5.3.5).
        # Absent on a receipt written before Task 6 existed; `()` then,
        # same as an unmediated program.
        mediator_nodes=tuple(p.get("mediator_nodes", ())))
    blocks = tuple(tuple(b) for b in p["blocks"])
    clamped = tuple(p["clamped_nodes"])
    clamp_values = {int(k): v for k, v in p["clamp_values"].items()}
    return SamplingProgram(im, blocks, schedule=p["schedule"], form=p["form"],
                           clamped=clamped, clamp_values=clamp_values)


def _physical_clamp(enc, nodes: tuple[str, ...], clamp: Mapping[str, int]) -> dict:
    idx = {n: i for i, n in enumerate(nodes)}
    return {idx[name]: value for name, value in enc.encode_clamp(clamp).items()}


def simulate(receipt_dir, *, n_chains: int = 32, n_samples: int = 200,
            n_warmup: int = 400, steps_per_sample: int = 2,
            beta: float | None = None, seed: int = 0,
            clamp: Mapping[str, int] | None = None,
            output_dir: str | Path | None = None):
    """Sample an already-compiled receipt's program with fresh parameters.

    `clamp`: a WORKLOAD-level {name: value} map. None (the default, no
    --clamp given) reuses the receipt's OWN clamp verbatim, including its
    ORIGINAL block partition -- zero recomputation beyond what
    `reconstruct_program` already did. Any other value (including {},
    explicitly unclamped) is translated through this receipt's own encoding
    and the graph is repartitioned for it via `analyse`/`build_program` (see
    the module docstring for why that repartition is not "recompiling").

    `beta`: overrides the compiled model's own beta for this run only --
    beta is a scalar multiplier on the SAME fixed energy model, so this
    never touches nodes/edges/weights/biases/offset. Task 6/spec 5.3.5: if
    the receipt's own program carries mediator spins (`base.ising.
    mediator_nodes`), their couplings are baked in at the beta they were
    computed at -- an override to any OTHER beta would silently sample the
    WRONG couplings, so `assert_beta_consistent` refuses it outright rather
    than letting it through.

    `output_dir`: where `simulation.json` is written. None (the default)
    writes it into `receipt_dir` itself, unchanged from before this
    parameter existed -- every current CLI/test caller relies on this.
    RP-1: `receipt_dir` may be frozen, git-tracked compile-time evidence
    (e.g. demo/receipts/small, read live by demo/lattice_app.py), and
    `simulate()` is called from that app once per pin change -- a caller in
    that position must pass an `output_dir` of its own so ordinary sampling
    never mutates the receipt directory it was only supposed to read.
    Created if it does not already exist.

    Returns `(simulation.json path, flattened samples array, the IsingModel
    actually sampled)` -- the array is (n_chains*n_samples, n_spins), column
    order matching the returned IsingModel's own `nodes`, handed back mainly
    so callers (and tests) can inspect the raw draws without re-reading the
    file this also writes.
    """
    from .backends.thrml_backend import sample_chains
    from .passes.route import assert_beta_consistent

    d = Path(receipt_dir)
    base = reconstruct_program(d)
    if beta is not None:
        assert_beta_consistent(base.ising, float(beta))
    im = base.ising if beta is None else replace(base.ising, beta=float(beta))

    encoding = _selected_encoding(d)
    spec = load_spec(str(d / "spec.yaml"))
    enc = encode(spec, encoding)

    if clamp is None:
        prog_json = json.loads((d / "program.json").read_text())
        workload_clamp = dict(prog_json.get("clamp") or {})
        prog = SamplingProgram(im, base.blocks, schedule=base.schedule,
                               form=base.form, clamped=base.clamped,
                               clamp_values=base.clamp_values)
    else:
        workload_clamp = dict(clamp)
        physical_clamp = _physical_clamp(enc, im.nodes, workload_clamp)
        report = analyse(im)
        prog = build_program(im, report, clamp=physical_clamp)

    params = dict(n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
                 steps_per_sample=steps_per_sample, beta=float(im.beta), seed=seed)

    t0 = time.perf_counter()
    chains = sample_chains(prog, n_chains=n_chains, n_samples=n_samples,
                           n_warmup=n_warmup, steps_per_sample=steps_per_sample,
                           seed=seed)
    wall_time_s = time.perf_counter() - t0
    got = chains.reshape(-1, chains.shape[-1])

    ok = 0
    codeword_violations = 0
    best_valid = None
    best_failing = None
    for row in got:
        bits = dict(zip(im.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            codeword_violations += 1
            continue
        decoded = enc.decode(bits)
        result = spec.contract.validate(decoded)
        if result.ok:
            ok += 1
            if best_valid is None:
                best_valid = decoded
        elif best_failing is None:
            best_failing = (decoded, result.violations)

    task_validity = ok / len(got)
    codeword_violation_rate = codeword_violations / len(got)

    if best_valid is not None:
        example = DecodedSample(decoded=best_valid, is_codeword=True,
                                task_valid=True, violations=(), seed=seed,
                                clamp=dict(workload_clamp))
    elif best_failing is not None:
        f_decoded, f_violations = best_failing
        example = DecodedSample(decoded=f_decoded, is_codeword=True,
                                task_valid=False, violations=tuple(f_violations),
                                seed=seed, clamp=dict(workload_clamp))
    else:
        example = DecodedSample(
            decoded=None, is_codeword=False, task_valid=False, violations=(),
            seed=seed, clamp=dict(workload_clamp),
            note=f"no codeword was drawn in this run ({len(got)} draws, "
                 f"{codeword_violations} codeword violation(s))")

    doc = {
        "receipt": str(d),
        "params": params,
        "clamp": dict(workload_clamp),
        "task_validity": task_validity,
        "codeword_violation_rate": codeword_violation_rate,
        "decoded_example": example.to_dict(),
        "wall_time_s": wall_time_s,
    }
    out_dir = Path(output_dir) if output_dir is not None else d
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulation.json"
    out_path.write_text(json.dumps(doc, indent=2))
    return out_path, got, im
