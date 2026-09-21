# Honest verdicts, a solutions readout, and a slim artifact

**Date:** 2026-09-21
**Status:** design, approved for planning
**Drives:** `docs/superpowers/plans/2026-09-21-*.md`

## 1. What is being built, and what is not

A three-part adversarial review (Grok, 2026-09-20) graded this repository
**revise, not reject**, with one sentence that organises everything below:

> The physics core is not the thing on fire. The referent of the word Z1 is.

This spec covers four workstreams plus a declutter. It does **not** cover the
compiler IDE. The IDE is the destination and is scoped in section 8 only far
enough to make sure nothing here blocks it.

**The thesis this artifact can support, and must not exceed:** a host-side
refusal compiler for a *reconstructed* Z1-shaped profile, with provenance on
every limit. Not Thermalizers. Not a decision procedure for what runs on the
die.

## 2. Triage: the review is partly stale

Work landed 2026-09-17 that the review predates. Every row below was checked
against the code, and the two P0 items were reproduced by measurement rather
than accepted on the reviewer's word.

| Review finding | Status |
|---|---|
| minor-embed "unwired, optional import" | **Stale.** Wired into `preflight`; `minorminer` is a hard dependency |
| "placement failure prints FAIL" | **Stale.** `preflight` returns `effort`; CLI exits 3 |
| "913 tests" | **Stale.** 1,002 |
| `insert_mediators` does not refuse `A > cap` | **LIVE, measured:** J=5.7, β=1 returns max&#124;J&#124;=6.0466 against cap 6.0 |
| colouring gate is not 2-colourability | **LIVE, measured:** a triangle has `bipartite=False` and the gate **passes** |
| chains absent from `search.py` (`tsuc compile`) | **LIVE.** Only `preflight` was wired |
| `peak > cap` admits exactly 6.0 | **LIVE** |
| Onsager βc used as a regime marker | **LIVE:** `cli.py`, `preflight/render.py`, `preflight/sweep.py` |
| distinct-J count folds sign through `abs()` | **LIVE** |
| identity soup; "Z1" vs "z1-profile" | **LIVE** |

## 3. Workstream A — truthfulness fixes

Prerequisite to everything else. Packaging on top of a measured hole would
ship a lie.

### A1. The mediation cap must refuse at the door, not in a hallway

`insert_mediators(ising, report)` takes no target, so it cannot check a cap.
The check currently lives only in `search._try` and `preflight`. Anyone
composing `insert_mediators → build_program` — fixtures, audit scripts, a
future backend — emits a model with an illegal coupling.

**Change:** `insert_mediators(ising, report, target=None)`. When `target` is
given, raise `CompileError` with a `GateFailure` if any mediated coupling
exceeds `max_abs_coupling`. When it is `None`, behave exactly as today, so
existing callers are unaffected and the new guard is opt-in at the call sites
that know a target.

**Closed-form safe bound**, reported whenever any edge will be mediated:
at β=1 with cap 6, the largest pre-mediation coupling that survives is
`|J| ≤ ½·ln(cosh 12) ≈ 5.653`.

### A2. The boundary at exactly `|J| = 6.0`

An edge at exactly the cap passes the pre-gate and *always* exceeds it after
mediation, because `A > |J|` strictly for every nonzero `J`.

The logical gate is **left as it is**: `peak > cap`, because 6.0 is genuinely
legal on the published hardware and failing it would be a false refusal. What
changes is that a model containing an edge at or near the cap, on a graph that
will need mediation, is warned *before* placement using A1's closed-form bound
(`|J| ≤ 5.653` at β=1). The mediated re-gate then refuses it on the real
number rather than a boundary rule.

### A3. `two_colourable` as its own gate

The existing `colouring` gate asks whether a DSATUR colouring is proper. Z1's
block Gibbs has two colour classes, not k. Measured: a triangle passes.

**Change:** keep `colouring` (proper colouring, housekeeping) and add
`two_colourable`, which fails on a non-bipartite graph pre-mediation and passes
on the graph that will actually be sampled. Both appear in the gate table.

### A4. Chains in `search.py`

`preflight` falls back to chain embedding; `tsuc compile` does not. Two doors
give two answers for the same model, which is its own integrity defect.

**Change:** mirror the `preflight` logic in `compile_spec` — direct placement
first, chains as fallback, chains never attempted when a gate has already
failed. Receipts gain the chain fields.

### A5. Onsager stops being a thermometer for this fabric

Onsager's `Kc = 0.440687` is exact for the 4-neighbour square lattice. Z1's
profile is degree 16 with knight-like offsets, where ordering sets in at a
different coupling. R10 already flagged the coordination mismatch.

**Change:** remove every critical-point annotation that uses Onsager's number
from any path whose graph is not literally the 4-neighbour grid. Where the
graph *is* that grid (the `watch` lattice control), keep it and say so. No
substitute βc is computed here — computing one for the offset graph is a
separate piece of work and is listed as a non-goal.

### A6. Distinct couplings counted with sign

`np.unique(np.round(np.abs(weights), 9))` treats `+J` and `−J` as one value.
Sign is not optional on an Ising edge.

**Change:** count signed values, and rename the reported field so it says what
it counts. The accompanying note already states that the *mapping* is
unmodelled; that stays.

## 4. Workstream D — solutions readout

`tsuc simulate` already reconstructs a program from a receipt, samples it, and
computes `task_validity`, `codeword_violation_rate` and one `decoded_example`.
Nothing saves the *set* of solutions.

### D1. What "valid" means, settled

Four layers, only two of which are properties of a solution:

| Layer | Asks | Property of | Checked by |
|---|---|---|---|
| 1 Hardware | degree, &#124;J&#124;, &#124;b&#124;, budget, 2-colourable | the **model** | `tsuc check` |
| 2 Codeword | is this bit pattern a value at all? | the **draw** | `codeword_violation_rate` |
| 3 Task | does the assignment obey declared rules? | the **assignment** | `TaskContract.validate` |
| 4 Energy | is it any good? | the assignment | ranking, **not validity** |

Layers 2 and 3 are reported as **independent columns, never merged.** They fail
in opposite ways: a task violation is loud and checkable; a codeword violation
is silent and self-laundering. Measured on a `k=3` domain-wall chain, the
pattern `(0,1)` is not a codeword and the decoder returns `c=1` for it anyway —
one pattern in four, with no flag. Merging the two would hide the silent
failure inside the loud one.

With no contract declared, `task_valid` reads `"no contract declared"`, never
`true`, so an edge-list input cannot imply a check that never ran.

**Out of scope, by the owner's decision:** per-workload semantic validity
("is this a good codon sequence"). That is a separate system, built per
workload. The compiler owes honest validity per the hardware and the maths,
and nothing more.

### D2. `tsuc solve`

New verb. Takes a receipt, samples, decodes, evaluates layers 2 and 3, dedupes
by assignment, ranks by energy, writes `solutions.json` and `solutions.csv`.

Every solutions file carries, at the top, the codeword violation rate for the
run that produced it. A solutions file without it is not writable — the rate is
what makes the rows readable rather than decorative.

Sampling stays off in `tsuc check`. `solve` is the second product, as a
separate verb, so a fit-check can never emit a mixing number by accident.

## 5. Workstream B — the slim artifact

Delivered **through the repository**, not as an executable. The owner
withdrew the exe from the shipping path: 413 MB, unsigned, Windows-only, and
never run on another machine. `pip install -e .` then `tsuc check` is testable
on a clean machine, which the exe never was. The exe remains a local toy.

### B1. Three lines, always, in this order

```
TARGET     z1-profile   (not a pinout; see assumptions.json)

LIMITS     PASS   degree 12/16   |J| 2.50/6   |b| 2.99/6*   spins 31/269568
SEATING    SEARCH_GAVE_UP   grid: no   chains: not tried (--allow-chains)
ASSUME     one J per edge; 8 cores = one sheet; |b| cap is ours

* assumed cap, not an Extropic figure
```

### B2. Frozen `verdict.json` schema

```json
{
  "schema": "tsuc-receipt/v1",
  "target": "z1-profile",
  "silicon": false,
  "limits": "PASS",
  "seating": "SEARCH_GAVE_UP",
  "programmable_on_die": "UNKNOWN",
  "summary": "Limits clear; no seat found on the profile lattice; die mapping uncertified."
}
```

Allowed values, enforced by a test: `limits` ∈ {PASS, FAIL}; `seating` ∈
{FOUND, SEARCH_GAVE_UP, NOT_TRIED, NOT_NEEDED}; `programmable_on_die` is
**always** `UNKNOWN` until a sharing map exists. A `limits: FAIL` never
upgrades on the strength of seating.

### B3. Naming

The CLI target becomes `z1-profile`. `z1` remains accepted as an alias so
existing receipts and scripts keep working, and resolves to the same profile.
No public verdict may print a bare "Z1" as the thing that accepted or refused
a model.

### B4. Three canned examples, and a test that pins their headlines

- `examples/grid_8x8.json` — LIMITS PASS, SEATING FOUND (the happy path)
- `examples/too_dense.json` — LIMITS FAIL, names `degree` (refusal works)
- `examples/codon_tiny.json` — LIMITS PASS, SEATING FOUND via chains (honesty:
  limit-legal, and seat-found only because chains are on)

A test asserts these three headlines never drift. If they do not run on a clean
machine, it is not an artifact.

### B5. `LIMITS_OF_THIS_TOOL.md`

One page, at the front door, in plain language: this checks against a published
profile in software; it has never run on a chip; green LIMITS means not
obviously too big or too strong; green SEATING means chairs were found on *our*
wiring drawing; it does not mean the die can load those weights; odd loops need
hidden spins and those need stronger couplings, which is re-checked;
`SEARCH_GAVE_UP` means unknown, not impossible.

## 6. Workstream C — the gap report

`tsuc check --why` prints a ranked **blocker taxonomy**, never a hardware
recommendation:

1. `LIMITS` — a published cap refuses it (model or analog-range ask)
2. `SEATING` — limit-legal, no seat found (embedding-software ask)
3. `PROGRAMMABILITY` — unknown, sharing map unpublished (always present)
4. `ARRAY_SIZE` — spins exceed budget even at sane overhead (the rare honest
   "bigger array" sentence)

Named "what's blocking this?", never "recommended hardware". It must not emit a
part number, a p-bit count to build, or any sentence that reads as a spec for
silicon. The failure mode being avoided is a screenshot captioned *"their own
compiler says Z1 can't run Extropic's model"*.

## 7. Declutter

Measured: `demo/` is 891 tracked files against 40 in `src/`. The compiler is
buried under demos.

**Constraints from `CLAUDE.md`, which this spec does not override:**
`demo/extropic-pack/**` belongs to Grok and is not written to.
`demo/cascade.py`, `demo/cascade_runs/`, `receipts/EXP-G8-D2/`,
`receipts/EXP-G8-D3/`, `figures/z1_lab_screenshot.png` are untouchable.
`specs/emergence_8x8.yaml` and `pyproject-review.toml` are QWEN leftovers and
are left alone.

**Therefore the declutter is by front-door prominence, not deletion.** Nothing
listed above moves. What changes:

- `README.md` leads with what the tool is, what it is not, the three-line
  verdict, seating scope, and how to reproduce one receipt. Everything else
  moves below the fold or into `docs/`.
- `audit/VERDICT.md` is an old LATTICE-app verdict at the compiler's front
  door. It is moved under `docs/history/` with a header naming which artifact
  it judged.
- `audit/findings/` gains an `INDEX.md` stating, per finding, which artifact it
  concerns — compiler, Observatory, or game. 34 findings currently mix all
  three, and a reader cannot tell which one got the measurement.
- One test count, quoted in one place, generated rather than typed.

## 8. The IDE, scoped only enough not to be blocked

The reference is Sci-Compiler's IDE, whose five regions map onto this work:

| Sci-Compiler region | Analogue here |
|---|---|
| Menus and toolbars | file ops plus `check` / `compile` / `solve` |
| Resources and settings (left) | project tree of specs and receipts, plus context help on the gate or term under the cursor |
| Design canvas (centre) | the spec editor today; a block diagram later |
| **Board Configuration (centre-right)** | **the target profile** — caps, offsets, and each field's sourced-or-assumed provenance |
| Status bar (bottom) | the three-line verdict from B1 |

The Board Configuration analogue is the strongest fit and the reason to keep
the target profile declarative: swapping to a future part should be editing
data, not forking the compiler. Nothing in workstreams A–D may make the profile
less declarative.

The IDE is the next design cycle and is not planned here.

## 9. Non-goals

- Computing βc for the Z1 offset graph. A5 removes the wrong number rather
  than substituting a new one.
- Modelling the coupler-sharing map. `programmable_on_die` stays `UNKNOWN`.
- Mediator exactness at large N. The refusal to extrapolate from the 3- and
  5-spin measurements stands, and is stated rather than fixed.
- Sandboxing the Workbench's user-code execution. It is a separate trust
  boundary; the artifact simply does not ship it.
- Deleting or rewriting any reserved path from section 7.

## 10. Success criteria

1. No public function returns a success for a model whose mediated coupling
   exceeds the cap — asserted against J=5.7, β=1.
2. A triangle fails `two_colourable` pre-mediation and passes post-mediation.
3. `tsuc compile` and `tsuc check` agree on seating for the same model.
4. No Onsager critical point is annotated on a graph that is not the
   4-neighbour square lattice.
5. `tsuc solve` writes ranked solutions with codeword and task validity as
   separate columns, and refuses to write without the violation rate.
6. The three canned examples produce their pinned headlines from a clean
   checkout via `pip install -e .`.
7. `verdict.json` validates against the frozen schema, and
   `programmable_on_die` is `UNKNOWN` in every one.
8. The README's first screen contains no sentence that survives being
   screenshotted as "Extropic's chip said no".
