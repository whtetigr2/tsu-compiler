# Workstream A — Truthfulness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four measured holes where this compiler can report success for a model that is not legal, or report two different answers at two different doors.

**Architecture:** Four independent fixes to existing passes. Each adds a refusal or a distinction that the code currently lacks, each is pinned by a test that fails before the change, and none alters the public pipeline order. No new modules.

**Tech Stack:** Python 3.11+, numpy, networkx, pytest. Interpreter: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"`, run from the repo root.

**Spec:** `docs/superpowers/specs/2026-09-21-honest-compiler-and-artifact-design.md`

## Global Constraints

- Stage commits by explicit path. Never `git add -A`.
- Do not touch: `receipts/EXP-G8-D2/`, `receipts/EXP-G8-D3/`, `figures/z1_lab_screenshot.png`, `demo/cascade.py`, `demo/cascade_runs/`, `demo/extropic-pack/**`, `specs/emergence_8x8.yaml`, `pyproject-review.toml`.
- No hardware claims. Everything is simulation against a published profile.
- A control that cannot fail is not a control. Every task below asserts its guard *fires*, not merely that the happy path still works.
- Full compiler suite before each commit: `python -m pytest tests -q`. Baseline is **1002 passed, 8 skipped**.
- End commit messages with: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Dropped from this workstream after verification

**Onsager removal (spec A5) is already done and must not be "fixed" again.** `preflight/sweep.py:detect_uniform_square_lattice` returns three states (applies / checked-and-does-not / not determined); `preflight/render.py:395` prints Kc only when it is not None, with the sentence "because this graph is a uniform square lattice in zero field, the only condition under which it applies"; `cli.py:329` is the `watch --lattice` path, which constructs a square lattice, where Onsager is correct. Re-read those three sites before concluding otherwise.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/tsu_compiler/passes/route.py` | Gains an optional `target` parameter to `insert_mediators` and refuses when a mediated coupling exceeds the cap | 1 |
| `tests/test_mediation_cap_door.py` | New. Pins that the refusal fires at the door, not only in `search` | 1 |
| `src/tsu_compiler/gates.py` | Gains a `two_colourable` gate beside the existing `colouring` gate | 2 |
| `tests/test_two_colourable_gate.py` | New. Pins that a triangle fails pre-mediation and passes post-mediation | 2 |
| `src/tsu_compiler/passes/search.py` | Falls back to chain embedding, mirroring `preflight` | 3 |
| `tests/test_compile_uses_chains.py` | New. Pins that `compile_spec` and `preflight` agree on seating | 3 |
| `src/tsu_compiler/preflight/check.py` | Counts signed coupling values; renames the reported field | 4 |
| `tests/test_signed_coupling_count.py` | New. Pins that `+J` and `−J` count as two | 4 |

---

### Task 1: Refuse an illegal mediated coupling at the door

**Files:**
- Modify: `src/tsu_compiler/passes/route.py:171-172` (signature), and the success return at `src/tsu_compiler/passes/route.py:281`
- Test: `tests/test_mediation_cap_door.py`

**Interfaces:**
- Consumes: `mediator_coupling(absJ: float, beta: float) -> float` from `route.py:147`; `CompileError` and `GateFailure` from `tsu_compiler.failures`; `TargetProfile` from `tsu_compiler.target`.
- Produces: `insert_mediators(ising, report, target=None) -> tuple[IsingModel, MediationReport]`. Task 3 relies on this signature remaining backward-compatible when `target` is omitted.

**Background the implementer needs.** Mediation subdivides an edge through a hidden spin whose coupling is `A = acosh(exp(2·β·|J|)) / (2·β)`, which is strictly greater than `|J|` for every nonzero `J`. So a model at the legal boundary becomes illegal after mediation. Measured today: `J = 5.7`, `β = 1.0` yields `max|J| = 6.0466` against a cap of `6.0`, and `insert_mediators` returns it without complaint. The gate that would catch it runs only in `search._try` and `preflight`, so anyone composing `insert_mediators → build_program` ships an illegal model.

`insert_mediators` currently takes no target, which is why it cannot check anything. The fix adds an optional one. When omitted, behaviour is byte-for-byte what it is today, so existing callers and fixtures are untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mediation_cap_door.py`:

```python
"""The mediated coupling cap must refuse at the door, not in a hallway.

`A = acosh(exp(2*beta*|J|)) / (2*beta) > |J|` strictly, so a model sitting at
the legal boundary is illegal once mediated. The check lived only in
`search._try` and `preflight`, which left every other composition of
`insert_mediators -> build_program` free to emit a model the hardware refuses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsu_compiler.failures import CompileError  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.route import insert_mediators, mediator_coupling  # noqa: E402
from tsu_compiler.preflight.model import IsingModel  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

Z1 = PROFILES["z1"]


def _triangle(J: float, beta: float = 1.0) -> IsingModel:
    """Odd cycle, so mediation is forced."""
    edges = ((0, 1), (1, 2), (0, 2))
    return IsingModel(
        nodes=("a", "b", "c"), edges=edges,
        weights=np.full(3, J), biases=np.zeros(3),
        beta=beta, offset=0.0,
    )


def test_the_premise_an_illegal_mediated_coupling_is_reachable():
    """Guard the test's own setup: J=5.7 must really cross the cap once
    mediated, or the rest of this file proves nothing."""
    A = mediator_coupling(5.7, 1.0)
    assert A > Z1.max_abs_coupling.value
    assert 5.7 <= Z1.max_abs_coupling.value, "pre-mediation it is legal"


def test_it_refuses_when_a_target_is_supplied():
    m = _triangle(5.7)
    with pytest.raises(CompileError) as e:
        insert_mediators(m, analyse(m), Z1)
    assert "coupling" in str(e.value).lower()


def test_the_refusal_names_both_numbers():
    m = _triangle(5.7)
    with pytest.raises(CompileError) as e:
        insert_mediators(m, analyse(m), Z1)
    text = str(e.value)
    assert "6.04" in text, "the measured mediated coupling"
    assert "6.0" in text, "the cap it broke"


def test_it_states_the_safe_pre_mediation_bound():
    """A refusal that does not say what WOULD work is a dead end. At beta=1
    with cap 6 the closed form gives |J| <= 0.5*ln(cosh(12)) ~= 5.653."""
    m = _triangle(5.7)
    with pytest.raises(CompileError) as e:
        insert_mediators(m, analyse(m), Z1)
    assert "5.65" in str(e.value)


def test_a_legal_model_still_mediates():
    """The guard must not refuse everything."""
    m = _triangle(1.0)
    med, rep = insert_mediators(m, analyse(m), Z1)
    assert rep.mediator_count > 0
    assert float(np.abs(med.weights).max()) <= Z1.max_abs_coupling.value


def test_omitting_the_target_preserves_the_old_behaviour():
    """Existing callers pass two arguments and must be unaffected, including
    the illegal case, which they were always free to produce."""
    m = _triangle(5.7)
    med, rep = insert_mediators(m, analyse(m))
    assert rep.mediator_count > 0
    assert float(np.abs(med.weights).max()) > Z1.max_abs_coupling.value


def test_an_already_bipartite_model_is_untouched_by_the_guard():
    """No mediation means no mediated coupling to check, whatever |J| is."""
    m = IsingModel(nodes=("a", "b"), edges=((0, 1),),
                   weights=np.array([6.0]), biases=np.zeros(2),
                   beta=1.0, offset=0.0)
    med, rep = insert_mediators(m, analyse(m), Z1)
    assert rep.mediator_count == 0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_mediation_cap_door.py -q`

Expected: `test_the_premise_...`, `test_omitting_the_target...` and `test_an_already_bipartite...` PASS. The four refusal tests FAIL — `insert_mediators() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Add the optional target and the refusal**

In `src/tsu_compiler/passes/route.py`, change the signature at line 171:

```python
def insert_mediators(ising: IsingModel, report: GraphReport,
                     target: TargetProfile | None = None
                     ) -> tuple[IsingModel, MediationReport]:
```

Add these imports at the top of the module if not already present:

```python
from ..failures import CompileError, GateFailure, Remediation
from ..target import TargetProfile
```

Immediately before the success return at line 281 (`return med_ising, MediationReport(`), insert:

```python
    # The cap, checked HERE rather than only in search._try and preflight.
    #
    # A = acosh(exp(2*beta*|J|)) / (2*beta) is strictly greater than |J| for
    # every nonzero J, so a model sitting at the legal boundary is illegal the
    # moment it is mediated. Measured before this guard existed: J=5.7 at
    # beta=1 returned max|J| = 6.0466 against a cap of 6.0, and this function
    # returned it without complaint. Any caller composing
    # `insert_mediators -> build_program` -- fixtures, audit scripts, a future
    # backend -- could therefore emit a model the hardware refuses.
    #
    # `target` is optional so existing two-argument callers keep the old
    # behaviour exactly. The guard is opt-in at the call sites that know a
    # target, which is every call site that is about to claim something.
    if target is not None and mediator_count:
        cap = target.max_abs_coupling.value
        peak = float(np.abs(med_ising.weights).max())
        if peak > cap:
            # What the author can actually do about it: the largest
            # pre-mediation coupling that survives, from the same closed form.
            safe = math.log(math.cosh(2.0 * beta * cap)) / (2.0 * beta)
            raise CompileError(
                f"mediation raises the peak coupling to {peak:.4f}, over the "
                f"{cap} cap",
                [GateFailure(
                    gate="mediated_coupling_cap",
                    cause=(f"mediating this graph raises |J| to {peak:.4f}, "
                           f"above the {cap} cap; the largest pre-mediation "
                           f"|J| that survives mediation at beta={beta} is "
                           f"{safe:.4f}"),
                    measured=peak, limit=cap,
                    assumed=target.is_assumed("max_abs_coupling"),
                    remediations=(
                        Remediation(
                            "lower the coupling",
                            f"keep every |J| at or below {safe:.4f} so "
                            f"mediation stays inside the cap"),
                        Remediation(
                            "avoid mediation",
                            "make the interaction graph bipartite, which "
                            "needs no hidden spins at all"),
                    ))])
```

Add `import math` to the module imports if it is not already there.

- [ ] **Step 4: Run the test file**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_mediation_cap_door.py -q`

Expected: 7 passed.

- [ ] **Step 5: Run the whole compiler suite**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests -q`

Expected: 1009 passed, 8 skipped. If anything else fails, a caller was passing a target positionally where it did not intend to — read the failure before changing it.

- [ ] **Step 6: Commit**

```bash
git add src/tsu_compiler/passes/route.py tests/test_mediation_cap_door.py
git commit -m "route: refuse an illegal mediated coupling at the door

A = acosh(exp(2*beta*|J|))/(2*beta) exceeds |J| strictly, so a model at
the legal boundary is illegal once mediated. Measured before this: J=5.7
at beta=1 returned max|J| 6.0466 against a cap of 6.0, and
insert_mediators returned it without complaint. The check lived only in
search._try and preflight, leaving every other composition of
insert_mediators -> build_program free to emit a model the hardware
refuses.

target is optional, so two-argument callers keep the old behaviour and
the guard is opt-in wherever a target is known. The refusal names the
measured value, the cap, and the largest pre-mediation |J| that survives
mediation, because a refusal that does not say what would work is a dead
end.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `two_colourable` as its own gate

**Files:**
- Modify: `src/tsu_compiler/gates.py:171-183` (beside the existing `colouring` check)
- Test: `tests/test_two_colourable_gate.py`

**Interfaces:**
- Consumes: `GraphReport.bipartite` from `passes/analyse.py`; `GateCheck(gate, passed, measured, limit, assumed)` from `gates.py:37`.
- Produces: a `GateCheck` named `two_colourable` in the tuple returned by `gate_checks`. Later workstreams render it in the gate table.

**Background the implementer needs.** Z1's block Gibbs updates two colour classes. The existing `colouring` gate asks only whether the DSATUR colouring is *proper* — no two adjacent nodes share a colour — which any number of colours satisfies. Measured today: a triangle has `bipartite=False`, needs three colours, and **passes** the `colouring` gate. That gate name invites the wrong theorem.

Both gates stay. `colouring` remains housekeeping; `two_colourable` is the one that answers whether block Gibbs has a schedule.

- [ ] **Step 1: Write the failing test**

Create `tests/test_two_colourable_gate.py`:

```python
"""Z1's block Gibbs has two colour classes, not k.

The existing `colouring` gate asks whether a DSATUR colouring is proper, which
any number of colours satisfies. A triangle passes it while needing three.
`two_colourable` is the gate that answers whether block Gibbs has a schedule
on the graph that will actually be sampled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsu_compiler.gates import gate_checks  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.route import insert_mediators  # noqa: E402
from tsu_compiler.preflight.model import IsingModel  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

Z1 = PROFILES["z1"]


def _model(edges, n, J=0.5):
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(n)), edges=tuple(edges),
        weights=np.full(len(edges), J), biases=np.zeros(n),
        beta=1.0, offset=0.0,
    )


def _by_gate(model):
    return {c.gate: c for c in gate_checks(model, analyse(model), Z1)}


def test_a_triangle_fails_two_colourable():
    m = _model([(0, 1), (1, 2), (0, 2)], 3)
    assert analyse(m).bipartite is False, "premise: a triangle is odd-cycled"
    assert _by_gate(m)["two_colourable"].passed is False


def test_the_old_colouring_gate_still_passes_the_triangle():
    """Both gates ship. This pins that they are genuinely different questions
    rather than one renamed, which is the whole point of adding the second."""
    m = _model([(0, 1), (1, 2), (0, 2)], 3)
    assert _by_gate(m)["colouring"].passed is True


def test_a_path_passes_two_colourable():
    m = _model([(0, 1), (1, 2)], 3)
    assert _by_gate(m)["two_colourable"].passed is True


def test_the_mediated_triangle_passes():
    """Mediation exists to make this true, and the gate must see it on the
    graph that will actually be sampled."""
    m = _model([(0, 1), (1, 2), (0, 2)], 3)
    med, rep = insert_mediators(m, analyse(m))
    assert rep.mediator_count > 0
    assert _by_gate(med)["two_colourable"].passed is True


def test_the_gate_reports_what_it_measured():
    m = _model([(0, 1), (1, 2), (0, 2)], 3)
    g = _by_gate(m)["two_colourable"]
    assert g.limit == 2
    assert g.measured is not None, "say how many colours it needs"
    assert int(g.measured) >= 3


def test_it_is_sourced_not_assumed():
    """Two-colour chromatic block Gibbs is documented for Z1, so a refusal on
    this gate is not resting on one of this project's own guesses."""
    m = _model([(0, 1), (1, 2), (0, 2)], 3)
    assert _by_gate(m)["two_colourable"].assumed is False
```

- [ ] **Step 2: Run it and watch it fail**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_two_colourable_gate.py -q`

Expected: FAIL with `KeyError: 'two_colourable'` on five tests. `test_the_old_colouring_gate_still_passes_the_triangle` passes already — that is the measured defect this task exists to complement, not to remove.

- [ ] **Step 3: Add the gate**

In `src/tsu_compiler/gates.py`, immediately after the existing `colouring` block that ends at line 183 (the `fails.append(GateFailure(... gate="colouring" ...))` clause) and before `return tuple(checks), tuple(fails)`, insert:

```python
    # Z1's block Gibbs updates TWO colour classes. The `colouring` gate above
    # asks only whether the DSATUR colouring is PROPER, which any number of
    # colours satisfies -- a triangle passes it while needing three. That is
    # housekeeping, not the hardware's question.
    #
    # This gate asks the hardware's question: is the graph that will actually
    # be sampled 2-colourable? It fails on an odd-cycled graph pre-mediation
    # and passes on the mediated graph, which is exactly what mediation is
    # for. Both gates ship, because they are different questions and a reader
    # is entitled to see which one refused.
    n_colours = (len(set(report.colouring.values()))
                 if report.colouring else 0)
    checks.append(GateCheck("two_colourable", bool(report.bipartite),
                            n_colours, 2, False))
    if not report.bipartite:
        fails.append(GateFailure(
            gate="two_colourable",
            cause=(f"this graph is not 2-colourable -- it needs {n_colours} "
                   f"colour blocks and chromatic block Gibbs has two, so "
                   f"there is no schedule for it as written"),
            measured=n_colours, limit=2, assumed=False,
            remediations=(
                Remediation(
                    "mediate the odd cycles",
                    "inserting a hidden spin into an edge inside one colour "
                    "block makes the graph bipartite while preserving the "
                    "marginal distribution over the original spins"),
            )))
```

If `Remediation` is not already imported in `gates.py`, add it to the existing import from `.failures`.

- [ ] **Step 4: Run the test file**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_two_colourable_gate.py -q`

Expected: 6 passed.

- [ ] **Step 5: Run the whole compiler suite**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests -q`

Expected: some tests that count gates or assert on a full gate list will now see one more gate. That is a real contract change. Update those assertions to include `two_colourable` rather than deleting them, and read each one to confirm the new expectation is what the test meant to pin.

- [ ] **Step 6: Commit**

```bash
git add src/tsu_compiler/gates.py tests/test_two_colourable_gate.py
git commit -m "gates: two_colourable is a different question from colouring

Z1's block Gibbs updates two colour classes. The colouring gate asks only
whether a DSATUR colouring is proper, which any number of colours
satisfies -- measured, a triangle has bipartite=False, needs three
colours, and passes that gate. The name invited the wrong theorem.

Both gates now ship. colouring stays as housekeeping; two_colourable
answers whether block Gibbs has a schedule on the graph that will
actually be sampled, failing pre-mediation on an odd cycle and passing
after, which is what mediation is for.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `tsuc compile` falls back to chains, like `preflight` already does

**Files:**
- Modify: `src/tsu_compiler/passes/search.py`, the `place()` call and its `except CompileError` at line 271
- Test: `tests/test_compile_uses_chains.py`

**Interfaces:**
- Consumes: `find_chain_embedding(ising, target, *, host_side=None, seed=1, timeout=600.0) -> ChainEmbedding` and `EmbeddingUnavailable` from `tsu_compiler.passes.embed`; `ChainEmbedding` exposes `n_physical: int`, `max_chain: int`, `mean_chain: float`, `chains: dict`.
- Produces: `Compilation` gains chain fields so receipts can report them. `preflight` already sets `chain_cells`, `chain_max`, `chain_mean`; use those names.

**Background the implementer needs.** `preflight/check.py:351` already tries chain embedding when direct placement fails and no gate has failed. `compile_spec` does not, so the same model gets two different seating answers from two doors — which is its own integrity defect and the thing this task closes. Copy the shape of the `preflight` logic, including the guard: **chains are never attempted when a gate has already failed**, because reporting a placement for a model the hardware refuses invites exactly the misreading this project exists to prevent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compile_uses_chains.py`:

```python
"""compile and preflight must give the same seating answer.

`preflight` falls back to chain embedding when direct placement exhausts.
`compile_spec` did not, so the same model got two different answers from two
doors. Two doors with two answers is an integrity defect regardless of which
one is right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("minorminer", reason="chain embedding needs minorminer")

from tsu_compiler.preflight.check import preflight  # noqa: E402
from tsu_compiler.preflight.model import IsingModel  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

Z1 = PROFILES["z1"]


def _hard_model():
    """A 4-regular expander: every gate green, and not a lattice subgraph, so
    direct placement exhausts. Verified as the premise in the first test."""
    import networkx as nx
    G = nx.random_regular_graph(4, 40, seed=7)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in G.edges()))
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(40)), edges=edges,
        weights=np.full(len(edges), 0.5), biases=np.zeros(40),
        beta=1.0, offset=0.0,
    )


def test_the_premise_direct_placement_really_fails_on_this_model():
    r = preflight(_hard_model(), restarts=1, iters=200, allow_chains=False)
    assert r.placed is False
    assert not [g for g in r.gates if g.status == "fail"], (
        "premise: no gate refused it, so this is a seating question only")


def test_preflight_places_it_with_chains():
    r = preflight(_hard_model(), restarts=1, iters=200)
    assert r.placed is True
    assert r.embedding == "chain"


def test_compile_places_it_too():
    """The point of the task. Before this change compile returned a
    placement failure for a model preflight placed in under a second."""
    from tsu_compiler.passes.search import compile_ising
    comp = compile_ising(_hard_model(), Z1, restarts=1, iters=200)
    assert comp.verdict != "HARDWARE", "no gate refused this model"
    assert comp.placement is not None or comp.chain_cells is not None


def test_compile_reports_the_chain_cost():
    from tsu_compiler.passes.search import compile_ising
    comp = compile_ising(_hard_model(), Z1, restarts=1, iters=200)
    assert comp.chain_cells is not None and comp.chain_cells >= 40
    assert comp.chain_max is not None and comp.chain_max >= 1


def test_a_gate_failure_still_blocks_chains():
    """Chains fix high logical degree, which is one of the things minor
    embedding is FOR. But a model whose gates fail does not fit this target,
    and reporting a placement beside a refusal invites the exact misreading
    this project exists to prevent."""
    from tsu_compiler.passes.search import compile_ising
    n = 24
    edges = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    dense = IsingModel(
        nodes=tuple(f"v{i}" for i in range(n)), edges=edges,
        weights=np.full(len(edges), 0.5), biases=np.zeros(n),
        beta=1.0, offset=0.0,
    )
    comp = compile_ising(dense, Z1)
    assert comp.verdict == "HARDWARE"
    assert comp.chain_cells is None


def test_a_lattice_shaped_model_does_not_pay_for_chains():
    """Direct placement stays preferred. A model that seats one-spin-one-cell
    must not spend 2x the p-bits on reach it does not need."""
    from tsu_compiler.passes.search import compile_ising
    edges = tuple((i, i + 1) for i in range(7))
    path = IsingModel(
        nodes=tuple(f"v{i}" for i in range(8)), edges=edges,
        weights=np.full(len(edges), 0.5), biases=np.zeros(8),
        beta=1.0, offset=0.0,
    )
    comp = compile_ising(path, Z1)
    assert comp.verdict == "COMPILED"
    assert comp.chain_cells is None
```

- [ ] **Step 2: Run it and watch it fail**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_compile_uses_chains.py -q`

Expected: the first two PASS (they exercise `preflight`, already wired). The four `compile` tests FAIL — either on a missing `compile_ising` entry point or on `chain_cells` not existing on `Compilation`.

**If `compile_ising` does not exist:** `search.py` exposes `compile_spec(spec, target, ...)` which takes a `WorkloadSpec`, not an `IsingModel`. Read `search.py` and either add a thin `compile_ising(ising, target, **kw)` wrapper around the same internals, or rewrite these four tests to build a `WorkloadSpec`. Prefer the wrapper: the tests want to pin seating behaviour on a graph, and routing that through spec construction adds an encoding step that is not what is under test.

- [ ] **Step 3: Add the chain fallback and the fields**

Add three fields to the `Compilation` dataclass at `src/tsu_compiler/passes/search.py:75`, **after the existing defaulted fields** (the dataclass already has `allow_assumed`, `gate_checks`, `program`, `encoded`, `regime`, `placement`, `verification`, `clamp`, `sample` with defaults, so these go at the end):

Note there is no `placed` field on `Compilation` — placement is recorded as `placement: Any = None`. The tests above assert on `verdict` and `chain_cells` for that reason.

```python
    chain_cells: int | None = None
    chain_max: int | None = None
    chain_mean: float | None = None
```

In the `except CompileError` branch at line 271, after the existing failure bookkeeping and before the failure `Compilation` is returned, insert:

```python
        # Direct placement seats one logical spin on one p-bit, which R33
        # measures as only workable when the graph is already a subgraph of
        # the lattice: past four placed neighbours there is typically ONE
        # legal cell anywhere. Chains dissolve that at the cost of extra
        # p-bits, so they are the FALLBACK -- a model that seats directly
        # must not pay 2x the spins for reach it does not need.
        #
        # NOT attempted when a gate has already failed. Chains genuinely fix
        # high logical degree, but a model whose gates fail does not fit this
        # target, and reporting a placement beside a refusal invites exactly
        # the misreading this project exists to prevent.
        chain = None
        if not any(c.gate for c in checks if not c.passed):
            try:
                from .embed import EmbeddingUnavailable, find_chain_embedding
                chain = find_chain_embedding(ising, target)
            except EmbeddingUnavailable:
                chain = None
            except Exception:       # noqa: BLE001 - no embedding found
                chain = None
```

Then, where the failure `Compilation` is built, when `chain is not None`, return a placed compilation instead, carrying `chain_cells=chain.n_physical`, `chain_max=chain.max_chain`, `chain_mean=round(chain.mean_chain, 3)`, and a verdict that is not `HARDWARE`.

**Read the surrounding code before writing this.** `search.py`'s verdict logic has several branches (`COMPILED`, `EFFORT`, `HARDWARE`, `COMPILER_ERROR`) and this fallback must slot into the `EFFORT` path only. Do not change what `HARDWARE` means.

- [ ] **Step 4: Run the test file**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_compile_uses_chains.py -q`

Expected: 6 passed.

- [ ] **Step 5: Run the whole compiler suite**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests -q`

Expected: all green. Tests that assert a model reports `EFFORT` may now report a placement instead — that is the intended change, but read each one and confirm the model really is gate-clean before updating the expectation.

- [ ] **Step 6: Commit**

```bash
git add src/tsu_compiler/passes/search.py tests/test_compile_uses_chains.py
git commit -m "search: compile falls back to chains, like preflight already did

preflight tries chain embedding when direct placement exhausts;
compile_spec did not, so the same model got two different seating
answers from two doors. Two doors with two answers is an integrity
defect regardless of which one is right.

Direct placement stays preferred, so a lattice-shaped model does not pay
2x the p-bits for reach it does not need, and chains are never attempted
when a gate has already failed -- chains do fix high logical degree, but
reporting a placement beside a hardware refusal invites exactly the
misreading this project exists to prevent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Count coupling values with their sign

**Files:**
- Modify: `src/tsu_compiler/preflight/check.py:140` (inside `_coupling_note`)
- Test: `tests/test_signed_coupling_count.py`

**Interfaces:**
- Consumes: `IsingModel.weights` (numpy array); `TargetProfile.coupling_parameters`.
- Produces: `_coupling_note(ising, target) -> tuple[int | None, str | None]` with the same signature. Only the counting rule and the note's wording change.

**Background the implementer needs.** The current line is:

```python
distinct = int(np.unique(np.round(np.abs(np.asarray(ising.weights, dtype=float)), 9)).size)
```

`np.abs` folds `+J` and `−J` into one value, while the note that reports the number calls them "distinct coupling values". A ferromagnetic and an antiferromagnetic edge are different programs on an Ising machine; sign is not optional. Either the count or the label is wrong, and the count is the one to fix.

This changes a reported number. It does not change any pass/fail outcome unless a model sits exactly at the parameter budget, which no shipped model does.

- [ ] **Step 1: Write the failing test**

Create `tests/test_signed_coupling_count.py`:

```python
"""+J and -J are different programs on an Ising machine.

The distinct-coupling count folded sign through np.abs while the note
reporting it said "distinct coupling values". Sign is not optional on an
Ising edge, so the count was wrong for its own label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsu_compiler.preflight.check import _coupling_note  # noqa: E402
from tsu_compiler.preflight.model import IsingModel  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

Z1 = PROFILES["z1"]


def _model(weights):
    w = np.asarray(weights, dtype=float)
    edges = tuple((i, i + 1) for i in range(len(w)))
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(len(w) + 1)), edges=edges,
        weights=w, biases=np.zeros(len(w) + 1), beta=1.0, offset=0.0,
    )


def test_opposite_signs_count_as_two():
    distinct, _ = _coupling_note(_model([2.5, -2.5]), Z1)
    assert distinct == 2


def test_identical_values_still_count_as_one():
    distinct, _ = _coupling_note(_model([2.5, 2.5, 2.5]), Z1)
    assert distinct == 1


def test_a_ferro_and_antiferro_mix_is_counted_fully():
    distinct, _ = _coupling_note(_model([1.0, -1.0, 2.0, -2.0, 1.0]), Z1)
    assert distinct == 4


def test_the_note_no_longer_says_magnitudes_when_it_counts_signed_values():
    _, note = _coupling_note(_model([2.5, -2.5]), Z1)
    assert note is not None
    assert "signed" in note.lower()


def test_the_unpublished_mapping_caveat_survives():
    """The count was never the binding question; the sharing map is, and it
    is unpublished. That sentence must not be lost in this change."""
    _, note = _coupling_note(_model([2.5, -2.5]), Z1)
    assert "unpublished" in note.lower()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_signed_coupling_count.py -q`

Expected: `test_opposite_signs_count_as_two` FAILS with `assert 1 == 2`. `test_a_ferro_and_antiferro_mix_is_counted_fully` FAILS with `assert 2 == 4`. `test_the_note_no_longer_says_magnitudes...` FAILS on the missing word.

- [ ] **Step 3: Count with sign**

In `src/tsu_compiler/preflight/check.py`, replace line 140-141:

```python
    distinct = int(np.unique(np.round(np.asarray(ising.weights,
                                                 dtype=float), 9)).size)
```

Note the removal of `np.abs`. Then update the note's first clause so the label matches what is counted:

```python
    note = (f"this model needs {distinct:,} distinct SIGNED coupling values "
            f"against {int(budget):,} programmable parameters on the die, so "
            f"the COUNT is not binding. The MAPPING is unmodelled: "
            f"~{2135904/budget:.1f} edges share each parameter on Z1 and the "
            f"sharing structure is unpublished, so whether these values can "
            f"be ASSIGNED is not checked here "
            f"(target.per_edge_independent_J is an assumption).")
```

Leave the `distinct > budget` branch's wording alone apart from inserting `signed` the same way.

- [ ] **Step 4: Run the test file**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests/test_signed_coupling_count.py -q`

Expected: 5 passed.

- [ ] **Step 5: Run the whole compiler suite**

Run: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests -q`

Expected: any test asserting a specific distinct-coupling number on a mixed-sign model will now see a larger number. Update those to the signed count.

- [ ] **Step 6: Commit**

```bash
git add src/tsu_compiler/preflight/check.py tests/test_signed_coupling_count.py
git commit -m "preflight: count coupling values with their sign

np.abs folded +J and -J into one value while the note reporting the
number called them distinct coupling values. A ferromagnetic and an
antiferromagnetic edge are different programs on an Ising machine, so
either the count or the label was wrong, and the count is the one to
fix.

This changes a reported number and no pass/fail outcome: the count was
never the binding question. The unpublished sharing map is, and that
caveat is unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** A1 → Task 1. A2 (the near-cap warning) is folded into Task 1: the refusal states the safe pre-mediation bound `5.653`, which is the actionable half of the warning, and `preflight` already emits `mediated_coupling_cap` from the same closed form before placement runs, so a separate warning path would be a third place to keep in sync. A3 → Task 2. A4 → Task 3. A5 verified already done and documented as dropped, with the three call sites named so a future reader does not re-open it. A6 → Task 4.

**Placeholder scan.** No TBD or TODO. Task 3 Step 2 and Step 3 carry explicit "read the surrounding code first" instructions rather than pretending the `search.py` verdict branches are simpler than they are — that is a real fork the implementer must resolve against the code, and the instruction names both acceptable resolutions.

**Type consistency.** `chain_cells` / `chain_max` / `chain_mean` match the names `preflight/check.py` already uses. `insert_mediators(ising, report, target=None)` keeps the two-argument form Task 3's fallback path does not use. `GateCheck(gate, passed, measured, limit, assumed)` matches `gates.py:37`.

## Subsequent plans

These are separate plans, written after A lands, because each produces working testable software on its own:

- **D — solutions readout.** `tsuc solve`: sample, decode, evaluate codeword and task validity as independent columns, dedupe, rank by energy, write `solutions.json` and `solutions.csv`, refuse to write without the codeword violation rate.
- **B — slim artifact.** `tsuc check`, the frozen `verdict.json` schema, `RECEIPT.md`, `z1-profile` naming with `z1` as an alias, three canned examples with pinned headlines, `LIMITS_OF_THIS_TOOL.md`.
- **C — gap report.** `tsuc check --why`, the four-blocker taxonomy, never a hardware recommendation.
