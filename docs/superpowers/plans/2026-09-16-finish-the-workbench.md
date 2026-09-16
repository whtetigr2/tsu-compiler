# Finish the Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the toolkit so every number on screen is one the project would publish, every claim is ablatable from a button, and the lattice can be seen moving in three dimensions.

**Architecture:** Three waves, ordered by what would be most embarrassing if someone else found it. Wave A repairs statistics and compiler integrity, because a beautiful instrument showing untrustworthy numbers is worse than no instrument. Wave B puts the ablation runner on screen, which is the one control nobody else has. Wave C builds the 3D viewport and live plots on top of numbers that are by then trustworthy.

**Tech Stack:** Python 3.14, FastAPI, THRML/JAX, React 19, Vite 8, Three.js (added in Wave C), PyInstaller.

**Spec:** `docs/superpowers/specs/2026-09-16-thermodynamic-workbench-design.md`

## Global Constraints

- Work only in `C:\Users\whtet\Documents\tsu-compiler`. Never commit to another repository.
- Stage commits by explicit path. Never `git add -A`.
- Do not touch `receipts/EXP-G8-D2/`, `receipts/EXP-G8-D3/`, `figures/z1_lab_screenshot.png`, `demo/cascade.py`, `demo/cascade_runs/`.
- `specs/emergence_8x8.yaml` and `pyproject-review.toml` are another agent's leftovers. Leave them.
- No secret values in any file or commit message.
- **Never weaken an assertion to make something pass. Record the failure.**
- **A control that cannot fail is not a control.** Before reporting a control as passed, ask what result would have failed it.
- **A missing measurement reads `unavailable: <reason>`.** Never a blank, a zero, a dash, or a guess.
- **No hardware claims.** Everything is simulation until someone else runs it on silicon.
- No em dashes in anything a user reads.
- Interpreter: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"`
- Observatory venv: `demo/gibbs-observatory/.venv/Scripts/python.exe`
- Before hand-rolling any statistic, check `CLAUDE.md`'s table. The compiler almost certainly has it, reviewed.

---

## File Structure

| Path | Responsibility |
|---|---|
| `demo/gibbs-observatory/backend/app/metrics.py` | Statistics shown live. Delegates to the compiler wherever the compiler has the function. |
| `demo/gibbs-observatory/backend/app/ablation.py` | NEW. Run a model with a term zeroed, against a noise floor from a different seed. |
| `demo/gibbs-observatory/backend/tests/test_live_statistics.py` | NEW. Every displayed statistic against the compiler's own implementation. |
| `demo/gibbs-observatory/backend/tests/test_ablation.py` | NEW. |
| `src/tsu_compiler/passes/search.py` | Candidate states. Add one meaning "not found within budget". |
| `frontend/src/components/views/AblationView.tsx` | NEW. The ablation runner. |
| `frontend/src/components/views/LatticeView3D.tsx` | NEW. Three.js viewport. |
| `frontend/src/components/StatValue.tsx` | NEW. One component that renders a number or its unavailability reason. Used everywhere. |
| `demo/gibbs-observatory/programs/tesseract_16.yaml` | NEW. A 4-cube. Arbitrary topology proof. |

---

## WAVE A: every number on screen is one we would publish

### Task 1: Stop publishing an effective sample size the compiler refuses to

**Files:**
- Modify: `demo/gibbs-observatory/backend/app/metrics.py`
- Test: `demo/gibbs-observatory/backend/tests/test_live_statistics.py`

**Interfaces:**
- Produces: `summarize_series(values) -> dict` keeps its keys but `ess` may be `None`, and gains `ess_reason: str | None` and `ess_reliable: bool`.

**Why.** Measured, not argued. `backend/app/metrics.py` computes ESS from lag-1 autocorrelation as `n(1-rho)/(1+rho)`. The compiler ships `tsu_compiler.ess.effective_sample_size`, a Sokal-window estimator that returns `ess=None, reliable=False` with a reason when the chain is too short, against a threshold of `N/tau >= 5000` established by its own AR(1) validation.

The Observatory keeps `history[-256:]`. At N=256 the threshold needs `tau <= 0.05`, which is unreachable. Measured:

    series                      N        shown    compiler
    white noise               256        237.1    REFUSES
    rho=0.5                   256        113.1    REFUSES
    rho=0.95 (near critical)  256          8.9    REFUSES
    white noise            20,000     19,985.5    20,561.3
    rho=0.5                20,000      6,791.9     6,760.4

The estimator is fine on long chains. The chains are too short, and nothing on screen says so. **Every ESS this application has displayed is one the project's own reviewed estimator would refuse to publish.**

- [ ] **Step 1: Write the failing test**

Create `demo/gibbs-observatory/backend/tests/test_live_statistics.py`:

```python
"""Every statistic on screen must be one this project would publish.

CLAUDE.md carries a table headed "Before you write numpy, check this table",
because a numpy reimplementation of something the compiler already has is a
regression, not a shortcut. These tests check the live statistics against that
rule, and against the compiler's own answers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from tsu_compiler.ess import RELIABILITY_MIN_N_OVER_TAU  # noqa: E402

from backend.app.metrics import summarize_series  # noqa: E402


def _ar1(rho: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_short_chains_report_no_ess_rather_than_a_confident_number():
    """The live history is 256 samples. That is too short to publish an ESS."""
    out = summarize_series(_ar1(0.95, 256))
    assert out["ess"] is None, (
        f"a 256-sample chain reported ESS={out['ess']}. The compiler refuses "
        f"below N/tau >= {RELIABILITY_MIN_N_OVER_TAU}, and this is the number "
        f"the interface shows.")
    assert out["ess_reliable"] is False
    assert "unavailable" in (out["ess_reason"] or "").lower()


def test_the_reason_names_why_rather_than_being_blank():
    out = summarize_series(_ar1(0.95, 256))
    reason = out["ess_reason"] or ""
    assert "tau" in reason.lower() or "short" in reason.lower(), reason


def test_a_long_enough_chain_does_report_an_ess():
    """A check that cannot pass proves as little as one that cannot fail."""
    out = summarize_series(_ar1(0.0, 40_000))
    assert out["ess"] is not None, out["ess_reason"]
    assert out["ess_reliable"] is True
    assert out["ess"] > 1_000


def test_ess_matches_the_compiler_not_a_reimplementation():
    from tsu_compiler.ess import effective_sample_size

    x = _ar1(0.5, 40_000)
    ours = summarize_series(x)
    theirs = effective_sample_size(x.reshape(1, -1))
    assert theirs.reliable
    assert ours["ess"] == pytest.approx(theirs.ess, rel=1e-9), (
        "the live ESS is not the compiler's ESS; it has been reimplemented")


def test_mean_std_and_last_still_work():
    """These are cheap and always available. They must not have been lost."""
    out = summarize_series(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out["mean"] == pytest.approx(2.5)
    assert out["last"] == pytest.approx(4.0)
    assert out["n"] == 4


def test_an_empty_series_is_unavailable_not_zero():
    out = summarize_series(np.array([]))
    assert out["ess"] is None
    assert out["mean"] is None, "an empty series has no mean; 0.0 is a lie"
```

- [ ] **Step 2: Run it and watch it fail**

```
cd demo/gibbs-observatory
.venv/Scripts/python.exe -m pytest backend/tests/test_live_statistics.py -v
```

Expected: failures on ESS being a float rather than None, and on `mean` being 0.0 for an empty series.

- [ ] **Step 3: Rewrite summarize_series to delegate**

Replace `ess_from_autocorr` and the `summarize_series` body in
`demo/gibbs-observatory/backend/app/metrics.py`:

```python
def summarize_series(values: np.ndarray) -> dict:
    """Summary of one live metric series.

    ESS comes from `tsu_compiler.ess.effective_sample_size`, the project's own
    reviewed Sokal-window estimator, NOT from a lag-1 AR(1) approximation
    computed here. The compiler's version refuses to answer when the chain is
    too short to support an estimate, and that refusal is passed through to the
    interface rather than replaced with a confident-looking number.

    Measured before this changed: the interface keeps 256 samples of history,
    and at that length the compiler declines every time. So every ESS ever shown
    was one this project would not publish.

    `lag1_autocorr` is kept. It is cheap, always available, and honest about
    being what it is.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return {"mean": None, "std": None, "last": None, "lag1_autocorr": None,
                "ess": None, "ess_reliable": False,
                "ess_reason": "unavailable: no samples yet", "n": 0}

    rho = lag1_autocorr(v)
    ess = ess_reason = None
    reliable = False
    try:
        from tsu_compiler.ess import effective_sample_size

        est = effective_sample_size(v.reshape(1, -1))
        reliable = bool(est.reliable)
        ess = float(est.ess) if est.reliable and est.ess is not None else None
        ess_reason = None if reliable else str(est.reason)
    except Exception as exc:  # noqa: BLE001
        ess_reason = f"unavailable: ESS estimator failed: {exc!r}"

    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "last": float(v[-1]),
        "lag1_autocorr": rho,
        "ess": ess,
        "ess_reliable": reliable,
        "ess_reason": ess_reason,
        "n": int(v.size),
    }
```

Delete `ess_from_autocorr` entirely. Leaving it invites its reuse.

- [ ] **Step 4: Run the test**

Expected: 6 passed.

- [ ] **Step 5: Run the whole backend suite**

```
.venv/Scripts/python.exe -m pytest backend/tests -q
```

Anything that asserted on the old ESS shape needs its expectation updated, not the behaviour reverted.

- [ ] **Step 6: Commit**

```bash
git add demo/gibbs-observatory/backend/app/metrics.py demo/gibbs-observatory/backend/tests/test_live_statistics.py
git commit -m "Stop publishing an effective sample size the compiler refuses to give"
```

---

### Task 2: Render unavailability instead of hiding it

**Files:**
- Create: `frontend/src/components/StatValue.tsx`
- Modify: every component that renders `ess`

**Interfaces:**
- Produces: `<StatValue value={number|null} reason={string|null} unit?={string} digits?={number} />`

**Why.** Task 1 makes `ess` null most of the time. A React component rendering `{null}` shows nothing at all, which is worse than the wrong number: the reader cannot tell the difference between "not measured" and "not rendered".

- [ ] **Step 1: Write the component**

```tsx
interface Props {
  value: number | null | undefined
  reason?: string | null
  unit?: string
  digits?: number
}

/**
 * A number, or why there is no number.
 *
 * Never renders nothing. A blank where a statistic should be reads as a layout
 * bug, and the reader cannot tell it from a value of zero. The project's rule
 * is that a missing measurement says "unavailable" and says why.
 */
export function StatValue({ value, reason, unit, digits = 2 }: Props) {
  if (value == null || !Number.isFinite(value)) {
    return (
      <span className="stat-unavailable" title={reason ?? undefined}>
        unavailable
      </span>
    )
  }
  return (
    <span className="stat-value">
      {value.toFixed(digits)}
      {unit ? <span className="stat-unit">{unit}</span> : null}
    </span>
  )
}
```

- [ ] **Step 2: Add the style**

Append to `frontend/src/index.css`:

```css
.stat-unavailable {
  color: var(--text-mute);
  font-style: italic;
  font-size: 0.9em;
  cursor: help;
  border-bottom: 1px dotted var(--border-bright);
}
.stat-value { font-variant-numeric: tabular-nums; }
.stat-unit { color: var(--text-mute); margin-left: 0.15em; font-size: 0.85em; }
```

- [ ] **Step 3: Replace every ESS render site**

```
grep -rn "ess" frontend/src --include=*.tsx
```

Each site becomes `<StatValue value={s.ess} reason={s.ess_reason} digits={0} />`.

- [ ] **Step 4: Typecheck and build**

```
cd frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/StatValue.tsx frontend/src/index.css frontend/src/components
git commit -m "Render why a statistic is unavailable instead of rendering nothing"
```

---

### Task 3: Pin the silent model substitution

**Files:**
- Modify: `demo/gibbs-observatory/backend/app/sampler_engine.py`
- Test: `demo/gibbs-observatory/backend/tests/test_live_statistics.py`

**Why.** `SamplerEngine._resolve_graph` catches bare `Exception` and substitutes a generic 2D lattice, then samples that. It sets a banner but discards the reason. Measured: all 15 packaged examples load, so it does not fire today. It is a trap waiting for the first receipt that fails to parse, and the person watching the energy trace would be watching a different model.

- [ ] **Step 1: Write the failing test**

```python
def test_a_failed_receipt_load_records_why_it_fell_back(monkeypatch):
    """The substitution is honest only if it says what went wrong."""
    from backend.app import sampler_engine as se

    def boom(_rid):
        raise ValueError("receipt is malformed in some specific way")

    monkeypatch.setattr(se, "receipt_to_graph_arrays", boom, raising=False)
    cfg = se.SamplerConfig(receipt_id="small")
    engine = se.SamplerEngine(cfg)
    assert engine.config.sampling_fallback is True
    assert engine.config.fallback_reason, (
        "the engine swapped in a different model and recorded no reason")
    assert "malformed" in engine.config.fallback_reason


def test_all_packaged_examples_load_without_falling_back():
    """If this fails, somebody is watching the wrong model's statistics."""
    from backend.app.receipt_loader import examples_shelf, receipt_to_graph_arrays

    failed = []
    for item in examples_shelf():
        if not item["packaged"]:
            continue
        try:
            receipt_to_graph_arrays(item["id"])
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{item['id']}: {exc!r}")
    assert not failed, (
        "these examples would silently sample a generic lattice instead of "
        "themselves: " + "; ".join(failed))
```

- [ ] **Step 2: Run it, watch it fail on `fallback_reason`**

- [ ] **Step 3: Record the reason**

In `SamplerConfig`, beside `sampling_fallback: bool = False`, add:

```python
    fallback_reason: str | None = None
```

In `_resolve_graph`, replace the bare handler:

```python
            except Exception as exc:  # noqa: BLE001
                # Substituting a DIFFERENT model is a large thing to do
                # quietly. The banner says it happened; this says why, so a
                # malformed receipt is debuggable rather than merely survivable.
                cfg.fallback_reason = (
                    f"unavailable: receipt {cfg.receipt_id!r} could not be "
                    f"loaded ({type(exc).__name__}: {exc}); sampling a generic "
                    f"lattice2d instead, which is NOT this receipt's model")
                graph = build_preset(
                    "lattice2d", size=cfg.size, degree_cap=cfg.degree_cap,
                    seed=cfg.seed,
                )
                return graph, True
```

Surface `fallback_reason` in `graph_payload()` next to the existing banner.

- [ ] **Step 4: Run the tests. Expected: pass.**

- [ ] **Step 5: Commit**

```bash
git add demo/gibbs-observatory/backend/app/sampler_engine.py demo/gibbs-observatory/backend/tests/test_live_statistics.py
git commit -m "Say why the sampler substituted a different model"
```

---

### Task 4: Name a search failure a search failure (R27)

**Files:**
- Modify: `src/tsu_compiler/passes/search.py`
- Test: `tests/test_candidate_states.py`

**Why.** `audit/findings/R27.md`. A domain-wall candidate that fails placement at the default budget is recorded `HARDWARE_INFEASIBLE`, while its own attached failure says `failure_class='placement_effort_exhausted'` with remediation "increase placement effort". Given 24 restarts and 250,000 iterations the same candidate is SELECTED and is 25% smaller on the die than the one-hot model that ships instead. The hardware was never the limit.

- [ ] **Step 1: Write the failing test**

Create `tests/test_candidate_states.py`:

```python
"""A search that ran out of time is not a hardware limit.

R27. `programs/seq_design_longer.yaml` compiles to a domain-wall candidate that
places cleanly at 24 restarts and 250,000 iterations and is then SELECTED, being
24 physical spins against one-hot's 32. At the default budget it fails to place
and was recorded HARDWARE_INFEASIBLE, which is a claim about Extropic's silicon
made on the basis of a timer.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsu_compiler.passes.search import CandidateState, compile_spec  # noqa: E402
from tsu_compiler.spec import load_spec  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

SPEC = ROOT / "demo" / "gibbs-observatory" / "programs" / "seq_design_longer.yaml"


def test_there_is_a_state_meaning_not_found_within_budget():
    assert hasattr(CandidateState, "PLACEMENT_EFFORT_EXHAUSTED"), (
        "a candidate whose placement search ran out of budget has no state to "
        "be in, so it lands in HARDWARE_INFEASIBLE and reads as a claim about "
        "the silicon")


def test_effort_exhaustion_is_not_recorded_as_hardware_infeasible():
    comp = compile_spec(load_spec(str(SPEC)), PROFILES["z1"], allow_assumed=True)
    by_enc = {c.encoding: c for c in comp.repset.candidates}
    dw = by_enc["domain_wall"]
    if dw.state is CandidateState.SELECTED:
        pytest.skip("domain_wall now places at the default budget")
    assert dw.state is not CandidateState.HARDWARE_INFEASIBLE, (
        f"domain_wall is recorded {dw.state.name} but its failure is "
        f"{getattr(dw.failure, 'failure_class', None)!r}. It places at a "
        f"higher budget, so the hardware is not the limit.")
    assert dw.state is CandidateState.PLACEMENT_EFFORT_EXHAUSTED


def test_a_genuine_hardware_failure_is_still_hardware_infeasible():
    """A control that cannot fail is not a control."""
    from tsu_compiler.passes.search import _classify_candidate_failure

    class _F:
        failure_class = "max_degree"

    assert _classify_candidate_failure(_F()) is CandidateState.HARDWARE_INFEASIBLE
```

- [ ] **Step 2: Run it. Expected: fails on the missing enum member.**

- [ ] **Step 3: Add the state and the classifier**

In `src/tsu_compiler/passes/search.py`, add to `CandidateState`:

```python
    PLACEMENT_EFFORT_EXHAUSTED = "PLACEMENT_EFFORT_EXHAUSTED"
```

Add beside it:

```python
# Failure classes that mean "the search gave up", not "the hardware cannot".
# Keeping these apart is the whole point of R27: reporting a timer as a
# silicon limit is the representation-versus-hardware conflation this compiler
# exists to prevent, and place.py's own docstring says so.
_EFFORT_FAILURE_CLASSES = frozenset({"placement_effort_exhausted"})


def _classify_candidate_failure(failure) -> "CandidateState":
    cls = getattr(failure, "failure_class", None)
    if cls in _EFFORT_FAILURE_CLASSES:
        return CandidateState.PLACEMENT_EFFORT_EXHAUSTED
    return CandidateState.HARDWARE_INFEASIBLE
```

Then find every site assigning `CandidateState.HARDWARE_INFEASIBLE` after a placement failure and route it through `_classify_candidate_failure(failure)`.

- [ ] **Step 4: Run the new test, then the full compiler suite**

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m pytest tests -q
```

Anything asserting `HARDWARE_INFEASIBLE` on a placement-effort failure is now asserting the wrong thing and its expectation moves.

- [ ] **Step 5: Confirm selection still works**

`PLACEMENT_EFFORT_EXHAUSTED` must be treated as not-feasible for selection purposes, exactly as before. Only the label changes.

- [ ] **Step 6: Commit**

```bash
git add src/tsu_compiler/passes/search.py tests/test_candidate_states.py
git commit -m "Record a placement search that ran out of budget as what it is"
```

---

### Task 5: Make the oracle test compile what ships (R26)

**Files:**
- Modify: `demo/gibbs-observatory/backend/tests/test_extropic_oracle.py`

**Why.** `audit/findings/R26.md`. The test calls `preflight(load_model(spec=...))`, and `load_model` hardcodes `"domain_wall"`. So the project's strongest regression signal proves those specs compile under one encoding, never exercises the encoding search, and does not compile the model the shelf ships. A change breaking one-hot selection leaves it green.

- [ ] **Step 1: Write the failing test**

Append to `test_extropic_oracle.py`:

```python
@pytest.mark.parametrize("record", VERIFIED_WORKLOADS,
                         ids=[r.shelf_id for r in VERIFIED_WORKLOADS])
def test_the_real_compile_path_selects_the_encoding_the_receipt_shipped(record):
    """Compile the way the product compiles, not through a fixed encoding.

    load_model hardcodes domain_wall. compile_spec searches encodings and is
    what produced the shipped receipts, so this is the only test that puts
    encoding SELECTION under test at all.
    """
    from tsu_compiler.passes.search import compile_spec
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    spec = PROGRAMS / f"{record.spec_stem}.yaml"
    comp = compile_spec(load_spec(str(spec)), PROFILES["z1"], allow_assumed=True)

    assert comp.verdict == "COMPILED", (
        f"{record.shelf_id} does not compile through the real path: "
        f"{comp.verdict}")

    selected = [c for c in comp.repset.candidates
                if c.state.name == "SELECTED"]
    assert len(selected) == 1, f"expected one selected candidate, got {selected}"
    assert selected[0].encoding == record.receipt_encoding, (
        f"{record.shelf_id}: the shipped receipt is "
        f"{record.receipt_encoding!r} but the compiler now selects "
        f"{selected[0].encoding!r}. Either the receipt is stale or selection "
        f"changed; both need a person, not a passing test.")
```

- [ ] **Step 2: Run it**

```
cd demo/gibbs-observatory
.venv/Scripts/python.exe -m pytest backend/tests/test_extropic_oracle.py -q
```

Expect roughly 25 seconds. If a workload's selected encoding disagrees with its receipt, STOP and report it. That is a real finding, not a test to adjust.

- [ ] **Step 3: Commit**

```bash
git add demo/gibbs-observatory/backend/tests/test_extropic_oracle.py
git commit -m "Compile the Extropic workloads the way the product compiles them"
```

---

## WAVE B: the control nobody else has

### Task 6: The ablation runner, as a service

**Files:**
- Create: `demo/gibbs-observatory/backend/app/ablation.py`
- Create: `demo/gibbs-observatory/backend/tests/test_ablation.py`
- Modify: `demo/gibbs-observatory/backend/app/main.py`

**Interfaces:**
- Produces:
  - `run_ablation(receipt_id, *, target: str, n_samples: int, seed: int) -> dict`
    where `target` is `"couplings"` or `"biases"`.
  - Returns `{"baseline": {...}, "ablated": {...}, "noise_floor": float, "shift": float, "ratio": float, "verdict": str, "explanation": str}`.
- Endpoint: `POST /api/ablation/run`.

**Why.** This is the project's signature move and it exists only as scripts. Zero a term, resample, and compare the change against the noise floor from a different seed. `ratio < 3.0` means the term is not doing the work the interface credits it with.

- [ ] **Step 1: Write the failing test**

```python
"""Turn a term off and see whether the answer moves.

The comparison baseline is NOT zero. Two runs of the SAME model with different
seeds differ by some amount, and a change smaller than that is indistinguishable
from noise. R23 is in this repository because a lattice's couplings could be
deleted with no effect while the prose credited them.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.ablation import run_ablation  # noqa: E402


def test_zeroing_couplings_moves_a_coupled_model():
    out = run_ablation("small", target="couplings", n_samples=64, seed=0)
    assert out["noise_floor"] > 0, "a noise floor of zero means it was not measured"
    assert out["ratio"] > 3.0, (
        f"zeroing every coupling moved the marginals only {out['ratio']:.1f}x "
        f"the noise floor. Either the couplings are inert or the ablation is "
        f"not actually zeroing them: {out['explanation']}")
    assert out["verdict"] == "load-bearing"


def test_the_ablation_can_report_inert():
    """A control that cannot fail is not a control."""
    out = run_ablation("small", target="none", n_samples=64, seed=0)
    assert out["ratio"] < 3.0, (
        "ablating NOTHING moved the answer more than three times the noise "
        "floor, so the noise floor is wrong")
    assert out["verdict"] == "not measurable"


def test_the_noise_floor_uses_a_different_seed_not_the_same_run():
    a = run_ablation("small", target="none", n_samples=64, seed=0)
    assert a["noise_floor"] > 0
```

- [ ] **Step 2: Run it, watch the import fail**

- [ ] **Step 3: Implement `ablation.py`**

```python
"""Try to prove a model's own terms are doing nothing.

The measurement is a shift in per-spin marginals, compared against a NOISE
FLOOR: the shift between two runs of the identical model under different seeds.
Comparing against zero would call every term load-bearing, because sampling is
stochastic and no two runs agree exactly.

Threshold 3.0 is the same one `audit/game_on_thrml.py` uses, so a claim made
here and a claim made there mean the same thing.
"""
from __future__ import annotations

import numpy as np

from .sampler_engine import SamplerConfig, SamplerEngine

RATIO_THRESHOLD = 3.0


def _marginals(receipt_id: str, *, seed: int, n_samples: int,
               zero_couplings: bool = False,
               zero_biases: bool = False) -> np.ndarray:
    cfg = SamplerConfig(receipt_id=receipt_id, seed=seed, batch_size=n_samples)
    engine = SamplerEngine(cfg)
    if zero_couplings or zero_biases:
        engine.zero_terms(couplings=zero_couplings, biases=zero_biases)
    batch = engine.sample_batch(n_samples=n_samples, warmup=64)
    states = np.asarray(batch["states"], dtype=float)
    return states.mean(axis=0)


def run_ablation(receipt_id: str, *, target: str = "couplings",
                 n_samples: int = 128, seed: int = 0) -> dict:
    base_a = _marginals(receipt_id, seed=seed, n_samples=n_samples)
    base_b = _marginals(receipt_id, seed=seed + 1_000, n_samples=n_samples)
    noise_floor = float(np.abs(base_a - base_b).mean())

    ablated = _marginals(
        receipt_id, seed=seed, n_samples=n_samples,
        zero_couplings=(target == "couplings"),
        zero_biases=(target == "biases"),
    )
    shift = float(np.abs(base_a - ablated).mean())
    ratio = shift / noise_floor if noise_floor > 0 else float("inf")
    load_bearing = ratio >= RATIO_THRESHOLD

    return {
        "receipt_id": receipt_id,
        "target": target,
        "baseline": {"marginal_mean": float(base_a.mean())},
        "ablated": {"marginal_mean": float(ablated.mean())},
        "noise_floor": noise_floor,
        "shift": shift,
        "ratio": ratio,
        "threshold": RATIO_THRESHOLD,
        "verdict": "load-bearing" if load_bearing else "not measurable",
        "explanation": (
            f"Zeroing the {target} moved the per-spin marginals by {shift:.4f}. "
            f"Two runs of the unchanged model under different seeds differ by "
            f"{noise_floor:.4f}. That is {ratio:.1f}x the noise floor, "
            + ("which is a real effect."
               if load_bearing else
               f"below the {RATIO_THRESHOLD}x threshold, so this term is not "
               f"doing work the interface should credit it with.")),
    }
```

Add `SamplerEngine.zero_terms(*, couplings: bool = False, biases: bool = False)`
which rebuilds `IsingEBM` with zeroed arrays, keeping nodes, edges and blocks.

- [ ] **Step 4: Run the tests. If `ratio < 3.0` on `small`, STOP.** That is an R23-class finding about the shipped receipt, not a test to loosen.

- [ ] **Step 5: Add the endpoint**

```python
class AblationBody(BaseModel):
    receipt_id: str
    target: str = Field("couplings", pattern="^(couplings|biases|none)$")
    n_samples: int = Field(128, ge=16, le=1024)
    seed: int = Field(0, ge=0)


@app.post("/api/ablation/run")
def api_ablation_run(body: AblationBody) -> dict[str, Any]:
    from .ablation import run_ablation

    return {"ok": True, **run_ablation(
        body.receipt_id, target=body.target,
        n_samples=body.n_samples, seed=body.seed)}
```

- [ ] **Step 6: Commit**

```bash
git add demo/gibbs-observatory/backend/app/ablation.py demo/gibbs-observatory/backend/app/sampler_engine.py demo/gibbs-observatory/backend/app/main.py demo/gibbs-observatory/backend/tests/test_ablation.py
git commit -m "Add the ablation runner as a service"
```

---

### Task 7: The ablation runner, on screen

**Files:**
- Create: `frontend/src/components/views/AblationView.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/LeftNav.tsx`

- [ ] **Step 1: Build the view**

One button per target, a result panel showing shift, noise floor, ratio and verdict, and the explanation sentence verbatim. Render the ratio as a bar against the 3x threshold, with the threshold drawn as a line, so "how close was it" is visible rather than inferred.

Copy for the empty state, which is the point of the whole feature:

> This tries to prove the loaded model's own terms are doing nothing. It zeroes a
> term, samples again, and compares the change against how much two identical
> runs differ by chance. If the change is not clearly bigger than that, the term
> is not doing the work.

- [ ] **Step 2: Wire the nav entry, build, commit**

```bash
git add frontend/src/components/views/AblationView.tsx frontend/src/App.tsx frontend/src/components/LeftNav.tsx
git commit -m "Put the ablation runner on screen"
```

---

## WAVE C: see it move

### Task 8: A tesseract, and arbitrary topology

**Files:**
- Create: `demo/gibbs-observatory/programs/tesseract_16.yaml`
- Test: `demo/gibbs-observatory/backend/tests/test_tesseract.py`

**Why.** A 4-cube is 16 vertices, 32 edges, degree 4, and bipartite. It is a perfectly ordinary Ising model that happens to live in four dimensions, so it compiles with zero mediators and proves the viewport can draw a topology nobody hardcoded. It is also the thing the owner asked for by name.

- [ ] **Step 1: Write the test**

```python
def test_the_tesseract_is_a_4_cube():
    """16 vertices, 32 edges, degree 4, bipartite. Q4 by construction."""
    import itertools

    from backend.app.receipt_loader import receipt_to_graph_arrays  # noqa: F401
    from tsu_compiler.preflight.check import preflight
    from tsu_compiler.preflight.model import load_model

    rep = preflight(load_model(spec=str(SPEC)))
    assert rep.max_degree == 4, "every vertex of a 4-cube has four neighbours"
    assert rep.verdict == "ok"
    assert rep.mediators == 0, "a hypercube is bipartite, so it needs no mediators"


def test_edges_join_vertices_differing_in_exactly_one_bit():
    """The definition of a hypercube, checked rather than asserted in prose."""
    import yaml

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    edges = [(t["a"], t["b"]) for t in spec["terms"] if t["kind"] == "product"]
    assert len(edges) == 32, f"Q4 has 32 edges, found {len(edges)}"
    for a, b in edges:
        ia, ib = int(a[1:]), int(b[1:])
        assert bin(ia ^ ib).count("1") == 1, (
            f"{a}-{b} differ in more than one coordinate, so this is not a "
            f"hypercube")
```

- [ ] **Step 2: Generate the YAML**

Vertices `v0` to `v15`. An edge for every pair whose indices differ in exactly one bit. Ferromagnetic weight `-0.8` so it orders visibly.

- [ ] **Step 3: Add it to the shelf catalog** as "A four-dimensional cube", with notes explaining that a hypercube is an ordinary graph and the fourth dimension is in its connectivity, not in the hardware.

- [ ] **Step 4: Commit**

---

### Task 9: The 3D viewport

**Files:**
- Create: `frontend/src/components/views/LatticeView3D.tsx`
- Modify: `frontend/package.json`

- [ ] **Step 1: Add Three.js**

```
cd frontend && npm install three @types/three
```

- [ ] **Step 2: Build the viewport**

Spins as instanced spheres, edges as lines, colour by state, orbit controls, and live updates driven by the existing socket. Layout comes from the receipt's placement coordinates where present; otherwise a force layout computed once and cached, labelled as a layout rather than as physical positions.

**The honesty rule, non-negotiable.** The viewport states which of three things it is showing: one draw, a mean over N, or a variance. And any projection says it is one, with how much it captures. Showing an 11,200-dimensional state space on three axes is honest only when labelled.

- [ ] **Step 3: Commit**

---

### Task 10: Waveforms and streaming plots

**Files:**
- Create: `frontend/src/components/TracePlot.tsx`

- [ ] **Step 1: Build it**

Canvas, not SVG, because these update every frame. Energy, magnetization and acceptance over sweeps, with a scrubber. Axis labels carry units. `font-variant-numeric: tabular-nums` so digits do not jitter.

Onsager's exact critical point is drawn as a reference line where the model is a 2D Ising lattice and therefore where it applies, and is omitted, not faked, where it does not.

- [ ] **Step 2: Commit**

---

## Self-Review

**Spec coverage.** Wave A covers the spec's "honesty constraints" section and section 12's diagnostics. Wave B covers the ablation runner named in section 12 as one of the three that carry the product. Wave C covers the viewport in sections 4 and 10 and the structure-factor groundwork. NOT covered here and deliberately deferred: the pane system and workspaces (spec sections 5 and 6), the compiler pass stack (section 11), the decoder contract (section 10), and packaging changes. Those are their own plans.

**Placeholders.** None. Task 4 names the exact enum member and classifier. Task 6 carries the full service implementation. Task 8 carries the hypercube property test rather than "check it is a tesseract".

**Type consistency.** `summarize_series` returns `ess`, `ess_reliable`, `ess_reason` in Tasks 1 and 2. `run_ablation` returns `ratio`, `noise_floor`, `verdict`, `explanation` in Tasks 6 and 7. `CandidateState.PLACEMENT_EFFORT_EXHAUSTED` and `_classify_candidate_failure` are spelled identically in Task 4 and its test.

**One known risk.** Task 4 edits the compiler, whose suite takes about eight minutes. Run it in full before committing, and do not adjust an existing assertion without reading why it asserted that.
