"""demo/frontier.py -- the capacity frontier for an already-compiled receipt.

Not "PASS": how much room is left on the hardware, and what the NEXT
increment (one more terrain value, or one more rule on an existing value)
would cost. Audience is a hardware team -- an honest limit is worth more
than a flattering number.

Two named laws this module predicts from (spec sections 4.2 and 4.8 of
`2026-08-28-lattice-design.md`), both established for the ONE-HOT encoding
specifically (the spec's own measurements were "one-hot unless stated"):

  degree(v) = (k-1) + g*|partners(v)|            -- section 4.2
  |b|_floor(P, k) = P*(k-2)/2  (a LOWER bound;    -- section 4.8
      the workload's own terms add a bit above it)

`predicted_degree_one_hot` and `predicted_field_one_hot_floor` below are
that arithmetic, TDD'd in tests/test_frontier.py against the spec's own
recorded measurement table before this module used them for anything.

EVERY number this module derives from those two functions is a PREDICTION
FROM A NAMED LAW -- never presented as a measurement (see `Prediction`
below, which carries its own law name in every instance). Section 4.2's law
was confirmed on one-hot; a companion project measurement
(`.superpowers/sdd/2026-08-28-lattice-prerequisites/task-3-report.md`,
"Reading the numbers against the degree law") found DOMAIN-WALL does *not*
follow the same law -- its shared chain spins accumulate degree from more
than one value's rules at once. `demo/receipts/small` (the receipt this
module reads by default) was compiled with domain_wall, not one_hot. This
module therefore never asserts the one-hot law describes THIS model's own
future cost -- it states the law's prediction, labelled, and separately
VERIFIES it by actually building the larger model with `encode`/`lower`/
`analyse` (milliseconds, no compile, no sampling) and reading the real
number back -- `verify_increment` below. Predicted and observed are always
shown side by side; a mismatch is reported as a mismatch, not smoothed over.

|J| <= 6.0 (this module reads it from the receipt's own gates.json / from
`tsu_compiler.target.Z1.max_abs_coupling`, never hardcodes it) is an Extropic-documented
Z1 hardware cap (Thermalizers paper, Fig. 12 cap-sweep axis annotated
"6 (Z1)"). `tsu_compiler.target.Z1.max_abs_bias`, the |b| <= 6.0 cap, remains an
ASSUMED project value, not a sourced Extropic figure -- every display below
says which is which.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsu_compiler.spec import WorkloadSpec, load_spec  # noqa: E402
from tsu_compiler.passes.encode import ONE_HOT_PENALTY, encode  # noqa: E402
from tsu_compiler.passes.lower import lower  # noqa: E402
from tsu_compiler.passes.analyse import GraphReport, analyse  # noqa: E402
from tsu_compiler.target import Z1  # noqa: E402

DEFAULT_RECEIPT_DIR = REPO_ROOT / "demo" / "receipts" / "small"

# g -- the grid's own per-cell edge degree (every interior 2-D grid cell has
# 4 neighbours). Spec section 4.2 states the law with this symbol; the value
# is fixed by the workload's own generator (`kind: grid`), never guessed.
GRID_DEGREE = 4


# ---------------------------------------------------------------------------
# The two named laws, as pure arithmetic. TDD'd in tests/test_frontier.py
# against the spec's own measured table BEFORE this module used them for
# anything else. Neither function ever measures -- both take the model's
# own parameters (k, partner count, penalty) and return what the named law
# says should follow.
# ---------------------------------------------------------------------------

def predicted_degree_one_hot(k: int, partners: int, g: int = GRID_DEGREE) -> int:
    """degree(v) = (k-1) + g*|partners(v)| -- spec section 4.2. Confirmed on
    five one-hot configurations in the spec (three of them predicted BEFORE
    measurement). `partners` is |partners(v)|: the count of distinct values
    v appears with in any rule, including itself."""
    return (k - 1) + g * partners


def predicted_degree_increment_one_hot(g: int = GRID_DEGREE) -> int:
    """One more rule on an existing value (p -> p+1): the law predicts a
    flat +g, independent of k or the starting p -- spec section 4.2."""
    return g


def predicted_field_one_hot_floor(penalty: float, k: int) -> float:
    """P*(k-2)/2 -- spec section 4.8. The one-hot exactly-one penalty
    `P*(sum_v x_v - 1)**2` gives every one-hot spin this linear coefficient
    BEFORE any workload term is added -- a LOWER BOUND on |b|max, not an
    exact value (the spec's own table shows observed running slightly above
    it: 15.00 predicted vs 15.31-16.25 observed at P=10,k=5 -- the excess is
    the workload's own contribution). At P=10, k=5 this floor is 15.0; at
    k=3 it is 5.0."""
    return penalty * (k - 2) / 2.0


# ---------------------------------------------------------------------------
# Gate headroom for the compiled model, read verbatim from the receipt's own
# gates.json / metrics.json -- never recomputed, never a guess.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateHeadroom:
    gate: str
    measured: float
    limit: float
    pct_used: float | None  # None when the limit is non-finite or zero --
                             # no percentage is fabricated for those.

    def describe(self) -> str:
        if self.pct_used is None:
            return f"{self.gate}: {self.measured} of {self.limit}"
        return (f"{self.gate}: {self.measured} of {self.limit} "
                f"({self.pct_used:.0f}% used)")


def _pct_used(measured: float, limit: Any) -> float | None:
    if not isinstance(limit, (int, float)):
        return None
    if limit in (0, float("inf")) or limit != limit:  # zero, inf, or NaN
        return None
    return 100.0 * float(measured) / float(limit)


def headroom_from_receipt(gates: list[dict], metrics: dict) -> list[GateHeadroom]:
    """Per-gate headroom for THIS compiled model.

    degree / coupling_cap / field_cap are read straight from gates.json --
    what the `gate_checks` pass actually evaluated and gated the compile on
    (pre-mediation for a mediated model, since gate_checks runs before
    `route.insert_mediators`).

    node_budget's own MEASURED count is instead read from metrics.json's
    `n_nodes` rather than gates.json's own node_budget row: gates.json's
    node_budget check ran on the pre-mediation logical spin count (128 for
    demo/receipts/small), but metrics.json's n_nodes is the FINAL placed
    physical node count (192, +64 mediators) -- the number that actually
    occupies hardware p-bits, which is the honest thing to show as "p-bits
    used" even though it was never itself gated against the budget
    post-mediation."""
    by_gate = {g["gate"]: g for g in gates}
    out = []
    for name in ("degree", "coupling_cap", "field_cap"):
        g = by_gate[name]
        out.append(GateHeadroom(name, g["measured"], g["limit"],
                                _pct_used(g["measured"], g["limit"])))
    node_limit = by_gate["node_budget"]["limit"]
    n_nodes = metrics["n_nodes"]
    out.append(GateHeadroom("node_budget", n_nodes, node_limit,
                            _pct_used(n_nodes, node_limit)))
    return out


# ---------------------------------------------------------------------------
# Current model shape -- k, and partners(v) for every value, read from the
# spec's own generated `product_over_edges` rules (spec section 4.2's own
# definition: "the set of distinct values v appears with in any rule,
# including itself").
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelShape:
    k: int
    partners: dict[int, frozenset[int]]

    @property
    def worst_value(self) -> int:
        return max(self.partners, key=lambda v: len(self.partners[v]))

    @property
    def worst_partners(self) -> int:
        return len(self.partners[self.worst_value])


def _raw_spec_dict(spec: WorkloadSpec) -> dict:
    return yaml.safe_load(spec.source_text)


def read_model_shape(spec: WorkloadSpec) -> ModelShape:
    raw = _raw_spec_dict(spec)
    gen = raw.get("generate")
    if not gen or gen.get("kind") != "grid" \
            or gen.get("variable_domain", {}).get("domain") != "categorical":
        raise ValueError(
            "read_model_shape only understands a grid-generated categorical "
            "workload (spec's own `generate: {kind: grid, variable_domain: "
            "{domain: categorical}}`); this spec has a different shape")
    k = int(gen["variable_domain"]["k"])
    partners: dict[int, set[int]] = {}
    for t in raw.get("terms", []):
        if t.get("kind") != "product_over_edges":
            continue
        a, b = int(t["a_value"]), int(t["b_value"])
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return ModelShape(k=k, partners={v: frozenset(s) for v, s in partners.items()})


# ---------------------------------------------------------------------------
# Verification: actually build the larger model (encode -> lower -> analyse,
# milliseconds, no compile, no sampling) and read the REAL numbers back.
# ---------------------------------------------------------------------------

def _spec_with_k(spec: WorkloadSpec, k: int) -> WorkloadSpec:
    """Same workload, `k` bumped -- no new rule added, so this isolates the
    representation's own structural cost of one more terrain value with no
    rule of its own yet (exactly what section 4.8's field law describes)."""
    raw = copy.deepcopy(_raw_spec_dict(spec))
    raw["generate"]["variable_domain"]["k"] = k
    raw["name"] = f"{raw['name']}__frontier_k{k}"
    return _load_raw(raw)


def _spec_with_extra_rule(spec: WorkloadSpec, a_value: int, b_value: int,
                          weight: float = 0.2) -> WorkloadSpec:
    """Same workload, one more `product_over_edges` rule added on an
    EXISTING pair of values -- one more distinct partner for whichever value
    wasn't already paired with the other."""
    raw = copy.deepcopy(_raw_spec_dict(spec))
    raw["terms"] = list(raw.get("terms", [])) + [
        {"kind": "product_over_edges", "a_value": a_value, "b_value": b_value,
         "weight": weight}]
    raw["name"] = f"{raw['name']}__frontier_rule_{a_value}_{b_value}"
    return _load_raw(raw)


def _load_raw(raw: dict) -> WorkloadSpec:
    fd, path_s = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    path = Path(path_s)
    try:
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_spec(str(path))
    finally:
        path.unlink(missing_ok=True)


def observe(spec: WorkloadSpec, encoding: str, coefficient_scale: float = 1.0
           ) -> GraphReport:
    """The ONLY function in this module that measures rather than predicts:
    actually runs `encode` -> `lower` -> `analyse` on `spec` and returns the
    real GraphReport. Cheap (milliseconds at this workload's size) and needs
    no compile, no placement, no sampling."""
    enc = encode(spec, encoding, coefficient_scale)
    im = lower(enc.model)
    return analyse(im)


@dataclass(frozen=True)
class VerifiedIncrement:
    label: str               # e.g. "k: 3 -> 4 (one more terrain value)"
    encoding: str
    law: str                 # which named law this predicted from
    predicted_degree: float
    observed_degree: int
    predicted_field_floor: float | None
    observed_field: float
    degree_matches: bool     # observed degree == predicted (one-hot law is exact)
    field_at_or_above_floor: bool | None  # None when no floor was predicted


def verify_k_increment(spec: WorkloadSpec, shape: ModelShape, encoding: str,
                       penalty: float, coefficient_scale: float = 1.0
                       ) -> VerifiedIncrement:
    k2 = shape.k + 1
    bigger = _spec_with_k(spec, k2)
    rep = observe(bigger, encoding, coefficient_scale)
    pred_degree = predicted_degree_one_hot(k2, shape.worst_partners)
    pred_field = predicted_field_one_hot_floor(penalty * coefficient_scale, k2)
    return VerifiedIncrement(
        label=f"k: {shape.k} -> {k2} (one more terrain value, no new rule)",
        encoding=encoding, law="one-hot law, spec section 4.2/4.8",
        predicted_degree=pred_degree, observed_degree=rep.max_degree,
        predicted_field_floor=pred_field, observed_field=rep.max_abs_b,
        degree_matches=(rep.max_degree == pred_degree),
        field_at_or_above_floor=(rep.max_abs_b >= pred_field))


def verify_p_increment(spec: WorkloadSpec, shape: ModelShape, encoding: str,
                       coefficient_scale: float = 1.0) -> VerifiedIncrement:
    v = shape.worst_value
    # The cheapest new partner: the smallest value not already in v's
    # partner set (falls back to a value >= k if v is already paired with
    # every existing value -- rare, but never silently skipped).
    existing = shape.partners[v]
    candidates = [x for x in range(shape.k) if x not in existing]
    new_partner = candidates[0] if candidates else shape.k
    bigger = _spec_with_extra_rule(spec, v, new_partner)
    rep = observe(bigger, encoding, coefficient_scale)
    p2 = shape.worst_partners + 1
    pred_degree = predicted_degree_one_hot(shape.k, p2)
    return VerifiedIncrement(
        label=(f"p: {shape.worst_partners} -> {p2} (value {v} gets one more "
               f"rule, with value {new_partner})"),
        encoding=encoding, law="one-hot law, spec section 4.2",
        predicted_degree=pred_degree, observed_degree=rep.max_degree,
        predicted_field_floor=None, observed_field=rep.max_abs_b,
        degree_matches=(rep.max_degree == pred_degree),
        field_at_or_above_floor=None)


# ---------------------------------------------------------------------------
# Which gate binds first -- walked from the LAW's own arithmetic (a
# PREDICTION), never from a search over real encodes (that would be slow at
# scale and this module states plainly that it is a prediction, not a
# guarantee, for any encoding other than one-hot).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BindingForecast:
    field_binds_at_k: int | None   # smallest k (partners held at current
                                    # worst) at which the field law's floor
                                    # is predicted to exceed the cap
    degree_binds_at_k: int | None  # smallest k (partners held fixed) at
                                    # which the degree law is predicted to
                                    # exceed the cap
    degree_binds_at_p: int | None  # smallest p (k held fixed) at which the
                                    # degree law is predicted to exceed the cap
    headline: str


def predict_first_binding_gate(shape: ModelShape, penalty: float,
                                degree_cap: float, field_cap: float,
                                g: int = GRID_DEGREE,
                                search_limit: int = 128) -> BindingForecast:
    """Walk the ONE-HOT LAW's own arithmetic (never a real encode) outward
    along two independent axes -- growing k alone (partners held at the
    model's current worst) and growing p alone (k held fixed) -- to find the
    smallest k or p at which each law-predicted quantity is predicted to
    exceed its cap. This is a PREDICTION about what the one-hot law implies,
    stated for comparison; whether it holds for this receipt's own encoding
    is a separate, VERIFIED question (see `verify_k_increment`/
    `verify_p_increment`)."""
    k0, p0 = shape.k, shape.worst_partners

    field_binds_at_k = None
    for k in range(k0, k0 + search_limit):
        if predicted_field_one_hot_floor(penalty, k) > field_cap:
            field_binds_at_k = k
            break

    degree_binds_at_k = None
    for k in range(k0, k0 + search_limit):
        if predicted_degree_one_hot(k, p0, g) > degree_cap:
            degree_binds_at_k = k
            break

    degree_binds_at_p = None
    for p in range(p0, p0 + search_limit):
        if predicted_degree_one_hot(k0, p, g) > degree_cap:
            degree_binds_at_p = p
            break

    candidates = []
    if field_binds_at_k is not None:
        candidates.append(("field_cap", "k", field_binds_at_k, field_binds_at_k - k0))
    if degree_binds_at_k is not None:
        candidates.append(("degree", "k", degree_binds_at_k, degree_binds_at_k - k0))
    if degree_binds_at_p is not None:
        candidates.append(("degree", "p", degree_binds_at_p, degree_binds_at_p - p0))

    if not candidates:
        headline = (f"under the one-hot law, no gate is predicted to bind "
                    f"within {search_limit} increments of k or p")
    else:
        gate, axis, at, steps = min(candidates, key=lambda c: c[3])
        headline = (f"under the one-hot law (section 4.2/4.8), {gate} is "
                    f"predicted to bind FIRST, at {axis}={at} "
                    f"({steps} increment(s) from the current model)")

    return BindingForecast(field_binds_at_k, degree_binds_at_k,
                           degree_binds_at_p, headline)


# ---------------------------------------------------------------------------
# Assembling the full report for one receipt.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierReport:
    receipt_dir: Path
    encoding: str
    coefficient_scale: float
    headroom: list[GateHeadroom]
    shape: ModelShape
    penalty: float
    law_predicted_degree_next_k: int
    law_predicted_field_floor_next_k: float
    law_predicted_degree_next_p: int
    binding: BindingForecast
    verified: list[VerifiedIncrement]
    encoding_note: str


def build_frontier_report(receipt_dir: Path = DEFAULT_RECEIPT_DIR) -> FrontierReport:
    receipt_dir = Path(receipt_dir)

    def load(name):
        return json.loads((receipt_dir / name).read_text())

    gates = load("gates.json")
    metrics = load("metrics.json")
    passes = load("passes.json")
    spec = load_spec(str(receipt_dir / "spec.yaml"))

    selected = next((c for c in passes["candidates"] if c["state"] == "SELECTED"),
                    None)
    encoding = selected["encoding"] if selected else "unavailable: no SELECTED candidate"
    coefficient_scale = float(passes.get("coefficient_scale", 1.0))

    headroom = headroom_from_receipt(gates, metrics)
    shape = read_model_shape(spec)
    penalty = ONE_HOT_PENALTY

    pred_degree_next_k = predicted_degree_one_hot(shape.k + 1, shape.worst_partners)
    pred_field_next_k = predicted_field_one_hot_floor(
        penalty * coefficient_scale, shape.k + 1)
    pred_degree_next_p = predicted_degree_one_hot(shape.k, shape.worst_partners + 1)

    degree_cap = Z1.degree.value
    # F-R13: field_cap gates |b| (predicted_field_one_hot_floor, fed to
    # predict_first_binding_gate just below, is a prediction of |b|, never
    # |J|) -- so per gates.py's own convention (field_cap <-> max_abs_bias,
    # coupling_cap <-> max_abs_coupling) this must read max_abs_bias, not
    # max_abs_coupling. Same conflation F-R2 fixed one file over, in
    # demo/layers.py's FIELD_CAP.
    field_cap = Z1.max_abs_bias.value
    binding = predict_first_binding_gate(shape, penalty * coefficient_scale,
                                         degree_cap, field_cap)

    verified = [
        verify_k_increment(spec, shape, encoding, penalty, coefficient_scale),
        verify_p_increment(spec, shape, encoding, coefficient_scale),
    ]
    if encoding != "one_hot":
        # Also verify the law against the encoding it was actually confirmed
        # for, on this SAME model -- the strongest available check, since it
        # tests the law on a configuration the spec's own table never
        # covered rather than only re-citing the spec's past measurements.
        verified.append(verify_k_increment(spec, shape, "one_hot", penalty,
                                           coefficient_scale))
        verified.append(verify_p_increment(spec, shape, "one_hot",
                                           coefficient_scale))

    if encoding == "one_hot":
        encoding_note = (
            "this receipt's SELECTED encoding is one_hot -- the encoding "
            "section 4.2/4.8's law was confirmed on, so the law predictions "
            "above are expected to describe this model directly (see the "
            "verified rows below).")
    else:
        encoding_note = (
            f"this receipt's SELECTED encoding is {encoding!r}, NOT one_hot. "
            f"Section 4.2/4.8's law was confirmed for one-hot only; a prior "
            f"project measurement "
            f"(task-3-report.md, \"Reading the numbers against the degree "
            f"law\") found domain_wall does NOT follow the same degree law "
            f"(measured 18 vs predicted 16 at p=3 in that measurement). The "
            f"law predictions above are shown for reference, labelled as "
            f"predictions; this model's OWN next-increment cost is the "
            f"VERIFIED {encoding} rows below, not the one-hot numbers.")

    return FrontierReport(
        receipt_dir=receipt_dir, encoding=encoding,
        coefficient_scale=coefficient_scale, headroom=headroom, shape=shape,
        penalty=penalty, law_predicted_degree_next_k=pred_degree_next_k,
        law_predicted_field_floor_next_k=pred_field_next_k,
        law_predicted_degree_next_p=pred_degree_next_p, binding=binding,
        verified=verified, encoding_note=encoding_note)


def render_text(r: FrontierReport) -> str:
    lines = [f"FRONTIER -- {r.receipt_dir.name}  (encoding={r.encoding}, "
             f"coefficient_scale={r.coefficient_scale})", ""]
    lines.append("headroom (this compiled model):")
    for h in r.headroom:
        lines.append(f"  {h.describe()}")
    lines.append("")
    lines.append(f"model shape: k={r.shape.k}, worst value={r.shape.worst_value} "
                 f"(partners={sorted(r.shape.partners[r.shape.worst_value])}, "
                 f"p={r.shape.worst_partners})")
    lines.append("")
    lines.append("next-increment cost -- PREDICTED from the one-hot law "
                 "(spec section 4.2/4.8), not a measurement:")
    lines.append(f"  k -> {r.shape.k + 1} (one more terrain value): "
                 f"predicted degree {r.law_predicted_degree_next_k}, "
                 f"predicted |b| floor {r.law_predicted_field_floor_next_k:.2f}")
    lines.append(f"  p -> {r.shape.worst_partners + 1} (one more rule on an "
                 f"existing value): predicted degree "
                 f"{r.law_predicted_degree_next_p}")
    lines.append("")
    lines.append(r.binding.headline)
    lines.append("")
    lines.append(r.encoding_note)
    lines.append("")
    lines.append("verified (real encode/lower/analyse -- predicted vs observed):")
    for v in r.verified:
        match = "MATCHES" if v.degree_matches else "DIVERGES"
        lines.append(f"  [{v.encoding}] {v.label}")
        lines.append(f"    degree: predicted {v.predicted_degree} vs "
                     f"observed {v.observed_degree}  ({match})")
        if v.predicted_field_floor is not None:
            ok = "at/above floor" if v.field_at_or_above_floor else "BELOW floor"
            lines.append(f"    |b|max: predicted floor "
                         f"{v.predicted_field_floor:.2f} vs observed "
                         f"{v.observed_field:.2f}  ({ok})")
        else:
            lines.append(f"    |b|max observed: {v.observed_field:.2f} "
                         f"(no field-law prediction for a p-increment)")
    lines.append("")
    lines.append("|J| <= 6.0 is an Extropic-documented Z1 hardware cap "
                 "(Thermalizers paper, Fig. 12 cap-sweep, annotated "
                 "\"6 (Z1)\"); |b| <= 6.0 remains an ASSUMED project value, "
                 "not a sourced Extropic figure (spec section 4.9 / project "
                 "note).")
    return "\n".join(lines)


def main():
    report = build_frontier_report()
    print(render_text(report))


if __name__ == "__main__":
    main()
