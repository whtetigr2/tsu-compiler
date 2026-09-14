"""Task 4 of the all-bipartite-world plan
(SPR/docs/superpowers/plans/2026-09-07-all-bipartite-world.md): compose
Task 2's three all-binary-stack layers (specs/binary_stack_l0/l1/l2.yaml)
into one of 8 states per cell, then measure what it actually costs to
relocate the base layer's hard adjacency rule -- "value 0 never adjacent to
value 1" -- from a compiled energy term into a cross-layer bias patch.

`compose_state` (Steps 1-3) is a pure function, tested without sampling in
tests/test_binary_stack.py. Everything below it (Steps 4-6) is measurement
code exercised by this module's own __main__, not by the pytest suite --
sampling a real compiled program is measurement work, not composition-logic
verification (see tests/test_binary_stack.py's own module docstring).

WHY THE PATCH TARGETS LAYER 0 ONLY (Step 4's design, stated up front, not
after the fact): compose_state's own convention (bits[0] + 2*bits[1] +
4*bits[2]) makes composed state 0 = bits (0,0,0) and composed state 1 =
bits (1,0,0) -- they differ ONLY in layer 0's bit, and BOTH require layers
1 and 2 to be 0. So "composed 0 never adjacent to composed 1" is entirely a
statement about layer 0's bit, conditioned on layers 1 and 2 already being
0 at both cells of an edge. This is not a general mechanism for an
arbitrary pair of composed states (a pair differing in layer 1 or 2's bit
would need THAT layer patched instead) -- it is the one relocation Task 4
Step 4 actually asks for, and no more is claimed.

DESIGN, revised after a first attempt measured nothing: a SINGLE fixed
backdrop (one draw each of layers 1/2, mirroring demo/stacked_world.py's
own `grid1 = grids1[0]` precedent) was tried first and rejected -- measured
directly, each of layers 1/2's own self-rule rewards ONLY the 1-1 agreeing
pair (never 0-0), so each layer's marginal P(bit=0) sits well under 50%,
making "eligible" cells (both layers 0, the only cells that can read as
composed 0 or 1) rare, and an ADJACENT PAIR of eligible cells (the only
configuration the relocated rule can ever fire on) rarer still -- present
in only a small fraction of single backdrops. A single backdrop is a
DEGENERATE instance for this measurement (see this module's Step 5-6
section for the numbers): most contain zero adjacent-eligible pairs, so a
"0% violation rate" from one backdrop is a structurally empty measurement,
not evidence the patch does anything -- the false-negative failure mode
the plan's Global Constraints warn about by name. The fix keeps the same
mechanism (layer 0 resampled, patched by a bias derived from layers 1/2's
decoded values) but pools over MANY independent backdrop realizations (not
one), so the rare adjacent-eligible-pair events accumulate a real sample.

MANDATORY CAVEAT (see demo/layers.py's module docstring for the full
derivation, repeated here because this module leans on it just as hard):
biasing layer 0's resample from layers 1/2's decoded backdrop is the same
DIRECTED / ANCESTRAL conditioning demo/layers.py and demo/elevation.py use
-- p(layer1) * p(layer2) * p(layer0 | layer1, layer2) -- never the joint
Boltzmann distribution over all three layers, and the backdrop layers can
never be influenced by layer 0's draw.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

import numpy as np

from tsu_compiler.spec import load_spec
from tsu_compiler.passes.encode import encode
from tsu_compiler.passes.lower import lower
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.place import place
from tsu_compiler.passes.route import route
from tsu_compiler.passes.program import build_program
from tsu_compiler.target import PROFILES
from tsu_compiler.backends.thrml_backend import sample as thrml_sample

from layers import bias_patch, FieldCapExceeded, FIELD_CAP

W = H = 8
SPEC_PATHS = {
    0: "specs/binary_stack_l0.yaml",
    1: "specs/binary_stack_l1.yaml",
    2: "specs/binary_stack_l2.yaml",
}

# Same scale demo/stacked_world.py already uses for an 8x8 binary/categorical
# receipt (~0.4-1.4s per call, measured there): n_chains * n_samples = 240
# draws per call, well past Task 4 Step 5's own ">=200 draws" requirement.
SAMPLE_PARAMS = dict(n_chains=8, n_samples=30, n_warmup=1200, steps_per_sample=4)

# Sweep points for Step 6. 0.0-0.35 matches demo/stacked_world.py's own
# ALPHA sweep range so "where the degenerate zone begins" is checked
# against the SAME earlier-measured threshold (~0.2) first; extended
# repeatedly past it (0.5 through 5.0) because each pass measured the
# violation rate still falling, then a minimum, then rising again --
# Step 6 asks to "confirm or refute" the ~0.2 figure, which requires
# actually looking past it rather than stopping at the earlier study's own
# ceiling (see audit/bipartite_matrix.md's Task 4 Step 6 for the resulting
# U-shaped curve and where its minimum falls for this configuration).
STRENGTH_SWEEP = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


# ---------------------------------------------------------------------------
# Steps 1-3: composition (see tests/test_binary_stack.py)
# ---------------------------------------------------------------------------

def compose_state(bits: list[int]) -> int:
    """Compose three independently-decoded binary layer values (each 0 or 1)
    into one of 8 distinguishable composed cell states -- a pure function of
    the three decoded layers, no sampling involved. `bits[0]` is
    least-significant (specs/binary_stack_l0.yaml is "bit 0" by that file's
    own docstring), `bits[2]` most-significant: state = bits[0] + 2*bits[1]
    + 4*bits[2]. This is the only place a bit-order convention needs
    picking -- every consumer in this module reasons about composed states
    purely through this function's own output."""
    if len(bits) != 3:
        raise ValueError(
            f"compose_state expects exactly 3 binary layer bits; got "
            f"{len(bits)}: {bits!r}")
    if any(b not in (0, 1) for b in bits):
        raise ValueError(
            f"compose_state expects each bit to be 0 or 1; got {bits!r}")
    return bits[0] + 2 * bits[1] + 4 * bits[2]


# ---------------------------------------------------------------------------
# Step 4a: compile each layer directly through the pass pipeline (encode ->
# lower -> analyse -> place -> route -> build_program), never through
# `tsuc compile`/`compile_spec`'s full multi-encoding search -- the plan's
# Global Constraints forbid running `tsuc compile` in this work, and every
# layer here is binary, where domain_wall is a no-op (one encoding, nothing
# to search over; see audit/bipartite_matrix.md's own note). This mirrors
# audit/bipartite_routes.py::bipartite_at's own encode/lower/place sequence,
# extended only with the route/build_program steps that function doesn't
# need (it never samples).
# ---------------------------------------------------------------------------

def _compile_layer(path: str):
    """THE SAFETY RULE, restated for this module: refuse to place a graph
    that `analyse` did not first confirm bipartite. Every binary_stack_l*
    spec was already measured bipartite at 8x8 (Task 2) -- this assertion
    exists so a future change to those specs cannot silently reintroduce a
    6-32-minute placement failure here, not because this call expects to
    ever see it trip."""
    spec = load_spec(path)
    enc = encode(spec, "domain_wall")
    ising = lower(enc.model)
    report = analyse(ising)
    assert report.bipartite is True, (
        f"{path}: analyse() reports bipartite=False -- refusing place() "
        f"per THE SAFETY RULE; this IS the finding, not a bug to route "
        f"around (see plan Global Constraints)")
    target = PROFILES["z1"]
    placement = place(ising, report, target, restarts=12, iters=200_000)
    assert placement.mediation is None, (
        f"{path}: place() mediated a graph analyse() reported bipartite; "
        f"that should be unreachable (see place()'s own docstring) and "
        f"this module has no logic to consume mediated_ising/mediated_report")
    ising = route(ising, report, target)
    prog = build_program(ising, report)
    return spec, enc, prog


def _grid_of(d: dict) -> np.ndarray:
    return np.array([[int(d[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])


def _decode_draws(rows, im, enc) -> list[np.ndarray]:
    """Every draw as a decoded WxH grid. Binary layers have no domain-wall
    chain (encode.py's _build_categorical never chains a Binary variable),
    so `is_codeword` is trivially true for every draw here -- called anyway,
    never assumed, so this stays correct if that ever changes."""
    out = []
    for row in rows:
        bits = dict(zip(im.nodes, row.tolist()))
        if enc.is_codeword(bits):
            out.append(_grid_of(enc.decode(bits)))
    return out


# ---------------------------------------------------------------------------
# Step 4b: the relocated rule as a cross-layer bias patch on layer 0.
# ---------------------------------------------------------------------------

def _neighbours(x: int, y: int):
    if x > 0:
        yield (y, x - 1)
    if x + 1 < W:
        yield (y, x + 1)
    if y > 0:
        yield (y - 1, x)
    if y + 1 < H:
        yield (y + 1, x)


def build_relocation_patch(l0_backdrop: np.ndarray, l1_backdrop: np.ndarray,
                           l2_backdrop: np.ndarray,
                           strength: float) -> dict[tuple[str, int], float]:
    """Build the {(cell, 1): weight} patch (demo/layers.py::bias_patch's own
    convention: weight > 0 ENCOURAGES layer 0 == 1 at that cell) that
    relocates "composed state 0 never adjacent to composed state 1" onto
    layer 0's own resample, given FIXED layers-1/2 backdrops.

    A cell `c` is ELIGIBLE to become composed state 0 or 1 iff layers 1 and
    2 are BOTH 0 there in the backdrop (state = bit0 + 2*bit1 + 4*bit2, so
    state in {0, 1} iff bit1 == bit2 == 0) -- for an eligible cell, its
    composed state after layer 0's resample is EXACTLY its own (possibly
    new) bit0. For each of `c`'s (up to 4) grid neighbours `n`:
      - `n` eligible and reads composed 1 there (`l0_backdrop[n] == 1`): if
        `c` ends up 0, that is a "0 next to 1" violation -- ENCOURAGE `c`'s
        bit0 toward 1 (contribute +1).
      - `n` eligible and reads composed 0 there (`l0_backdrop[n] == 0`):
        symmetric -- DISCOURAGE `c`'s bit0 toward 1 (contribute -1).
      - `n` not eligible (its composed state cannot be 0 or 1 regardless of
        its own bit0): no violation possible via this neighbour,
        contributes 0.
    A non-eligible `c` gets no patch entry at all: its own composed state
    cannot be 0 or 1 regardless of its (resampled) bit0, so biasing that
    bit toward this rule would change nothing the rule cares about.

    `l0_backdrop` supplies each NEIGHBOUR's already-decoded bit0 (used only
    to classify the neighbour as reading composed-0 or composed-1); it is
    never read for the cell BEING patched, whose own bit0 is exactly what
    layer 0's upcoming resample will decide.

    `strength` scales the summed per-neighbour contribution (max magnitude
    4 * strength, an interior cell's full neighbour count) -- the same
    "unscaled per-cell nudge multiplied by strength" convention
    demo/elevation.py::band_patch and demo/stacked_world.py's ALPHA already
    use for this same bias_patch mechanism.
    """
    strength = float(strength)
    eligible = (l1_backdrop == 0) & (l2_backdrop == 0)
    patch: dict[tuple[str, int], float] = {}
    for y in range(H):
        for x in range(W):
            if not eligible[y, x]:
                continue
            total = 0.0
            for (ny, nx) in _neighbours(x, y):
                if not eligible[ny, nx]:
                    continue
                total += 1.0 if l0_backdrop[ny, nx] == 1 else -1.0
            patch[(f"g{x}_{y}", 1)] = strength * total
    return patch


# ---------------------------------------------------------------------------
# Step 5-6: measure the violation rate.
#
# MEASURED, NOT ASSUMED, BEFORE DESIGNING THIS SECTION: each of layers 1/2's
# own self-rule REWARDS a 1-1 agreeing pair but never a 0-0 one (an
# asymmetric, not a symmetric, clumping term), so at low temperature each
# layer's own marginal P(bit=0) is small -- measured directly (10 backdrop
# draws per layer, /tmp diagnostic, not committed) at roughly 7-19% for
# layer 1 and 11-25% for layer 2, consistently below 50% across every seed
# tried, never a one-off. A cell is ELIGIBLE to read as composed state 0 or
# 1 only when BOTH layers 1 and 2 read 0 there (compose_state: state in
# {0, 1} iff bit1 == bit2 == 0) -- so eligibility is the PRODUCT of two
# already-small marginals, measured at ~1.2/64 cells per backdrop, and an
# ADJACENT PAIR of eligible cells (the only configuration that can ever
# violate "0 never touches 1") measured at just ~0.08 per backdrop, present
# in only ~7.5% of backdrops at all (240-backdrop diagnostic).
#
# A single fixed backdrop (this module's first draft) therefore measures
# next to nothing: most single backdrops contain ZERO adjacent-eligible
# pairs, so "0% violations" from one backdrop is a STRUCTURALLY EMPTY
# measurement, not evidence the patch works -- exactly the false-negative
# failure mode this project's own epistemic discipline warns about
# (Global Constraints: "a degenerate instance has already produced both a
# false positive and a false negative"). The fix is not to change the rule
# or the layers to make eligibility more common (that would be tuning the
# instance to get a nicer answer) -- it is to pool over MANY independent
# backdrop realizations so the rare adjacent-eligible-pair events actually
# accumulate a usable sample size, which is what follows.
# ---------------------------------------------------------------------------

def _all_adjacent_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Every unordered adjacent cell-pair position in the WxH grid, each
    counted once (matches Task 1/2's own `n_edges` convention for an 8x8
    4-neighbour grid: W*(H-1) vertical + H*(W-1) horizontal = 112)."""
    pairs = []
    for y in range(H):
        for x in range(W):
            if x + 1 < W:
                pairs.append(((y, x), (y, x + 1)))
            if y + 1 < H:
                pairs.append(((y, x), (y + 1, x)))
    return pairs


ALL_PAIRS = _all_adjacent_pairs()


def _eligible_adjacent_pairs(l1: np.ndarray, l2: np.ndarray):
    """The subset of ALL_PAIRS where BOTH members are eligible (layers 1
    and 2 both read 0 there) under this one backdrop -- the only positions
    where a 0-vs-1 composed-state violation can possibly occur, since any
    pair with a non-eligible member has composed state outside {0, 1}
    regardless of layer 0's value."""
    eligible = (l1 == 0) & (l2 == 0)
    return [(p1, p2) for (p1, p2) in ALL_PAIRS if eligible[p1] and eligible[p2]]


def _count_violations(l0_grids: list[np.ndarray],
                      pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> int:
    """How many (l0_grid, pair) combinations read {0, 1} at that pair's two
    positions -- non-eligible pairs are never passed in (they can never
    violate, by construction; see _eligible_adjacent_pairs), so this only
    ever checks positions that COULD violate, not every edge of every
    grid."""
    bad = 0
    for g0 in l0_grids:
        for (p1, p2) in pairs:
            if {int(g0[p1]), int(g0[p2])} == {0, 1}:
                bad += 1
    return bad


def main():
    print("=== Task 4: compiling the three binary layers ===")
    spec0, enc0, prog0 = _compile_layer(SPEC_PATHS[0])
    spec1, enc1, prog1 = _compile_layer(SPEC_PATHS[1])
    spec2, enc2, prog2 = _compile_layer(SPEC_PATHS[2])

    print("=== drawing one large UNCONDITIONED batch per layer "
          "(the ensemble of backdrops + the zero-nudge/no-op-patch pool) ===")
    l0_batch = _decode_draws(thrml_sample(prog0, seed=10, **SAMPLE_PARAMS), prog0.ising, enc0)
    l1_batch = _decode_draws(thrml_sample(prog1, seed=11, **SAMPLE_PARAMS), prog1.ising, enc1)
    l2_batch = _decode_draws(thrml_sample(prog2, seed=22, **SAMPLE_PARAMS), prog2.ising, enc2)
    n_backdrops = min(len(l0_batch), len(l1_batch), len(l2_batch))
    assert n_backdrops >= 200, (
        f"only {n_backdrops} codewords decoded across the three layers, "
        f"need >= 200 (Task 4 Step 5's own requirement) -- increase "
        f"SAMPLE_PARAMS")
    l0_batch, l1_batch, l2_batch = l0_batch[:n_backdrops], l1_batch[:n_backdrops], l2_batch[:n_backdrops]

    p0_zero_l1 = float((np.stack(l1_batch) == 0).mean())
    p0_zero_l2 = float((np.stack(l2_batch) == 0).mean())
    print(f"measured marginal P(bit=0): layer1={p0_zero_l1:.3f}  layer2={p0_zero_l2:.3f}  "
          f"(both self-rules reward ONLY the 1-1 agreeing pair, never 0-0 -- "
          f"an asymmetric clumping term, so P(bit=0) < 0.5 is the expected, "
          f"not surprising, direction)")

    backdrop_pairs = [_eligible_adjacent_pairs(l1_batch[i], l2_batch[i])
                      for i in range(n_backdrops)]
    n_eligible_cells = [int(((l1_batch[i] == 0) & (l2_batch[i] == 0)).sum())
                        for i in range(n_backdrops)]
    active = [i for i in range(n_backdrops) if backdrop_pairs[i]]
    total_pairs_available = sum(len(p) for p in backdrop_pairs)
    print(f"{n_backdrops} independent backdrop realizations (layers 1+2 each): "
          f"mean eligible cells/backdrop={np.mean(n_eligible_cells):.3f}, "
          f"mean adjacent-eligible pairs/backdrop={total_pairs_available/n_backdrops:.4f}, "
          f"{len(active)}/{n_backdrops} backdrops ({100*len(active)/n_backdrops:.1f}%) "
          f"have >=1 adjacent-eligible pair, {total_pairs_available} total "
          f"adjacent-eligible-pair SLOTS across the whole ensemble")
    if not active:
        print("!!! ZERO backdrops contain an adjacent-eligible pair in this "
              "ensemble -- the relocated rule cannot be measured against "
              "this instance at all; reporting this as the finding, not "
              "fabricating a rate.")
        return

    # ---- zero-nudge (strength 0.0, no patch at all) --------------------
    # Every backdrop's eligible-pair slots, checked against the FULL
    # unconditioned l0_batch (n_backdrops draws) -- a valid composed "world"
    # for ANY (l0 draw, backdrop) pairing, since p(l0)*p(l1)*p(l2) makes
    # every combination a genuine sample of the unconditioned joint. This
    # reuses data already drawn above -- no extra sampling calls.
    zero_bad = sum(_count_violations(l0_batch, backdrop_pairs[i]) for i in active)
    zero_slots = sum(len(backdrop_pairs[i]) for i in active) * n_backdrops
    zero_total_pairs = n_backdrops * n_backdrops * len(ALL_PAIRS)
    zero_rate_overall = zero_bad / zero_total_pairs
    zero_rate_conditional = zero_bad / zero_slots if zero_slots else float("nan")
    print()
    print(f"strength=0.0 (zero nudge): overall {zero_bad}/{zero_total_pairs} "
          f"({zero_rate_overall*100:.4f}%)  |  conditional-on-eligible "
          f"{zero_bad}/{zero_slots} ({zero_rate_conditional*100:.2f}%)")

    # ---- patched strengths ----------------------------------------------
    # Only the `active` backdrops need a fresh patched draw: for every other
    # backdrop the patch dict is EMPTY (no eligible cells at all), and
    # demo/layers.py::bias_patch on an empty patch is proven a no-op
    # (tests/test_layers.py::test_empty_patch_is_a_noop) -- so those
    # backdrops' contribution is IDENTICAL to the zero-nudge case (0 bad,
    # by construction, since they have no eligible-pair slots at all) and
    # is not re-sampled.
    results: dict[float, dict] = {0.0: dict(bad=zero_bad, slots=zero_slots,
                                            total_pairs=zero_total_pairs,
                                            rate_overall=zero_rate_overall,
                                            rate_conditional=zero_rate_conditional)}
    for strength in STRENGTH_SWEEP:
        if strength == 0.0:
            continue
        bad = 0
        slots = 0
        skipped = 0
        bmax_seen = 0.0
        for i in active:
            patch = build_relocation_patch(l0_batch[i], l1_batch[i], l2_batch[i], strength)
            try:
                patched0 = bias_patch(prog0, enc0, patch, field_cap=FIELD_CAP)
            except FieldCapExceeded:
                skipped += 1
                continue
            bmax_seen = max(bmax_seen, float(np.abs(patched0.ising.biases).max()))
            got0 = thrml_sample(patched0, seed=1000 + int(round(strength * 1000)) + i,
                                **SAMPLE_PARAMS)
            l0_draws = _decode_draws(got0, patched0.ising, enc0)
            bad += _count_violations(l0_draws, backdrop_pairs[i])
            slots += len(backdrop_pairs[i]) * len(l0_draws)
        total_pairs = n_backdrops * n_backdrops * len(ALL_PAIRS)
        rate_overall = bad / total_pairs
        rate_conditional = bad / slots if slots else float("nan")
        results[strength] = dict(bad=bad, slots=slots, total_pairs=total_pairs,
                                 rate_overall=rate_overall,
                                 rate_conditional=rate_conditional,
                                 skipped_field_cap=skipped, bmax=bmax_seen)
        print(f"strength={strength:<5} (of {len(active)} active backdrops"
              f"{f', {skipped} skipped: field cap exceeded' if skipped else ''}) "
              f"|b|max_seen={bmax_seen:.4f}: "
              f"overall {bad}/{total_pairs} ({rate_overall*100:.4f}%)  |  "
              f"conditional-on-eligible {bad}/{slots} ({rate_conditional*100:.2f}%)")

    print()
    print("=== Comparison (the price of this plan) ===")
    print("hard energy term, current 8x8 base (specs/lattice_small_8x8_k3.yaml): "
          "0% by construction (contract-validated, never sampled)")
    for s in STRENGTH_SWEEP:
        if s not in results:
            continue
        r = results[s]
        print(f"relocated as a bias patch, strength={s}: overall "
              f"{r['rate_overall']*100:.4f}%  ({r['bad']}/{r['total_pairs']} "
              f"pairs)  |  conditional-on-eligible {r['rate_conditional']*100:.2f}%  "
              f"({r['bad']}/{r['slots']} pairs)")


if __name__ == "__main__":
    main()
