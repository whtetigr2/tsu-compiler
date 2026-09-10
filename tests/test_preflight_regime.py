"""Task 4: locating the transition by finite-size scaling.

The Binder cumulant's two limits are known exactly, so it is checked against
them rather than against our own sampler."""
import inspect
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

from tsu.ess import RELIABILITY_MIN_N_OVER_TAU
from tsu.passes.lower import IsingModel
from tsu.preflight.diagnostics import RHAT_THRESHOLD
from tsu.preflight.sweep import (RegimeRow, binder, crossing, susceptibility,
                                 usable_band, sweep, SATURATION)


def test_binder_of_a_gaussian_order_parameter_is_zero():
    """Deep in the disordered phase m is Gaussian about zero, where
    <m^4> = 3<m^2>^2 exactly, so U = 1 - 3/3 = 0."""
    m = np.random.default_rng(0).standard_normal(400_000)
    assert binder(m) == pytest.approx(0.0, abs=0.01)


def test_binder_of_a_two_delta_order_parameter_is_two_thirds():
    """Deep in the ordered phase m sits at +-m0, so <m^4> = m0^4 and
    <m^2> = m0^2, giving U = 1 - 1/3 = 2/3. These two limits are why the
    cumulant locates a transition without knowing anything about the model."""
    m = np.where(np.random.default_rng(1).random(400_000) < 0.5, -0.7, 0.7)
    assert binder(m) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_binder_of_a_constant_is_two_thirds():
    assert binder(np.full(1000, 0.4)) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_binder_of_an_all_zero_order_parameter_is_defined():
    """A degenerate sample must not divide by zero -- it should report the
    disordered limit rather than a NaN that propagates into the band search."""
    assert binder(np.zeros(100)) == pytest.approx(0.0)


def test_susceptibility_of_a_gaussian_matches_its_closed_form():
    """chi = N(<m^2> - <|m|>^2). For m ~ Normal(0, sigma), <m^2> = sigma^2 and
    <|m|> = sigma*sqrt(2/pi), so chi = N*sigma^2*(1 - 2/pi) EXACTLY -- about
    0.3634*N*sigma^2. Pinned against that closed form rather than asserted to be
    merely 'bigger when the spread is bigger': an inequality that follows
    algebraically from the definition cannot fail, and proportionality to N is
    likewise true by construction, so neither would catch a wrong estimator."""
    n_spins, sigma = 100, 0.2
    m = np.random.default_rng(2).standard_normal(400_000) * sigma
    expected = n_spins * sigma ** 2 * (1.0 - 2.0 / np.pi)
    assert susceptibility(m, n_spins) == pytest.approx(expected, rel=0.02)


def _rows(size, us, couplings):
    return [RegimeRow(beta_j=b, size=size, abs_m=0.0, abs_m_err=0.0, chi=0.0,
                      binder=u, tau=1.0, n_eff=100.0, r_hat=1.0,
                      ess_reason="ok", ess_unavailable=False, provisional=False,
                      n_samples_used=2000)
            for b, u in zip(couplings, us)]


def test_crossing_finds_where_two_binder_curves_meet():
    """Curves for different sizes cross at the critical coupling: below it the
    larger system is MORE disordered, above it more ordered. The crossing is the
    measurement -- 'where |m| looks like it jumps' is not."""
    cs = [0.1, 0.2, 0.3, 0.4]
    small = _rows(8, [0.10, 0.20, 0.40, 0.60], cs)
    large = _rows(16, [0.02, 0.15, 0.50, 0.64], cs)
    x = crossing(small, large)
    # The difference small-large is [+0.08, +0.05, -0.10, -0.04], so it flips
    # between 0.2 and 0.3 and linear interpolation gives
    # 0.2 + 0.1*0.05/(0.05+0.10) = 0.2 + 0.1/3 exactly. Asserting a range here
    # would pass for any implementation that merely lands in the right cell.
    assert x == pytest.approx(0.2 + 0.1 / 3.0, abs=1e-12)


def test_crossing_returns_none_when_the_curves_never_meet():
    """Reporting a transition that was not observed would be worse than
    reporting none, so a sweep that never brackets one says so."""
    cs = [0.1, 0.2, 0.3]
    assert crossing(_rows(8, [0.1, 0.2, 0.3], cs),
                    _rows(16, [0.05, 0.15, 0.25], cs)) is None


def test_usable_bands_lower_edge_is_the_crossing_value():
    """WHAT THIS PINS: `usable_band`'s lower edge IS the `crossing` value
    passed in -- not a value `usable_band` derives on its own from
    `binder`. This test replaces
    `test_usable_band_starts_at_ordering_and_ends_at_saturation`, which
    encoded the exact defect a real coordinator review caught: this
    function used to derive its own lower edge from a hardcoded
    `binder > 0.1` literal, invented while writing the sweep plan and never
    checked against the design doc, which already says the lower edge IS
    the crossing. That literal fired on a 16-node 1D Ising ring -- a model
    with NO finite-temperature transition at any coupling (Ising 1925) --
    reporting a confident-looking band where none exists (see
    `test_usable_band_is_none_without_a_crossing_on_a_real_1d_ising_ring`
    below for the live reproduction). The old test asserted only `lo < hi`
    and `hi <= 0.5`, which the vacuous 0.1-derived band ALSO satisfied --
    exactly the "trivially true" shape this project's review process
    exists to catch, so this replacement pins an exact tuple instead.
    HOW IT FAILS: `usable_band` internally deriving its own lower edge
    (from `binder` or anything else) instead of using the `crossing`
    argument verbatim makes `band[0] != 0.2` here.
    PROVENANCE: hand-traced arithmetic below."""
    cs = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=0.01, chi=1.0,
                      binder=u, tau=1.0, n_eff=100.0, r_hat=1.0,
                      ess_reason="ok", ess_unavailable=False, provisional=False,
                      n_samples_used=2000)
            for b, m, u in zip(cs, [0.05, 0.12, 0.40, 0.80, 0.97],
                               [0.02, 0.10, 0.35, 0.58, 0.66])]
    # lo = crossing = 0.2 (given, not derived). Rows at/above 0.2:
    # (0.2, 0.12), (0.3, 0.40), (0.4, 0.80), (0.5, 0.97) -- only 0.5
    # reaches SATURATION (0.9), so hi = the largest sub-saturation beta_j
    # at/above lo among the rest, 0.4.
    band = usable_band(rows, crossing=0.2)
    assert band == (0.2, 0.4)


def test_usable_band_is_open_above_when_no_row_saturates():
    """WHAT THIS PINS: the upper edge is a REAL saturation observation, not
    "the last coupling the sweep happened to try". This is the other half
    of the defect the coordinator's review caught: on the 1D ring, the
    reported upper edge (0.6) was exactly the sweep's own `--beta-max`, not
    a measured saturation -- none of that sweep's rows ever reached
    SATURATION. Here, no row at or above the crossing reaches SATURATION
    (0.9) either, so the band must be open above (`(crossing, None)`), not
    `(crossing, 0.4)` (the largest beta_j actually swept).
    HOW IT FAILS: reverting to `below[-1].beta_j` as the upper edge
    whenever `below` is non-empty (the old code's shape), without first
    checking whether anything actually SATURATED, makes this return
    `(0.2, 0.4)` instead of `(0.2, None)`.
    PROVENANCE: hand-traced arithmetic below (max abs_m in this fixture is
    0.30, well under SATURATION=0.9)."""
    cs = [0.1, 0.2, 0.3, 0.4]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=0.01, chi=1.0,
                      binder=0.1, tau=1.0, n_eff=100.0, r_hat=1.0,
                      ess_reason="ok", ess_unavailable=False, provisional=False,
                      n_samples_used=2000)
            for b, m in zip(cs, [0.05, 0.12, 0.20, 0.30])]
    assert all(r.abs_m < SATURATION for r in rows)
    band = usable_band(rows, crossing=0.2)
    assert band == (0.2, None)


def test_a_provisional_row_is_excluded_from_the_band():
    """A row whose chains disagree (R-hat above threshold) must not silently
    set the band's upper edge -- even though its own abs_m would (if
    wrongly included) look like the right answer. `provisional` alone
    (never `ess_unavailable`) is what disqualifies a row; see
    `test_ess_unavailable_row_still_counts_toward_the_band` for the
    complementary case.

    Reworked for the `crossing` parameter `usable_band` now requires (see
    `test_usable_bands_lower_edge_is_the_crossing_value`): `crossing=0.1`
    puts the lower edge at the first row. Without the provisional row
    (beta_j=0.35, deliberately given a LOW abs_m=0.10 so it would extend
    the band's upper edge if wrongly counted), the upper edge is the last
    sub-saturation row among {0.1, 0.3, 0.4}, which is 0.3 (0.4 saturates
    at abs_m=0.95). If the provisional filter were broken, 0.35's abs_m
    (0.10, well under SATURATION) would push the upper edge to 0.35
    instead -- this test pins the exact tuple so that regression is
    visible, not just "a band exists"."""
    cs = [0.1, 0.3, 0.35, 0.4]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=0.01, chi=1.0,
                      binder=0.5, tau=1.0, n_eff=100.0, r_hat=rh,
                      ess_reason="ok", ess_unavailable=False,
                      provisional=rh > RHAT_THRESHOLD, n_samples_used=2000)
            for b, m, rh in zip(cs, [0.05, 0.60, 0.10, 0.95],
                                [1.0, 1.0, 1.9, 1.0])]
    band = usable_band(rows, crossing=0.1)
    assert band == (0.1, 0.3), \
        "the provisional row at beta_j=0.35 must not set the upper edge"


def test_ess_unavailable_row_still_counts_toward_the_band():
    """An R-hat-clean row whose ESS estimate is unavailable must still count
    toward the band -- refusing an ERROR BAR does not mean the MEAN (abs_m)
    is wrong, and tau genuinely diverges near a transition (critical
    slowing down), which is exactly the region this function exists to
    locate. Conflating the two (the shape this task's brief originally
    shipped: `provisional = r_hat>threshold or not reliable`) would make
    usable_band refuse to find the band precisely where it is.

    Same beta_j/abs_m fixture as
    test_usable_bands_lower_edge_is_the_crossing_value (crossing=0.3 here
    reproduces that test's own derived lower edge, giving the same
    expected band (0.3, 0.4)), but every row here has tau/n_eff/abs_m_err
    =None and ess_unavailable=True, provisional=False throughout -- so a
    usable_band that (incorrectly) filtered on ess_unavailable too would
    collapse `good` to the empty list and return None instead of
    (0.3, 0.4). Verified this fails under that reversion (`good = [r for r
    in rows if not (r.provisional or r.ess_unavailable)]`) before
    confirming it passes against the actual (provisional-only) filter."""
    cs = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=None, chi=1.0,
                      binder=0.3, tau=None, n_eff=None, r_hat=1.0,
                      ess_reason="unavailable: N/tau below the reliability "
                                 "threshold (largest attempt: n_samples=32000, "
                                 "budget max_samples=32000)",
                      ess_unavailable=True, provisional=False, n_samples_used=32_000)
            for b, m in zip(cs, [0.05, 0.12, 0.40, 0.80, 0.97])]
    band = usable_band(rows, crossing=0.3)
    assert band is not None, \
        "an ESS-unavailable (but R-hat-clean) row must still count toward the band"
    assert band == (0.3, 0.4)


@pytest.mark.slow
def test_usable_band_is_none_without_a_crossing_on_a_real_1d_ising_ring():
    """WHAT THIS PINS: a real sweep of a 16-node 1D Ising ring (uniform
    ferromagnetic J, zero bias) -- which has NO finite-temperature phase
    transition at any coupling strength (exact result, Ising 1925) -- must
    report no usable band when `crossing` is None, even though the ring's
    own MEASURED Binder cumulant rises past the deleted `0.1` literal well
    within this sweep's range. A band reported here would be a false
    positive with a KNOWN-CORRECT answer (no band exists, because no
    transition exists).

    This is the exact live finding from a coordinator review: `tsu regime
    --edges` on a 16-node ring reported "No Binder crossing was observed"
    immediately followed by "Usable band: (0.433, 0.6)" three lines later
    -- self-contradictory, both symptoms of the same deleted literal.

    HOW IT FAILS: reverting `usable_band` to derive its lower edge from
    `binder > 0.1` (ignoring the `crossing` argument) makes this fail --
    the fixture-sanity assertion below confirms the ring's own binder
    really does cross 0.1 within this sweep, so the reverted code would
    report a spurious band instead of None. Verified directly: temporarily
    restored the old `started = [r for r in good if r.binder > 0.1]` /
    `lo = started[0].beta_j` logic (ignoring `crossing`) and re-ran this
    test -- it failed with a non-None band, before confirming it passes
    against the actual crossing-gated implementation.
    PROVENANCE: Ising, E. (1925), "Beitrag zur Theorie des
    Ferromagnetismus" -- the 1D Ising chain/ring has no finite-temperature
    ordering transition at any nonzero temperature, for any coupling
    strength."""
    n = 16
    edges = tuple((i, (i + 1) % n) for i in range(n))

    def model_fn(size, beta_j):
        return IsingModel(nodes=tuple(f"x{i}" for i in range(n)), edges=edges,
                          weights=np.full(n, 1.0), biases=np.zeros(n),
                          beta=beta_j, offset=0.0)

    rows = sweep(model_fn, sizes=[n], couplings=[0.05, 0.2, 0.4, 0.6], seed=0,
                n_chains=8, n_samples=1000, n_warmup=1000, steps=4,
                max_samples=8000)

    crossed_the_deleted_literal = [r for r in rows if r.binder > 0.1]
    assert crossed_the_deleted_literal, (
        "fixture sanity: this test needs the ring's REAL measured binder "
        "to cross the deleted 0.1 literal somewhere in the sweep, or it "
        "would not actually catch a reversion to that logic -- got binder "
        f"values {[r.binder for r in rows]}")

    assert usable_band(rows, None) is None


def test_sweep_defaults_clear_the_ess_reliability_floor():
    """n_chains * n_samples must clear tsu.ess's own reliability floor in the
    BEST case (tau ~ 1) -- otherwise every row sweep() produces at its own
    defaults is unconditionally ESS-unavailable regardless of how well the
    chain mixes, independent of mixing quality entirely. This is exactly the
    contradiction this task's Concern 1 found (8*400=3,200 against a floor of
    5,000, even at tau~1).

    Reads the floor from tsu.ess directly (not a hardcoded 5000) and the
    defaults off sweep's own signature via inspect.signature (not restated
    literals), so this test tracks either constant if it is ever re-derived
    rather than silently drifting from the code."""
    sig = inspect.signature(sweep)
    n_chains = sig.parameters["n_chains"].default
    n_samples = sig.parameters["n_samples"].default
    assert n_chains * n_samples >= RELIABILITY_MIN_N_OVER_TAU


@pytest.mark.slow
def test_sweep_runs_end_to_end_and_carries_the_ess_refusal_through():
    """A real (tiny) sampling run through sweep() itself -- not just the
    synthetic RegimeRow objects the tests above construct by hand -- to
    confirm analyse -> build_program -> sample_chains -> tsu.ess/r_hat are
    actually wired together correctly and produce well-typed rows.

    Makes NO claim about where a transition sits on this toy: a 3x3 lattice
    sampled for 40 draws/chain is far too small/short for that, and asserting
    a location here would be exactly the fabrication this module exists to
    refuse. What IS checked is the refusal itself: n_chains*n_samples = 160 is
    far below tsu.ess's own reliability floor (RELIABILITY_MIN_N_OVER_TAU =
    5000 -- see tsu/ess.py), so effective_sample_size MUST return
    ess=None/iat=None regardless of how well the chain mixes, and sweep() must
    carry that refusal through as None fields and ess_unavailable=True rather
    than substituting a raw standard error (trap #2 in this task's brief).
    `provisional` is a SEPARATE question (R-hat alone, see FIX 1) -- this toy
    run's R-hat is not pinned to any particular value, so `provisional` is
    checked for internal consistency against R-hat directly rather than
    against a fixed expectation.

    `max_samples=n_samples` pins this to a SINGLE attempt (FIX 3's escalation
    loop breaks immediately once the next doubling, 80, would exceed the
    40-sample budget) -- this test is about the refusal surfacing correctly,
    not about escalation, which gets its own tests below.
    """
    from world.fields import compile_layer

    def model_fn(size, beta_j):
        _, _, ising, _ = compile_layer(size, beta_j)
        return ising

    rows = sweep(model_fn, sizes=[3], couplings=[0.2, 0.6], seed=0,
                n_chains=4, n_samples=40, n_warmup=100, steps=2,
                max_samples=40)

    assert len(rows) == 2
    for r in rows:
        assert isinstance(r, RegimeRow)
        assert r.size == 3
        assert 0.0 <= r.abs_m <= 1.0
        assert -1e-9 <= r.binder <= 2.0 / 3.0 + 1e-9
        assert r.r_hat > 0.0
        # 4*40 = 160 total draws is nowhere near tsu.ess's reliability floor
        # of 5000, at ANY tau -- this is not a statement about mixing quality,
        # so it is unconditionally true regardless of what this toy's R-hat
        # happens to be. max_samples=40 forbids any escalation, so this stays
        # a single-attempt check.
        assert r.n_eff is None
        assert r.tau is None
        assert r.abs_m_err is None
        assert r.ess_unavailable is True
        assert r.ess_reason != ""
        assert r.n_samples_used == 40
        # provisional reflects R-hat ALONE (FIX 1) -- checked as a relation to
        # r.r_hat, not a fixed True/False, since this toy's actual R-hat isn't
        # pinned by this test.
        assert r.provisional == (r.r_hat > RHAT_THRESHOLD)


def _independent_spins_model(size, beta_j):
    """Zero edges, zero bias: every spin an independent fair coin flip on
    every draw, so tau ~= 1 EXACTLY (Sokal-estimated, not merely nominally --
    see the module docstring on `sample_chains`' `_edgeless_sample` path,
    which draws i.i.d. and bypasses thrml/JAX MCMC entirely). Deliberately
    used here instead of a real ferromagnet: the three escalation tests below
    are about the LOOP's bookkeeping (does it retry, does it stop, does it
    skip retrying), not about genuine critical slowing down -- which the
    already-slow `test_sweep_runs_end_to_end_and_carries_the_ess_refusal_through`
    test above exercises with a real thrml sampling run instead. `beta_j` is
    accepted (to match `sweep`'s `model_fn(size, beta_j)` contract) and
    unused, since there is nothing for it to couple."""
    return IsingModel(nodes=tuple(f"n{i}" for i in range(size)),
                      edges=(), weights=np.array([], dtype=float),
                      biases=np.zeros(size), beta=1.0, offset=0.0)


def test_a_refused_row_escalates_until_it_clears():
    """A row that refuses at the base draw count must be resampled at larger
    n_samples rather than accepted as unavailable outright -- tau grows near
    a transition (critical slowing down), so a fixed draw count is guaranteed
    to fail exactly where the measurement matters most. n_chains=8,
    n_samples=100 (n_total=800) is far short of the 5000 floor at any tau, so
    the first attempt is certain to refuse; doubling (100->200->400->800)
    reaches n_total=6400, comfortably over the floor at tau~1.

    Mutation-checked: accepting the first refusal instead of escalating
    (`break` right after the first `effective_sample_size` call, before the
    `if est.reliable` check) makes this test fail, since n_samples_used would
    then equal the starting 100 and tau/n_eff would stay None. Verified by
    temporarily reverting `sweep`'s loop and re-running this test."""
    rows = sweep(_independent_spins_model, sizes=[4], couplings=[0.0], seed=0,
                n_chains=8, n_samples=100, n_warmup=10, steps=1,
                max_samples=20_000)
    r = rows[0]
    assert r.n_samples_used > 100, \
        "a row that refuses at n_samples=100 must escalate past it"
    assert r.tau is not None
    assert r.n_eff is not None
    assert r.ess_unavailable is False


def test_escalation_stops_at_the_max_samples_budget():
    """Escalation must not loop forever chasing a floor a tiny budget can
    never reach: n_chains=4, n_samples=50 (n_total=200) refuses by roughly an
    order of magnitude regardless of tau, and max_samples=60 is barely above
    the starting n_samples, so the very next doubling (100) already exceeds
    it. The row must come back unavailable rather than escalating past the
    stated budget, and `n_samples_used`/`ess_reason` must reflect the actual
    (failed) attempt made, not a number that was never sampled."""
    rows = sweep(_independent_spins_model, sizes=[4], couplings=[0.0], seed=0,
                n_chains=4, n_samples=50, n_warmup=10, steps=1,
                max_samples=60)
    r = rows[0]
    assert r.ess_unavailable is True
    assert r.n_samples_used <= 60
    assert "n_samples=" in r.ess_reason, \
        "the refusal reason must name the largest attempt actually made"


def test_a_row_that_clears_on_the_first_attempt_does_not_escalate():
    """The complementary case to both tests above: an easy row (n_chains=8,
    n_samples=1000, n_total=8000, well over the 5000 floor at tau~1) must NOT
    be resampled at all. This is the test that would catch an escalation loop
    that always iterates at least once regardless of the first result --
    which would silently spend the whole compute budget even on rows that
    never needed it."""
    rows = sweep(_independent_spins_model, sizes=[4], couplings=[0.0], seed=0,
                n_chains=8, n_samples=1000, n_warmup=10, steps=1,
                max_samples=32_000)
    r = rows[0]
    assert r.n_samples_used == 1000
    assert r.ess_unavailable is False
    assert r.tau is not None


# ---------------------------------------------------------------------------
# F7 (branch review): escalation must not retry a PERMANENT refusal (one no
# number of draws can cure) and must not report it as if a compute budget
# were the limit. The review's live repro: a single spin, --beta-steps 3,
# escalated 2000 -> 4000 -> ... -> 32,000 (16x the compute, three times
# over) before reporting "series is constant (zero variance)...
# (largest attempt: n_samples=32000, budget max_samples=32000)" -- text
# that reads to any user as "we ran out of budget; raise --max-samples",
# which would not help: a numerically constant series has no variance for
# ANY number of draws to reveal.
# ---------------------------------------------------------------------------

def _frozen_spin_model(size, beta_j):
    """A single spin with a bias large enough that beta_j*b clears ~18.4 --
    the point past which `p_plus = 1/(1+exp(-2*beta*b))` rounds to EXACTLY
    1.0 in float64 (verified directly: `np.random.default_rng(0).random()`
    draws are always < 1.0, so `draws < p_plus` is unconditionally True) --
    reproducing tsu.ess's 'series is constant (zero variance)' refusal
    deterministically, the same shape as the branch review's own
    single-spin live repro, without depending on a specific --beta-steps
    value or a near-miss numeric coincidence."""
    return IsingModel(nodes=("only",), edges=(),
                      weights=np.array([], dtype=float),
                      biases=np.array([1000.0]), beta=beta_j, offset=0.0)


def test_a_permanent_refusal_is_not_retried_and_does_not_blame_the_budget():
    """WHAT THIS PINS: a refusal that MORE DRAWS CANNOT CURE (a numerically
    constant series) is not escalated past the STARTING n_samples, and its
    reason does not claim a budget was exhausted.
    HOW IT FAILS: an escalation loop that doubles on EVERY refusal
    (structural or not) makes `r.n_samples_used` equal `max_samples`
    (32,000) instead of the starting 2,000 -- it always escalates all the
    way to the budget on a series that can never clear the reliability
    floor at any draw count. And an unconditional
    `f"{reason} (largest attempt: ..., budget max_samples=...)"` suffix
    makes `r.ess_reason` contain 'budget'/'largest attempt' even for this
    permanent, unfixable refusal, implying to a reader that raising
    --max-samples would help.
    PROVENANCE: tsu.ess.effective_sample_size's own reason string for a
    constant series ('unavailable: series is constant (zero variance);
    autocorrelation is undefined'); branch review F7's single-spin live
    reproduction (beta_j=0.05, beta-steps=3)."""
    rows = sweep(_frozen_spin_model, sizes=[1], couplings=[0.05], seed=0,
                n_chains=8, n_samples=2000, n_warmup=10, steps=1,
                max_samples=32_000)
    r = rows[0]
    assert r.ess_unavailable is True
    assert "series is constant" in r.ess_reason
    assert r.n_samples_used == 2000, \
        "a permanent (uncurable) refusal must not be escalated past the " \
        "starting n_samples"
    assert "budget" not in r.ess_reason, \
        "a permanent refusal's reason must not imply a budget shortfall"
    assert "largest attempt" not in r.ess_reason, \
        "a permanent refusal's reason must not name an escalation attempt " \
        "that was never actually needed"


def test_a_curable_refusal_still_escalates_and_still_names_its_budget():
    """Complementary case: N/tau-below-floor (curable in principle by more
    draws) must still escalate and still carry the 'largest attempt'
    budget wording -- F7's fix must distinguish the two refusal REASONS,
    not stop escalating altogether. `max_samples=150` (vs. n_samples=50)
    is chosen so ONE doubling (50 -> 100) fits inside the budget before
    the next (100 -> 200) would exceed it -- confirming an actual
    escalation happens, not merely that the loop exits immediately (which
    `test_escalation_stops_at_the_max_samples_budget`'s own tighter budget
    already covers)."""
    rows = sweep(_independent_spins_model, sizes=[4], couplings=[0.0], seed=0,
                n_chains=4, n_samples=50, n_warmup=10, steps=1,
                max_samples=150)
    r = rows[0]
    assert r.ess_unavailable is True
    assert r.n_samples_used == 100, \
        "a curable refusal (N/tau below the floor) must still escalate " \
        "(one doubling fits inside this fixture's budget)"
    assert "n_samples=" in r.ess_reason
    assert "budget max_samples=" in r.ess_reason


# ---------------------------------------------------------------------------
# F6 (branch review): N_eff must never exceed the number of draws actually
# taken. tsu.ess's windowed tau estimate can dip below 1.0 on a fast-mixing
# chain (routine, not a bug in tsu.ess -- see ess.py's own module docstring),
# which makes N_eff = N_total/tau print LARGER than N_total. "your effective
# sample size exceeds your sample size" is flagged in the branch review as
# the one line that ends a conversation with a reviewer, even though the
# review independently confirmed (exact Boltzmann enumeration, RMS sigma_off
# 1.07 over 10 couplings) that the resulting error bars are still correctly
# calibrated -- this is a presentation/estimator-artifact defect, not a wrong
# number. tsu.ess itself is deliberately NOT touched here (it is validated
# against arviz/statsmodels with its own test suite); the floor belongs at
# the one call site that renders tau/N_eff to a reader.
# ---------------------------------------------------------------------------

def test_reported_ess_is_floored_and_never_exceeds_the_total_draw_count(monkeypatch):
    """WHAT THIS PINS: `sweep()` floors the REPORTED tau at 1.0 (and, with
    it, N_eff = N_total/tau) before it goes into a RegimeRow, even when
    tsu.ess's own estimator reports a raw tau below 1. tau_A = 1 +
    2*sum_k rho_k is exactly 1.0 for i.i.d. draws (every rho_k is 0 beyond
    lag 0), so a windowed ESTIMATE below 1 is an artifact of Sokal's
    finite-window truncation on a fast-mixing chain -- no physical process
    mixes better than white noise. Flooring is CONSERVATIVE: it can only
    shrink an over-large N_eff back toward N_total (never inflate it
    further) and can only widen `abs_m_err` (never understate it) --
    checked here by asserting `stderr_from_ess` is called with the FLOORED
    ess, not tsu.ess's raw (inflated) one.

    `effective_sample_size` is monkeypatched (not tsu.ess itself, which
    this task's brief says must stay untouched) to return a controlled
    EssEstimate with tau=0.8 -- exactly the "tau dips below 1" shape the
    branch review reproduced live (N_eff=35,153 from N_total=32,000 on a
    16-node ring).

    HOW IT FAILS: without a floor, `r.tau`/`r.n_eff` pass tsu.ess's raw
    EssEstimate straight through, so `r.tau == 0.8` (fails `>= 1.0`) and
    `r.n_eff == n_total/0.8 == 1.25*n_total` (exceeds n_total). And if the
    floor were applied to `tau`/`n_eff` for display only but NOT threaded
    into the `stderr_from_ess` call, the captured `ess` argument would be
    the raw, inflated `n_total/0.8` instead of `n_total` -- an error bar
    computed from a too-large ESS is a too-SMALL (falsely precise) stderr,
    exactly the overstatement the review's own "conservative" framing
    rules out.
    PROVENANCE: tsu.ess's module docstring (Sokal 1989 sec. 3.3) for why
    tau_A = 1 for i.i.d. draws; N_eff=35,153/N_total=32,000 is the branch
    review's own live reproduction (F6)."""
    import tsu.preflight.sweep as sweep_mod
    from tsu.ess import EssEstimate

    n_chains, n_samples = 8, 100
    n_total = n_chains * n_samples

    def fake_ess(chains, *a, **k):
        # A raw tau below 1: the windowing artifact this floor exists to
        # neutralise at the reporting boundary, not a real chain property.
        return EssEstimate(ess=n_total / 0.8, iat=0.8, n_total=n_total,
                           reliable=True, reason="")
    monkeypatch.setattr(sweep_mod, "effective_sample_size", fake_ess)

    captured = {}
    real_stderr = sweep_mod.stderr_from_ess

    def spying_stderr(x, ess):
        captured["ess"] = ess
        return real_stderr(x, ess)
    monkeypatch.setattr(sweep_mod, "stderr_from_ess", spying_stderr)

    rows = sweep_mod.sweep(_independent_spins_model, sizes=[4], couplings=[0.0],
                           seed=0, n_chains=n_chains, n_samples=n_samples,
                           n_warmup=10, steps=1, max_samples=n_samples)
    r = rows[0]
    assert r.tau == pytest.approx(1.0), \
        "a raw tau below 1.0 must be floored to 1.0 when reported"
    assert r.n_eff == pytest.approx(n_total), \
        "N_eff must never be reported larger than the total draws taken"
    assert captured["ess"] == pytest.approx(n_total), \
        "abs_m_err must be computed from the FLOORED ess, not tsu.ess's raw, " \
        "artificially-inflated one -- otherwise flooring tau/n_eff for " \
        "display would leave the error bar just as falsely precise as before"


def test_report_legend_states_the_tau_and_n_eff_convention(tmp_path):
    """WHAT THIS PINS: report.md's legend states the tau/N_eff convention
    this tool actually uses (tau_A = 1 + 2*sum_k rho_k, Sokal/emcee; N_eff =
    N_total/tau_A) so a reader is not left to guess which of several
    published conventions (the branch review notes the design spec itself
    is ambiguous about this) the printed numbers follow.
    HOW IT FAILS: a report.md whose legend never states the formula makes
    every substring assertion below fail.
    PROVENANCE: branch review F6 ("report.md prints a tau column and an
    N_eff column and never states the convention... Add one line to the
    legend")."""
    from tsu.preflight.render import write_regime
    rows = _rows(16, [0.3], [0.2])
    write_regime(rows, None, None, tmp_path, onsager=None)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "tau_A" in text or "1 + 2" in text, \
        "the legend must state the tau_A = 1 + 2*sum rho_k convention"
    assert "N_eff" in text and "tau_A" in text, \
        "the legend must state N_eff = N_total / tau_A"


# ---------------------------------------------------------------------------
# F1 (branch review): `regime.json["onsager_note"]` was hardcoded to "not
# applicable to this graph" whenever `onsager` was None -- collapsing "I did
# not check" into "it does not apply" is a false claim about the exact
# constant this project's spec opens with (an 8x8 uniform square lattice --
# the one graph Onsager solved -- printed that false note on every run,
# because the code never checked). `detect_uniform_square_lattice` replaces
# the hardcoded `onsager=None` in `tsu.cli.sweep_single_model` with an actual,
# deliberately conservative check: three distinct claims (applies / checked
# and does not apply / not determined), never a binary. No test anywhere in
# the suite previously read `regime.json["onsager_note"]` at all (the review's
# own grep confirmed this) -- every test below does.
# ---------------------------------------------------------------------------

def _open_grid(n: int, j: float = 0.4, b: float = 0.0) -> IsingModel:
    """An n x n open (non-periodic) 4-neighbour square lattice, uniform |J|,
    zero bias by default -- the exact graph class Onsager solved. Duplicated
    from test_preflight_check.py's/test_preflight_render.py's own `grid()`
    fixture rather than imported, per this project's stated convention (no
    tests/__init__.py, no cross-test-file imports elsewhere in the suite)."""
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in idx:
                e.append((i, idx[(x + dx, y + dy)]))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)),
                      edges=tuple(e), weights=np.full(len(e), j),
                      biases=np.full(n * n, b), beta=1.0, offset=0.0)


def test_onsager_betac_matches_the_textbook_constant():
    """betac = ln(1+sqrt(2)) / (2*|J|max) -- Onsager's exact result (Onsager
    1944), computed from its closed form here, not asserted against
    `demo.lattice_app`'s copy (which now imports THIS function -- see
    test_frontier.py's existing `test_onsager_betac_matches_the_textbook_constant`
    for the demo-side re-export check)."""
    import math
    from tsu.preflight.sweep import onsager_betac
    j_max = 1.7
    expected = math.log(1.0 + math.sqrt(2.0)) / 2.0 / j_max
    assert onsager_betac(j_max) == pytest.approx(expected, rel=1e-12)


def test_detect_uniform_square_lattice_applies_on_an_open_grid():
    """WHAT THIS PINS: an 8x8 open uniform square lattice in zero field --
    the one graph Onsager solved -- is DETECTED, and the returned Kc matches
    the closed form in THIS model's own beta*J units (j=1, so kc ==
    onsager_betac(1.0) numerically). This is the exact live case the branch
    review reproduced: chi peaked at beta*J=0.4778 against Kc=0.4407 while
    `regime.json["onsager_note"]` read "not applicable to this graph".
    HOW IT FAILS: a detection that still hardcodes None (or one that is
    too timid and reports "not determined" even on this textbook case)
    makes `kc` None here, failing the `is not None` assertion below.
    PROVENANCE: branch review F1's own reproduction (8x8 lattice, Kc =
    0.4407 = ln(1+sqrt(2))/2)."""
    from tsu.preflight.sweep import detect_uniform_square_lattice, onsager_betac
    model = _open_grid(8, j=1.0, b=0.0)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is not None, f"must detect the 8x8 open grid as a square lattice, got note={note!r}"
    assert kc == pytest.approx(onsager_betac(1.0))
    assert kc == pytest.approx(0.4407, abs=1e-4)
    assert "applies" in note.lower() or "square lattice" in note.lower()
    assert "not applicable" not in note.lower()


def test_detect_uniform_square_lattice_scales_kc_with_j():
    """kc must be reported in the SWEPT model's own beta*J units, i.e.
    onsager_betac(j_uniform) -- not the unit-coupling constant regardless of
    the model's actual |J|."""
    from tsu.preflight.sweep import detect_uniform_square_lattice, onsager_betac
    model = _open_grid(6, j=2.5, b=0.0)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is not None
    assert kc == pytest.approx(onsager_betac(2.5))


def test_detect_uniform_square_lattice_does_not_apply_on_nonzero_field():
    """A nonzero bias breaks Onsager's zero-field precondition -- this must
    be a POSITIVELY VERIFIED 'does not apply', never 'not determined' (the
    reason is known, not merely unchecked)."""
    from tsu.preflight.sweep import detect_uniform_square_lattice
    model = _open_grid(4, j=0.4, b=0.1)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is None
    assert "does not apply" in note.lower()
    assert "field" in note.lower() or "bias" in note.lower()


def test_detect_uniform_square_lattice_does_not_apply_on_nonuniform_coupling():
    """Non-uniform |J| breaks Onsager's uniform-coupling precondition."""
    from tsu.preflight.sweep import detect_uniform_square_lattice
    model = _open_grid(4, j=0.4, b=0.0)
    weights = np.array(model.weights, dtype=float)
    weights[0] *= 3.0
    model = IsingModel(nodes=model.nodes, edges=model.edges, weights=weights,
                       biases=model.biases, beta=model.beta, offset=model.offset)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is None
    assert "does not apply" in note.lower()
    assert "uniform" in note.lower()


def test_detect_uniform_square_lattice_does_not_apply_on_a_1d_ring():
    """A 1-D ring is bipartite-adjacent-ish but is not a 2-D square lattice
    at all -- it must not be misdetected as one. This is the same fixture
    `usable_band`'s own tests use to pin the "no transition exists" case
    (Ising 1925): if this were ever misdetected as a square lattice, its
    report would print a fabricated Kc next to a model that provably has
    no finite-temperature transition at any coupling."""
    from tsu.preflight.sweep import detect_uniform_square_lattice
    n = 16
    edges = tuple((i, (i + 1) % n) for i in range(n))
    model = IsingModel(nodes=tuple(f"x{i}" for i in range(n)), edges=edges,
                       weights=np.full(n, 1.0), biases=np.zeros(n),
                       beta=1.0, offset=0.0)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is None
    assert "does not apply" in note.lower() or "not determined" in note.lower()


def test_detect_uniform_square_lattice_does_not_apply_on_two_components():
    """Two disjoint grids are not "a single connected uniform square
    lattice" -- Onsager's result is for one connected lattice."""
    from tsu.preflight.sweep import detect_uniform_square_lattice
    a = _open_grid(3, j=0.4, b=0.0)
    b = _open_grid(3, j=0.4, b=0.0)
    offset = len(a.nodes)
    nodes = a.nodes + tuple(f"m{i}" for i in range(len(b.nodes)))
    edges = a.edges + tuple((u + offset, v + offset) for u, v in b.edges)
    weights = np.concatenate([a.weights, b.weights])
    biases = np.concatenate([a.biases, b.biases])
    model = IsingModel(nodes=nodes, edges=edges, weights=weights,
                       biases=biases, beta=1.0, offset=0.0)
    kc, note = detect_uniform_square_lattice(model)
    assert kc is None
    assert "does not apply" in note.lower()
    assert "component" in note.lower() or "connected" in note.lower()


def test_detect_uniform_square_lattice_reports_not_determined_when_the_embed_search_is_inconclusive(monkeypatch):
    """WHAT THIS PINS: when the bounded grid-embedding search cannot find
    an embedding within its step budget, that is NOT proof the graph is
    not a square lattice (`_try_grid_embed`'s own docstring: it "never
    claims the graph is NOT grid-embeddable"). Reporting "does not apply"
    here would be exactly F1's mistake one level down -- turning "I could
    not confirm it" into "it is false". Must report "not determined".
    HOW IT FAILS: treating a None from `_try_grid_embed` as a confirmed
    negative (instead of an inconclusive search) makes `note` say "does
    not apply" instead of "not determined"."""
    import tsu.preflight.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "_try_grid_embed", lambda g, **k: None)
    model = _open_grid(4, j=0.4, b=0.0)
    kc, note = sweep_mod.detect_uniform_square_lattice(model)
    assert kc is None
    assert "not determined" in note.lower()


def test_write_regime_honours_an_explicit_onsager_note(tmp_path):
    """WHAT THIS PINS: `write_regime`'s `onsager_note` keyword, when
    supplied, is written into `regime.json["onsager_note"]` VERBATIM --
    the caller's three-state judgement (from `detect_uniform_square_lattice`)
    must not be collapsed back down by `write_regime`'s own fallback
    binary logic. The branch review's own grep (`grep -rn "onsager"
    tests/`) found NO test anywhere reading `regime.json["onsager_note"]`
    at all -- this is that test (see test_cli.py for the full end-to-end
    check through a real `tsu regime` run).
    HOW IT FAILS: `write_regime` ignoring `onsager_note` and falling back
    to its own onsager-is-None binary guess makes the JSON field read
    "not determined: this call did not report..." (the new default)
    instead of the exact string passed in below."""
    import json
    from tsu.preflight.render import write_regime
    rows = _rows(16, [0.3], [0.2])
    custom_note = "checked and does not apply: test stub reason"
    paths = write_regime(rows, None, None, tmp_path, onsager=None,
                         onsager_note=custom_note)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["onsager_note"] == custom_note


def test_write_regime_defaults_to_not_determined_rather_than_not_applicable(tmp_path):
    """WHAT THIS PINS: F1's exact regression -- when a caller supplies
    `onsager=None` WITHOUT an explicit `onsager_note` (i.e. never checked
    applicability at all), `write_regime` must NOT claim "not applicable to
    this graph" (a false claim about a fact it never checked). It must say
    it does not know.
    HOW IT FAILS: reverting to the old two-branch literal keyed only on
    `onsager is not None` makes `data["onsager_note"] ==
    "not applicable to this graph"` -- exactly F1's shipped defect.
    PROVENANCE: branch review F1, live reproduction on an 8x8 uniform
    square lattice (`onsager_kc`: None, `onsager_note`: "not applicable to
    this graph", while chi peaked at beta*J=0.4778 against Kc=0.4407)."""
    import json
    from tsu.preflight.render import write_regime
    rows = _rows(16, [0.3], [0.2])
    paths = write_regime(rows, None, None, tmp_path, onsager=None)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["onsager_note"] != "not applicable to this graph"
    assert "not determined" in data["onsager_note"].lower()
