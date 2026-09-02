"""Task 5 & 6: demo/scope.py's pure, headlessly-tested statistics -- the
temperature readout (Task 5) and the SCOPE panel's four readouts (Task 6:
autocorrelation, magnetization, energy histogram, local-field/sigmoid
response). Every test here checks against a signal with a KNOWN ANALYTIC
ANSWER (white noise decorrelates immediately, a slow sine stays correlated,
an all-ones/balanced draw has a known magnetization, a hand-picked bias/
edge model has an exactly computable local field), never against
scope.py's own output -- see the module docstring in demo/scope.py for why.
"""
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))


# ---------------------------------------------------------------------------
# Task 5: beta_to_temperature
# ---------------------------------------------------------------------------

def test_temperature_is_the_reciprocal_of_beta():
    from scope import beta_to_temperature
    assert beta_to_temperature(1.0) == 1.0
    assert beta_to_temperature(4.0) == 0.25


def test_zero_or_negative_beta_is_refused():
    """beta <= 0 is not a colder or hotter model, it is a meaningless one --
    exp(-beta*E) would invert or flatten the distribution."""
    from scope import beta_to_temperature
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            beta_to_temperature(bad)


# ---------------------------------------------------------------------------
# Task 6: autocorrelation
# ---------------------------------------------------------------------------

def test_autocorrelation_of_white_noise_decays_immediately():
    """Independent draws have no memory: acf[0] == 1 and every later lag sits
    near zero. Uses a fixed seed so the assertion is reproducible."""
    import random
    from scope import autocorrelation
    rng = random.Random(0)
    acf = autocorrelation([rng.gauss(0, 1) for _ in range(4000)], max_lag=10)
    assert acf[0] == 1.0
    assert all(abs(v) < 0.1 for v in acf[1:])


def test_autocorrelation_of_a_slow_sine_stays_high_at_short_lag():
    """A strongly correlated series keeps its memory across short lags."""
    from scope import autocorrelation
    series = [math.sin(i / 50.0) for i in range(4000)]
    acf = autocorrelation(series, max_lag=5)
    assert acf[1] > 0.9


# ---------------------------------------------------------------------------
# Task 6: magnetization
# ---------------------------------------------------------------------------

def test_magnetization_of_all_ones_is_one():
    from scope import magnetization
    assert magnetization([[1, 1, 1, 1]]) == [1.0]


def test_magnetization_of_balanced_draw_is_zero():
    from scope import magnetization
    assert magnetization([[1, 0, 1, 0]]) == [0.0]


# ---------------------------------------------------------------------------
# Task 6: energy_histogram
# ---------------------------------------------------------------------------

def test_energy_histogram_counts_sum_to_the_series_length():
    from scope import energy_histogram
    edges, counts = energy_histogram([1.0, 2.0, 2.0, 3.0], bins=3)
    assert sum(counts) == 4
    assert len(edges) == len(counts) + 1


# ---------------------------------------------------------------------------
# Task 6 step 6: local_field_response -- the measured sigmoid
# ---------------------------------------------------------------------------

class _FakeIsing:
    """A minimal stand-in carrying exactly the attributes
    local_field_response reads (nodes/edges/weights/biases/beta) -- the
    same shape as tsu.passes.lower.IsingModel, without pulling the whole
    compiler in for a pure-arithmetic test."""

    def __init__(self, nodes, edges, weights, biases, beta):
        self.nodes = nodes
        self.edges = edges
        self.weights = weights
        self.biases = biases
        self.beta = beta


def test_local_field_response_computes_the_exact_field_from_neighbours():
    """Two spins, one edge, no randomness: every draw is the SAME row
    ([1, 0], i.e. s0=+1, s1=-1), so each node's local field is an EXACT,
    hand-computable number (gamma_i = beta*(b_i + sum_j J_ij*s_j)), not a
    statistical estimate -- gamma0 = 1*(0.3 + 0.5*(-1)) = -0.2, gamma1 =
    1*(-0.1 + 0.5*(1)) = 0.4. Pooled over both nodes and all 40 draws, the
    field range is exactly [-0.2, 0.4]; with bins=2 the (hand-computable)
    bin edges are [-0.2, 0.1, 0.4] and centres [-0.05, 0.25]. Every one of
    node 0's 40 (gamma=-0.2) points falls in bin 0 with s0 always 1
    (P=1.0); every one of node 1's 40 (gamma=0.4) points falls in bin 1
    with s1 always 0 (P=0.0) -- an exact, not approximate, known answer."""
    from scope import local_field_response
    ising = _FakeIsing(nodes=("n0", "n1"), edges=((0, 1),), weights=[0.5],
                       biases=[0.3, -0.1], beta=1.0)
    draws = [[1, 0]] * 40
    centers, probs, counts = local_field_response(draws, ising, bins=2)
    assert len(centers) == len(probs) == len(counts) == 2
    assert centers[0] == pytest.approx(-0.05, abs=1e-9)
    assert centers[1] == pytest.approx(0.25, abs=1e-9)
    assert counts == [40, 40]
    assert probs[0] == pytest.approx(1.0)   # bin 0 = node 0's field, s0 always 1
    assert probs[1] == pytest.approx(0.0)   # bin 1 = node 1's field, s1 always 0


def test_local_field_response_recovers_the_analytic_sigmoid_from_synthetic_draws():
    """A single, edgeless spin with bias b under beta: the field is the
    CONSTANT gamma = beta*b, and P(s=1) is the exactly known
    sigmoid(2*gamma) by the model's own conditional-Gibbs definition (see
    demo/scope.py's module docstring / the thrml skill's P(s=1|nb)=
    sigmoid(2*gamma) convention). Draws are generated directly from that
    analytic Bernoulli(sigmoid(2*gamma)) law with a fixed seed -- so the
    'known answer' this test checks against is the closed-form sigmoid,
    not scope.py's own prior output."""
    import random
    from scope import local_field_response
    beta, b = 1.0, 0.6
    gamma = beta * b
    p_analytic = 1.0 / (1.0 + math.exp(-2.0 * gamma))
    rng = random.Random(42)
    n = 4000
    draws = [[1 if rng.random() < p_analytic else 0] for _ in range(n)]
    ising = _FakeIsing(nodes=("n0",), edges=(), weights=[], biases=[b], beta=beta)
    centers, probs, counts = local_field_response(draws, ising, bins=1)
    assert counts == [n]
    assert centers[0] == pytest.approx(gamma, abs=1e-9)
    # binomial standard error at n=4000: sqrt(p(1-p)/n) ~= 0.0077; 5 sigma
    # is a generous, non-flaky band for a fixed-seed reproducible draw.
    se = math.sqrt(p_analytic * (1 - p_analytic) / n)
    assert probs[0] == pytest.approx(p_analytic, abs=5 * se)


def test_local_field_response_reports_a_low_count_bin_as_unavailable_not_a_number():
    """A bin with too few samples to be meaningful must report its count,
    never a spuriously precise probability -- so its probability entry is
    NaN (never a plausible-looking float) while its count is exact."""
    from scope import local_field_response
    ising = _FakeIsing(nodes=("n0",), edges=(), weights=[], biases=[0.0], beta=1.0)
    draws = [[1], [0], [1], [1], [0]]  # 5 draws, well under any sane threshold
    centers, probs, counts = local_field_response(draws, ising, bins=1)
    assert counts == [5]
    assert math.isnan(probs[0])
