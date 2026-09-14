"""Item 3: `tsuc simulate` -- compile once, sample many.

Reconstructs the SamplingProgram straight from a receipt's own program.json
(nodes/edges/weights/biases/beta/offset, verbatim) and samples it with fresh
parameters, WITHOUT re-running encode/lower/gate_checks/place/route/search --
`compile_spec` must never be called by any of this."""
import json

import numpy as np
import pytest

from tsu_compiler.cli import main
from tsu_compiler.spec import load_spec
from tsu_compiler.target import Z1
from tsu_compiler.passes.search import compile_spec
from tsu_compiler.backends.thrml_backend import exact_conditional_distribution, sample_chains
from tsu_compiler.passes.verify import tv_noise_floor
from tsu_compiler.receipt import write_receipt
from tsu_compiler.simulate import reconstruct_program, simulate


def _toy_receipt(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    return write_receipt(c, tmp_path / "r")


# ---------------------------------------------------------------------------
# reconstruction: the program is READ from the receipt, never re-derived
# ---------------------------------------------------------------------------

def test_reconstruct_program_matches_program_json_bit_for_bit(tmp_path):
    d = _toy_receipt(tmp_path)
    prog_json = json.loads((d / "program.json").read_text())
    prog = reconstruct_program(d)
    assert list(prog.ising.nodes) == prog_json["nodes"]
    assert [list(e) for e in prog.ising.edges] == prog_json["edges"]
    assert np.allclose(prog.ising.weights, prog_json["weights"])
    assert np.allclose(prog.ising.biases, prog_json["biases"])
    assert prog.ising.beta == pytest.approx(prog_json["beta"])
    assert prog.ising.offset == pytest.approx(prog_json["offset"])
    assert [list(b) for b in prog.blocks] == prog_json["blocks"]


def test_simulate_never_calls_compile_spec(tmp_path, monkeypatch):
    """The whole point: `simulate` samples a receipt's ALREADY-compiled
    program -- it must never re-run the search/gate/placement pipeline that
    produced it. Patch `compile_spec` in the module where search.py's own
    machinery lives so ANY call anywhere in the simulate path blows up
    loudly rather than silently recompiling."""
    d = _toy_receipt(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("compile_spec was called -- simulate() recompiled")

    import tsu_compiler.passes.search as search_mod
    monkeypatch.setattr(search_mod, "compile_spec", _boom)

    path, got, im = simulate(d, n_chains=2, n_samples=5, n_warmup=20, seed=0)
    assert path.exists()
    assert got.shape == (10, len(im.nodes))


# ---------------------------------------------------------------------------
# simulation.json: parameters, clamp, task validity, codeword violations,
# a decoded example
# ---------------------------------------------------------------------------

def test_simulate_writes_simulation_json_with_the_required_fields(tmp_path):
    d = _toy_receipt(tmp_path)
    path, got, im = simulate(d, n_chains=4, n_samples=20, n_warmup=50, seed=1)
    assert path == d / "simulation.json"
    doc = json.loads(path.read_text())
    assert doc["params"]["seed"] == 1
    assert doc["params"]["n_chains"] == 4
    assert doc["params"]["n_samples"] == 20
    assert doc["params"]["n_warmup"] == 50
    assert doc["clamp"] == {}
    assert 0.0 <= doc["task_validity"] <= 1.0
    assert 0.0 <= doc["codeword_violation_rate"] <= 1.0
    assert isinstance(doc["decoded_example"], dict)
    assert set(doc["decoded_example"]["decoded"]) == {"a", "b", "c"}


def test_reconstruct_program_refuses_a_receipt_with_no_program(tmp_path):
    """A LOGICAL/HARDWARE receipt's program.json is not empty -- it still
    carries a bare {"clamp": {}} (write_receipt always writes that key) --
    so this must not be fooled by that into thinking a program is present."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    d = write_receipt(c, tmp_path / "r")
    with pytest.raises(ValueError, match="no program"):
        reconstruct_program(d)


def test_cli_simulate_subcommand(tmp_path):
    d = _toy_receipt(tmp_path)
    rc = main(["simulate", str(d), "--samples", "20", "--chains", "4",
              "--warmup", "50", "--seed", "2"])
    assert rc == 0
    assert (d / "simulation.json").exists()


# ---------------------------------------------------------------------------
# acceptance: reproduces the compiled distribution within the noise floor
# ---------------------------------------------------------------------------

def test_simulate_reproduces_the_compiled_distribution_within_noise_floor(tmp_path):
    """The RECONSTRUCTED program must be the same physical object the
    compile produced -- sampling it with a generous budget and comparing
    against the exact (clamp-aware) reference must land at or below the
    same noise floor the compiler's own `_verify` is held to."""
    d = _toy_receipt(tmp_path)
    prog = reconstruct_program(d)
    states, probs = exact_conditional_distribution(prog)

    chains = sample_chains(prog, n_chains=32, n_samples=300, n_warmup=400,
                           steps_per_sample=2, seed=7)
    got = chains.reshape(-1, chains.shape[-1])

    state_index = {tuple(int(x) for x in row): i for i, row in enumerate(states)}
    idx = np.array([state_index[tuple(int(x) for x in row)] for row in got])
    hist = np.bincount(idx, minlength=len(probs)).astype(float)
    hist /= hist.sum()
    execution_tv = float(0.5 * np.abs(hist - probs).sum())
    floor = tv_noise_floor(probs, len(got))
    assert execution_tv < max(5 * floor, 0.05)


# ---------------------------------------------------------------------------
# a different seed yields a different draw
# ---------------------------------------------------------------------------

def test_simulate_different_seed_yields_a_different_draw(tmp_path):
    d = _toy_receipt(tmp_path)
    _, got_a, _ = simulate(d, n_chains=8, n_samples=20, n_warmup=50, seed=0)
    _, got_b, _ = simulate(d, n_chains=8, n_samples=20, n_warmup=50, seed=1)
    assert not np.array_equal(got_a, got_b)


# ---------------------------------------------------------------------------
# a --clamp is honoured in every returned sample
# ---------------------------------------------------------------------------

def test_simulate_clamp_is_honoured_in_every_returned_sample(tmp_path):
    d = _toy_receipt(tmp_path)
    path, got, im = simulate(d, n_chains=6, n_samples=30, n_warmup=50, seed=3,
                             clamp={"a": 1})
    idx = {n: i for i, n in enumerate(im.nodes)}
    assert np.all(got[:, idx["a"]] == 1)
    doc = json.loads(path.read_text())
    assert doc["clamp"] == {"a": 1}
    assert doc["decoded_example"]["decoded"]["a"] == 1


def test_simulate_clamp_to_zero_is_also_honoured(tmp_path):
    """Regression against 'clamped silently means clamped to 1' (same bug
    class test_clamp.py guards against for the compiler proper)."""
    d = _toy_receipt(tmp_path)
    _, got, im = simulate(d, n_chains=6, n_samples=30, n_warmup=50, seed=4,
                          clamp={"a": 0})
    idx = {n: i for i, n in enumerate(im.nodes)}
    assert np.all(got[:, idx["a"]] == 0)


def test_simulate_with_no_clamp_argument_reuses_the_receipts_own_clamp(tmp_path):
    """`clamp=None` (the default -- no --clamp given) must reproduce
    whatever the receipt itself was compiled under, not silently unclamp
    it."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    d = write_receipt(c, tmp_path / "r")
    path, got, im = simulate(d, n_chains=6, n_samples=20, n_warmup=50, seed=5)
    idx = {n: i for i, n in enumerate(im.nodes)}
    assert np.all(got[:, idx["a"]] == 1)
    doc = json.loads(path.read_text())
    assert doc["clamp"] == {"a": 1}


def test_simulate_explicit_empty_clamp_overrides_the_receipts_own_clamp(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    d = write_receipt(c, tmp_path / "r")
    path, got, im = simulate(d, n_chains=16, n_samples=100, n_warmup=100,
                             seed=6, clamp={})
    idx = {n: i for i, n in enumerate(im.nodes)}
    a_vals = got[:, idx["a"]]
    assert a_vals.min() != a_vals.max(), \
        "an explicit empty clamp must actually unclamp 'a', not reuse the receipt's"


# ---------------------------------------------------------------------------
# a beta override changes the sampled temperature without recompiling
# ---------------------------------------------------------------------------

def test_simulate_beta_override_is_recorded_and_used(tmp_path):
    d = _toy_receipt(tmp_path)
    path, got, im = simulate(d, n_chains=4, n_samples=10, n_warmup=30, seed=8,
                             beta=2.5)
    assert im.beta == pytest.approx(2.5)
    doc = json.loads(path.read_text())
    assert doc["params"]["beta"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# RP-1: a receipt directory can be git-tracked, frozen compile-time evidence
# (demo/receipts/small is exactly this in the live app) -- simulate() must
# be able to leave it completely untouched when a caller supplies its own
# output_dir, rather than being forced to write simulation.json into the
# same directory it was asked to read.
# ---------------------------------------------------------------------------

def test_simulate_with_output_dir_leaves_the_receipt_directory_untouched(tmp_path):
    d = _toy_receipt(tmp_path)
    before = sorted(p.name for p in d.iterdir())
    out_dir = tmp_path / "sim_output"
    path, got, im = simulate(d, n_chains=2, n_samples=5, n_warmup=20, seed=0,
                             output_dir=out_dir)
    assert path == out_dir / "simulation.json"
    assert path.exists()
    assert not (d / "simulation.json").exists()
    assert sorted(p.name for p in d.iterdir()) == before
