"""C1: clamping -- conditional resampling with variables held fixed.

Clamping is a WORKLOAD-level concept ("variable X is pinned to value v") that
must survive `encode` (a clamped categorical becomes clamped chain spins),
partition `build_program`'s blocks (a clamped node is never resampled, so it
must never appear in a free block), and reach the thrml backend's `sample`/
`sample_chains` (thrml supports this natively via `IsingSamplingProgram`'s
`clamped_blocks` and `sample_states`' `state_clamp`). No workload vocabulary
anywhere here -- clamping serves any spec, generic over what "variable X" means.
"""
import numpy as np
import pytest

from tsu.ir import Binary, Categorical, EnergyModel, Linear, LinearForm, Product, Var, VarRef
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse
from tsu.passes.program import build_program
from tsu.backends.thrml_backend import sample, sample_chains
from tsu.spec import load_spec, TaskContract, WorkloadSpec
from tsu.target import IDEAL, Z1
from tsu.passes.search import compile_spec


# --------------------------------------------------------------------------
# encode: a clamp survives into physical spin values
# --------------------------------------------------------------------------

def test_encode_clamp_maps_a_binary_variable_directly():
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    assert enc.encode_clamp({"a": 1}) == {"a": 1}
    assert enc.encode_clamp({"a": 0}) == {"a": 0}


def test_encode_clamp_maps_a_categorical_value_to_its_chain_domain_wall():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "domain_wall")
    # c has k=3; value 1 -> n_0=1, n_1=0 (delta_1 = n_0 - n_1)
    assert enc.encode_clamp({"c": 1}) == {"c__dw0": 1, "c__dw1": 0}


def test_encode_clamp_maps_a_categorical_value_to_its_chain_one_hot():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    assert enc.encode_clamp({"c": 2}) == {"c__oh0": 0, "c__oh1": 0, "c__oh2": 1}


def test_encode_clamp_is_a_partial_map_only_over_the_named_variables():
    """Clamping "a" must not mention "b" or "c" at all -- the whole point is
    that a caller can clamp a SUBSET at run time."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    out = enc.encode_clamp({"a": 1})
    assert set(out) == {"a"}


def test_encode_clamp_rejects_an_unknown_variable_name():
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    with pytest.raises(KeyError):
        enc.encode_clamp({"nonexistent": 1})


def test_encode_clamp_agrees_with_encode_assignment_on_a_full_assignment():
    """Clamping every variable must be identical to encoding a full assignment
    -- two different entry points into the SAME mapping, not two mappings."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    for asg in s.assignments():
        assert enc.encode_clamp(asg) == enc.encode_assignment(asg)


# --------------------------------------------------------------------------
# build_program: a clamped node must never appear in a free block
# --------------------------------------------------------------------------

def _ring_ising(n):
    return lower(EnergyModel(
        tuple(Var(f"x{i}", Binary()) for i in range(n)),
        tuple(Product(LinearForm({VarRef(f"x{i}"): 1.0}),
                      LinearForm({VarRef(f"x{(i + 1) % n}"): 1.0}), 1.0)
              for i in range(n)), 1.0))


def test_build_program_excludes_clamped_nodes_from_every_free_block():
    im = _ring_ising(6)
    report = analyse(im)
    prog = build_program(im, report, clamp={0: 1, 3: 0})
    clamped_members = {n for b in prog.blocks for n in b if n in (0, 3)}
    assert clamped_members == set(), "a clamped node appeared in a free block"
    assert set(prog.clamped) == {0, 3}


def test_build_program_free_blocks_plus_clamp_still_cover_every_node_once():
    im = _ring_ising(6)
    report = analyse(im)
    prog = build_program(im, report, clamp={2: 1})
    free_members = [n for b in prog.blocks for n in b]
    assert sorted(free_members) == [0, 1, 3, 4, 5]
    assert sorted(free_members + list(prog.clamped)) == list(range(6))


def test_build_program_records_the_clamp_values():
    im = _ring_ising(4)
    report = analyse(im)
    prog = build_program(im, report, clamp={1: 1, 2: 0})
    assert prog.clamp_values == {1: 1, 2: 0}


def test_build_program_with_no_clamp_behaves_exactly_as_before():
    """Regression: an absent/empty clamp must reproduce the old unclamped
    behaviour bit for bit -- clamping is additive, not a rewrite."""
    im = _ring_ising(5)
    report = analyse(im)
    plain = build_program(im, report)
    explicit_empty = build_program(im, report, clamp={})
    assert plain.blocks == explicit_empty.blocks
    assert plain.clamped == () and plain.clamp_values == {}


# --------------------------------------------------------------------------
# sample()/sample_chains(): every returned sample honours the clamp
# --------------------------------------------------------------------------

def _clamped_triangle_program(clamp):
    """3 spins, a triangle of repulsive couplings (so the free spins are not
    trivially independent), one spin clamped."""
    m = EnergyModel(
        (Var("a", Binary()), Var("b", Binary()), Var("c", Binary())),
        (Product(LinearForm({VarRef("a"): 1.0}), LinearForm({VarRef("b"): 1.0}), 1.5),
         Product(LinearForm({VarRef("b"): 1.0}), LinearForm({VarRef("c"): 1.0}), 1.5),
         Product(LinearForm({VarRef("a"): 1.0}), LinearForm({VarRef("c"): 1.0}), 1.5),
         Linear(LinearForm({VarRef("a"): 1.0}), 0.3)),
        1.0)
    im = lower(m)
    report = analyse(im)
    idx = {n: i for i, n in enumerate(im.nodes)}
    phys_clamp = {idx[n]: v for n, v in clamp.items()}
    prog = build_program(im, report, clamp=phys_clamp)
    return prog, m, idx


def test_sample_chains_every_draw_honours_the_clamped_value():
    prog, m, idx = _clamped_triangle_program({"a": 1})
    chains = sample_chains(prog, n_chains=6, n_samples=40, n_warmup=50,
                            steps_per_sample=2, seed=0)
    assert np.all(chains[:, :, idx["a"]] == 1)


def test_sample_chains_honours_a_clamp_to_zero_too():
    """Regression against a bug class where "clamped" silently means "clamped
    to 1" -- a clamp to 0 must be just as binding."""
    prog, m, idx = _clamped_triangle_program({"a": 0})
    chains = sample_chains(prog, n_chains=6, n_samples=40, n_warmup=50,
                            steps_per_sample=2, seed=0)
    assert np.all(chains[:, :, idx["a"]] == 0)


def test_sample_honours_the_clamp_in_its_flattened_form_too():
    prog, m, idx = _clamped_triangle_program({"b": 1})
    got = sample(prog, n_chains=8, n_samples=50, n_warmup=50,
                  steps_per_sample=2, seed=1)
    assert np.all(got[:, idx["b"]] == 1)


def test_unclamped_variables_still_vary_under_a_clamp():
    """A clamp must not accidentally freeze the WHOLE model -- the free spins
    must still show variation across draws."""
    prog, m, idx = _clamped_triangle_program({"a": 1})
    chains = sample_chains(prog, n_chains=16, n_samples=100, n_warmup=100,
                            steps_per_sample=2, seed=2)
    b_vals = chains[:, :, idx["b"]]
    assert b_vals.min() != b_vals.max(), \
        "the unclamped spin never varied across any draw -- clamp is over-binding"


def test_zero_edge_model_honours_a_clamp_too():
    """The edgeless closed-form sampling path (C3 in thrml_backend's own
    history) needs its own clamp support -- it is a SEPARATE code path from
    the general IsingSamplingProgram route."""
    m = EnergyModel((Var("a", Binary()), Var("b", Binary())),
                    (Linear(LinearForm({VarRef("a"): 1.0}), -1.5),
                     Linear(LinearForm({VarRef("b"): 1.0}), 0.2)), 1.0)
    im = lower(m)
    report = analyse(im)
    idx = {n: i for i, n in enumerate(im.nodes)}
    prog = build_program(im, report, clamp={idx["a"]: 1})
    chains = sample_chains(prog, n_chains=4, n_samples=20, n_warmup=5,
                            steps_per_sample=1, seed=0)
    assert np.all(chains[:, :, idx["a"]] == 1)
    b_vals = chains[:, :, idx["b"]]
    assert b_vals.min() != b_vals.max()


def test_sample_chains_refuses_zero_warmup_even_when_clamped():
    prog, m, idx = _clamped_triangle_program({"a": 1})
    with pytest.raises(AssertionError, match="n_warmup"):
        sample_chains(prog, n_chains=4, n_samples=4, n_warmup=0,
                      steps_per_sample=1, seed=0)


# --------------------------------------------------------------------------
# the unclamped variables reproduce their CONDITIONAL distribution
# --------------------------------------------------------------------------

def test_unclamped_variables_reproduce_the_conditional_distribution():
    """The reference here is computed directly from the IR's own `energy` --
    the same technique test_backends.py already uses for its unconditioned
    exact reference -- never a hand-rolled Boltzmann loop standing in for a
    backend. Conditioning on a=1, the reference is the Boltzmann distribution
    over (b, c) restricted to that slice, renormalised."""
    prog, m, idx = _clamped_triangle_program({"a": 1})
    chains = sample_chains(prog, n_chains=48, n_samples=400, n_warmup=500,
                            steps_per_sample=2, seed=7)
    got = chains.reshape(-1, chains.shape[-1])

    # exact conditional reference, computed from the IR model directly
    states = [(b, c) for b in (0, 1) for c in (0, 1)]
    weights = np.array([
        np.exp(-m.energy({"a": 1, "b": b, "c": c})) for b, c in states])
    probs = weights / weights.sum()

    hist = np.zeros(len(states))
    for row in got:
        b, c = int(row[idx["b"]]), int(row[idx["c"]])
        hist[states.index((b, c))] += 1
    hist /= hist.sum()

    assert np.abs(hist - probs).max() < 0.06


# --------------------------------------------------------------------------
# compile_spec: the parameter-form entry point, and the receipt
# --------------------------------------------------------------------------

def test_compile_spec_accepts_a_clamp_parameter_and_still_compiles():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    assert c.verdict == "COMPILED"


def test_compile_spec_with_no_clamp_is_unaffected_by_the_new_parameter():
    """Regression: adding the parameter must not change the unclamped path."""
    baseline = compile_spec(load_spec("specs/toy.yaml"), Z1)
    clamped_empty = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={})
    assert baseline.verdict == clamped_empty.verdict == "COMPILED"
    assert baseline.program.blocks == clamped_empty.program.blocks


def test_compile_spec_clamp_produces_samples_that_all_honour_it():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    assert c.verdict == "COMPILED"
    idx = {n: i for i, n in enumerate(c.program.ising.nodes)}
    assert idx["a"] in c.program.clamped
    assert c.program.clamp_values[idx["a"]] == 1


def test_receipt_records_which_variables_were_clamped_and_to_what(tmp_path):
    from tsu.receipt import write_receipt
    import json
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    d = write_receipt(c, tmp_path / "r")
    program = json.loads((d / "program.json").read_text())
    assert program["clamp"] == {"a": 1}


def test_receipt_of_an_unclamped_compile_records_an_empty_clamp(tmp_path):
    """A sample obtained under a clamp must never be mistakable for an
    unconditioned one -- the converse must hold too: an unclamped receipt must
    visibly record that NOTHING was clamped, not omit the field."""
    from tsu.receipt import write_receipt
    import json
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    program = json.loads((d / "program.json").read_text())
    assert program["clamp"] == {}
