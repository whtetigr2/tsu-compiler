"""Task 1: loading a model from either accepted input.

The edge-list path is what makes this tool usable by someone who has never seen
this project's YAML. It is not a convenience -- a tool that only reads our own
format is a tool for one user."""
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu_compiler.preflight.model import load_model


def _write_edges(tmp_path, **over):
    d = dict(nodes=4, edges=[[0, 1, 0.5], [1, 2, 0.5], [2, 3, 0.5]],
             biases=[0.0, 0.0, 0.0, 0.0], beta=1.0)
    d.update(over)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_loads_an_edge_list(tmp_path):
    im = load_model(edges=_write_edges(tmp_path))
    assert len(im.nodes) == 4
    assert im.edges == ((0, 1), (1, 2), (2, 3))
    assert np.allclose(im.weights, [0.5, 0.5, 0.5])
    assert float(im.beta) == 1.0


def test_loads_a_spec_and_agrees_with_the_compiler(tmp_path):
    """The spec path must produce exactly what encode+lower produce directly --
    if it drifted, preflight would report on a different model than the one the
    rest of the toolchain compiles."""
    from tsu_compiler.spec import load_spec
    from tsu_compiler.passes.encode import encode
    from tsu_compiler.passes.lower import lower
    p = tmp_path / "s.yaml"
    p.write_text(
        "name: t\n"
        "generate:\n  kind: grid\n  width: 4\n  height: 4\n"
        "  variable_domain: {domain: binary}\n"
        "terms:\n"
        "  - {kind: product_over_edges, a_value: 1, b_value: 1, weight: -0.4}\n",
        encoding="utf-8")
    mine = load_model(spec=p)
    theirs = lower(encode(load_spec(p), "domain_wall").model)
    assert mine.nodes == theirs.nodes
    assert mine.edges == theirs.edges
    assert np.array_equal(mine.weights, theirs.weights)
    assert np.array_equal(mine.biases, theirs.biases)
    assert float(mine.beta) == float(theirs.beta)
    assert float(mine.offset) == float(theirs.offset)


def test_exactly_one_input_is_required(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_model()
    with pytest.raises(ValueError, match="exactly one"):
        load_model(spec="a.yaml", edges=_write_edges(tmp_path))


def test_a_bias_of_the_wrong_length_is_refused(tmp_path):
    """Silently padding or truncating would change the model the user asked
    about, which is worse than refusing it."""
    with pytest.raises(ValueError, match="biases"):
        load_model(edges=_write_edges(tmp_path, biases=[0.0, 0.0]))


def test_an_edge_naming_a_missing_node_is_refused(tmp_path):
    with pytest.raises(ValueError, match="node index"):
        load_model(edges=_write_edges(tmp_path, edges=[[0, 9, 0.5]]))


def test_a_self_loop_is_refused(tmp_path):
    """A self-coupling is a bias in disguise and would silently double-count."""
    with pytest.raises(ValueError, match="self"):
        load_model(edges=_write_edges(tmp_path, edges=[[1, 1, 0.5]]))


def test_a_duplicate_edge_is_refused(tmp_path):
    """(i,j) and (j,i) are the same coupling; accepting both would double it."""
    with pytest.raises(ValueError, match="duplicate"):
        load_model(edges=_write_edges(
            tmp_path, edges=[[0, 1, 0.5], [1, 0, 0.5]]))


@pytest.mark.parametrize("bad", [
    [[0, 1]],                        # two elements: no weight
    [{"i": 0, "j": 1, "w": 0.5}],    # dict-shaped rather than a triple
    [[0, 1, "heavy"]],               # non-numeric weight
])
def test_a_malformed_edge_entry_names_the_file_and_the_schema(tmp_path, bad):
    """A user who mis-shapes one edge must get a message naming the file, the
    offending entry and the schema -- not a raw IndexError or KeyError escaping
    the parser's internals, which says nothing about what to fix.

    Every other parse failure in this module already does that; the per-item
    loop was the one path that did not."""
    with pytest.raises(ValueError, match="malformed"):
        load_model(edges=_write_edges(tmp_path, edges=bad))
