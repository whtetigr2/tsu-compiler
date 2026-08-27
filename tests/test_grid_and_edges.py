"""A1-A3: the `grid` shape generator, the `product_over_edges` term template, and
the `forbid_value_pair_over_edges` validation rule -- all generic mechanisms over
a spec's edge set, following the `clique` generator's precedent ("a graph SHAPE,
not a problem domain"). No workload vocabulary here; the assertions are entirely
structural.
"""
import os
import tempfile
import textwrap

import pytest

from tsu.ir import Binary, Categorical, LinearForm, Product, VarRef
from tsu.spec import load_spec


def _write(body: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        f.write(textwrap.dedent(body))
        return f.name


def test_grid_generator_produces_4_neighbour_edges_on_a_2x2_lattice():
    p = _write("""
        name: grid_shape
        generate:
          kind: grid
          width: 2
          height: 2
          variable_domain: {domain: categorical, k: 3}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        assert len(s.variables) == 4
        assert all(isinstance(v.domain, Categorical) and v.domain.k == 3
                   for v in s.variables)
        # a 2x2 grid has exactly 4 edges: two horizontal, two vertical
        assert len(s.edges) == 4
        names = {v.name for v in s.variables}
        for u, v in s.edges:
            assert u in names and v in names
            assert u != v
        # no duplicate edges, undirected (each pair appears once)
        undirected = {frozenset(e) for e in s.edges}
        assert len(undirected) == 4
    finally:
        os.unlink(p)


def test_grid_generator_supports_binary_domain_too():
    p = _write("""
        name: grid_binary
        generate:
          kind: grid
          width: 3
          height: 1
          variable_domain: {domain: binary}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        assert len(s.variables) == 3
        assert all(isinstance(v.domain, Binary) for v in s.variables)
        assert len(s.edges) == 2   # a 3x1 strip: 2 horizontal edges, 0 vertical
    finally:
        os.unlink(p)


def test_clique_generator_still_exposes_its_edge_set():
    """Regression: A1 restructures _generated_terms to return edges alongside
    variables/terms for every generator, clique included."""
    p = _write("""
        name: clique_shape
        generate:
          kind: clique
          n: 5
          weight: 1.0
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        assert len(s.edges) == 10   # C(5, 2)
        assert len(s.terms) == 10
    finally:
        os.unlink(p)


def test_hand_written_spec_has_an_empty_edge_set_by_default():
    s = load_spec("specs/toy.yaml")
    assert s.edges == ()


def test_product_over_edges_expands_one_product_per_edge_both_orientations_by_default():
    p = _write("""
        name: edge_terms
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: categorical, k: 3}
        terms:
          - {kind: product_over_edges, a_value: 0, b_value: 2, weight: 8.0}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        assert len(s.edges) == 1
        u, v = s.edges[0]
        # symmetric default True: both orientations present
        assert len(s.terms) == 2
        expected_fwd = Product(LinearForm({VarRef(u, 0): 1.0}),
                               LinearForm({VarRef(v, 2): 1.0}), 8.0)
        expected_rev = Product(LinearForm({VarRef(u, 2): 1.0}),
                               LinearForm({VarRef(v, 0): 1.0}), 8.0)
        assert expected_fwd in s.terms
        assert expected_rev in s.terms
    finally:
        os.unlink(p)


def test_product_over_edges_symmetric_false_emits_only_the_declared_orientation():
    p = _write("""
        name: edge_terms_directed
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: categorical, k: 3}
        terms:
          - {kind: product_over_edges, a_value: 0, b_value: 2, weight: 8.0,
             symmetric: false}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        assert len(s.terms) == 1
        u, v = s.edges[0]
        expected = Product(LinearForm({VarRef(u, 0): 1.0}),
                           LinearForm({VarRef(v, 2): 1.0}), 8.0)
        assert s.terms[0] == expected
    finally:
        os.unlink(p)


def test_product_over_edges_supports_binary_variables_value_1_is_occupancy():
    p = _write("""
        name: edge_terms_binary
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: product_over_edges, a_value: 1, b_value: 1, weight: 3.0,
             symmetric: false}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        u, v = s.edges[0]
        expected = Product(LinearForm({VarRef(u): 1.0}),
                           LinearForm({VarRef(v): 1.0}), 3.0)
        assert s.terms[0] == expected
    finally:
        os.unlink(p)


def test_forbid_value_pair_over_edges_reports_each_specific_violating_edge():
    p = _write("""
        name: edge_contract
        generate:
          kind: grid
          width: 2
          height: 2
          variable_domain: {domain: categorical, k: 3}
        contract:
          validate:
            - {rule: forbid_value_pair_over_edges, a_value: 0, b_value: 2,
               message: "value 0 adjacent to value 2"}
    """)
    try:
        s = load_spec(p)
        names = sorted(v.name for v in s.variables)
        # set every variable to 0 except the last, which is 2 -- adjacent pairs
        # touching it should each be flagged individually
        assignment = {n: 0 for n in names}
        assignment[names[-1]] = 2
        result = s.contract.validate(assignment)
        violating_edges = [e for e in s.edges if names[-1] in e]
        assert not result.ok
        assert len(result.violations) == len(violating_edges)
        for v in result.violations:
            assert "value 0 adjacent to value 2" in v
        # every violation names its specific edge -- not a bare count
        for u, w in violating_edges:
            assert any(u in msg and w in msg for msg in result.violations)
    finally:
        os.unlink(p)


def test_forbid_value_pair_over_edges_symmetric_default_catches_either_order():
    p = _write("""
        name: edge_contract_sym
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: categorical, k: 3}
        contract:
          validate:
            - {rule: forbid_value_pair_over_edges, a_value: 0, b_value: 2,
               message: "clash"}
    """)
    try:
        s = load_spec(p)
        u, v = s.edges[0]
        assert not s.contract.validate({u: 2, v: 0}).ok
        assert not s.contract.validate({u: 0, v: 2}).ok
        assert s.contract.validate({u: 1, v: 1}).ok
    finally:
        os.unlink(p)


def test_forbid_value_pair_over_edges_symmetric_false_is_directional():
    p = _write("""
        name: edge_contract_directed
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: categorical, k: 3}
        contract:
          validate:
            - {rule: forbid_value_pair_over_edges, a_value: 0, b_value: 2,
               message: "clash", symmetric: false}
    """)
    try:
        s = load_spec(p)
        u, v = s.edges[0]
        assert not s.contract.validate({u: 0, v: 2}).ok
        assert s.contract.validate({u: 2, v: 0}).ok
    finally:
        os.unlink(p)
