import pytest
from tsu_compiler.spec import load_spec, ValidationResult
from tsu_compiler.ir import Binary, Categorical


SPECS = "specs"


def test_toy_parses_with_three_variables_and_three_terms():
    s = load_spec(f"{SPECS}/toy.yaml")
    assert s.name == "toy"
    assert {v.name for v in s.variables} == {"a", "b", "c"}
    assert isinstance(s.var("c").domain, Categorical)
    assert s.var("c").domain.k == 3
    assert isinstance(s.var("a").domain, Binary)
    assert len(s.terms) == 3


def test_categorical_indicator_syntax_parses_to_a_varref_with_a_value():
    s = load_spec(f"{SPECS}/toy.yaml")
    forbid = s.terms[2]
    refs = forbid.refs()
    assert any(r.name == "c" and r.value == 0 for r in refs)


def test_validate_returns_specific_violations_not_a_bool():
    s = load_spec(f"{SPECS}/toy.yaml")
    bad = s.contract.validate({"a": 1, "b": 1, "c": 0})
    assert isinstance(bad, ValidationResult)
    assert bad.ok is False
    assert len(bad.violations) == 2
    assert any("a and b" in v for v in bad.violations)
    assert any("c=0" in v for v in bad.violations)


def test_validate_passes_a_legal_assignment():
    s = load_spec(f"{SPECS}/toy.yaml")
    good = s.contract.validate({"a": 1, "b": 0, "c": 2})
    assert good.ok is True
    assert good.violations == ()


def test_a_variable_named_const_is_rejected_not_silently_swallowed():
    """MINOR (final review): a variable declared 'const' and referenced in a
    term's form parses to LinearForm(coeffs={}, const=<its weight>) -- the
    variable vanishes from the term with no error, and energy({const: 0,
    b: 0}) silently returns a wrong nonzero value. Same silent-wrongness class
    as the '__dw' chain-name collision, guarded the same way: loudly, at parse
    time."""
    import tempfile, os, textwrap
    body = textwrap.dedent("""
        name: reserved_word
        variables:
          const: {domain: binary}
          b: {domain: binary}
        terms:
          - {kind: linear, form: {const: 1.0, b: 1.0}, weight: 1.0}
        contract:
          validate: []
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body); p = f.name
    try:
        with pytest.raises(ValueError, match="const"):
            load_spec(p)
    finally:
        os.unlink(p)


def test_a_contract_without_messages_is_rejected_at_schema_level():
    """A verdict without the violation that caused it is not actionable."""
    import tempfile, os, textwrap
    body = textwrap.dedent("""
        name: nomsg
        variables: {a: {domain: binary}}
        terms: []
        contract:
          validate:
            - {rule: forbid_both, vars: [a, a]}
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body); p = f.name
    try:
        with pytest.raises(ValueError, match="message"):
            load_spec(p)
    finally:
        os.unlink(p)


# ---------------------------------------------------------------------------
# G1: FormulationRationale -- WHY a spec construct expanded into the IR shape
# it did, recorded at expansion time (spec.py's own term generation), not
# reconstructed after the fact.
# ---------------------------------------------------------------------------

def test_literal_terms_carry_a_formulation_record_with_no_invented_reason():
    """toy.yaml's terms are hand-written product/linear terms declared
    directly -- there is no structural template choosing their shape, so the
    honest rationale is an ABSENT reason, never invented prose standing in
    for one."""
    from tsu_compiler.spec import FormulationRationale
    s = load_spec(f"{SPECS}/toy.yaml")
    assert s.formulation, "toy.yaml's hand-written terms must still be recorded"
    by_construct = {r.construct: r for r in s.formulation}
    assert "product" in by_construct and "linear" in by_construct
    product = by_construct["product"]
    assert isinstance(product, FormulationRationale)
    assert product.ir_shape == "Product"
    assert product.count == 2          # toy.yaml declares 2 literal product terms
    assert product.reason == "", \
        "a hand-written term has no structural template reason to invent"
    linear = by_construct["linear"]
    assert linear.ir_shape == "Linear"
    assert linear.count == 1
    assert linear.reason == ""


def test_product_over_edges_formulation_record_states_the_structural_reason():
    """A2: the pairwise penalty over an edge set becomes one Product per edge
    (per orientation) because a two-variable constraint IS a product of two
    value-indicator linear forms -- that structural fact is the recorded
    reason, sourced at the point spec.py actually builds the Product terms."""
    from tsu_compiler.spec import FormulationRationale
    s = load_spec(f"{SPECS}/adjacency_2x2_k3.yaml")
    poe = next(r for r in s.formulation if r.construct == "product_over_edges")
    assert isinstance(poe, FormulationRationale)
    assert poe.count == len(s.terms)          # every term here comes from A2
    assert str(len(s.edges)) in poe.scope      # "N-edge generated set"
    assert poe.reason, "A2 has a genuine structural reason; it must not be absent"
    assert "product" in poe.reason.lower() or "indicator" in poe.reason.lower()


def test_conserve_over_edges_formulation_record_states_the_structural_reason():
    """C2: a conservation constraint is an equality; squaring a linear form
    keeps the penalty pairwise regardless of how many flow variables the
    form itself sums over -- the recorded structural reason."""
    import os, tempfile, textwrap
    from tsu_compiler.spec import FormulationRationale
    body = textwrap.dedent("""
        name: flow_formulation
        generate:
          kind: grid
          width: 3
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {}, sinks: {}}
        contract:
          validate: []
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body); p = f.name
    try:
        s = load_spec(p)
        coe = next(r for r in s.formulation if r.construct == "conserve_over_edges")
        assert isinstance(coe, FormulationRationale)
        assert coe.count == 3   # one term per node of the 3x1 strip
        assert coe.reason, "C2 has a genuine structural reason; it must not be absent"
        assert "equality" in coe.reason.lower() or "squar" in coe.reason.lower()
    finally:
        os.unlink(p)


def test_clique_formulation_record_states_the_structural_reason():
    import os, tempfile, textwrap
    from tsu_compiler.spec import FormulationRationale
    body = textwrap.dedent("""
        name: clique_formulation
        generate:
          kind: clique
          n: 4
          weight: 1.0
        contract:
          validate: []
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body); p = f.name
    try:
        s = load_spec(p)
        c = next(r for r in s.formulation if r.construct == "clique")
        assert isinstance(c, FormulationRationale)
        assert c.count == 6   # C(4, 2)
        assert c.reason, "a clique's own shape is a genuine structural reason"
    finally:
        os.unlink(p)


def test_a_grid_generator_with_no_hand_terms_has_no_fabricated_formulation():
    """`grid` alone generates variables/edges but no terms -- there is
    nothing to explain, so no formulation record is fabricated for it."""
    import os, tempfile, textwrap
    body = textwrap.dedent("""
        name: bare_grid
        generate:
          kind: grid
          width: 2
          height: 2
          variable_domain: {domain: binary}
        contract:
          validate: []
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body); p = f.name
    try:
        s = load_spec(p)
        assert s.formulation == ()
    finally:
        os.unlink(p)
