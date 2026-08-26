import pytest
from tsu.spec import load_spec, ValidationResult
from tsu.ir import Binary, Categorical


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
