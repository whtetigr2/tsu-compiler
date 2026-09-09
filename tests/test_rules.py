"""Rule schema: a Rule object carries its class, scope, and provenance so the
compiler can reason about a rule (expressibility, cost) rather than merely
execute it. See src/tsu/rules.py for the module docstring on why four of the
twelve `RULE_CLASSES` read as domain vocabulary while remaining pure data.
"""


def test_rule_carries_its_class_and_provenance():
    from tsu.rules import Rule, load_rules
    rules = load_rules([{
        "id": "min_patch", "rule_class": "morphology", "scope": "neighbourhood",
        "hard": False, "weight": 2.5,
        "measurement": {"kind": "neighbourhood_count", "value": 1, "radius": 1},
        "provenance": "design intent, 2026-09-04",
    }])
    assert len(rules) == 1
    assert rules[0].rule_class == "morphology"
    assert rules[0].provenance == "design intent, 2026-09-04"


def test_unknown_rule_class_is_rejected_not_silently_accepted():
    """A typo'd class must fail loudly. Silently accepting it would let a rule
    skip expressibility analysis entirely, which is the one thing this schema
    exists to prevent."""
    import pytest
    from tsu.rules import load_rules
    with pytest.raises(ValueError, match="unknown rule_class"):
        load_rules([{"id": "x", "rule_class": "morphologyy", "scope": "cell",
                     "hard": True, "weight": 1.0, "measurement": {},
                     "provenance": "typo"}])
