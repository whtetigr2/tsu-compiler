"""The expressibility pass: tells the truth about what a declared Rule
becomes once it is forced through the pairwise IR (Linear/Product only).
See src/tsu/passes/expressibility.py for the EXACT / DISTORTED /
INEXPRESSIBLE verdict semantics -- DISTORTED must name its distortion in
plain words and must never be silently promoted to EXACT.
"""


def test_squared_deviation_is_exact():
    """(target - sum)^2 is a squared linear form -> Product(L, L, w) -> pairwise,
    with no change of meaning. This is the honest EXACT case."""
    from tsu.rules import Rule
    from tsu.passes.expressibility import analyse_rule
    r = Rule(id="balance", rule_class="statistical", scope="global", hard=False,
             weight=1.0,
             measurement={"kind": "squared_deviation", "target": 4, "value": 1},
             provenance="test")
    v = analyse_rule(r)
    assert v.status == "EXACT"
    assert v.distortion is None


def test_unregistered_measurement_kind_is_unknown_not_inexpressible():
    """C4 (code review, 2026-09-04-lattice-rule-taxonomy): `analyse_rule` used
    to return INEXPRESSIBLE for any measurement kind it simply had no
    dispatch for -- including `neighbourhood_count`, the generic term
    template this branch ships (`src/tsu/passes/encode.py`'s
    `_neighbourhood_count_terms`) and which `audit/expressibility_matrix.md`
    row #2 measures as EXACT. INEXPRESSIBLE is a claim ("no pairwise form
    exists") this pass can only make when it has actually analysed the
    kind's mathematical shape; "I have no dispatch entry for this string" is
    a different, weaker fact and must say so under its own name (UNKNOWN),
    never borrow INEXPRESSIBLE's vocabulary."""
    from tsu.rules import Rule
    from tsu.passes.expressibility import analyse_rule
    r = Rule(id="nb_count", rule_class="neighbourhood", scope="neighbourhood",
             hard=False, weight=0.5,
             measurement={"kind": "neighbourhood_count", "target": 2, "value": 1},
             provenance="test")
    v = analyse_rule(r)
    assert v.status == "UNKNOWN"
    assert v.status != "INEXPRESSIBLE"
    assert "neighbourhood_count" in v.reason


def test_squared_deviation_cost_does_not_hardcode_a_wrong_pairwise_term_count():
    """I8 (code review, 2026-09-04-lattice-rule-taxonomy): `Verdict.cost` used
    to hardcode `{"pairwise_terms": 1}` for EVERY squared_deviation rule --
    but `audit/expressibility_matrix.md` row #11 (`statistical`) measured
    2016 pairwise EDGES for the identical IR shape (Product(L, L, w) with L
    summing 64 variables). A `Rule`'s `measurement` mapping carries no field
    for how many variables its own LinearForm sums over, so `analyse_rule`
    has no way to compute the real post-lowering edge count from the Rule
    alone -- hardcoding 1 regardless is not a measurement, it is a guess
    dressed up as one. Per this plan's own rule ("Verification never
    fabricates. Any unmeasured quantity reads `unavailable: <reason>`"),
    the honest cost must say it cannot be computed from a bare Rule, not
    assert a specific number it has no basis for."""
    from tsu.rules import Rule
    from tsu.passes.expressibility import analyse_rule
    r = Rule(id="balance", rule_class="statistical", scope="global", hard=False,
             weight=1.0,
             measurement={"kind": "squared_deviation", "target": 4, "value": 1},
             provenance="test")
    v = analyse_rule(r)
    assert v.cost["pairwise_terms"] != 1
    assert isinstance(v.cost["pairwise_terms"], str)
    assert v.cost["pairwise_terms"].startswith("unavailable:")


def test_one_sided_threshold_cost_does_not_hardcode_a_wrong_pairwise_term_count():
    """Same defect (I8), same fix, for the DISTORTED path -- `_one_sided_threshold`
    hardcoded the identical wrong constant."""
    from tsu.rules import Rule
    from tsu.passes.expressibility import analyse_rule
    r = Rule(id="min_patch", rule_class="morphology", scope="neighbourhood",
             hard=False, weight=2.5,
             measurement={"kind": "one_sided_threshold", "target": 4, "value": 1},
             provenance="test")
    v = analyse_rule(r)
    assert v.cost["pairwise_terms"] != 1
    assert isinstance(v.cost["pairwise_terms"], str)
    assert v.cost["pairwise_terms"].startswith("unavailable:")


def test_one_sided_threshold_is_distorted_and_names_the_distortion():
    """max(0, target - N) has a kink and is not a polynomial at any degree.
    The nearest pairwise form is symmetric, which changes the rule's meaning.
    The verdict must SAY SO rather than quietly substituting it."""
    from tsu.rules import Rule
    from tsu.passes.expressibility import analyse_rule
    r = Rule(id="min_patch", rule_class="morphology", scope="neighbourhood",
             hard=False, weight=2.5,
             measurement={"kind": "one_sided_threshold", "target": 4, "value": 1},
             provenance="test")
    v = analyse_rule(r)
    assert v.status == "DISTORTED"
    assert v.distortion is not None
    assert "symmetric" in v.distortion.lower()
    assert "exactly" in v.distortion.lower()
