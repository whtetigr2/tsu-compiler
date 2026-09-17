"""Chain embedding, and the validator that decides whether to believe it.

The validator is the point of these tests. `minorminer` is a mature library and
it is still a third party: if this project reports a model as placed on Z1, the
evidence has to be a check this project ran, not a success flag someone else
returned. So the validator is tested against embeddings that are deliberately
broken in each of the three ways an embedding can be wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsu_compiler.passes.embed import (  # noqa: E402
    build_host,
    find_chain_embedding,
    validate_embedding,
)
from tsu_compiler.preflight.model import IsingModel  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

minorminer = pytest.importorskip("minorminer", reason="minorminer not installed")

Z1 = PROFILES["z1"]


def _model(edges, n):
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(n)),
        edges=tuple(edges),
        weights=np.full(len(edges), 0.5),
        biases=np.zeros(n),
        beta=1.0, offset=0.0,
    )


class TestTheHost:
    def test_the_host_is_the_declared_offset_lattice(self):
        H = build_host(10, Z1.offsets.value)
        offs = set(Z1.offsets.value)
        for (a, b) in H.edges():
            assert (b[0] - a[0], b[1] - a[1]) in offs or \
                   (a[0] - b[0], a[1] - b[1]) in offs

    def test_interior_cells_have_the_full_degree(self):
        H = build_host(12, Z1.offsets.value)
        assert H.degree((6, 6)) == len(Z1.offsets.value)


class TestTheValidatorCatchesEachFailure:
    """Three ways an embedding can be wrong, and the validator must catch all
    three. A validator that only ever passes is not evidence."""

    def test_a_disconnected_chain_is_caught(self):
        H = build_host(12, Z1.offsets.value)
        chains = {0: [(0, 0), (9, 9)], 1: [(1, 0)]}
        fails = validate_embedding([(0, 1)], H, chains)
        assert any("disconnected" in f for f in fails)

    def test_a_shared_cell_is_caught(self):
        """Two logical spins on one p-bit silently identifies variables the
        model intends to be distinct."""
        H = build_host(12, Z1.offsets.value)
        chains = {0: [(0, 0)], 1: [(0, 0)]}
        fails = validate_embedding([(0, 1)], H, chains)
        assert any("used by both" in f for f in fails)

    def test_an_unrealized_edge_is_caught(self):
        """The coupling the model asked for does not exist on the die, so the
        energy sampled is not the energy compiled."""
        H = build_host(12, Z1.offsets.value)
        chains = {0: [(0, 0)], 1: [(7, 7)]}
        fails = validate_embedding([(0, 1)], H, chains)
        assert any("not realized" in f for f in fails)

    def test_a_missing_chain_is_caught(self):
        H = build_host(12, Z1.offsets.value)
        fails = validate_embedding([(0, 1)], H, {0: [(0, 0)]})
        assert any("no chain" in f for f in fails)

    def test_a_good_embedding_passes(self):
        H = build_host(12, Z1.offsets.value)
        dx, dy = Z1.offsets.value[0]
        chains = {0: [(5, 5)], 1: [(5 + dx, 5 + dy)]}
        assert validate_embedding([(0, 1)], H, chains) == ()


class TestItEmbedsWhatDirectPlacementCannot:
    def test_a_triangle_embeds_despite_a_bipartite_host(self):
        """The Z1 lattice is triangle-free, so a triangle has no SUBGRAPH
        embedding at all. It has a MINOR embedding, because contracting a
        6-cycle yields a triangle -- which is the whole reason chains work."""
        emb = find_chain_embedding(_model([(0, 1), (1, 2), (0, 2)], 3), Z1, seed=3)
        assert emb.n_logical == 3
        assert emb.max_chain >= 2, "at least one spin must span cells"

    def test_a_clique_embeds(self):
        n = 8
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        emb = find_chain_embedding(_model(edges, n), Z1, seed=1)
        assert emb.n_physical >= n

    def test_an_isolated_spin_still_gets_a_cell(self):
        """It has no edges, so the search never sees it. A spin with no
        location is not placed."""
        m = _model([(0, 1)], 3)      # spin 2 is isolated
        emb = find_chain_embedding(m, Z1, seed=1)
        assert len(emb.chains[2]) == 1


class TestItRefusesRatherThanGuesses:
    def test_a_target_with_no_lattice_is_refused(self):
        ideal = PROFILES["ideal"]
        with pytest.raises(RuntimeError, match="no lattice offsets"):
            find_chain_embedding(_model([(0, 1)], 2), ideal)

    def test_the_result_is_always_validated_before_return(self):
        """The contract: nothing is returned that this module did not check.
        Forcing the validator to report a failure must abort the call."""
        import tsu_compiler.passes.embed as mod
        real = mod.validate_embedding
        try:
            mod.validate_embedding = lambda *a, **k: ("forced failure",)
            with pytest.raises(RuntimeError, match="does not validate"):
                find_chain_embedding(_model([(0, 1)], 2), Z1)
        finally:
            mod.validate_embedding = real


class TestItIsWiredIntoPreflight:
    """The capability existing is not the same as the compiler using it.

    `passes/embed.py` was written before `preflight` called it, and for a while
    `tsuc preflight` still reported models as unplaced that the embedder could
    place in a second. These pin the integration.
    """

    def _hard(self):
        """A model direct placement genuinely cannot do.

        NOT a triangle: `place()` mediates first, and a mediated triangle is a
        6-cycle, which embeds directly. Mediation already covers odd cycles.
        What it does not cover is a graph whose local structure is nothing like
        the lattice's, which is R33's subject -- so this is a random 4-regular
        expander, verified below to fail without chains.
        """
        import networkx as nx
        G = nx.random_regular_graph(4, 40, seed=7)
        edges = tuple(sorted((min(u, v), max(u, v)) for u, v in G.edges()))
        return _model(edges, 40)

    def test_a_model_direct_placement_cannot_do_now_places(self):
        from tsu_compiler.preflight.check import preflight
        m = self._hard()
        without = preflight(m, restarts=1, iters=200, allow_chains=False)
        assert without.placed is False, "premise: this must defeat direct placement"
        r = preflight(m, restarts=1, iters=200)
        assert r.placed is True
        assert r.verdict != "effort"

    def test_the_report_says_chains_were_used_and_what_they_cost(self):
        from tsu_compiler.preflight.check import preflight
        r = preflight(self._hard(), restarts=1, iters=200)
        assert r.embedding == "chain"
        assert r.chain_cells and r.chain_cells >= 40
        assert r.chain_max and r.chain_max >= 1

    def test_direct_placement_is_still_preferred_when_it_works(self):
        """A model that places directly must not pay 2x the spins for reach it
        does not need, so chains are the fallback and not the default."""
        from tsu_compiler.preflight.check import preflight
        path = _model([(0, 1)], 2)          # trivially lattice-shaped
        r = preflight(path)
        assert r.placed is True
        assert r.embedding != "chain"
        assert r.chain_cells is None

    def test_chains_can_be_switched_off(self):
        """So the direct-only behaviour remains measurable, which is what R33
        was measured against."""
        from tsu_compiler.preflight.check import preflight
        r = preflight(self._hard(), restarts=1, iters=200, allow_chains=False)
        assert r.placed is False
        assert r.verdict == "effort"
        assert r.chain_cells is None
