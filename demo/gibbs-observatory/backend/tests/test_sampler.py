"""Smoke tests for THRML sampler engine."""

from backend.app.sampler_engine import SamplerConfig, SamplerEngine


def test_lattice_samples():
    eng = SamplerEngine(
        SamplerConfig(preset="lattice2d", size=8, warmup=5, batch_size=4, seed=1)
    )
    g = eng.graph_payload()
    assert g["n_nodes"] == 64
    assert len(g["color0"]) + len(g["color1"]) == 64
    batch = eng.sample_batch()
    assert batch["type"] == "batch"
    assert len(batch["last_state"]) == 64
    assert len(batch["energies"]) == 4


def test_chain_clamp():
    eng = SamplerEngine(
        SamplerConfig(preset="chain1d", size=12, clamp=True, warmup=2, batch_size=2)
    )
    batch = eng.sample_batch()
    # endpoints clamped to +1 (True => 1)
    assert batch["last_state"][0] == 1
    assert batch["last_state"][-1] == 1


def test_sparse_builds():
    eng = SamplerEngine(
        SamplerConfig(preset="sparse", size=20, degree_cap=4, warmup=2, batch_size=2)
    )
    assert eng.graph_payload()["n_nodes"] == 20
    batch = eng.sample_batch()
    assert len(batch["magnetizations"]) == 2
