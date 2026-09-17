"""THRML-backed block-Gibbs sampling engine for Gibbs Observatory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from .graph_presets import GraphSpec, build_from_receipt_arrays, build_preset
from .metrics import ising_energy, magnetization, summarize_series
from .structure_factor import recover_lattice, structure_factor


@dataclass
class SamplerConfig:
    preset: str = "lattice2d"
    size: int = 16
    degree_cap: int = 16
    beta: float = 0.5
    J: float = 1.0
    h: float = 0.0
    warmup: int = 50
    steps_per_sample: int = 2
    batch_size: int = 8
    seed: int = 0
    clamp: bool = False
    clamp_value: bool = True  # True => +1
    receipt_id: str | None = None
    # When receipt cannot drive THRML, engine falls back to preset but UI
    # still shows receipt inspect data.
    sampling_fallback: bool = False
    # Why a DIFFERENT model was substituted. Set alongside the flag, because
    # swapping the model out from under a live energy trace is a large thing
    # to do quietly, and a banner saying it happened without saying why
    # leaves a malformed receipt merely survivable rather than debuggable.
    fallback_reason: str | None = None


@dataclass
class EngineState:
    graph: GraphSpec
    nodes: list[Any]
    free_blocks: list[Block]
    clamped_blocks: list[Block]
    model: IsingEBM
    program: IsingSamplingProgram
    key: jax.Array
    init_free: list[jnp.ndarray]
    clamp_state: list[jnp.ndarray]
    edge_idx: list[tuple[int, int]]
    biases_np: np.ndarray
    weights_np: np.ndarray
    beta: float
    step: int = 0
    history_energy: list[float] = field(default_factory=list)
    history_mag: list[float] = field(default_factory=list)
    last_state: np.ndarray | None = None
    active_block: int = 0


class SamplerEngine:
    """Owns THRML program; structure static, params rebuildable."""

    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        self.state: EngineState | None = None
        self.reset(self.config)

    def _resolve_graph(self, cfg: SamplerConfig) -> tuple[GraphSpec, bool]:
        """Return (graph, used_fallback)."""
        if cfg.receipt_id:
            try:
                from .receipt_loader import receipt_to_graph_arrays

                data = receipt_to_graph_arrays(cfg.receipt_id)
                graph = build_from_receipt_arrays(data)
                # Force β from receipt when mediated / fixed
                if data.get("beta_fixed") or data.get("beta") is not None:
                    cfg.beta = float(data["beta"])
                return graph, False
            except Exception as exc:  # noqa: BLE001
                # Honest fallback to lattice2d, now saying what went wrong.
                cfg.fallback_reason = (
                    f"unavailable: receipt {cfg.receipt_id!r} could not be "
                    f"loaded ({type(exc).__name__}: {exc}); sampling a generic "
                    f"lattice2d instead, which is NOT this receipt's model, so "
                    f"every statistic on screen describes something else")
                graph = build_preset(
                    "lattice2d",
                    size=cfg.size,
                    degree_cap=cfg.degree_cap,
                    seed=cfg.seed,
                )
                return graph, True
        graph = build_preset(
            cfg.preset,  # type: ignore[arg-type]
            size=cfg.size,
            degree_cap=cfg.degree_cap,
            seed=cfg.seed,
        )
        return graph, False

    def reset(self, config: SamplerConfig | None = None) -> dict:
        if config is not None:
            self.config = config
        cfg = self.config
        graph, used_fallback = self._resolve_graph(cfg)
        cfg.sampling_fallback = used_fallback

        nodes = [SpinNode() for _ in range(graph.n_nodes)]
        edges = [(nodes[i], nodes[j]) for i, j in graph.edges]

        if graph.weights is not None and graph.biases is not None:
            biases = jnp.asarray(graph.biases, dtype=jnp.float32)
            weights = jnp.asarray(graph.weights, dtype=jnp.float32)
            beta_val = float(cfg.beta)
        else:
            biases = jnp.full((graph.n_nodes,), float(cfg.h), dtype=jnp.float32)
            weights = jnp.full((len(graph.edges),), float(cfg.J), dtype=jnp.float32)
            beta_val = float(cfg.beta)

        beta = jnp.array(beta_val, dtype=jnp.float32)
        model = IsingEBM(nodes, edges, biases, weights, beta)

        if cfg.clamp and graph.clamp_indices:
            clamp_set = set(graph.clamp_indices)
            free0 = [nodes[i] for i in graph.color0 if i not in clamp_set]
            free1 = [nodes[i] for i in graph.color1 if i not in clamp_set]
            clamped_nodes = [nodes[i] for i in graph.clamp_indices]
            free_blocks = []
            if free0:
                free_blocks.append(Block(free0))
            if free1:
                free_blocks.append(Block(free1))
            if not free_blocks:
                # fallback: nothing free, keep tiny free set
                free_blocks = [Block([nodes[i] for i in graph.color0[:1]])]
            clamped_blocks = [Block(clamped_nodes)]
            clamp_state = [
                jnp.full((len(clamped_nodes),), bool(cfg.clamp_value), dtype=jnp.bool_)
            ]
        else:
            free_blocks = [
                Block([nodes[i] for i in graph.color0]),
                Block([nodes[i] for i in graph.color1]),
            ]
            clamped_blocks = []
            clamp_state = []

        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=clamped_blocks)
        key = jax.random.key(int(cfg.seed))
        key, k_init = jax.random.split(key)
        init_free = hinton_init(k_init, model, free_blocks, ())

        self.state = EngineState(
            graph=graph,
            nodes=nodes,
            free_blocks=free_blocks,
            clamped_blocks=clamped_blocks,
            model=model,
            program=program,
            key=key,
            init_free=init_free,
            clamp_state=clamp_state,
            edge_idx=list(graph.edges),
            biases_np=np.asarray(biases),
            weights_np=np.asarray(weights),
            beta=beta_val,
        )
        return self.graph_payload()

    def update_params(
        self,
        *,
        beta: float | None = None,
        J: float | None = None,
        h: float | None = None,
        clamp: bool | None = None,
        warmup: int | None = None,
        steps_per_sample: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Update dynamic params. Structure-changing flags trigger full reset."""
        cfg = self.config
        structure_change = False
        # Receipt-backed mediated models: β is FIXED, ignore beta patches
        beta_locked = bool(cfg.receipt_id) and (
            self.state is not None and self.state.graph.beta_fixed
        )
        if beta is not None and not beta_locked:
            cfg.beta = float(beta)
        if J is not None and self.state is not None and self.state.graph.weights is None:
            cfg.J = float(J)
        if h is not None and self.state is not None and self.state.graph.biases is None:
            cfg.h = float(h)
        if warmup is not None:
            cfg.warmup = int(warmup)
        if steps_per_sample is not None:
            cfg.steps_per_sample = int(steps_per_sample)
        if batch_size is not None:
            cfg.batch_size = int(batch_size)
        if seed is not None and seed != cfg.seed:
            cfg.seed = int(seed)
            structure_change = True
        if clamp is not None and bool(clamp) != cfg.clamp:
            cfg.clamp = bool(clamp)
            structure_change = True

        if structure_change or self.state is None:
            return self.reset(cfg)

        # Rebuild model/program with new biases/weights/beta; keep spin state.
        st = self.state
        graph = st.graph
        nodes = st.nodes
        edges = [(nodes[i], nodes[j]) for i, j in graph.edges]

        if graph.weights is not None and graph.biases is not None:
            # Receipt: keep per-edge / per-node params; only β may change if unlocked
            biases = jnp.asarray(graph.biases, dtype=jnp.float32)
            weights = jnp.asarray(graph.weights, dtype=jnp.float32)
        else:
            biases = jnp.full((graph.n_nodes,), float(cfg.h), dtype=jnp.float32)
            weights = jnp.full((len(graph.edges),), float(cfg.J), dtype=jnp.float32)

        beta_arr = jnp.array(float(cfg.beta), dtype=jnp.float32)
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        program = IsingSamplingProgram(model, st.free_blocks, clamped_blocks=st.clamped_blocks)
        st.model = model
        st.program = program
        st.biases_np = np.asarray(biases)
        st.weights_np = np.asarray(weights)
        st.beta = float(cfg.beta)
        return self.graph_payload()

    def zero_terms(self, *, couplings: bool = False,
                   biases: bool = False) -> None:
        """Rebuild the model with a term zeroed, keeping everything else.

        For ablation. The graph, the node identities and the colour blocks are
        untouched, so the ONLY difference between this model and the one it came
        from is the term being tested. Anything else changing would make the
        measured shift attributable to two causes at once, which is the failure
        mode the ablation exists to avoid.
        """
        assert self.state is not None, "reset() before zeroing terms"
        st = self.state
        nodes = st.nodes
        edges = [(nodes[i], nodes[j]) for i, j in st.graph.edges]

        b = (jnp.zeros_like(jnp.asarray(st.biases_np, dtype=jnp.float32))
             if biases else jnp.asarray(st.biases_np, dtype=jnp.float32))
        w = (jnp.zeros_like(jnp.asarray(st.weights_np, dtype=jnp.float32))
             if couplings else jnp.asarray(st.weights_np, dtype=jnp.float32))

        model = IsingEBM(nodes, edges, b, w,
                         jnp.array(float(st.beta), dtype=jnp.float32))
        st.model = model
        st.program = IsingSamplingProgram(
            model, st.free_blocks, clamped_blocks=st.clamped_blocks)
        st.biases_np = np.asarray(b)
        st.weights_np = np.asarray(w)


    def _structure_factor(self, samples) -> dict | None:
        """S(k) for this batch, or None when there is no lattice to transform.

        Uses the model's OWN site coordinates, read from names the grid
        generator wrote. Never placement coordinates: those say where a spin
        landed on the die, and transforming them would picture the router's
        output rather than the physics.
        """
        assert self.state is not None
        lattice = recover_lattice(self.state.graph.node_names)
        if lattice is None:
            return None
        try:
            spins = np.asarray(samples)
            if spins.ndim == 3:
                spins = spins.reshape(-1, spins.shape[-1])
            return structure_factor(spins.tolist(), lattice)
        except Exception as exc:  # noqa: BLE001
            return {"available": False,
                    "reason": f"unavailable: transform failed: {exc!r}"}

    def graph_payload(self) -> dict:
        assert self.state is not None
        g = self.state.graph
        cfg = self.config
        banner = None
        if cfg.receipt_id and cfg.sampling_fallback:
            # The reason, not just the fact. A banner saying a different model
            # is being sampled, without saying why, leaves a malformed receipt
            # survivable but not debuggable.
            banner = cfg.fallback_reason or (
                "sampling fallback: a generic lattice is being sampled, not "
                "this receipt's model")
        return {
            "preset": g.name,
            "receipt_id": cfg.receipt_id or g.receipt_id,
            "n_nodes": g.n_nodes,
            "edges": g.edges,
            "edge_weights": g.weights,
            "biases": g.biases,
            "color0": g.color0,
            "color1": g.color1,
            "layout": g.layout,
            "shape": list(g.shape),
            "positions": g.positions,
            "clamp_indices": g.clamp_indices if cfg.clamp else [],
            "clamp_enabled": cfg.clamp,
            "degree_cap": g.degree_cap,
            "world_idx": g.world_idx,
            "mediator_idx": g.mediator_idx,
            "node_names": g.node_names,
            "beta_fixed": g.beta_fixed,
            "kernel": g.kernel,
            "sampling_fallback": cfg.sampling_fallback,
            "sampling_banner": banner,
            "params": {
                "beta": cfg.beta,
                "J": cfg.J,
                "h": cfg.h,
                "warmup": cfg.warmup,
                "steps_per_sample": cfg.steps_per_sample,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "size": cfg.size,
                "receipt_id": cfg.receipt_id,
            },
            "label": "JAX/THRML simulation, not Extropic silicon",
        }

    def _assemble_full_state(self, free_states: list[jnp.ndarray]) -> np.ndarray:
        """Map free (+ clamp) block states into a full bool vector."""
        assert self.state is not None
        st = self.state
        full = np.zeros(st.graph.n_nodes, dtype=bool)
        cfg = self.config
        clamp_set = set(st.graph.clamp_indices) if cfg.clamp else set()
        free_index_groups: list[list[int]] = []
        if cfg.clamp and st.graph.clamp_indices:
            g0 = [i for i in st.graph.color0 if i not in clamp_set]
            g1 = [i for i in st.graph.color1 if i not in clamp_set]
            if g0:
                free_index_groups.append(g0)
            if g1:
                free_index_groups.append(g1)
        else:
            free_index_groups = [st.graph.color0, st.graph.color1]

        for idxs, arr in zip(free_index_groups, free_states, strict=False):
            a = np.asarray(arr, dtype=bool).ravel()
            for k, node_i in enumerate(idxs):
                if k < a.size:
                    full[node_i] = bool(a[k])

        if cfg.clamp and st.clamp_state:
            clamp_vals = np.asarray(st.clamp_state[0], dtype=bool).ravel()
            for k, node_i in enumerate(st.graph.clamp_indices):
                if k < clamp_vals.size:
                    full[node_i] = bool(clamp_vals[k])
        return full

    def sample_batch(self, *, n_samples: int | None = None, warmup: int | None = None) -> dict:
        """Run one THRML sample_states batch and return UI payload."""
        assert self.state is not None
        st = self.state
        cfg = self.config
        n = int(n_samples if n_samples is not None else cfg.batch_size)
        n = max(1, min(n, 64))
        warm = int(warmup if warmup is not None else (cfg.warmup if st.step == 0 else 0))
        warm = max(0, warm)
        sps = max(1, int(cfg.steps_per_sample))

        schedule = SamplingSchedule(n_warmup=warm, n_samples=n, steps_per_sample=sps)
        st.key, k_samp = jax.random.split(st.key)
        samples_list = sample_states(
            k_samp,
            st.program,
            schedule,
            st.init_free,
            st.clamp_state,
            [Block(st.nodes)],
        )
        samples = np.asarray(samples_list[0], dtype=bool)  # (n, N)

        # Carry last free-block state forward for continuous streaming
        last = samples[-1]
        clamp_set = set(st.graph.clamp_indices) if cfg.clamp else set()
        new_init: list[jnp.ndarray] = []
        if cfg.clamp and st.graph.clamp_indices:
            groups = [
                [i for i in st.graph.color0 if i not in clamp_set],
                [i for i in st.graph.color1 if i not in clamp_set],
            ]
            groups = [g for g in groups if g]
        else:
            groups = [st.graph.color0, st.graph.color1]
        for g in groups:
            new_init.append(jnp.array([bool(last[i]) for i in g], dtype=jnp.bool_))
        st.init_free = new_init

        energies = ising_energy(
            samples, st.edge_idx, st.biases_np, st.weights_np, st.beta
        )
        mags = magnetization(samples)
        st.history_energy.extend(float(x) for x in energies)
        st.history_mag.extend(float(x) for x in mags)
        # cap history
        max_hist = 2000
        if len(st.history_energy) > max_hist:
            st.history_energy = st.history_energy[-max_hist:]
            st.history_mag = st.history_mag[-max_hist:]

        st.step += n
        st.last_state = last
        # active block alternates with steps (visual cue for chromatic sweeps)
        st.active_block = (st.active_block + sps) % 2

        e_sum = summarize_series(np.asarray(st.history_energy[-256:]))
        m_sum = summarize_series(np.asarray(st.history_mag[-256:]))

        banner = None
        if cfg.receipt_id and cfg.sampling_fallback:
            # The reason, not just the fact. A banner saying a different model
            # is being sampled, without saying why, leaves a malformed receipt
            # survivable but not debuggable.
            banner = cfg.fallback_reason or (
                "sampling fallback: a generic lattice is being sampled, not "
                "this receipt's model")

        return {
            "type": "batch",
            "step": st.step,
            "states": samples.astype(np.uint8).tolist(),  # 0/1
            "last_state": last.astype(np.uint8).tolist(),
            "energies": energies.tolist(),
            "magnetizations": mags.tolist(),
            # S(k) when the model's own site names give a real-space lattice.
            # Absent, not faked, when they do not: a transform over an arbitrary
            # spin ordering produces a convincing square picture of nothing.
            "structure_factor": self._structure_factor(samples),
            "active_block": st.active_block,
            "metrics": {
                "energy": e_sum,
                "magnetization": m_sum,
            },
            "history": {
                "energy": st.history_energy[-256:],
                "magnetization": st.history_mag[-256:],
            },
            "receipt_id": cfg.receipt_id,
            "sampling_fallback": cfg.sampling_fallback,
            "sampling_banner": banner,
            "label": "JAX/THRML simulation, not Extropic silicon",
        }
