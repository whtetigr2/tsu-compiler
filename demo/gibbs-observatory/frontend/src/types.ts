export type PresetId = 'lattice2d' | 'chain1d' | 'sparse'

export type LangMode = 'sci' | 'prog'

export type NavView =
  | 'overview'
  | 'spins'
  | 'couplings'
  | 'connectivity'
  | 'schedule'
  | 'statespace'
  | 'residuals'
  | 'scope'
  | 'notepad'
  | 'alloylab'
  | 'ebmlab'
  | 'worlds'

export interface SimParams {
  preset: PresetId
  size: number
  degree_cap: number
  beta: number
  J: number
  h: number
  warmup: number
  steps_per_sample: number
  batch_size: number
  seed: number
  clamp: boolean
  receipt_id?: string | null
}

export interface GraphPayload {
  type?: string
  preset: string
  receipt_id?: string | null
  n_nodes: number
  edges: [number, number][]
  edge_weights?: number[] | null
  biases?: number[] | null
  color0: number[]
  color1: number[]
  layout: string
  shape: number[]
  positions: [number, number][]
  clamp_indices: number[]
  clamp_enabled: boolean
  degree_cap: number | null
  world_idx?: number[]
  mediator_idx?: number[]
  node_names?: string[]
  beta_fixed?: boolean
  kernel?: string | null
  sampling_fallback?: boolean
  sampling_banner?: string | null
  params: Partial<SimParams>
  label: string
}

export interface MetricSummary {
  /** null when the series is empty. An empty series has no mean; 0 is a lie. */
  mean: number | null
  std: number | null
  last: number | null
  lag1_autocorr: number | null
  /**
   * null unless the chain is long enough for the compiler's Sokal estimator to
   * stand behind it, which needs N/tau >= 5000. The live history is 256
   * samples, so this is null in the interface today and `ess_reason` says why.
   */
  ess: number | null
  ess_reliable: boolean
  ess_reason: string | null
  n: number
}

export interface BatchPayload {
  type: 'batch'
  step: number
  states: number[][]
  last_state: number[]
  energies: number[]
  magnetizations: number[]
  active_block: number
  metrics: {
    energy: MetricSummary
    magnetization: MetricSummary
  }
  history: {
    energy: number[]
    magnetization: number[]
  }
  receipt_id?: string | null
  sampling_fallback?: boolean
  sampling_banner?: string | null
  label: string
}

export interface GateRow {
  gate: string
  passed: boolean
  measured: number | string | null
  limit: number | string | null
  assumed?: boolean
  downgraded?: boolean
}

export interface ReceiptSummary {
  id: string
  path: string
  verdict: string | null
  encoding: string | null
  n_nodes: number | null
  mediator_count: number | null
  bipartite: boolean | null
}

export interface ReceiptInspect {
  id: string
  path: string
  verdict: string | null
  encoding: string | null
  beta: number | null
  beta_fixed: boolean
  kernel: string | null
  gates: GateRow[]
  spins: {
    n_nodes: number | null
    world: number | null
    mediators: number | null
    world_idx: number[]
    mediator_idx: number[]
    node_names: string[] | null
  }
  edges: {
    count: number | null
    pairs: [number, number][] | null
    weights: number[] | null
  }
  biases: number[] | null
  blocks: { color0: number[]; color1: number[] } | null
  positions: [number, number][] | null
  frontier: Record<string, unknown> | null
  connectivity: {
    logical: Record<string, unknown>
    physical: Record<string, unknown>
    notes: string[]
  }
  mediation: Record<string, unknown> | null
  fabric_tax: FabricTax | null
  schedule: ScheduleInfo | null
  residuals: ResidualsInfo | null
  workload: Record<string, unknown> | null
  metrics: Record<string, unknown> | null
  verification: {
    task_validity: number | null
    codeword_violation_rate: number | null
    ess: { available: boolean; value: number | null; reason: string | null }
  } | null
  regime: Record<string, unknown> | null
  target: { name?: string; bipartite?: boolean | null } | null
  sampling: {
    thrml_ready: boolean
    fallback: string | null
    banner: string | null
    beta: number | null
    beta_fixed: boolean
    seed: number | null
  }
  claim_badges: string[]
  label: string
  files_available: string[]
  files_unavailable: string[]
  spec_yaml?: string | null
}


export interface FabricTaxPair {
  world_u: number
  world_v: number
  mediator: number
  world_u_name: string | null
  world_v_name: string | null
  mediator_name: string | null
}

export interface FabricTax {
  derived: boolean
  method: string
  pair_count: number
  pairs: FabricTaxPair[]
  by_edge_key: Record<string, FabricTaxPair>
  notes: string[]
}

export interface ScheduleBlock {
  colour: number
  size: number
  indices: number[]
}

export interface ScheduleInfo {
  kernel: string | null
  n_colours: number | null
  blocks: ScheduleBlock[] | null
  notes: string[]
}

export interface ResidualBar {
  id: string
  label: string
  value: number
  utilisation?: number | null
  derived: boolean
  sources?: string[]
  assumed?: boolean
}

export interface ResidualHeatCell {
  row: string
  col: string
  value: number
  utilisation?: number | null
  derived: boolean
}

export interface ResidualsInfo {
  bars: ResidualBar[]
  heatmap: ResidualHeatCell[]
  unavailable: string[]
  notes: string[]
  has_receipt_residual_matrix: boolean
}

export interface ExampleShelfItem {
  id: string
  title: string
  source: string
  kind: string
  extropic: boolean
  notes: string
  packaged: boolean
  status: 'ready' | 'stub' | 'missing' | string
  message: string | null
  receipt: ReceiptSummary | null
  /** Written name. Never the directory name. */
  plain_name: string
  /**
   * True only when an oracle test compiles this workload. Derived from
   * backend/app/verified_workloads.py, never typed into a catalog, so the
   * badge cannot claim more than the test suite enforces.
   */
  verified: boolean
  /** What does the verifying, and at what placement effort. */
  verified_by: string | null
  published_by: string | null
  one_line: string | null
  /** The real-world question, before any physics. */
  problem: string | null
  /** How that question becomes an energy. Written out. */
  math: string | null
  /** What it costs on Z1. Measured, not estimated. */
  hardware: string | null
  citation: string | null
}

export const DEFAULT_PARAMS: SimParams = {
  preset: 'lattice2d',
  size: 16,
  degree_cap: 8,
  beta: 0.44,
  J: 1.0,
  h: 0.0,
  warmup: 40,
  steps_per_sample: 2,
  batch_size: 8,
  seed: 42,
  clamp: false,
  receipt_id: 'small',
}
