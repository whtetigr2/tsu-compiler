import { useCallback, useEffect, useMemo, useState } from 'react'
import type { BatchPayload, GraphPayload, LangMode, ReceiptInspect } from '../../types'

interface DecodedSample {
  occupancy: number[][]
  width: number
  height: number
  m_s: number
  abs_m_s: number
  concentration_B: number
  unlike_edge_fraction: number
  energy_config: number
  energy_ising?: number | null
  sample_index?: number | null
}

interface BatchResult {
  ok?: boolean
  samples: DecodedSample[]
  histogram_m_s: { bins: number[]; counts: number[]; edges: number[] }
  best: DecodedSample | null
  caption?: string
  n_sampled?: number
  all_m_s?: number[]
}

interface Props {
  graph: GraphPayload | null
  batch: BatchPayload | null
  receipt: ReceiptInspect | null
  lang: LangMode
  onLoadAlloyReceipt?: () => void
}

const A_COLOR = '#c47a3a' // copper / amber
const B_COLOR = '#6a8fa8' // zinc / steel-blue
const BG = '#0c0e10'

function OccupancyGrid({
  grid,
  size = 280,
  title,
  highlight,
}: {
  grid: number[][] | null
  size?: number
  title?: string
  highlight?: boolean
}) {
  if (!grid || !grid.length) {
    return (
      <div
        className="alloy-grid empty"
        style={{ width: size, height: size, background: BG }}
      >
        <span className="dim">no occupancy</span>
      </div>
    )
  }
  const h = grid.length
  const w = grid[0]?.length ?? h
  const cell = Math.floor(size / Math.max(w, h))
  const pad = Math.max(0, (size - cell * w) / 2)

  return (
    <div
      className={`alloy-grid ${highlight ? 'highlight' : ''}`}
      style={{ width: size, height: size, background: BG }}
      title={title}
    >
      <svg width={size} height={size} role="img" aria-label={title ?? 'occupancy'}>
        <rect width={size} height={size} fill={BG} />
        {grid.map((row, y) =>
          row.map((v, x) => {
            const fill = v ? B_COLOR : A_COLOR
            return (
              <rect
                key={`${x}-${y}`}
                x={pad + x * cell + 1}
                y={pad + y * cell + 1}
                width={Math.max(1, cell - 2)}
                height={Math.max(1, cell - 2)}
                rx={Math.min(4, cell * 0.15)}
                fill={fill}
                opacity={0.92}
              />
            )
          }),
        )}
        {/* subtle grid lines */}
        {Array.from({ length: w + 1 }, (_, i) => (
          <line
            key={`vx${i}`}
            x1={pad + i * cell}
            y1={pad}
            x2={pad + i * cell}
            y2={pad + h * cell}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth={1}
          />
        ))}
      </svg>
    </div>
  )
}

function Gauge({ label, value, max = 1, format }: { label: string; value: number; max?: number; format?: (n: number) => string }) {
  const pct = Math.max(0, Math.min(100, (Math.abs(value) / max) * 100))
  const display = format ? format(value) : value.toFixed(3)
  return (
    <div className="alloy-gauge">
      <div className="alloy-gauge-head">
        <span className="k">{label}</span>
        <span className="v mono">{display}</span>
      </div>
      <div className="alloy-gauge-track">
        <div className="alloy-gauge-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function Histogram({ bins, counts }: { bins: number[]; counts: number[] }) {
  const max = Math.max(1, ...counts)
  const w = 220
  const h = 72
  const barW = w / Math.max(1, counts.length)
  return (
    <svg className="alloy-hist" width={w} height={h} role="img" aria-label="m_s histogram">
      <rect width={w} height={h} fill="transparent" />
      {counts.map((c, i) => {
        const bh = (c / max) * (h - 8)
        const x = i * barW
        const mid = bins[i] ?? 0
        const warm = Math.abs(mid) > 0.4
        return (
          <rect
            key={i}
            x={x + 1}
            y={h - bh}
            width={Math.max(1, barW - 2)}
            height={bh}
            fill={warm ? A_COLOR : B_COLOR}
            opacity={0.75}
          />
        )
      })}
      <line x1={0} y1={h - 1} x2={w} y2={h - 1} stroke="var(--border)" />
    </svg>
  )
}

function localDecode(state: number[], width: number, height: number): DecodedSample {
  const n = width * height
  const flat = state.slice(0, n)
  while (flat.length < n) flat.push(0)
  const occupancy: number[][] = []
  for (let y = 0; y < height; y++) {
    occupancy.push(flat.slice(y * width, (y + 1) * width).map((v) => (v > 0.5 ? 1 : 0)))
  }
  let m = 0
  let cb = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = occupancy[y][x]
      const s = v ? 1 : -1
      const stagger = (x + y) % 2 === 0 ? 1 : -1
      m += stagger * s
      cb += v
    }
  }
  m /= n
  cb /= n
  // unlike edges
  let edges = 0
  let unlike = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x + 1 < width) {
        edges++
        if (occupancy[y][x] !== occupancy[y][x + 1]) unlike++
      }
      if (y + 1 < height) {
        edges++
        if (occupancy[y][x] !== occupancy[y + 1][x]) unlike++
      }
    }
  }
  // config energy with default weights
  const wAA = 0.45
  const wBB = 0.45
  const wAB = -0.2
  let e = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = occupancy[y][x]
      if (x + 1 < width) {
        const b = occupancy[y][x + 1]
        e += a === b ? (a ? wBB : wAA) : wAB
      }
      if (y + 1 < height) {
        const b = occupancy[y + 1][x]
        e += a === b ? (a ? wBB : wAA) : wAB
      }
    }
  }
  return {
    occupancy,
    width,
    height,
    m_s: m,
    abs_m_s: Math.abs(m),
    concentration_B: cb,
    unlike_edge_fraction: edges ? unlike / edges : 0,
    energy_config: e,
  }
}

export function AlloyLabView({
  graph,
  batch,
  receipt,
  onLoadAlloyReceipt,
}: Props) {
  const isAlloy =
    receipt?.id === 'prog_alloy_ordering_8x8' ||
    graph?.receipt_id === 'prog_alloy_ordering_8x8'
  // Receipt graphs often report shape=[n_nodes]; alloy program is always 8×8.
  const shape = graph?.shape ?? []
  const width =
    isAlloy || (shape.length === 1 && shape[0] === 64)
      ? 8
      : shape.length >= 2
        ? shape[0]
        : shape[0] && shape[0] <= 32
          ? shape[0]
          : 8
  const height =
    isAlloy || (shape.length === 1 && shape[0] === 64)
      ? 8
      : shape.length >= 2
        ? shape[1]
        : width

  const live = useMemo(() => {
    const st = batch?.last_state
    if (!st || !st.length) return null
    return localDecode(st, width, height)
  }, [batch?.last_state, width, height])

  const [gallery, setGallery] = useState<BatchResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [selected, setSelected] = useState<DecodedSample | null>(null)
  const [nSamples, setNSamples] = useState(16)

  const hero = selected ?? live

  const runBatch = useCallback(async () => {
    setBusy(true)
    setErr(null)
    try {
      const res = await fetch('/api/lab/alloy/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n: nSamples,
          width,
          height,
          receipt_id: 'prog_alloy_ordering_8x8',
          rank_by: 'abs_m_s',
          top_k: 8,
          beta: 0.9,
          warmup: 60,
          steps_per_sample: 4,
          seed: 42,
        }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const data = (await res.json()) as BatchResult
      setGallery(data)
      if (data.best) setSelected(data.best)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'batch failed')
    } finally {
      setBusy(false)
    }
  }, [nSamples, width, height])

  // Seed gallery from current batch states if present
  useEffect(() => {
    if (!batch?.states?.length || gallery) return
    const decoded = batch.states.map((s) => localDecode(s, width, height))
    decoded.sort((a, b) => b.abs_m_s - a.abs_m_s)
    const ms = decoded.map((d) => d.m_s)
    const nBins = 21
    const edges: number[] = []
    for (let i = 0; i <= nBins; i++) edges.push(-1 + (2 * i) / nBins)
    const counts = new Array(nBins).fill(0)
    const bins = edges.slice(0, -1).map((e, i) => 0.5 * (e + edges[i + 1]))
    for (const v of ms) {
      let idx = Math.floor(((v + 1) / 2) * nBins)
      if (idx < 0) idx = 0
      if (idx >= nBins) idx = nBins - 1
      counts[idx]++
    }
    setGallery({
      samples: decoded.slice(0, 8),
      histogram_m_s: { bins, counts, edges },
      best: decoded[0] ?? null,
      n_sampled: decoded.length,
      all_m_s: ms,
      caption: 'Distribution Lab · binary ordering alloy · THRML sim · not silicon',
    })
  }, [batch?.states, width, height, gallery])

  return (
    <div className="view alloy-lab-view">
      <div className="alloy-caption">
        Distribution Lab · binary ordering alloy · THRML sim · not silicon
        {!isAlloy ? (
          <button
            type="button"
            className="btn tiny phosphor"
            style={{ marginLeft: '0.75rem' }}
            onClick={onLoadAlloyReceipt}
          >
            Load prog_alloy_ordering_8x8
          </button>
        ) : null}
      </div>

      <div className="alloy-layout">
        <div className="alloy-hero">
          <OccupancyGrid
            grid={hero?.occupancy ?? null}
            size={360}
            title="Occupancy A/B"
            highlight
          />
          <div className="alloy-species-legend">
            <span>
              <span className="swatch" style={{ background: A_COLOR }} /> A · Cu-like
            </span>
            <span>
              <span className="swatch" style={{ background: B_COLOR }} /> B · Zn-like
            </span>
          </div>
        </div>

        <div className="alloy-side">
          <h3 className="alloy-side-title">Order metrics</h3>
          <Gauge label="|m_s| staggered order" value={hero?.abs_m_s ?? 0} />
          <Gauge
            label="m_s (signed)"
            value={hero?.m_s ?? 0}
            format={(n) => n.toFixed(3)}
          />
          <Gauge
            label="c_B concentration"
            value={hero?.concentration_B ?? 0}
            format={(n) => `${(n * 100).toFixed(1)}%`}
          />
          <Gauge
            label="unlike-edge % (SRO)"
            value={hero?.unlike_edge_fraction ?? 0}
            format={(n) => `${(n * 100).toFixed(1)}%`}
          />
          <Gauge
            label="E_config (YAML weights)"
            value={hero ? Math.abs(hero.energy_config) : 0}
            max={30}
            format={() => (hero ? hero.energy_config.toFixed(2) : ', ')}
          />
          {hero?.energy_ising != null ? (
            <p className="empty-hint mono">E_ising {Number(hero.energy_ising).toFixed(3)}</p>
          ) : null}

          <div className="alloy-batch-controls">
            <label className="dim">
              Best-of-N{' '}
              <input
                type="number"
                min={4}
                max={64}
                value={nSamples}
                onChange={(e) => setNSamples(Number(e.target.value))}
                className="alloy-n-input mono"
              />
            </label>
            <button
              type="button"
              className="btn phosphor"
              disabled={busy}
              onClick={() => void runBatch()}
            >
              {busy ? 'Sampling…' : 'Sample batch (THRML)'}
            </button>
          </div>
          {err ? <p className="status-error">{err}</p> : null}
        </div>
      </div>

      {gallery ? (
        <div className="alloy-gallery-section">
          <div className="alloy-gallery-head">
            <h3>Best-of-N gallery</h3>
            <span className="dim mono">
              n={gallery.n_sampled ?? gallery.samples.length} · ranked by |m_s|
            </span>
          </div>
          <div className="alloy-gallery">
            {gallery.samples.map((s, i) => (
              <button
                key={i}
                type="button"
                className={`alloy-thumb ${selected === s ? 'active' : ''}`}
                onClick={() => setSelected(s)}
                title={`|m_s|=${s.abs_m_s.toFixed(3)}`}
              >
                <OccupancyGrid grid={s.occupancy} size={88} />
                <span className="mono">{s.abs_m_s.toFixed(2)}</span>
              </button>
            ))}
          </div>
          <div className="alloy-hist-block">
            <h4>Histogram · m_s</h4>
            <Histogram
              bins={gallery.histogram_m_s.bins}
              counts={gallery.histogram_m_s.counts}
            />
          </div>
        </div>
      ) : (
        <p className="empty-hint">
          Run the sampler or click <strong>Sample batch</strong> for a ranked gallery.
        </p>
      )}
    </div>
  )
}
