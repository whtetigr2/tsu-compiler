import { useCallback, useEffect, useMemo, useState } from 'react'
import type { GraphPayload, LangMode, ReceiptInspect } from '../../types'
import { Sparkline } from '../Sparkline'

interface EbmDecoded {
  image?: number[][]
  occupancy?: number[][]
  width: number
  height: number
  density: number
  row_purity: number
  col_purity: number
  bar_stripe_score: number
  is_pure: boolean
  energy_ising?: number | null
  sample_index?: number | null
}

interface EbmInfo {
  ok?: boolean
  caption?: string
  honesty?: string
  train?: {
    final_cd_moment_l1?: number
    pure_rate?: number
    mean_bar_stripe_score?: number
    gate_passed?: boolean
    n_epochs?: number
    curve?: { epoch: number; cd_moment_l1: number }[]
    checkpoint_loaded?: boolean
  }
  cached_data_examples?: number[][][]
  cached_model_samples?: number[][][]
  dataset?: { name?: string; grid?: string; positive?: string; n_visible?: number; n_hidden?: number }
  model?: { kind?: string; graph?: string }
  paths?: Record<string, string>
}

interface Props {
  graph: GraphPayload | null
  receipt: ReceiptInspect | null
  lang: LangMode
  onLoadEbmReceipt?: () => void
}

const ON = '#e8eef2'
const OFF = '#1a1e24'
const BG = '#0c0e10'
const ACCENT = '#3ee0b0'

function BinaryTile({
  grid,
  size = 120,
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
      <div className="ebm-tile empty" style={{ width: size, height: size, background: BG }}>
        <span className="dim">, </span>
      </div>
    )
  }
  const h = grid.length
  const w = grid[0]?.length ?? h
  const cell = Math.floor(size / Math.max(w, h))
  const pad = Math.max(0, (size - cell * w) / 2)
  return (
    <div
      className={`ebm-tile ${highlight ? 'highlight' : ''}`}
      style={{ width: size, height: size, background: BG }}
      title={title}
    >
      <svg width={size} height={size} role="img" aria-label={title ?? 'binary image'}>
        <rect width={size} height={size} fill={BG} />
        {grid.map((row, y) =>
          row.map((v, x) => (
            <rect
              key={`${x}-${y}`}
              x={pad + x * cell + 0.5}
              y={pad + y * cell + 0.5}
              width={Math.max(1, cell - 1)}
              height={Math.max(1, cell - 1)}
              rx={Math.min(2, cell * 0.12)}
              fill={v ? ON : OFF}
              opacity={0.95}
            />
          )),
        )}
      </svg>
    </div>
  )
}

function Gauge({
  label,
  value,
  max = 1,
  format,
}: {
  label: string
  value: number
  max?: number
  format?: (n: number) => string
}) {
  const pct = Math.max(0, Math.min(100, (Math.abs(value) / max) * 100))
  const display = format ? format(value) : value.toFixed(3)
  return (
    <div className="lab-gauge">
      <div className="lab-gauge-head">
        <span className="k">{label}</span>
        <span className="v mono">{display}</span>
      </div>
      <div className="lab-gauge-track">
        <div className="lab-gauge-fill" style={{ width: `${pct}%`, background: ACCENT }} />
      </div>
    </div>
  )
}

export function EbmLabView({ graph, receipt, lang: _lang, onLoadEbmReceipt }: Props) {
  const isEbm =
    receipt?.id === 'prog_ebm_bars_stripes' || graph?.receipt_id === 'prog_ebm_bars_stripes'

  const [info, setInfo] = useState<EbmInfo | null>(null)
  const [samples, setSamples] = useState<EbmDecoded[]>([])
  const [dataTiles, setDataTiles] = useState<number[][][]>([])
  const [selected, setSelected] = useState<EbmDecoded | null>(null)
  const [busy, setBusy] = useState(false)
  const [trainBusy, setTrainBusy] = useState(false)
  const [n, setN] = useState(16)
  const [err, setErr] = useState<string | null>(null)
  const [lossCurve, setLossCurve] = useState<number[]>([])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/lab/ebm/info')
        if (!res.ok) throw new Error(`info ${res.status}`)
        const data = (await res.json()) as EbmInfo
        if (cancelled) return
        setInfo(data)
        setDataTiles(data.cached_data_examples || [])
        const curve = (data.train?.curve || []).map((c) => c.cd_moment_l1)
        setLossCurve(curve)
        if (!samples.length && data.cached_model_samples?.length) {
          setSamples(
            data.cached_model_samples.map((img, i) => {
              const h = img.length
              const w = img[0]?.length ?? h
              const n = Math.max(1, h * w)
              return {
                image: img,
                occupancy: img,
                width: w,
                height: h,
                density: img.flat().reduce((a, b) => a + b, 0) / n,
                row_purity: 0,
                col_purity: 0,
                bar_stripe_score: 0,
                is_pure: false,
                sample_index: i,
              }
            }),
          )
        }
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onSample = useCallback(async () => {
    setBusy(true)
    setErr(null)
    try {
      const res = await fetch('/api/lab/ebm/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n,
          receipt_id: 'prog_ebm_bars_stripes',
          warmup: 80,
          beta: 1.2,
        }),
      })
      if (!res.ok) {
        const t = await res.text()
        throw new Error(t || `sample ${res.status}`)
      }
      const data = await res.json()
      const list = (data.samples || []) as EbmDecoded[]
      setSamples(list)
      setSelected(list[0] ?? null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [n])

  const onRetrainLite = useCallback(async () => {
    setTrainBusy(true)
    setErr(null)
    try {
      const res = await fetch('/api/lab/ebm/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lite: true, epochs: 12, force: true }),
      })
      if (!res.ok) throw new Error(`train ${res.status}`)
      const data = await res.json()
      const curve = (data.curve || []).map((c: { cd_moment_l1: number }) => c.cd_moment_l1)
      if (curve.length) setLossCurve(curve)
      setInfo((prev) => ({
        ...(prev || {}),
        train: {
          ...(prev?.train || {}),
          final_cd_moment_l1: data.final_cd_moment_l1,
          pure_rate: data.pure_rate,
          mean_bar_stripe_score: data.mean_bar_stripe_score,
          gate_passed: data.gate_passed,
          n_epochs: data.n_epochs,
          curve: data.curve,
          checkpoint_loaded: true,
        },
      }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setTrainBusy(false)
    }
  }, [])

  const hero = useMemo(() => {
    if (selected?.image || selected?.occupancy) return selected.image || selected.occupancy || null
    if (samples[0]?.image) return samples[0].image
    if (dataTiles[0]) return dataTiles[0]
    return null
  }, [selected, samples, dataTiles])

  const metrics = selected || samples[0] || null

  return (
    <div className="view lab-lab-view ebm-lab-view">
      <div className="lab-caption ebm-honesty">
        THRML/JAX SIM · trained RBM · not Extropic silicon · not Z1T
      </div>

      {!isEbm ? (
        <div className="lab-side" style={{ marginBottom: '0.75rem' }}>
          <p className="dim" style={{ margin: 0 }}>
            Load the bars-and-stripes RBM receipt (4×4 visibles + hiddens) to sample through SamplerEngine.
          </p>
          {onLoadEbmReceipt ? (
            <button type="button" className="btn phosphor" onClick={onLoadEbmReceipt}>
              Load prog_ebm_bars_stripes
            </button>
          ) : null}
        </div>
      ) : null}

      <div className="lab-layout ebm-layout">
        <div className="lab-hero">
          <BinaryTile grid={hero} size={280} title="Selected 4×4 visibles" highlight />
          <div className="lab-species-legend">
            <span>
              <span style={{ color: ON }}>■</span> 1 (on)
            </span>
            <span>
              <span style={{ color: OFF, outline: '1px solid #333' }}>■</span> 0 (off)
            </span>
          </div>
        </div>

        <div className="lab-side">
          <h3 className="lab-side-title">Bars &amp; stripes decode (4×4 RBM)</h3>
          <Gauge label="bar/stripe score" value={metrics?.bar_stripe_score ?? 0} />
          <Gauge label="row purity" value={metrics?.row_purity ?? 0} />
          <Gauge label="col purity" value={metrics?.col_purity ?? 0} />
          <Gauge label="density" value={metrics?.density ?? 0} />
          {metrics?.energy_ising != null ? (
            <div className="mono dim" style={{ fontSize: '0.8rem' }}>
              E_ising = {Number(metrics.energy_ising).toFixed(3)}
            </div>
          ) : null}
          {info?.train?.pure_rate != null ? (
            <Gauge label="pure_rate (train gate ≥0.40)" value={Number(info.train.pure_rate)} />
          ) : null}
          {info?.train?.mean_bar_stripe_score != null ? (
            <div className="mono dim" style={{ fontSize: '0.8rem' }}>
              mean bar/stripe score ≈ {Number(info.train.mean_bar_stripe_score).toFixed(3)}
              {info.train.gate_passed ? ' · gate PASS' : ''}
            </div>
          ) : null}
          {info?.train?.final_cd_moment_l1 != null ? (
            <div className="mono dim" style={{ fontSize: '0.8rem' }}>
              CD moment L1 ≈ {Number(info.train.final_cd_moment_l1).toFixed(4)}
              {info.train.n_epochs != null ? ` · ${info.train.n_epochs} ep` : ''}
            </div>
          ) : null}
          {info?.model?.kind ? (
            <div className="mono dim" style={{ fontSize: '0.75rem' }}>
              {info.model.kind}
              {info.dataset?.n_hidden != null ? ` · ${info.dataset.n_visible ?? 16}v+${info.dataset.n_hidden}h` : ''}
            </div>
          ) : null}

          {lossCurve.length > 0 ? (
            <div className="ebm-loss-block">
              <div className="lab-gallery-head">
                <h3>Train loss (CD moment L1)</h3>
              </div>
              <Sparkline data={lossCurve} color={ACCENT} height={56} />
            </div>
          ) : null}

          <div className="lab-batch-controls">
            <label className="dim">
              N{' '}
              <input
                className="lab-n-input mono"
                type="number"
                min={4}
                max={64}
                value={n}
                onChange={(e) => setN(Number(e.target.value) || 16)}
              />
            </label>
            <button type="button" className="btn phosphor" disabled={busy} onClick={() => void onSample()}>
              {busy ? 'Sampling…' : 'Sample (THRML)'}
            </button>
            <button
              type="button"
              className="btn tiny"
              disabled={trainBusy}
              onClick={() => void onRetrainLite()}
              title="Short PCD retrain, does not recompile receipt"
            >
              {trainBusy ? 'Training…' : 'Retrain lite'}
            </button>
          </div>
          {err ? (
            <div className="dim" style={{ color: 'var(--warn)', fontSize: '0.8rem' }}>
              {err}
            </div>
          ) : null}
        </div>
      </div>

      <div className="lab-gallery-section">
        <div className="lab-gallery-head">
          <h3>Train examples</h3>
          <span className="dim mono">pure bars ∪ stripes</span>
        </div>
        <div className="lab-gallery ebm-gallery">
          {dataTiles.slice(0, 12).map((g, i) => (
            <button
              key={`d${i}`}
              type="button"
              className="lab-thumb ebm-thumb"
              onClick={() => {
                const h = g.length
                const w = g[0]?.length ?? h
                setSelected({
                  image: g,
                  width: w,
                  height: h,
                  density: g.flat().reduce((a, b) => a + b, 0) / Math.max(1, h * w),
                  row_purity: 1,
                  col_purity: 1,
                  bar_stripe_score: 1,
                  is_pure: true,
                })
              }}
            >
              <BinaryTile grid={g} size={72} />
            </button>
          ))}
          {!dataTiles.length ? <span className="dim">No cached data tiles, run train script.</span> : null}
        </div>
      </div>

      <div className="lab-gallery-section">
        <div className="lab-gallery-head">
          <h3>Model samples</h3>
          <span className="dim mono">decode → 4×4 visibles</span>
        </div>
        <div className="lab-gallery ebm-gallery">
          {samples.slice(0, 16).map((s, i) => {
            const g = s.image || s.occupancy || null
            return (
              <button
                key={`s${i}`}
                type="button"
                className={`lab-thumb ebm-thumb ${selected === s ? 'active' : ''}`}
                onClick={() => setSelected(s)}
                title={s.is_pure ? 'pure bar/stripe' : `score ${s.bar_stripe_score?.toFixed(2)}`}
              >
                <BinaryTile grid={g} size={72} highlight={s.is_pure} />
              </button>
            )
          })}
          {!samples.length ? <span className="dim">Click Sample to draw from the trained EBM.</span> : null}
        </div>
      </div>

      <p className="dim" style={{ fontSize: '0.78rem', marginTop: '0.5rem' }}>
        RBM (not pairwise): train → export V↔H Ising → compile/receipt → inspect/sample/decode visibles.
        Won&apos;t beat ImageNet, proves the loop. {info?.dataset?.positive ?? ''}
      </p>
    </div>
  )
}
