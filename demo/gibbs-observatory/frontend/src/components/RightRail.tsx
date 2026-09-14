import { useState } from 'react'
import type { BatchPayload, GraphPayload, LangMode, ReceiptInspect } from '../types'
import { label, tip } from '../lib/glossary'
import { ClaimHygienePanel } from './ClaimHygienePanel'

interface Props {
  receipt: ReceiptInspect | null
  batch: BatchPayload | null
  graph: GraphPayload | null
  lang: LangMode
  running: boolean
  connected: boolean
  selectedSpin: number | null
}

function fmt(x: number | null | undefined, d = 3) {
  if (x == null || !Number.isFinite(x)) return '—'
  return x.toFixed(d)
}

export function RightRail({
  receipt,
  batch,
  graph,
  lang,
  running,
  connected,
  selectedSpin,
}: Props) {
  const [hygieneOpen, setHygieneOpen] = useState(false)
  const [gatesOpen, setGatesOpen] = useState(false)

  const e = batch?.metrics.energy
  const m = batch?.metrics.magnetization
  const ess = receipt?.verification?.ess
  const spinIdx = selectedSpin ?? graph?.world_idx?.[0] ?? 0
  const spinVal =
    batch?.last_state && spinIdx < batch.last_state.length
      ? batch.last_state[spinIdx]
      : null
  const nodeName =
    graph?.node_names?.[spinIdx] ??
    receipt?.spins.node_names?.[spinIdx] ??
    `s${spinIdx}`
  const bias = graph?.biases?.[spinIdx] ?? receipt?.biases?.[spinIdx] ?? null

  const readyLabel = !connected
    ? 'offline'
    : running
      ? 'streaming'
      : 'ready'

  return (
    <aside className="right-rail" data-tour="right-rail">
      <div className="sampler-card">
        <div className="sampler-card-head">
          <h2>Sampler state</h2>
          <span
            className={`sampler-ready ${
              !connected ? 'offline' : running ? 'streaming' : ''
            }`}
          >
            {readyLabel}
          </span>
        </div>

        <div className="metrics-grid">
          <div className="metric-cell" title={tip('energy')}>
            <span className="mk">{label('energy', lang)}</span>
            <span className="mv">{e ? fmt(e.last, 2) : '—'}</span>
            <span className="ms">
              μ {e ? fmt(e.mean, 2) : '—'} · ρ₁ {e ? fmt(e.lag1_autocorr, 2) : '—'}
            </span>
          </div>
          <div className="metric-cell" title={tip('mag')}>
            <span className="mk">{label('mag', lang)}</span>
            <span className="mv">{m ? fmt(m.last, 3) : '—'}</span>
            <span className="ms">μ {m ? fmt(m.mean, 3) : '—'}</span>
          </div>
          <div className="metric-cell">
            <span className="mk">Sweep</span>
            <span className="mv">{batch?.step ?? 0}</span>
            <span className="ms">
              ESS{' '}
              {ess?.available
                ? fmt(ess.value, 1)
                : e
                  ? `≈${fmt(e.ess, 0)}`
                  : '—'}
            </span>
          </div>
          <div className="metric-cell">
            <span className="mk">Active colour</span>
            <span className="mv">
              {batch != null ? `V${(batch.active_block ?? 0) + 1}` : '—'}
            </span>
            <span className="ms">{label('blocks', lang)} chromatic</span>
          </div>
        </div>

        <div className="inspector-card">
          <div className="ik">
            Selected spin · {nodeName}
          </div>
          <div className="inspector-grid">
            <div className="metric-cell">
              <span className="mk">State</span>
              <span className="mv">
                {spinVal == null ? '—' : spinVal > 0 ? '+1' : '−1'}
              </span>
            </div>
            <div className="metric-cell">
              <span className="mk">Local field h</span>
              <span className="mv">{bias == null ? '—' : fmt(bias, 3)}</span>
            </div>
          </div>
          <p className="inspector-hint">
            Click a labeled node on small programs to inspect. Large lattices
            default to the first world spin.
          </p>
        </div>
      </div>

      <section>
        <button
          type="button"
          className="collapse-toggle"
          aria-expanded={gatesOpen}
          onClick={() => setGatesOpen((v) => !v)}
        >
          <span>Gate ledger</span>
          <span>{gatesOpen ? '▾' : '▸'}</span>
        </button>
        {gatesOpen ? (
          <div className="collapse-body">
            <div className="gate-list">
              {(receipt?.gates ?? []).map((g) => (
                <div
                  key={g.gate}
                  className={`gate-row ${g.passed ? 'pass' : 'fail'}`}
                >
                  <span className="light" />
                  <span className="name">{g.gate}</span>
                  <span className="meas mono">
                    {g.measured == null ? '—' : String(g.measured)}
                    <span className="dim">
                      {' '}
                      / {g.limit == null ? '—' : String(g.limit)}
                    </span>
                  </span>
                  {g.assumed ? <span className="assumed">assumed</span> : null}
                </div>
              ))}
              {!receipt?.gates?.length ? (
                <p className="empty-hint">Load a receipt to populate gates.</p>
              ) : null}
            </div>
          </div>
        ) : null}
      </section>

      <section className="claim-rail-section">
        <button
          type="button"
          className="collapse-toggle"
          aria-expanded={hygieneOpen}
          onClick={() => setHygieneOpen((v) => !v)}
        >
          <span>Claim hygiene</span>
          <span>{hygieneOpen ? '▾' : '▸'}</span>
        </button>
        {hygieneOpen ? (
          <div className="collapse-body">
            <ClaimHygienePanel receipt={receipt} compact />
          </div>
        ) : null}
      </section>
    </aside>
  )
}
