import type { BatchPayload } from '../types'
import { Sparkline } from './Sparkline'

interface Props {
  batch: BatchPayload | null
}

function fmt(x: number, d = 3) {
  if (!Number.isFinite(x)) return ', '
  return x.toFixed(d)
}

export function MetricsPanel({ batch }: Props) {
  const e = batch?.metrics.energy
  const m = batch?.metrics.magnetization
  return (
    <aside className="panel metrics-col">
      <h2>Metrics</h2>

      <div className="metric-card">
        <div className="title">Energy</div>
        <div className="big">{e ? fmt(e.last, 2) : ', '}</div>
        <div className="sub">
          <span>mean {e ? fmt(e.mean, 2) : ', '}</span>
          <span>std {e ? fmt(e.std, 2) : ', '}</span>
          <span>ρ₁ {e ? fmt(e.lag1_autocorr, 3) : ', '}</span>
          <span>ESS≈ {e ? fmt(e.ess, 1) : ', '}</span>
        </div>
        <Sparkline data={batch?.history.energy ?? []} color="#3ee0b0" />
      </div>

      <div className="metric-card">
        <div className="title">Magnetization</div>
        <div className="big">{m ? fmt(m.last, 3) : ', '}</div>
        <div className="sub">
          <span>mean {m ? fmt(m.mean, 3) : ', '}</span>
          <span>std {m ? fmt(m.std, 3) : ', '}</span>
          <span>ρ₁ {m ? fmt(m.lag1_autocorr, 3) : ', '}</span>
          <span>ESS≈ {m ? fmt(m.ess, 1) : ', '}</span>
        </div>
        <Sparkline data={batch?.history.magnetization ?? []} color="#4cc9f0" />
      </div>

      <div className="metric-card">
        <div className="title">Sampling</div>
        <div className="sub" style={{ gridTemplateColumns: '1fr' }}>
          <span>step {batch?.step ?? 0}</span>
          <span>active block {batch?.active_block ?? 0}</span>
          <span>window n={e?.n ?? 0}</span>
        </div>
      </div>

      <p className="status-line">
        ESS uses lag-1 AR(1) approximation on a rolling window, diagnostic, not
        a formal MCMC ESS.
      </p>
    </aside>
  )
}
