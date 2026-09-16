import type { BatchPayload } from '../types'
import { TracePlot } from './TracePlot'
import { StatValue } from './StatValue'

interface Props {
  batch: BatchPayload | null
}

function fmt(x: number | null | undefined, d = 3) {
  if (x == null || !Number.isFinite(x)) return 'unavailable'
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
        <div className="big">{e ? fmt(e.last, 2) : 'unavailable'}</div>
        <div className="sub">
          <span>mean {e ? fmt(e.mean, 2) : 'unavailable'}</span>
          <span>std {e ? fmt(e.std, 2) : 'unavailable'}</span>
          <span>ρ₁ {e ? fmt(e.lag1_autocorr, 3) : 'unavailable'}</span>
          <span>ESS <StatValue value={e?.ess} reason={e?.ess_reason} digits={0} /></span>
        </div>
        <TracePlot data={batch?.history.energy ?? []} label="energy over sweeps"
                   color="#7ec8c0" step={batch?.step} />
      </div>

      <div className="metric-card">
        <div className="title">Magnetization</div>
        <div className="big">{m ? fmt(m.last, 3) : 'unavailable'}</div>
        <div className="sub">
          <span>mean {m ? fmt(m.mean, 3) : 'unavailable'}</span>
          <span>std {m ? fmt(m.std, 3) : 'unavailable'}</span>
          <span>ρ₁ {m ? fmt(m.lag1_autocorr, 3) : 'unavailable'}</span>
          <span>ESS <StatValue value={m?.ess} reason={m?.ess_reason} digits={0} /></span>
        </div>
        <TracePlot data={batch?.history.magnetization ?? []}
                   label="magnetization over sweeps" color="#4cc9f0"
                   step={batch?.step} />
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
