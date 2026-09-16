import type { BatchPayload, LangMode, ReceiptInspect } from '../../types'
import { label, tip } from '../../lib/glossary'
import { TracePlot } from '../TracePlot'

interface Props {
  batch: BatchPayload | null
  receipt: ReceiptInspect | null
  lang: LangMode
}

function fmt(x: number | null | undefined, d = 3) {
  if (x == null || !Number.isFinite(x)) return 'unavailable'
  return x.toFixed(d)
}

export function ScopeView({ batch, receipt, lang }: Props) {
  const e = batch?.metrics.energy
  const m = batch?.metrics.magnetization
  const ess = receipt?.verification?.ess
  return (
    <div className="view scope-view">
      <div className="view-caption">
        Scope rack, live series. ESS only when honest (receipt contract).
      </div>
      <div className="scope-grid">
        <div className="metric-card">
          <div className="title">{label('energy', lang)}</div>
          <div className="big">{fmt(e?.last, 2)}</div>
          <div className="sub">
            <span>mean {fmt(e?.mean, 2)}</span>
            <span>std {fmt(e?.std, 2)}</span>
            <span>ρ₁ {fmt(e?.lag1_autocorr, 3)}</span>
          </div>
          <TracePlot data={batch?.history.energy ?? []} label="energy over sweeps"
                     color="#7ec8c0" step={batch?.step} />
        </div>
        <div className="metric-card">
          <div className="title">{label('mag', lang)}</div>
          <div className="big">{fmt(m?.last, 3)}</div>
          <div className="sub">
            <span>mean {fmt(m?.mean, 3)}</span>
            <span>std {fmt(m?.std, 3)}</span>
            <span>ρ₁ {fmt(m?.lag1_autocorr, 3)}</span>
          </div>
          <TracePlot data={batch?.history.magnetization ?? []}
                     label="magnetization over sweeps" color="#4cc9f0"
                     step={batch?.step} />
        </div>
        <div className="metric-card">
          <div className="title" title={tip('ess')}>
            {label('ess', lang)}
          </div>
          {ess?.available ? (
            <div className="big">{fmt(ess.value, 1)}</div>
          ) : (
            <>
              <div className="big">unavailable</div>
              <p className="empty-hint">
                {ess?.reason ??
                  'Diagnostic lag-1 ESS on the stream is not a formal MCMC ESS; receipt ESS withheld when contract broken.'}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
