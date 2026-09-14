import type { BatchPayload, LangMode, ReceiptInspect } from '../../types'
import { label } from '../../lib/glossary'

interface Props {
  receipt: ReceiptInspect | null
  batch: BatchPayload | null
  lang: LangMode
}

function previewIndices(idxs: number[], limit = 24): string {
  if (!idxs.length) return '—'
  const head = idxs.slice(0, limit).join(', ')
  return idxs.length > limit ? `${head}, … (+${idxs.length - limit})` : head
}

export function ScheduleView({ receipt, batch, lang }: Props) {
  const schedule = receipt?.schedule
  const blocks = schedule?.blocks
  const active = batch?.active_block ?? null
  const kernel = schedule?.kernel ?? receipt?.kernel

  if (!receipt) {
    return (
      <div className="view">
        <div className="stub-panel">
          <h3>Schedule</h3>
          <p className="empty-hint">Load a receipt to inspect chromatic blocks.</p>
        </div>
      </div>
    )
  }

  if (!blocks || !blocks.length) {
    return (
      <div className="view">
        <div className="stub-panel">
          <h3>Schedule</h3>
          <p className="empty-hint">
            {(schedule?.notes ?? []).join(' ') ||
              'No chromatic blocks in program.json — timeline unavailable.'}
          </p>
        </div>
      </div>
    )
  }

  const maxSize = Math.max(...blocks.map((b) => b.size), 1)

  return (
    <div className="view schedule-view">
      <div className="view-caption">
        {label('blocks', lang)} timeline · kernel{' '}
        <span className="mono">{kernel ?? '—'}</span>
        {active != null ? (
          <>
            {' '}
            · live active colour{' '}
            <span className={`chip block-chip c${active}`}>V{active}</span>
          </>
        ) : (
          ' · idle (run/step to pulse active block)'
        )}
      </div>

      <div className="schedule-timeline" role="list">
        {blocks.map((b) => {
          const isActive = active === b.colour
          const widthPct = Math.max(8, (b.size / maxSize) * 100)
          return (
            <div
              key={b.colour}
              className={`schedule-block c${b.colour} ${isActive ? 'active' : ''}`}
              role="listitem"
              style={{ flexGrow: b.size, flexBasis: `${widthPct}%` }}
            >
              <div className="schedule-block-head">
                <strong>
                  Colour {b.colour} / V{b.colour}
                </strong>
                <span className="mono">{b.size} spins</span>
                {isActive ? <span className="chip">updating</span> : null}
              </div>
              <div className="schedule-bar" aria-hidden />
              <p className="schedule-indices mono">{previewIndices(b.indices)}</p>
            </div>
          )
        })}
      </div>

      <div className="fabric-card">
        <h3>Which spins update when</h3>
        <p className="dim" style={{ fontSize: '0.8rem', marginTop: 0 }}>
          Chromatic block-Gibbs alternates exclusive colour sets so neighbors
          never update in the same half-sweep. Live{' '}
          <span className="mono">active_block</span> comes from the WS batch
          (step parity), not a per-sweep THRML observer.
        </p>
        <dl className="kv">
          {blocks.map((b) => (
            <div key={b.colour}>
              <dt>
                V{b.colour} {active === b.colour ? '(active)' : ''}
              </dt>
              <dd className="mono">{b.size} indices</dd>
            </div>
          ))}
        </dl>
        {(schedule?.notes ?? []).map((n) => (
          <p key={n} className="empty-hint">
            {n}
          </p>
        ))}
      </div>
    </div>
  )
}
