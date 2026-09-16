import type { LangMode, ReceiptInspect } from '../../types'

interface Props {
  receipt: ReceiptInspect | null
  lang: LangMode
}

function fmt(v: number | null | undefined, d = 3): string {
  if (v == null || !Number.isFinite(v)) return ', '
  return Number.isInteger(v) ? String(v) : v.toFixed(d)
}

export function ResidualsView({ receipt }: Props) {
  const res = receipt?.residuals

  if (!receipt) {
    return (
      <div className="view">
        <div className="stub-panel">
          <h3>Residuals</h3>
          <p className="empty-hint">Load a receipt to inspect residuals / headroom.</p>
        </div>
      </div>
    )
  }

  if (!res || (!res.bars.length && !res.unavailable.length)) {
    return (
      <div className="view">
        <div className="stub-panel">
          <h3>Residuals</h3>
          <p className="empty-hint">
            No residual matrix or derivable headroom fields in this receipt.
            Observatory will not invent fake residual fields.
          </p>
          {(res?.notes ?? []).map((n) => (
            <p key={n} className="empty-hint">
              {n}
            </p>
          ))}
        </div>
      </div>
    )
  }

  const absMax = Math.max(
    ...res.bars.map((b) => Math.abs(b.value)),
    ...res.bars.map((b) => (b.utilisation != null ? b.utilisation : 0)),
    1e-9,
  )

  // Group heatmap cells by row for a simple grid
  const rows = Array.from(new Set(res.heatmap.map((c) => c.row)))

  return (
    <div className="view residuals-view">
      <div className="view-caption">
        Residuals MVP, derived headroom / connectivity deltas · no invented receipt
        fields
        {res.has_receipt_residual_matrix
          ? ''
          : ' · receipt has no residual matrix'}
      </div>

      <div className="residuals-layout">
        <div className="fabric-card">
          <h3>Bar chart (v0)</h3>
          <div className="residual-bars">
            {res.bars.map((b) => {
              const pct = Math.min(100, (Math.abs(b.value) / absMax) * 100)
              return (
                <div key={b.id} className="residual-bar-row">
                  <div className="residual-bar-label">
                    <span>{b.label}</span>
                    {b.derived ? <span className="chip derived">derived</span> : null}
                    {b.assumed ? <span className="chip assumed">assumed</span> : null}
                  </div>
                  <div className="residual-bar-track">
                    <div
                      className={`residual-bar-fill ${b.value < 0 ? 'neg' : 'pos'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="mono residual-bar-value">
                    {fmt(b.value, 4)}
                    {b.utilisation != null ? (
                      <span className="dim"> · util {fmt(b.utilisation * 100, 1)}%</span>
                    ) : null}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="fabric-card">
          <h3>Heatmap (v0 cells)</h3>
          {res.heatmap.length ? (
            <div className="residual-heat">
              {rows.map((row) => (
                <div key={row} className="residual-heat-row">
                  <div className="residual-heat-rowlabel mono">{row}</div>
                  <div className="residual-heat-cells">
                    {res.heatmap
                      .filter((c) => c.row === row)
                      .map((c) => {
                        const intensity = Math.min(
                          1,
                          Math.abs(c.value) / absMax,
                        )
                        return (
                          <div
                            key={`${c.row}-${c.col}`}
                            className="residual-heat-cell"
                            title={`${c.col}: ${c.value}`}
                            style={{
                              background: `rgba(62, 224, 176, ${0.12 + intensity * 0.7})`,
                            }}
                          >
                            <span className="mono">{c.col}</span>
                            <strong className="mono">{fmt(c.value, 2)}</strong>
                            {c.derived ? (
                              <span className="chip derived">derived</span>
                            ) : null}
                          </div>
                        )
                      })}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-hint">No heatmap cells.</p>
          )}
        </div>
      </div>

      {res.unavailable.length ? (
        <div className="fabric-card">
          <h3>Unavailable (honest)</h3>
          <ul className="prose-list">
            {res.unavailable.map((u) => (
              <li key={u}>{u}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {(res.notes ?? []).map((n) => (
        <p key={n} className="empty-hint">
          {n}
        </p>
      ))}
    </div>
  )
}
