import type { ExampleShelfItem, ReceiptSummary } from '../types'

interface Props {
  open: boolean
  receipts: ReceiptSummary[]
  examples: ExampleShelfItem[]
  currentId: string | null
  onClose: () => void
  onSelect: (id: string) => void
}

export function ReceiptPicker({
  open,
  receipts,
  examples,
  currentId,
  onClose,
  onSelect,
}: Props) {
  if (!open) return null

  const curated = examples.filter((e) => e.kind === 'curated' || e.kind === 'stub' || e.extropic)
  const localOnly = receipts.filter((r) => !examples.some((e) => e.id === r.id && e.packaged))

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <div className="drawer wide" role="dialog" onClick={(e) => e.stopPropagation()}>
        <header>
          <h3>Open receipt / examples</h3>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </header>

        <h4>Extropic / curated examples</h4>
        <ul className="receipt-list">
          {curated.map((ex) => {
            const disabled = !ex.packaged || ex.status === 'stub'
            return (
              <li key={ex.id}>
                <button
                  type="button"
                  className={ex.id === currentId ? 'active' : ''}
                  disabled={disabled}
                  title={disabled ? ex.message ?? ex.notes : ex.notes}
                  onClick={() => {
                    if (disabled) return
                    onSelect(ex.id)
                    onClose()
                  }}
                >
                  <span className="mono">{ex.id}</span>
                  {ex.extropic ? <span className="chip">extropic</span> : null}
                  <span className={`chip ${ex.status}`}>{ex.status}</span>
                  {ex.receipt?.verdict ? (
                    <span className="chip">{ex.receipt.verdict}</span>
                  ) : null}
                  <span className="dim">
                    {ex.title}
                    {ex.receipt?.n_nodes != null ? ` · ${ex.receipt.n_nodes} spins` : ''}
                    {disabled && ex.message ? `, ${ex.message}` : ''}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
        {!curated.length ? (
          <p className="empty-hint">No shelf entries.</p>
        ) : null}

        <h4>All on-disk receipts</h4>
        <ul className="receipt-list">
          {receipts.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                className={r.id === currentId ? 'active' : ''}
                onClick={() => {
                  onSelect(r.id)
                  onClose()
                }}
              >
                <span className="mono">{r.id}</span>
                <span className="chip">{r.verdict ?? '?'}</span>
                <span className="dim">
                  {r.encoding ?? ', '} · {r.n_nodes ?? '?'} spins
                </span>
              </button>
            </li>
          ))}
        </ul>
        {!receipts.length ? (
          <p className="empty-hint">No curated receipts under /receipts.</p>
        ) : null}
        {localOnly.length ? (
          <p className="empty-hint">
            Extra local receipts not in the Extropic catalog are listed above.
          </p>
        ) : null}
      </div>
    </div>
  )
}
