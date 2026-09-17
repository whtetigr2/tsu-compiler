import { useEffect, useState } from 'react'
import type { ExampleShelfItem, ReceiptSummary } from '../types'

interface Props {
  open: boolean
  receipts: ReceiptSummary[]
  examples: ExampleShelfItem[]
  currentId: string | null
  onClose: () => void
  onSelect: (id: string) => void
}

/**
 * Choosing what to open.
 *
 * Two things this must not do again. It used to lead every row with the
 * receipt's DIRECTORY NAME in monospace, so the first thing a newcomer read was
 * `prog_placement_8x8_exclusion`. And the shelf carried the problem each
 * workload solves, its math, its measured hardware cost and a citation, none of
 * which appeared anywhere: data that only the API could see is not shipped.
 *
 * Verified workloads lead, because they are the strongest thing here. Somebody
 * else published the problem, it compiles, and a standing test says so.
 */
export function ReceiptPicker({
  open,
  receipts,
  examples,
  currentId,
  onClose,
  onSelect,
}: Props) {
  const [focused, setFocused] = useState<string | null>(null)
  const [showRaw, setShowRaw] = useState(false)

  const verified = examples.filter((e) => e.verified)
  const others = examples.filter((e) => !e.verified)
  const detail =
    examples.find((e) => e.id === focused) ?? verified[0] ?? others[0] ?? null

  useEffect(() => {
    if (open) setFocused(currentId)
  }, [open, currentId])

  if (!open) return null

  const row = (ex: ExampleShelfItem) => {
    const disabled = !ex.packaged || ex.status === 'stub'
    return (
      <li key={ex.id}>
        <button
          type="button"
          className={`example-row${ex.id === focused ? ' focused' : ''}${
            ex.id === currentId ? ' active' : ''
          }`}
          disabled={disabled}
          onClick={() => setFocused(ex.id)}
          onDoubleClick={() => {
            if (disabled) return
            onSelect(ex.id)
            onClose()
          }}
        >
          <span className="example-name">{ex.plain_name ?? ex.title}</span>
          <span className="example-meta">
            {ex.verified ? <span className="chip verified">VERIFIED</span> : null}
            {typeof ex.reachable === 'number' && ex.reachable < 8 ? (
              <span className="chip warn" title="how many configurations this program can reach">
                {ex.reachable} states
              </span>
            ) : null}
            {ex.receipt?.n_nodes != null ? (
              <span className="dim">{ex.receipt.n_nodes} spins</span>
            ) : null}
          </span>
        </button>
      </li>
    )
  }

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <div
        className="drawer wide picker"
        role="dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <h3>Open an example</h3>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="picker-body">
          <div className="picker-list">
            {verified.length ? (
              <>
                <h4>
                  Verified workloads
                  <span className="dim"> published elsewhere, compiled here</span>
                </h4>
                <ul className="receipt-list">{verified.map(row)}</ul>
              </>
            ) : null}

            <h4>Examples</h4>
            <ul className="receipt-list">{others.map(row)}</ul>
            {!examples.length ? <p className="empty-hint">No examples.</p> : null}

            <button
              type="button"
              className="btn tiny"
              onClick={() => setShowRaw((v) => !v)}
            >
              {showRaw ? 'Hide' : 'Show'} all receipts on disk ({receipts.length})
            </button>
            {showRaw ? (
              <>
                <p className="empty-hint">
                  Working artifacts, including parameter sweeps and scratch
                  compiles. Reachable, but not examples of what this does.
                </p>
                <ul className="receipt-list">
                  {receipts.map((r) => (
                    <li key={r.id}>
                      <button
                        type="button"
                        className={`example-row${
                          r.id === currentId ? ' active' : ''
                        }`}
                        onClick={() => {
                          onSelect(r.id)
                          onClose()
                        }}
                      >
                        <span className="mono">{r.id}</span>
                        <span className="example-meta">
                          <span className="chip">{r.verdict ?? '?'}</span>
                          <span className="dim">{r.n_nodes ?? '?'} spins</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </div>

          <div className="picker-detail">
            {detail ? (
              <>
                <h4>{detail.plain_name ?? detail.title}</h4>
                {detail.verified ? (
                  <p className="verified-line">
                    <span className="chip verified">VERIFIED</span>{' '}
                    Published by {detail.published_by}. {detail.verified_by}.
                  </p>
                ) : null}
                <p className="detail-lead">{detail.one_line ?? detail.notes}</p>

                {detail.problem ? (
                  <section>
                    <h5>The problem</h5>
                    <p>{detail.problem}</p>
                  </section>
                ) : null}

                {detail.math ? (
                  <section>
                    <h5>The math</h5>
                    <pre className="detail-math">{detail.math}</pre>
                  </section>
                ) : null}

                {detail.hardware ? (
                  <section>
                    <h5>On the hardware</h5>
                    <p>{detail.hardware}</p>
                  </section>
                ) : null}

                <section>
                  <h5>Configurations it can reach</h5>
                  <p>
                    {typeof detail.reachable === 'number' ? (
                      <>
                        <strong>{detail.reachable.toLocaleString()}</strong>
                        {detail.reachable < 8 ? (
                          <>
                            {' '}states satisfy this program&rsquo;s constraints.
                            That is a very small space, so sampling it
                            demonstrates the constraint rather than the search.
                          </>
                        ) : ' distinct states satisfy its constraints.'}
                      </>
                    ) : typeof detail.reachable === 'string' ? (
                      <span className="stat-unavailable" title={detail.reachable}>
                        too large to enumerate
                      </span>
                    ) : (
                      <span className="stat-unavailable">unavailable</span>
                    )}
                  </p>
                </section>

                {detail.citation ? (
                  <section>
                    <h5>Where it comes from</h5>
                    <p className="detail-citation">{detail.citation}</p>
                  </section>
                ) : null}

                <footer className="picker-actions">
                  <button
                    type="button"
                    className="btn phosphor"
                    disabled={!detail.packaged || detail.status === 'stub'}
                    onClick={() => {
                      onSelect(detail.id)
                      onClose()
                    }}
                  >
                    Open this example
                  </button>
                  <span className="dim mono">{detail.id}</span>
                </footer>
              </>
            ) : (
              <p className="empty-hint">Pick something on the left.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
