import { useMemo, useState } from 'react'
import type { FabricTaxPair, LangMode, ReceiptInspect } from '../../types'
import { label } from '../../lib/glossary'

interface Props {
  receipt: ReceiptInspect | null
  lang: LangMode
}

function edgeKey(a: number, b: number): string {
  return `${Math.min(a, b)}-${Math.max(a, b)}`
}

export function ConnectivityView({ receipt, lang }: Props) {
  const logical = receipt?.connectivity.logical ?? {}
  const physical = receipt?.connectivity.physical ?? {}
  const notes = receipt?.connectivity.notes ?? []
  const fabric = receipt?.fabric_tax
  const hasLogical = Object.keys(logical).length > 0
  const hasPhysical = Object.keys(physical).length > 0
  const [selected, setSelected] = useState<FabricTaxPair | null>(null)
  const [selectedPhysical, setSelectedPhysical] = useState<[number, number] | null>(
    null,
  )

  const physicalEdges = receipt?.edges.pairs ?? []
  const medSet = useMemo(
    () => new Set(receipt?.spins.mediator_idx ?? []),
    [receipt?.spins.mediator_idx],
  )

  const mediatedPairs = fabric?.pairs ?? []

  const onClickMediated = (pair: FabricTaxPair) => {
    setSelected(pair)
    setSelectedPhysical(null)
  }

  const onClickPhysicalEdge = (a: number, b: number) => {
    setSelectedPhysical([a, b])
    // If this is a world–mediator edge, find the mediated pair for that mediator
    const med = medSet.has(a) ? a : medSet.has(b) ? b : null
    if (med != null && fabric) {
      const hit = fabric.pairs.find((p) => p.mediator === med) ?? null
      setSelected(hit)
      return
    }
    // Same-parity logical (derived): both world ends
    if (!medSet.has(a) && !medSet.has(b) && fabric?.by_edge_key) {
      const hit = fabric.by_edge_key[edgeKey(a, b)]
      setSelected(hit ?? null)
      return
    }
    setSelected(null)
  }

  if (!receipt) {
    return (
      <div className="view">
        <div className="stub-panel">
          <h3>Connectivity / Fabric Tax</h3>
          <p className="empty-hint">
            Load a compile receipt to compare logical (pre-mediate) vs physical
            (post-mediate) graph stats. No receipt loaded.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="view connectivity-view">
      <div className="view-caption">
        Fabric Tax — logical vs physical · click a mediated pair or physical edge
        for {label('mediators', lang)} callout
      </div>
      <div className="fabric-grid">
        <div className="fabric-card">
          <h3>Logical (pre-mediate)</h3>
          {hasLogical ? (
            <dl className="kv">
              {Object.entries(logical).map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd className="mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="empty-hint">
              Logical edge list not in receipt. Workload counts unavailable.
            </p>
          )}
        </div>
        <div className="fabric-card">
          <h3>Physical (post-mediate)</h3>
          {hasPhysical ? (
            <dl className="kv">
              {Object.entries(physical).map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd className="mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="empty-hint">Physical metrics missing from receipt.</p>
          )}
        </div>
      </div>

      <div className="fabric-card">
        <h3>
          Mediated logical pairs{' '}
          {fabric?.derived ? <span className="chip derived">derived</span> : null}
        </h3>
        {mediatedPairs.length ? (
          <>
            <p className="dim" style={{ fontSize: '0.75rem', marginTop: 0 }}>
              {fabric?.pair_count ?? mediatedPairs.length} pairs inferred from
              degree-2 {label('mediators', lang)} in program.json (not a stored
              logical edge list). Click a row for callout.
            </p>
            <div className="fabric-pair-list">
              {mediatedPairs.slice(0, 48).map((p) => {
                const active =
                  selected?.mediator === p.mediator &&
                  selected.world_u === p.world_u &&
                  selected.world_v === p.world_v
                return (
                  <button
                    key={`${p.mediator}-${p.world_u}-${p.world_v}`}
                    type="button"
                    className={`fabric-pair ${active ? 'active' : ''}`}
                    onClick={() => onClickMediated(p)}
                  >
                    <span className="mono">
                      {p.world_u}–{p.world_v}
                    </span>
                    <span className="dim">→ med</span>
                    <span className="mono accent">{p.mediator}</span>
                  </button>
                )
              })}
            </div>
            {mediatedPairs.length > 48 ? (
              <p className="empty-hint">
                Showing first 48 of {mediatedPairs.length} pairs.
              </p>
            ) : null}
          </>
        ) : (
          <p className="empty-hint">
            {(fabric?.notes ?? []).join(' ') ||
              'No derived mediator pairs (unmediated or structure unsupported).'}
          </p>
        )}
      </div>

      {physicalEdges.length ? (
        <div className="fabric-card">
          <h3>Physical edges (sample click targets)</h3>
          <p className="dim" style={{ fontSize: '0.75rem', marginTop: 0 }}>
            Click an edge incident to a mediator to surface its mediated world
            pair. Showing a capped sample for UI density.
          </p>
          <div className="fabric-pair-list">
            {physicalEdges.slice(0, 64).map(([a, b]) => {
              const active =
                selectedPhysical != null &&
                ((selectedPhysical[0] === a && selectedPhysical[1] === b) ||
                  (selectedPhysical[0] === b && selectedPhysical[1] === a))
              const touchesMed = medSet.has(a) || medSet.has(b)
              return (
                <button
                  key={`${a}-${b}`}
                  type="button"
                  className={`fabric-pair ${active ? 'active' : ''} ${
                    touchesMed ? 'touches-med' : ''
                  }`}
                  onClick={() => onClickPhysicalEdge(a, b)}
                >
                  <span className="mono">
                    {a}–{b}
                  </span>
                  {touchesMed ? <span className="chip">via med</span> : null}
                </button>
              )
            })}
          </div>
        </div>
      ) : null}

      <div className={`fabric-card callout ${selected ? 'lit' : ''}`}>
        <h3>Mediator callout</h3>
        {selected ? (
          <dl className="kv">
            <div>
              <dt>world u</dt>
              <dd className="mono">
                {selected.world_u}
                {selected.world_u_name ? ` (${selected.world_u_name})` : ''}
              </dd>
            </div>
            <div>
              <dt>world v</dt>
              <dd className="mono">
                {selected.world_v}
                {selected.world_v_name ? ` (${selected.world_v_name})` : ''}
              </dd>
            </div>
            <div>
              <dt>{label('mediators', lang).replace(/s$/, '')}</dt>
              <dd className="mono accent">
                {selected.mediator}
                {selected.mediator_name ? ` (${selected.mediator_name})` : ''}
              </dd>
            </div>
            <div>
              <dt>source</dt>
              <dd className="mono">derived · program.json edges</dd>
            </div>
          </dl>
        ) : (
          <p className="empty-hint">
            Select a mediated pair or a physical edge that touches a mediator.
            {selectedPhysical
              ? ` Selected physical edge ${selectedPhysical[0]}–${selectedPhysical[1]} has no mediator callout.`
              : ''}
          </p>
        )}
      </div>

      <div className="fabric-card">
        <h3>Mediation (passes.json)</h3>
        {receipt.mediation ? (
          <dl className="kv">
            {Object.entries(receipt.mediation).map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd className="mono">{String(v)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="empty-hint">
            No mediation block in passes.json (bipartite / unmediated or absent).
          </p>
        )}
        {notes.map((n) => (
          <p key={n} className="empty-hint">
            {n}
          </p>
        ))}
        {(fabric?.notes ?? []).map((n) => (
          <p key={n} className="empty-hint">
            {n}
          </p>
        ))}
      </div>
    </div>
  )
}
