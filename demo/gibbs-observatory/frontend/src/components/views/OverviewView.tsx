import type { BatchPayload, GraphPayload, LangMode, ReceiptInspect } from '../../types'
import { label } from '../../lib/glossary'
import { SpinField } from '../SpinField'

interface Props {
  graph: GraphPayload | null
  batch: BatchPayload | null
  receipt: ReceiptInspect | null
  lang: LangMode
  selectedSpin: number | null
  onSelectSpin: (i: number | null) => void
}

export function OverviewView({
  graph,
  batch,
  receipt,
  lang,
  selectedSpin,
  onSelectSpin,
}: Props) {
  const fabric = receipt?.fabric_tax
  const showFabric =
    Boolean(fabric) &&
    ((fabric?.pair_count ?? 0) > 0 ||
      Object.keys(receipt?.connectivity?.logical ?? {}).length > 0 ||
      Object.keys(receipt?.connectivity?.physical ?? {}).length > 0)

  const nColors =
    (graph?.color0?.length ? 1 : 0) + (graph?.color1?.length ? 1 : 0) || 2

  return (
    <div className="view overview-view">
      {showFabric ? (
        <div className="fabric-callout" role="note">
          <span className="fabric-accent">Fabric Tax</span>
          <span className="mono">
            {fabric && fabric.pair_count > 0
              ? `${fabric.pair_count} mediated pairs (derived)`
              : 'logical vs physical connectivity available'}
          </span>
          <span className="dim">
          , open Connectivity for {label('mediators', lang)} callouts
          </span>
        </div>
      ) : null}

      <div className="overview-grid">
        <div className="canvas-wrap">
          <SpinField
            graph={graph}
            state={batch?.last_state ?? null}
            activeBlock={batch?.active_block ?? 0}
            height={460}
            showEdges
            selectedSpin={selectedSpin}
            onSelectSpin={onSelectSpin}
            graphAesthetic
          />
        </div>
        {/* Fallback mission strip for narrow layouts (CSS shows on ≤1100px) */}
        <div className="overview-side">
          <div className="mission-strip">
            <div>
              <span className="k">spins</span>
              <span className="v mono">
                {receipt?.spins.n_nodes ?? graph?.n_nodes ?? ', '}
                {receipt?.spins.mediators != null
                  ? ` (${receipt.spins.world} world + ${receipt.spins.mediators} ${label('mediators', lang)})`
                  : ''}
              </span>
            </div>
            <div>
              <span className="k">edges</span>
              <span className="v mono">
                {receipt?.edges.count ?? graph?.edges.length ?? ', '}
              </span>
            </div>
            <div>
              <span className="k">{label('beta', lang)}</span>
              <span className="v mono">
                {receipt?.beta ?? graph?.params.beta ?? ', '}
                {receipt?.beta_fixed ? ' FIXED' : ''}
              </span>
            </div>
            <div>
              <span className="k">schedule</span>
              <span className="v mono">{nColors}-color chromatic</span>
            </div>
          </div>
          <div className="gate-lights">
            {(receipt?.gates ?? []).map((g) => (
              <div key={g.gate} className={`traffic ${g.passed ? 'pass' : 'fail'}`}>
                <span className="light" />
                <span>{g.gate}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
