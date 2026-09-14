import type { BatchPayload, GraphPayload, LangMode } from '../../types'
import { label } from '../../lib/glossary'
import { SpinField } from '../SpinField'

interface Props {
  graph: GraphPayload | null
  batch: BatchPayload | null
  lang: LangMode
  selectedSpin?: number | null
  onSelectSpin?: (i: number | null) => void
}

export function CouplingsView({
  graph,
  batch,
  lang,
  selectedSpin = null,
  onSelectSpin,
}: Props) {
  const weights = graph?.edge_weights
  const hasWeights = !!(weights && weights.length)
  const med = graph?.mediator_idx?.length ?? 0
  return (
    <div className="view">
      <div className="view-caption">
        Edges over lattice · stroke ∝ |{label('J', lang)}| · green ferro (J≥0) · red
        antiferro (J&lt;0)
        {!hasWeights ? ' · uniform |J| (preset)' : ''}
        {med
          ? ` · Fabric Tax: ${med} ${label('mediators', lang)} live with world (dashed = world–mediator)`
          : ''}
      </div>
      <div className="canvas-wrap">
        <SpinField
          graph={graph}
          state={batch?.last_state ?? null}
          activeBlock={batch?.active_block ?? 0}
          edgeMode="coupling"
          height={520}
          selectedSpin={selectedSpin}
          onSelectSpin={onSelectSpin}
        />
      </div>
      <div className="legend">
        <span>
          <span className="swatch" style={{ background: 'rgba(62,224,176,0.8)' }} /> ferro
        </span>
        <span>
          <span className="swatch" style={{ background: 'rgba(255,107,107,0.8)' }} />{' '}
          antiferro
        </span>
        {med > 0 && (
          <span>
            <span className="swatch" style={{ background: '#7ec8c0' }} /> Fabric Tax
            mediator
          </span>
        )}
      </div>
    </div>
  )
}
