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

export function SpinsView({
  graph,
  batch,
  lang,
  selectedSpin = null,
  onSelectSpin,
}: Props) {
  const med = graph?.mediator_idx?.length ?? 0
  return (
    <div className="view">
      <div className="view-caption">
        Physical lattice · world vs {label('mediators', lang)}
        {med
          ? ` · Fabric Tax: ${med} ${label('mediators', lang)} (auxiliary spins for bipartite Z1 fabric)`
          : ''}{' '}
        · chromatic pulse = active {label('blocks', lang)}
      </div>
      <div className="canvas-wrap">
        <SpinField
          graph={graph}
          state={batch?.last_state ?? null}
          activeBlock={batch?.active_block ?? 0}
          height={520}
          selectedSpin={selectedSpin}
          onSelectSpin={onSelectSpin}
        />
      </div>
      <div className="legend">
        <span>
          <span className="swatch" style={{ background: '#e8f0ef' }} /> world +1
        </span>
        <span>
          <span className="swatch" style={{ background: '#14181c' }} /> world −1
        </span>
        <span>
          <span className="swatch" style={{ background: '#7ec8c0' }} />{' '}
          {med ? 'Fabric Tax mediator' : 'mediator'}
        </span>
        <span>
          <span className="swatch" style={{ background: '#6eb8d4' }} /> block 0
        </span>
        <span>
          <span className="swatch" style={{ background: '#d47aa8' }} /> block 1
        </span>
      </div>
    </div>
  )
}
