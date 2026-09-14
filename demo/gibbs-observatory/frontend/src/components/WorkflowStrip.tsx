import type { NavView } from '../types'

export type WorkflowStepId = 'receipt' | 'inspect' | 'sample' | 'compare'

interface Props {
  active: WorkflowStepId
  onSelect: (id: WorkflowStepId) => void
  receiptLabel: string | null
}

const STEPS: { id: WorkflowStepId; n: number; label: string }[] = [
  { id: 'receipt', n: 1, label: 'Receipt' },
  { id: 'inspect', n: 2, label: 'Inspect' },
  { id: 'sample', n: 3, label: 'Sample' },
  { id: 'compare', n: 4, label: 'Compare' },
]

/** Map observatory views → workflow step for highlight */
export function workflowStepForView(view: NavView, running: boolean): WorkflowStepId {
  if (running) return 'sample'
  if (view === 'residuals' || view === 'scope' || view === 'statespace') return 'compare'
  if (view === 'notepad' || view === 'worlds') return 'receipt'
  return 'inspect'
}

export function WorkflowStrip({ active, onSelect, receiptLabel }: Props) {
  return (
    <div className="workflow-strip" data-tour="workflow-strip">
      <div className="workflow-copy">
        <h1>Inspect a compiled thermodynamic sampling program.</h1>
        <p>
          Load a receipt, read the chromatic schedule and gates, then stream
          live THRML block-Gibbs draws
          {receiptLabel ? ` · ${receiptLabel}` : ''}.
        </p>
      </div>
      <div className="workflow-stepper" role="tablist" aria-label="Workflow">
        {STEPS.map((s) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={active === s.id}
            className={`wf-step ${active === s.id ? 'active' : ''}`}
            data-tour={`wf-${s.id}`}
            onClick={() => onSelect(s.id)}
          >
            {s.n} · {s.label}
          </button>
        ))}
      </div>
    </div>
  )
}
