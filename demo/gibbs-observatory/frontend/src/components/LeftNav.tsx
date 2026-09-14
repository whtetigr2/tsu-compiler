import type { NavView, ReceiptInspect, SimParams } from '../types'

const ITEMS: { id: NavView; label: string; phase?: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'alloylab', label: 'Alloy Lab' },
  { id: 'ebmlab', label: 'EBM Lab' },
  { id: 'spins', label: 'Spins' },
  { id: 'couplings', label: 'Couplings' },
  { id: 'connectivity', label: 'Connectivity' },
  { id: 'schedule', label: 'Schedule' },
  { id: 'statespace', label: 'State space' },
  { id: 'residuals', label: 'Residuals' },
  { id: 'scope', label: 'Scope' },
  { id: 'notepad', label: 'Program notepad' },
  { id: 'worlds', label: 'Worlds', phase: 'drawer' },
]

interface Props {
  active: NavView
  onSelect: (v: NavView) => void
  receipt: ReceiptInspect | null
  params: SimParams
  setParams: (p: SimParams | ((prev: SimParams) => SimParams)) => void
  running: boolean
  connected: boolean
  onOpenReceipts: () => void
  onRun: () => void
  onPause: () => void
  onStep: () => void
  onReset: () => void
  speed: number
  setSpeed: (n: number) => void
}

export function LeftNav({
  active,
  onSelect,
  receipt,
  params,
  setParams,
  running,
  connected,
  onOpenReceipts,
  onRun,
  onPause,
  onStep,
  onReset,
  speed,
  setSpeed,
}: Props) {
  const betaFixed = Boolean(receipt?.beta_fixed)
  const beta = receipt?.beta ?? params.beta

  return (
    <nav className="left-rail" aria-label="Experiment and inspect">
      <div className="rail-section" data-tour="experiment-rail">
        <h2 className="rail-label">Experiment</h2>

        <div className="exp-field">
          <div className="field-head">
            <label>Receipt</label>
            {receipt?.verdict ? (
              <span className="verdict-chip">{receipt.verdict}</span>
            ) : null}
          </div>
          <button
            type="button"
            className="exp-select"
            onClick={onOpenReceipts}
            title={receipt?.path ?? 'Open receipt'}
          >
            {receipt ? `receipts/${receipt.id}` : 'Choose receipt…'}
          </button>
        </div>

        <div className="exp-field">
          <div className="field-head">
            <label>Inverse temperature β</label>
            <span className="field-val">
              {beta != null ? Number(beta).toFixed(2) : '—'}
              {betaFixed ? <span className="chip fixed">FIXED</span> : null}
            </span>
          </div>
          <input
            type="range"
            min={0.05}
            max={2.5}
            step={0.01}
            value={typeof beta === 'number' ? beta : 1}
            disabled={betaFixed || !connected}
            onChange={(e) =>
              setParams((p) => ({ ...p, beta: Number(e.target.value) }))
            }
            aria-label="Beta"
          />
          {betaFixed ? (
            <p className="empty-hint" style={{ margin: 0 }}>
              Mediated receipt — β locked to compile value.
            </p>
          ) : null}
        </div>

        <div className="exp-field">
          <div className="field-head">
            <label>Steps / sample</label>
            <span className="field-val mono">{speed}</span>
          </div>
          <input
            type="range"
            min={1}
            max={8}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            aria-label="Speed"
          />
        </div>

        <div className="exp-actions">
          {!running ? (
            <button
              className="btn block phosphor"
              type="button"
              onClick={onRun}
              disabled={!connected}
            >
              Run THRML sampler
            </button>
          ) : (
            <button className="btn block danger" type="button" onClick={onPause}>
              Pause sampler
            </button>
          )}
          <div className="btn-row">
            <button
              className="btn"
              type="button"
              onClick={onStep}
              disabled={!connected || running}
            >
              Step
            </button>
            <button
              className="btn"
              type="button"
              onClick={onReset}
              disabled={!connected}
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="rail-section">
        <h2 className="rail-label">Inspect</h2>
        <div className="inspect-nav">
          {ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${active === item.id ? 'active' : ''}`}
              data-tour={`nav-${item.id}`}
              onClick={() => onSelect(item.id)}
            >
              <span>{item.label}</span>
              {item.phase ? <span className="phase">{item.phase}</span> : null}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section">
        <div className="device-card" data-tour="honest-label">
          <div className="device-name">Device</div>
          <div className="device-status">CPU / THRML · silicon Unavailable</div>
          <p className="device-note">
            JAX + THRML simulation only. No Extropic silicon, no joule claims.
          </p>
        </div>
      </div>
    </nav>
  )
}
