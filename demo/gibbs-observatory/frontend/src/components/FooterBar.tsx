interface Props {
  running: boolean
  connected: boolean
  seed: number
  setSeed: (n: number) => void
  onRun: () => void
  onPause: () => void
  onStep: () => void
  onReset: () => void
  onSnapshot: () => void
  snapshotBusy?: boolean
  speed: number
  setSpeed: (n: number) => void
}

export function FooterBar({
  running,
  connected,
  seed,
  setSeed,
  onRun,
  onPause,
  onStep,
  onReset,
  onSnapshot,
  snapshotBusy = false,
  speed,
  setSpeed,
}: Props) {
  return (
    <footer className="footer-bar" data-tour="footer-bar">
      <div className="btn-row">
        {!running ? (
          <button className="btn primary" type="button" onClick={onRun} disabled={!connected}>
            Run
          </button>
        ) : (
          <button className="btn danger" type="button" onClick={onPause}>
            Pause
          </button>
        )}
        <button className="btn" type="button" onClick={onStep} disabled={!connected || running}>
          Step
        </button>
        <button className="btn" type="button" onClick={onReset} disabled={!connected}>
          Reset
        </button>
        <button
          className="btn ghost"
          type="button"
          onClick={onSnapshot}
          disabled={snapshotBusy}
          title="Export PNG of stage + JSON receipt slice"
        >
          {snapshotBusy ? 'Saving…' : 'Save snapshot'}
        </button>
      </div>
      <label className="footer-seed">
        Seed
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value) || 0)}
        />
      </label>
      <label className="footer-speed">
        Speed
        <input
          type="range"
          min={1}
          max={8}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        />
        <span className="mono">{speed}</span>
      </label>
      <div className="footer-meta mono">
        <span>CPU / THRML</span>
        <span className="warn">silicon Unavailable</span>
      </div>
    </footer>
  )
}
