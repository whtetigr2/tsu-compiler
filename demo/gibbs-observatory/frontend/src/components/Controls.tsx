import type { PresetId, SimParams } from '../types'

interface Props {
  params: SimParams
  setParams: (p: SimParams) => void
  onReset: () => void
  onRun: () => void
  onPause: () => void
  onStep: () => void
  running: boolean
  connected: boolean
}

export function Controls({
  params,
  setParams,
  onReset,
  onRun,
  onPause,
  onStep,
  running,
  connected,
}: Props) {
  const patch = <K extends keyof SimParams>(key: K, value: SimParams[K]) => {
    setParams({ ...params, [key]: value })
  }

  const sizeLabel =
    params.preset === 'lattice2d'
      ? `L=${params.size} (${params.size * params.size} spins)`
      : `${params.size} nodes`

  return (
    <aside className="panel">
      <h2>Controls</h2>

      <div className="section">
        <div className="control-row">
          <label>Preset</label>
        </div>
        <select
          value={params.preset}
          onChange={(e) =>
            patch('preset', e.target.value as PresetId)
          }
        >
          <option value="lattice2d">2D Ising lattice</option>
          <option value="chain1d">1D Ising chain</option>
          <option value="sparse">Sparse (deg-capped)</option>
        </select>
      </div>

      <div className="section">
        <div className="control-row">
          <label>Size</label>
          <span className="val">{sizeLabel}</span>
        </div>
        <input
          type="range"
          min={params.preset === 'lattice2d' ? 8 : 8}
          max={params.preset === 'lattice2d' ? 32 : 64}
          step={params.preset === 'lattice2d' ? 2 : 1}
          value={params.size}
          onChange={(e) => patch('size', Number(e.target.value))}
        />

        {params.preset === 'sparse' && (
          <>
            <div className="control-row">
              <label>Degree cap</label>
              <span className="val">{params.degree_cap}</span>
            </div>
            <input
              type="range"
              min={2}
              max={16}
              value={params.degree_cap}
              onChange={(e) => patch('degree_cap', Number(e.target.value))}
            />
          </>
        )}
      </div>

      <div className="section">
        <div className="control-row">
          <label>β (beta)</label>
          <span className="val">{params.beta.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0.05}
          max={2.5}
          step={0.01}
          value={params.beta}
          onChange={(e) => patch('beta', Number(e.target.value))}
        />

        <div className="control-row">
          <label>J (coupling)</label>
          <span className="val">{params.J.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={-2}
          max={2}
          step={0.05}
          value={params.J}
          onChange={(e) => patch('J', Number(e.target.value))}
        />

        <div className="control-row">
          <label>h (field)</label>
          <span className="val">{params.h.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={-1.5}
          max={1.5}
          step={0.05}
          value={params.h}
          onChange={(e) => patch('h', Number(e.target.value))}
        />
      </div>

      <div className="section">
        <div className="control-row">
          <label>Warmup</label>
          <span className="val">{params.warmup}</span>
        </div>
        <input
          type="range"
          min={0}
          max={400}
          step={5}
          value={params.warmup}
          onChange={(e) => patch('warmup', Number(e.target.value))}
        />

        <div className="control-row">
          <label>Steps / sample</label>
          <span className="val">{params.steps_per_sample}</span>
        </div>
        <input
          type="range"
          min={1}
          max={8}
          value={params.steps_per_sample}
          onChange={(e) => patch('steps_per_sample', Number(e.target.value))}
        />

        <div className="control-row">
          <label>Batch size</label>
          <span className="val">{params.batch_size}</span>
        </div>
        <input
          type="range"
          min={1}
          max={32}
          value={params.batch_size}
          onChange={(e) => patch('batch_size', Number(e.target.value))}
        />

        <div className="control-row">
          <label>Seed</label>
          <span className="val">{params.seed}</span>
        </div>
        <input
          type="number"
          value={params.seed}
          onChange={(e) => patch('seed', Number(e.target.value) || 0)}
        />
      </div>

      <div className="section">
        <label className="toggle">
          <input
            type="checkbox"
            checked={params.clamp}
            onChange={(e) => patch('clamp', e.target.checked)}
          />
          Clamp border / region (+1)
        </label>
      </div>

      <div className="section btn-row">
        <button className="btn" onClick={onReset} disabled={!connected}>
          Apply / Reset
        </button>
        {!running ? (
          <button className="btn primary" onClick={onRun} disabled={!connected}>
            Run
          </button>
        ) : (
          <button className="btn danger" onClick={onPause}>
            Pause
          </button>
        )}
        <button className="btn" onClick={onStep} disabled={!connected || running}>
          Step
        </button>
      </div>

      <p className="status-line">
        Structure is static; β / J / h update the THRML IsingEBM params.
        Clamp toggles free vs clamped blocks.
      </p>
    </aside>
  )
}
