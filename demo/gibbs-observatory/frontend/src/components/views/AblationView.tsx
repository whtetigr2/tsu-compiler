import { useState } from 'react'
import type { ReceiptInspect } from '../../types'

interface AblationResult {
  ok: boolean
  receipt_id: string
  target: string
  n_samples: number
  baseline_mean: number
  ablated_mean: number
  noise_floor: number
  shift: number
  ratio: number
  threshold: number
  verdict: 'load-bearing' | 'not measurable'
  explanation: string
}

interface Props {
  receipt: ReceiptInspect | null
}

const TARGETS: { id: string; label: string; blurb: string }[] = [
  {
    id: 'couplings',
    label: 'Couplings',
    blurb: 'Set every coupling between spins to zero, leaving the biases alone.',
  },
  {
    id: 'biases',
    label: 'Biases',
    blurb: 'Set every per-spin bias to zero, leaving the couplings alone.',
  },
  {
    id: 'none',
    label: 'Nothing (control)',
    blurb:
      'Change nothing at all. This must come out below the threshold. If it does not, the noise floor is wrong and no other result here means anything.',
  },
]

/**
 * The control this project is built around.
 *
 * It tries to prove the loaded model's own terms are doing nothing. The
 * comparison is against a noise floor, the difference between two runs of the
 * same model under different seeds, never against zero: sampling is stochastic
 * and no two runs agree exactly, so comparing to zero would call every term
 * load-bearing.
 */
export function AblationView({ receipt }: Props) {
  const [running, setRunning] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, AblationResult>>({})
  const [error, setError] = useState<string | null>(null)

  const run = async (target: string) => {
    if (!receipt?.id) return
    setRunning(target)
    setError(null)
    try {
      const res = await fetch('/api/ablation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          receipt_id: receipt.id,
          target,
          n_samples: 128,
        }),
      })
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
      const data: AblationResult = await res.json()
      setResults((r) => ({ ...r, [target]: data }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'ablation failed')
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="view ablation-view">
      <p className="view-lead">
        This tries to prove the loaded model&rsquo;s own terms are doing nothing.
        It zeroes a term, samples again, and compares the change against how much
        two identical runs differ by chance. If the change is not clearly bigger
        than that, the term is not doing the work.
      </p>

      {!receipt ? (
        <p className="empty-hint">Open an example first.</p>
      ) : (
        <>
          <div className="ablation-targets">
            {TARGETS.map((t) => {
              const r = results[t.id]
              return (
                <div key={t.id} className="ablation-card">
                  <div className="ablation-card-head">
                    <h4>{t.label}</h4>
                    <button
                      type="button"
                      className="btn tiny phosphor"
                      disabled={running !== null}
                      onClick={() => void run(t.id)}
                    >
                      {running === t.id ? 'sampling...' : 'Run'}
                    </button>
                  </div>
                  <p className="ablation-blurb">{t.blurb}</p>

                  {r ? (
                    <>
                      <div className={`ablation-verdict ${r.verdict === 'load-bearing' ? 'ok' : 'warn'}`}>
                        {r.verdict}
                      </div>
                      <RatioBar ratio={r.ratio} threshold={r.threshold} />
                      <dl className="ablation-numbers">
                        <div>
                          <dt>shift</dt>
                          <dd>{r.shift.toFixed(4)}</dd>
                        </div>
                        <div>
                          <dt>noise floor</dt>
                          <dd>{r.noise_floor.toFixed(4)}</dd>
                        </div>
                        <div>
                          <dt>ratio</dt>
                          <dd>{r.ratio.toFixed(1)}x</dd>
                        </div>
                      </dl>
                      <p className="ablation-explanation">{r.explanation}</p>
                    </>
                  ) : (
                    <p className="empty-hint">Not run yet.</p>
                  )}
                </div>
              )
            })}
          </div>

          {error ? <p className="ablation-error">{error}</p> : null}

          <p className="view-footnote">
            The threshold is {TARGETS.length ? '3.0x' : ''} the noise floor, the
            same figure audit/game_on_thrml.py uses, so a verdict here and a
            verdict there mean the same thing. Simulation on CPU through THRML,
            not Extropic silicon.
          </p>
        </>
      )}
    </div>
  )
}

/** The ratio against the threshold, drawn, so "how close was it" is visible. */
function RatioBar({ ratio, threshold }: { ratio: number; threshold: number }) {
  const span = Math.max(threshold * 2, ratio * 1.1, 1)
  const pct = Math.min(100, (ratio / span) * 100)
  const markPct = (threshold / span) * 100
  return (
    <div className="ratio-bar" title={`${ratio.toFixed(1)}x the noise floor`}>
      <div
        className={`ratio-fill ${ratio >= threshold ? 'ok' : 'warn'}`}
        style={{ width: `${pct}%` }}
      />
      <div className="ratio-threshold" style={{ left: `${markPct}%` }}>
        <span>{threshold.toFixed(0)}x</span>
      </div>
    </div>
  )
}
