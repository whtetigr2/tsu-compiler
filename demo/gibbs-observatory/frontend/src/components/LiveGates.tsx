import { useEffect, useRef, useState } from 'react'
import type { GateRow } from '../types'

/**
 * The gates, updating as you type.
 *
 * Compiling has two costs and they are nothing alike. Checking the gates takes
 * a millisecond or two; finding an embedding takes seconds. Measured warm on a
 * 6x6 grid, 1ms against 1401ms, and on ecology_lotka_lite 2ms against 7198ms.
 * So the cheap half runs on every pause and the expensive half runs when asked.
 *
 * What this strip must never do is look like a compile. It reports the model
 * AS WRITTEN: routing has not inserted mediators or split high-degree nodes,
 * and placement has not run at all, so a green strip says the model is within
 * the caps and says nothing whatsoever about whether it can be laid out. The
 * caveat is on screen rather than in a tooltip, because a reader who takes
 * green for COMPILED has been misled by the interface, not by their own
 * carelessness.
 */

export interface GatePreview {
  ok: boolean
  gates: GateRow[]
  n_spins?: number
  n_couplings?: number
  max_degree?: number
  bipartite?: boolean | null
  colour_blocks?: number | null
  placement_checked: boolean
  routed: boolean
  elapsed_seconds?: number
  note?: string
}

type State =
  | { kind: 'idle' }
  | { kind: 'checking' }
  | { kind: 'ready'; preview: GatePreview }
  | { kind: 'unreadable'; message: string }

/** How long the editor must be still before the gates are re-checked. */
const QUIET_MS = 500

export function LiveGates({ yaml, target = 'z1' }: { yaml: string; target?: string }) {
  const [state, setState] = useState<State>({ kind: 'idle' })
  // Every request carries a sequence number so a slow early reply cannot
  // overwrite a fast later one and leave the strip describing text that is no
  // longer in the editor.
  const seq = useRef(0)

  useEffect(() => {
    if (!yaml.trim()) {
      setState({ kind: 'idle' })
      return
    }
    const mine = ++seq.current
    const timer = setTimeout(async () => {
      setState({ kind: 'checking' })
      try {
        const res = await fetch('/api/program/gates', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ yaml, target, allow_assumed: false }),
        })
        const data = await res.json().catch(() => ({}))
        if (mine !== seq.current) return
        if (!res.ok) {
          const d = data?.detail
          const raw = typeof d === 'string' ? d : d?.message ?? ''
          setState({ kind: 'unreadable', message: plainly(raw) })
          return
        }
        setState({ kind: 'ready', preview: data as GatePreview })
      } catch (e) {
        if (mine === seq.current) {
          setState({ kind: 'unreadable', message: String(e) })
        }
      }
    }, QUIET_MS)
    return () => clearTimeout(timer)
  }, [yaml, target])

  if (state.kind === 'idle') return null

  if (state.kind === 'checking') {
    return (
      <div className="live-gates checking">
        <span className="live-dot pulse" />
        <span className="live-label">checking gates</span>
      </div>
    )
  }

  if (state.kind === 'unreadable') {
    // Half-typed YAML is not a failing program, and calling it one would make
    // the strip flash red at every keystroke.
    return (
      <div className="live-gates unreadable">
        <span className="live-dot" />
        <span className="live-label">not readable yet</span>
        <span className="live-detail">{state.message}</span>
      </div>
    )
  }

  const { preview } = state
  const failed = preview.gates.filter((g) => g.status === 'fail' || g.passed === false)

  return (
    <div className={`live-gates ${failed.length ? 'bad' : 'ok'}`}>
      <span className="live-dot" />
      <span className="live-label">
        {failed.length === 0
          ? `${preview.gates.length} gates pass`
          : `${failed.length} of ${preview.gates.length} gates fail`}
      </span>

      <span className="live-chips">
        {failed.map((g) => (
          <span className="live-chip bad" key={String(g.gate ?? g.name)}>
            {String(g.gate ?? g.name)} {fmt(g.measured ?? g.value)} / {fmt(g.limit)}
          </span>
        ))}
        {failed.length === 0 ? (
          <>
            <span className="live-chip">{preview.n_spins?.toLocaleString()} spins</span>
            <span className="live-chip">{preview.n_couplings?.toLocaleString()} couplings</span>
            <span className="live-chip">degree {preview.max_degree}</span>
          </>
        ) : null}
      </span>

      <span className="live-caveat">
        model as written &middot; not routed &middot; not placed
      </span>
    </div>
  )
}

/**
 * The useful half of a loader error.
 *
 * `spec load failed: ValueError: unknown generator 'gr'` tells the reader two
 * things about our plumbing and one thing about their program. Keep the third.
 */
function plainly(message: string): string {
  const trimmed = message
    .replace(/^spec load failed:\s*/i, '')
    .replace(/^[A-Za-z]*(Error|Exception):\s*/, '')
    .trim()
  return trimmed || 'not readable yet'
}

function fmt(v: unknown): string {
  if (typeof v === 'number') return v.toLocaleString()
  if (typeof v === 'string') return v
  return 'unavailable'
}
