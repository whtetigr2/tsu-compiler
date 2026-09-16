import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import type { NavView } from '../types'

export const WALKTHROUGH_STORAGE_KEY = 'gibbs-observatory-walkthrough-v1'

export interface WalkthroughStep {
  id: string
  title: string
  body: string
  /** CSS selector for spotlight target (prefer [data-tour="…"]) */
  target?: string
  /** Optional left-rail view to open for this step */
  view?: NavView
}

export const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    id: 'mission',
    title: 'Mission',
    body:
      'This inspects a compiled thermodynamic sampling program. Open an example, read its gates, spins, couplings and connectivity, then watch THRML sample it live. Everything on screen is computed on your CPU, not on Extropic hardware.',
    target: '[data-tour="brand"]',
    view: 'overview',
  },
  {
    id: 'program-bar',
    title: 'Workflow + program meta',
    body:
      'The workflow strip (Receipt → Inspect → Sample → Compare) and program meta show the loaded receipt path, encoding, β (FIXED when mediated), kernel, and compile verdict. COMPILED means the receipt passed gates and is ready to sample, still software / THRML, not silicon.',
    target: '[data-tour="program-bar"]',
    view: 'overview',
  },
  {
    id: 'receipts',
    title: 'Open a receipt',
    body:
      'Menu → File → Open example loads the shelf. Start with "Starter grid", which is the quickest. Three of the examples are marked VERIFIED: those are problems Extropic published, and a standing test compiles each one, so the badge means somebody can check it rather than that we typed it.',
    target: '[data-tour="menu-burger"]',
  },
  {
    id: 'overview',
    title: 'Overview · hero stage',
    body:
      'Overview is the hero stage: compiled sampling program graph with chromatic schedule subtitle. Fabric Tax accents appear when connectivity data exists. Experiment controls sit in the left rail.',
    target: '[data-tour="nav-overview"]',
    view: 'overview',
  },
  {
    id: 'spins',
    title: 'Spins',
    body:
      'Spins shows the live ±1 field with chromatic V1/V2 block pulse while streaming. World spins and mediator spins share the lattice layout from the receipt.',
    target: '[data-tour="nav-spins"]',
    view: 'spins',
  },
  {
    id: 'couplings',
    title: 'Couplings',
    body:
      'Couplings visualizes edge weights / biases from the compiled Ising program. Same numbers whether you label them Sci (J, b) or Prog (coupling, bias).',
    target: '[data-tour="nav-couplings"]',
    view: 'couplings',
  },
  {
    id: 'connectivity',
    title: 'Connectivity · Fabric Tax',
    body:
      'Connectivity contrasts logical vs physical graphs. Click a mediated pair or physical edge for a mediator callout, Fabric Tax is derived from program.json, not a separate energy claim.',
    target: '[data-tour="nav-connectivity"]',
    view: 'connectivity',
  },
  {
    id: 'schedule',
    title: 'Schedule',
    body:
      'Schedule shows chromatic block-Gibbs colouring (independent sets updated together). Active-block cues track step parity; they are not a per-sweep THRML observer.',
    target: '[data-tour="nav-schedule"]',
    view: 'schedule',
  },
  {
    id: 'transport',
    title: 'Run · Pause · Step',
    body:
      'Primary Run THRML sampler lives in the Experiment rail; footer mirrors Run / Step / Reset plus seed and speed. CPU / THRML · silicon Unavailable stays honest, simulation only.',
    target: '[data-tour="experiment-rail"]',
  },
  {
    id: 'lang',
    title: 'Sci | Prog',
    body:
      'Toggle Sci vs Prog labels in the top bar. Numbers stay identical, only the glossary wording changes (β ↔ temperature scale, J ↔ coupling, etc.).',
    target: '[data-tour="lang-toggle"]',
  },
  {
    id: 'notepad',
    title: 'Thermodynamic Program notepad',
    body:
      'Program notepad edits YAML, runs Preflight / ideal-first Compile via an optional sibling tsu package, then Apply only when a COMPILED receipt exists on disk. Revert restores the previous receipt.',
    target: '[data-tour="nav-notepad"]',
    view: 'notepad',
  },
  {
    id: 'hygiene',
    title: 'Claim hygiene',
    body:
      'Standing prohibitions: no silicon claims, no joule / energy product claims, mediated β stays FIXED, ESS only when the receipt contract allows. Device card and collapsed Claim hygiene on the right rail keep badges honest.',
    target: '[data-tour="honest-label"]',
    view: 'overview',
  },
  {
    id: 'snapshot',
    title: 'Snapshot',
    body:
      'File → Save snapshot, or the footer, exports a PNG of the stage plus a JSON slice of the receipt. These are simulation traces. Restart this tour any time from Help → Start walkthrough.',
    target: '[data-tour="footer-bar"]',
    view: 'overview',
  },
]

interface Rect {
  top: number
  left: number
  width: number
  height: number
}

interface Props {
  open: boolean
  onClose: (reason: 'finish' | 'skip' | 'esc') => void
  onNavigate?: (view: NavView) => void
}

function markSeen() {
  try {
    localStorage.setItem(WALKTHROUGH_STORAGE_KEY, '1')
  } catch {
    /* private mode / blocked storage */
  }
}

export function hasSeenWalkthrough(): boolean {
  try {
    return localStorage.getItem(WALKTHROUGH_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

export function Walkthrough({ open, onClose, onNavigate }: Props) {
  const [index, setIndex] = useState(0)
  const [hole, setHole] = useState<Rect | null>(null)
  const navigateRef = useRef(onNavigate)
  navigateRef.current = onNavigate

  const step = WALKTHROUGH_STEPS[index]
  const total = WALKTHROUGH_STEPS.length
  const isFirst = index === 0
  const isLast = index === total - 1

  const measure = useCallback(() => {
    if (!step?.target) {
      setHole(null)
      return
    }
    const el = document.querySelector(step.target)
    if (!el) {
      setHole(null)
      return
    }
    const r = el.getBoundingClientRect()
    const pad = 8
    setHole({
      top: Math.max(0, r.top - pad),
      left: Math.max(0, r.left - pad),
      width: Math.min(window.innerWidth, r.width + pad * 2),
      height: Math.min(window.innerHeight, r.height + pad * 2),
    })
  }, [step])

  useEffect(() => {
    if (!open) return
    setIndex(0)
  }, [open])

  useEffect(() => {
    if (!open || !step) return
    if (step.view && navigateRef.current) navigateRef.current(step.view)
    const t = window.setTimeout(measure, 40)
    return () => window.clearTimeout(t)
  }, [open, step, measure])

  useLayoutEffect(() => {
    if (!open) return
    measure()
    const onResize = () => measure()
    window.addEventListener('resize', onResize)
    window.addEventListener('scroll', onResize, true)
    return () => {
      window.removeEventListener('resize', onResize)
      window.removeEventListener('scroll', onResize, true)
    }
  }, [open, measure, index])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        markSeen()
        onClose('esc')
      } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
        e.preventDefault()
        if (isLast) {
          markSeen()
          onClose('finish')
        } else {
          setIndex((i) => Math.min(total - 1, i + 1))
        }
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setIndex((i) => Math.max(0, i - 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, isLast, total, onClose])

  if (!open || !step) return null

  const finish = (reason: 'finish' | 'skip' | 'esc') => {
    markSeen()
    onClose(reason)
  }

  const cardStyle: CSSProperties = (() => {
    if (!hole) {
      return { top: '20%', left: '50%', transform: 'translateX(-50%)' }
    }
    const below = hole.top + hole.height + 16
    const above = hole.top - 16
    const preferBelow = below + 220 < window.innerHeight
    const top = preferBelow ? below : Math.max(12, above - 200)
    let left = hole.left
    const maxLeft = window.innerWidth - 360 - 16
    left = Math.max(12, Math.min(left, maxLeft))
    return { top, left, transform: 'none' }
  })()

  return (
    <div className="walkthrough-root" role="dialog" aria-modal="true" aria-label="Guided walkthrough">
      <div className="walkthrough-shade" aria-hidden>
        {hole ? (
          <div
            className="walkthrough-hole"
            style={{
              top: hole.top,
              left: hole.left,
              width: hole.width,
              height: hole.height,
            }}
          />
        ) : null}
      </div>

      <div className="walkthrough-card" style={cardStyle}>
        <div className="walkthrough-meta mono">
          Step {index + 1} / {total}
        </div>
        <h3>{step.title}</h3>
        <p>{step.body}</p>
        <div className="walkthrough-actions">
          <button type="button" className="btn" onClick={() => finish('skip')}>
            Skip
          </button>
          <div className="walkthrough-nav">
            <button
              type="button"
              className="btn"
              disabled={isFirst}
              onClick={() => setIndex((i) => Math.max(0, i - 1))}
            >
              Back
            </button>
            {isLast ? (
              <button type="button" className="btn primary" onClick={() => finish('finish')}>
                Finish
              </button>
            ) : (
              <button
                type="button"
                className="btn primary"
                onClick={() => setIndex((i) => Math.min(total - 1, i + 1))}
              >
                Next
              </button>
            )}
          </div>
        </div>
        <p className="walkthrough-hint mono">Esc closes · ← → navigate</p>
      </div>
    </div>
  )
}
