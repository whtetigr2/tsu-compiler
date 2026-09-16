import { useEffect, useState } from 'react'
import type { ReceiptInspect } from '../types'
import { ClaimHygienePanel } from './ClaimHygienePanel'

export type HelpTab =
  | 'about'
  | 'walkthrough'
  | 'uimap'
  | 'receipts'
  | 'hygiene'

interface Props {
  open: boolean
  onClose: () => void
  receipt: ReceiptInspect | null
  initialTab?: HelpTab
  onStartWalkthrough?: () => void
}

const UI_MAP: { id: string; title: string; blurb: string }[] = [
  {
    id: 'overview',
    title: 'Overview',
    blurb:
      'Landing inspect view: gate ledger traffic lights, spin census (world vs mediators), and Fabric Tax accent when connectivity exists.',
  },
  {
    id: 'spins',
    title: 'Spins',
    blurb:
      'Live ±1 spin field with chromatic block pulse while THRML streams. Layout and world/mediator indices come from the receipt.',
  },
  {
    id: 'couplings',
    title: 'Couplings',
    blurb:
      'Edge weights and biases of the compiled Ising program. Sci|Prog only changes labels, not numbers.',
  },
  {
    id: 'connectivity',
    title: 'Connectivity',
    blurb:
      'Logical vs physical graph and Fabric Tax. Click a mediated pair or physical edge for a mediator callout derived from program.json.',
  },
  {
    id: 'schedule',
    title: 'Schedule',
    blurb:
      'Chromatic block-Gibbs colouring: independent sets updated together. Active-block cue is a step-parity visual, not a per-sweep observer.',
  },
  {
    id: 'statespace',
    title: 'State space',
    blurb:
      'Honest PCA: 3D embedding only for small n (≤16). Large models refuse 3D with a reason and show a 2D sample cloud only.',
  },
  {
    id: 'residuals',
    title: 'Residuals',
    blurb:
      'Residual / utilisation bars when the receipt provides them; unavailable slots are labeled honestly rather than invented.',
  },
  {
    id: 'scope',
    title: 'Scope',
    blurb:
      'Streaming scope for energy / magnetization traces from simulation, diagnostic curves, not joule product claims.',
  },
  {
    id: 'notepad',
    title: 'Program notepad',
    blurb:
      'Edit a Thermodynamic Program YAML, Preflight, ideal-first Compile via optional tsu, Apply only with a COMPILED receipt on disk.',
  },
  {
    id: 'worlds',
    title: 'Worlds (drawer)',
    blurb:
      'Demoted world-gen / World Studio. Available from ☰ or the left-rail Worlds entry, not part of the hero path.',
  },
  {
    id: 'rails',
    title: 'Rails & chrome',
    blurb:
      'Top ☰ menu (File / View / Worlds / Help), program bar (receipt · β · verdict), right rail (gates + claim hygiene), footer transport (Run / Pause / Step / snapshot).',
  },
]

export function HelpModal({
  open,
  onClose,
  receipt,
  initialTab = 'about',
  onStartWalkthrough,
}: Props) {
  const [tab, setTab] = useState<HelpTab>(initialTab)
  useEffect(() => {
    if (open) setTab(initialTab)
  }, [open, initialTab])

  if (!open) return null

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <div
        className="drawer wide help-drawer"
        role="dialog"
        aria-label="User guide"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <h3>Help · user guide</h3>
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="help-tabs wrap">
          <button
            type="button"
            className={tab === 'about' ? 'on' : ''}
            onClick={() => setTab('about')}
          >
            About
          </button>
          <button
            type="button"
            className={tab === 'walkthrough' ? 'on' : ''}
            onClick={() => setTab('walkthrough')}
          >
            Walkthrough
          </button>
          <button
            type="button"
            className={tab === 'uimap' ? 'on' : ''}
            onClick={() => setTab('uimap')}
          >
            UI map
          </button>
          <button
            type="button"
            className={tab === 'receipts' ? 'on' : ''}
            onClick={() => setTab('receipts')}
          >
            Receipts & sampling
          </button>
          <button
            type="button"
            className={tab === 'hygiene' ? 'on' : ''}
            onClick={() => setTab('hygiene')}
          >
            Claim hygiene
          </button>
        </div>

        {tab === 'about' ? (
          <div className="help-pane">
            <p>
              Nsight-style inspector for a <em>compiled thermodynamic sampling program</em>.
              Load a compile receipt, inspect gates / spins / couplings / connectivity, and
              stream live THRML block-Gibbs draws on CPU (JAX).
            </p>
            <p className="empty-hint">
              Hero path: Extropic / curated example → Overview → Connectivity (Fabric Tax) →
              Spins. Worlds / World Studio stay demoted in the menu, not required for the
              first 10 seconds.
            </p>
            <p className="empty-hint warn-inline">
              JAX/THRML simulation, not Extropic silicon. Mediated β is FIXED. No joule /
              energy product claims.
            </p>
            <p className="empty-hint">
              Clone-and-use: see <span className="mono">WALKTHROUGH.md</span> in the repo, or
              start the in-app tour from the Walkthrough tab.
            </p>
          </div>
        ) : null}

        {tab === 'walkthrough' ? (
          <div className="help-pane">
            <p>
              A short spotlight tour covers mission, program bar, receipts, left-rail views,
              transport, Sci|Prog, notepad, claim hygiene, and snapshot export.
            </p>
            <p className="empty-hint">
              First visit auto-starts once (localStorage key{' '}
              <span className="mono">gibbs-observatory-walkthrough-v1</span>). Restart anytime
              from here or ☰ → Help → Start walkthrough…
            </p>
            <button
              type="button"
              className="btn primary"
              onClick={() => {
                onClose()
                onStartWalkthrough?.()
              }}
            >
              Start walkthrough…
            </button>
          </div>
        ) : null}

        {tab === 'uimap' ? (
          <div className="help-pane">
            <p className="empty-hint">Left-nav views and chrome, one or two sentences each.</p>
            <dl className="ui-map">
              {UI_MAP.map((row) => (
                <div key={row.id} className="ui-map-row">
                  <dt>{row.title}</dt>
                  <dd>{row.blurb}</dd>
                </div>
              ))}
            </dl>
          </div>
        ) : null}

        {tab === 'receipts' ? (
          <div className="help-pane">
            <h4>Loading receipts</h4>
            <p>
              Curated receipts live under <span className="mono">receipts/</span> in the repo.
              ☰ → File → Open receipt… lists packaged ids; Open Extropic example… uses the
              examples shelf (<span className="mono">small</span>,{' '}
              <span className="mono">elev_band</span> ready; <span className="mono">codon_opt</span>{' '}
              stub until packaged).
            </p>
            <h4>What COMPILED means</h4>
            <p>
              Verdict <strong>COMPILED</strong> means the receipt passed compile gates and
              exposes a THRML-ready Ising program (nodes / edges / weights / biases / blocks).
              It does <em>not</em> mean silicon, hardware energy, or production Thermalizers.
            </p>
            <h4>THRML CPU honesty</h4>
            <p>
              Sampling is JAX + THRML on CPU in this Observatory. The amber badge and footer
              meta always say so. If a receipt is not THRML-ready, the program bar / banner
              explains the fallback, we do not silently pretend.
            </p>
            <h4>Notepad compile (optional)</h4>
            <p className="empty-hint">
              Program notepad needs an importable <span className="mono">tsu</span> package
              (sibling tree or <span className="mono">TSU_ROOT</span> /{' '}
              <span className="mono">PYTHONPATH=.:../tsu-compiler</span>). Apply only succeeds
              with a COMPILED receipt written under <span className="mono">receipts/notepad</span>.
            </p>
          </div>
        ) : null}

        {tab === 'hygiene' ? <ClaimHygienePanel receipt={receipt} /> : null}
      </div>
    </div>
  )
}
