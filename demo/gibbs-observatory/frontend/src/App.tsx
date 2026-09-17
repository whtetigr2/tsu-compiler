import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { FooterBar } from './components/FooterBar'
import { HelpModal, type HelpTab } from './components/HelpModal'
import { ITEMS as NAV_ITEMS, LeftNav } from './components/LeftNav'
import {
  Pane, Split, WORKSPACES, useWorkspaceLayout,
} from './components/Workspace'
import { ProgramBar } from './components/ProgramBar'
import { ReceiptPicker } from './components/ReceiptPicker'
import { RightRail } from './components/RightRail'
import { TopMenu } from './components/TopMenu'
import { Walkthrough, hasSeenWalkthrough } from './components/Walkthrough'
import {
  WorkflowStrip,
  workflowStepForView,
  type WorkflowStepId,
} from './components/WorkflowStrip'
import { WorldsDrawer } from './components/WorldsDrawer'
import { ConnectivityView } from './components/views/ConnectivityView'
import { CouplingsView } from './components/views/CouplingsView'
import { OverviewView } from './components/views/OverviewView'
import { ResidualsView } from './components/views/ResidualsView'
import { ScheduleView } from './components/views/ScheduleView'
import { ScopeView } from './components/views/ScopeView'
import { SpinsView } from './components/views/SpinsView'
import { NotepadView } from './components/views/NotepadView'
import { StateSpaceView } from './components/views/StateSpaceView'
import { AblationView } from './components/views/AblationView'
import { LatticeView3D } from './components/views/LatticeView3D'
import { StructureFactorView } from './components/views/StructureFactorView'
import { GameView } from './components/views/GameView'
import { EbmLabView } from './components/views/EbmLabView'
import { useGibbsSocket } from './hooks/useGibbsSocket'
import { exportSnapshot } from './lib/snapshot'
import {
  DEFAULT_PARAMS,
  type ExampleShelfItem,
  type LangMode,
  type NavView,
  type ReceiptInspect,
  type ReceiptSummary,
  type SimParams,
} from './types'

export default function App() {
  const {
    status,
    running,
    graph,
    batch,
    error,
    clearError,
    reset,
    updateParams,
    run,
    pause,
    step,
  } = useGibbsSocket()

  const [params, setParams] = useState<SimParams>(DEFAULT_PARAMS)
  const [lang, setLang] = useState<LangMode>('sci')
  const [view, setView] = useState<NavView>('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [worldsOpen, setWorldsOpen] = useState(false)
  const [receiptsOpen, setReceiptsOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [helpTab, setHelpTab] = useState<HelpTab>('about')
  const [walkthroughOpen, setWalkthroughOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const walkthroughBooted = useRef(false)
  const [snapshotBusy, setSnapshotBusy] = useState(false)
  const [snapshotMsg, setSnapshotMsg] = useState<string | null>(null)
  const stageRef = useRef<HTMLElement | null>(null)
  const [receipts, setReceipts] = useState<ReceiptSummary[]>([])
  const [examples, setExamples] = useState<ExampleShelfItem[]>([])
  const [receipt, setReceipt] = useState<ReceiptInspect | null>(null)
  const [speed, setSpeed] = useState(2)
  const [workspace, setWorkspace] = useState<string>('single')
  const [selectedSpin, setSelectedSpin] = useState<number | null>(null)
  const connected = status === 'open'
  const didInit = useRef(false)
  const resetRef = useRef(reset)
  resetRef.current = reset

  const loadReceipt = useCallback(async (id: string) => {
    const res = await fetch(`/api/receipts/${id}`)
    if (!res.ok) throw new Error(`receipt ${id} failed`)
    const data = (await res.json()) as ReceiptInspect
    setReceipt(data)
    setSelectedSpin(null)
    return data
  }, [])

  useEffect(() => {
    fetch('/api/receipts')
      .then((r) => r.json())
      .then((d) => setReceipts(d.receipts ?? []))
      .catch(() => setReceipts([]))
    fetch('/api/examples')
      .then((r) => r.json())
      .then((d) => setExamples(d.examples ?? []))
      .catch(() => setExamples([]))
  }, [])

  useEffect(() => {
    if (walkthroughBooted.current) return
    walkthroughBooted.current = true
    if (!hasSeenWalkthrough()) {
      const t = window.setTimeout(() => setWalkthroughOpen(true), 600)
      return () => window.clearTimeout(t)
    }
  }, [])

  useEffect(() => {
    if (!connected || didInit.current) return
    didInit.current = true
    const boot = async () => {
      const id = DEFAULT_PARAMS.receipt_id || 'small'
      try {
        const data = await loadReceipt(id)
        const next: SimParams = {
          ...DEFAULT_PARAMS,
          receipt_id: id,
          beta: data.beta ?? DEFAULT_PARAMS.beta,
          seed: data.sampling.seed ?? DEFAULT_PARAMS.seed,
          warmup: 20,
          batch_size: 4,
          steps_per_sample: 2,
        }
        setParams(next)
        resetRef.current(next)
      } catch {
        resetRef.current(DEFAULT_PARAMS)
      }
    }
    void boot()
  }, [connected, loadReceipt])

  useEffect(() => {
    if (!connected || !didInit.current) return
    const t = setTimeout(() => {
      const patch: Partial<SimParams> = {
        warmup: params.warmup,
        steps_per_sample: params.steps_per_sample,
        batch_size: params.batch_size,
      }
      if (!receipt?.beta_fixed) {
        patch.beta = params.beta
      }
      if (!params.receipt_id) {
        patch.J = params.J
        patch.h = params.h
      }
      updateParams(patch)
    }, 120)
    return () => clearTimeout(t)
  }, [
    connected,
    params.beta,
    params.J,
    params.h,
    params.warmup,
    params.steps_per_sample,
    params.batch_size,
    params.receipt_id,
    receipt?.beta_fixed,
    updateParams,
  ])

  const selectReceipt = async (id: string) => {
    const data = await loadReceipt(id)
    const next: SimParams = {
      ...params,
      receipt_id: id,
      beta: data.beta ?? 1.0,
      seed: data.sampling.seed ?? params.seed,
      warmup: 20,
      batch_size: 4,
      steps_per_sample: speed,
    }
    setParams(next)
    clearError()
    reset(next)
    if (id === 'prog_ebm_bars_stripes') {
      setView('ebmlab')
    } else {
      setView('overview')
    }
  }

  const onNav = (v: NavView) => {
    if (v === 'worlds') {
      setWorldsOpen(true)
      return
    }
    setView(v)
  }

  const onSpeed = (n: number) => {
    setSpeed(n)
    setParams((p) => ({ ...p, steps_per_sample: n }))
  }

  const onWorkflow = (id: WorkflowStepId) => {
    if (id === 'receipt') {
      setReceiptsOpen(true)
      return
    }
    if (id === 'inspect') {
      setView('overview')
      return
    }
    if (id === 'sample') {
      setView('overview')
      if (connected && !running) run()
      return
    }
    if (id === 'compare') {
      setView('residuals')
    }
  }

  const saveSnapshot = async () => {
    setSnapshotBusy(true)
    setSnapshotMsg(null)
    try {
      const result = await exportSnapshot({
        stageEl: stageRef.current,
        receipt,
        batch,
        view,
      })
      setSnapshotMsg(result.message)
    } catch (err) {
      setSnapshotMsg(err instanceof Error ? err.message : 'snapshot failed')
    } finally {
      setSnapshotBusy(false)
      window.setTimeout(() => setSnapshotMsg(null), 5000)
    }
  }

  const fallbackBanner =
    graph?.sampling_banner ||
    batch?.sampling_banner ||
    (receipt && !receipt.sampling.thrml_ready ? receipt.sampling.banner : null)

  const statusLabel =
    status === 'open'
      ? running
        ? 'streaming'
        : 'idle'
      : status === 'reconnecting'
        ? 'reconnecting…'
        : status

  const nColors =
    (graph?.color0?.length ? 1 : 0) + (graph?.color1?.length ? 1 : 0) || 2
  const scheduleLabel =
    receipt?.kernel?.includes('chromatic') || graph?.kernel?.includes('chromatic')
      ? `${nColors}-color chromatic schedule`
      : receipt?.kernel ?? graph?.kernel ?? 'sampling program'
  const heroId = receipt
    ? `#${receipt.id}`
    : graph?.receipt_id
      ? `#${graph.receipt_id}`
      : 'unavailable'

  const wfActive = receiptsOpen
    ? 'receipt'
    : workflowStepForView(view, running)

  // Any pane can ask for any editor, so this is a function of the view id
  // rather than a single `stage` computed for the one active view.
  const renderEditor = (which: NavView): ReactNode => {
    let stage: ReactNode = null
    switch (which) {
    case 'overview':
      stage = (
        <OverviewView
          graph={graph}
          batch={batch}
          receipt={receipt}
          lang={lang}
          selectedSpin={selectedSpin}
          onSelectSpin={setSelectedSpin}
        />
      )
      break
    case 'spins':
      stage = (
        <SpinsView
          graph={graph}
          batch={batch}
          lang={lang}
          selectedSpin={selectedSpin}
          onSelectSpin={setSelectedSpin}
        />
      )
      break
    case 'couplings':
      stage = (
        <CouplingsView
          graph={graph}
          batch={batch}
          lang={lang}
          selectedSpin={selectedSpin}
          onSelectSpin={setSelectedSpin}
        />
      )
      break
    case 'connectivity':
      stage = <ConnectivityView receipt={receipt} lang={lang} />
      break
    case 'schedule':
      stage = <ScheduleView receipt={receipt} batch={batch} lang={lang} />
      break
    case 'residuals':
      stage = <ResidualsView receipt={receipt} lang={lang} />
      break
    case 'scope':
      stage = <ScopeView batch={batch} receipt={receipt} lang={lang} />
      break
    case 'statespace':
      stage = (
        <StateSpaceView graph={graph} batch={batch} receipt={receipt} lang={lang} />
      )
      break
    case 'notepad':
      stage = (
        <NotepadView receipt={receipt} onApplyReceipt={selectReceipt} />
      )
      break
    case 'structurefactor':
      stage = <StructureFactorView batch={batch} />
      break
    case 'game':
      stage = <GameView />
      break
    case 'lattice3d':
      stage = <LatticeView3D graph={graph} batch={batch} />
      break
    case 'ablation':
      stage = <AblationView receipt={receipt} />
      break
    case 'ebmlab':
      stage = (
        <EbmLabView
          graph={graph}
          receipt={receipt}
          lang={lang}
          onLoadEbmReceipt={() => void selectReceipt('prog_ebm_bars_stripes')}
        />
      )
      break
    default:
      stage = null
    }
    return stage
  }

  const spec = WORKSPACES.find((w) => w.id === workspace) ?? WORKSPACES[0]
  const layout = useWorkspaceLayout(spec)
  const editorChoices = NAV_ITEMS
    .filter((i) => i.phase !== 'drawer')
    .map((i) => ({ id: i.id, label: i.label }))

  const stage = renderEditor(view)

  /** The workspace grid: columns of panes, each pane any editor. */
  const workspaceStage = (
    <Split direction="row" sizes={layout.colSizes} onResize={layout.setColSizes}>
      {layout.editors.map((column, ci) => (
        <Split
          key={ci}
          direction="column"
          sizes={layout.rowSizes[ci] ?? [1]}
          onResize={(next) => layout.setRowSizesAt(ci, next)}
        >
          {column.map((editorId, ri) => (
            <Pane
              key={ri}
              editor={editorId}
              editors={editorChoices}
              onChange={(next) => layout.setEditor(ci, ri, next)}
              render={renderEditor}
            />
          ))}
        </Split>
      ))}
    </Split>
  )

  const viewTitle: Record<string, string> = {
    overview: 'Compiled sampling program',
    spins: 'Spin field',
    couplings: 'Couplings',
    connectivity: 'Connectivity · Fabric Tax',
    schedule: 'Chromatic schedule',
    residuals: 'Residuals',
    scope: 'Scope',
    statespace: 'State space',
    notepad: 'Thermodynamic Program notepad',
    game: 'Walk a level solved by sampling',
    ablation: 'Ablation runner',
    lattice3d: 'Lattice in 3D',
    structurefactor: 'Structure factor S(k)',
    ebmlab: 'EBM Lab · Bars & Stripes RBM 4×4',
  }

  return (
    <div className="app-shell" onClick={() => menuOpen && setMenuOpen(false)}>
      <TopMenu
        lang={lang}
        setLang={setLang}
        onOpenReceipts={() => setReceiptsOpen(true)}
        onOpenExamples={() => setReceiptsOpen(true)}
        onOpenWorlds={() => setWorldsOpen(true)}
        onOpenHelp={() => {
          setHelpTab('about')
          setHelpOpen(true)
        }}
        onOpenClaimHygiene={() => {
          setHelpTab('hygiene')
          setHelpOpen(true)
        }}
        onStartWalkthrough={() => {
          setHelpOpen(false)
          setMenuOpen(false)
          setWalkthroughOpen(true)
        }}
        onOpenShortcuts={() => setShortcutsOpen(true)}
        onSaveSnapshot={() => void saveSnapshot()}
        menuOpen={menuOpen}
        setMenuOpen={setMenuOpen}
        running={running}
        connected={connected}
      />

      <WorkflowStrip
        active={wfActive}
        onSelect={onWorkflow}
        receiptLabel={
          receipt
            ? `${receipt.id} · ${receipt.spins.n_nodes ?? graph?.n_nodes ?? '?'} spins`
            : null
        }
      />

      {/* Hidden but kept for walkthrough data-tour="program-bar" */}
      <div className="workspace-bar">
        <div className="workspace-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={workspace === 'single'}
            className={workspace === 'single' ? 'active' : ''}
            onClick={() => setWorkspace('single')}
            title="One editor at a time, chosen from the rail"
          >
            Single
          </button>
          {WORKSPACES.map((w) => (
            <button
              key={w.id}
              type="button"
              role="tab"
              aria-selected={workspace === w.id}
              className={workspace === w.id ? 'active' : ''}
              onClick={() => setWorkspace(w.id)}
              title={w.blurb}
            >
              {w.label}
            </button>
          ))}
          <span className="workspace-blurb">
            {workspace === 'single'
              ? 'One editor at a time, chosen from the rail.'
              : spec.blurb}
          </span>
        </div>
      </div>
      <ProgramBar receipt={receipt} fallbackBanner={fallbackBanner} />

      <div className="shell-body">
        <LeftNav
          active={view}
          onSelect={onNav}
          receipt={receipt}
          params={params}
          setParams={setParams}
          running={running}
          connected={connected}
          onOpenReceipts={() => setReceiptsOpen(true)}
          onRun={run}
          onPause={pause}
          onStep={step}
          onReset={() => {
            clearError()
            reset(params)
          }}
          speed={speed}
          setSpeed={onSpeed}
        />
        <main className="main-stage" ref={stageRef}>
          {workspace !== 'single' ? (
            <div className="workspace-stage">{workspaceStage}</div>
          ) : (
          <>
          <div className="stage-header">
            <div className="hero-title-block">
              <span className="hero-eyebrow">
                {view === 'overview' ? 'Compiled Ising program' : view === 'ebmlab' ? 'Trained model, bars and stripes' : 'Observatory view'}
              </span>
              <h2 className="hero-title">
                {viewTitle[view] ?? view}
              </h2>
              <p className="hero-sub">
                {view === 'overview' ? scheduleLabel : statusLabel}
                {error ? (
                  <>
                    {' '}
                    · <span className="status-error">{error}</span>
                    <button
                      type="button"
                      className="btn tiny"
                      onClick={clearError}
                      title="Dismiss"
                    >
                      ×
                    </button>
                  </>
                ) : null}
              </p>
            </div>
            <div className="stage-header-right">
              <span className="hero-id mono">{heroId}</span>
              <span className="mono">
                <span
                  className={`dot ${
                    running ? 'on' : connected ? 'off' : 'warn'
                  }`}
                />
                {statusLabel}
                {' · '}
                {graph?.preset === 'receipt'
                  ? `${graph.n_nodes} spins`
                  : graph
                    ? `${graph.preset} · ${graph.n_nodes}`
                    : 'unavailable'}
              </span>
              {snapshotMsg ? (
                <span className="snapshot-toast mono">{snapshotMsg}</span>
              ) : null}
              {fallbackBanner ? (
                <span className="snapshot-toast mono" style={{ color: 'var(--warn)' }}>
                  {fallbackBanner}
                </span>
              ) : null}
            </div>
          </div>
          {receipt?.id === 'prog_ebm_bars_stripes' && view !== 'ebmlab' ? (
            <div className="lab-open-banner" role="note">
              <span>EBM bars-and-stripes loaded</span>
              <button
                type="button"
                className="btn tiny phosphor"
                onClick={() => setView('ebmlab')}
              >
                Open in EBM Lab
              </button>
            </div>
          ) : null}
          <div className="hero-panel">{stage}</div>
          </>
          )}
        </main>
        <RightRail
          receipt={receipt}
          batch={batch}
          graph={graph}
          lang={lang}
          running={running}
          connected={connected}
          selectedSpin={selectedSpin}
        />
      </div>

      <FooterBar
        running={running}
        connected={connected}
        seed={params.seed}
        setSeed={(n) => setParams((p) => ({ ...p, seed: n }))}
        onRun={run}
        onPause={pause}
        onStep={step}
        onReset={() => {
          clearError()
          reset(params)
        }}
        onSnapshot={() => void saveSnapshot()}
        snapshotBusy={snapshotBusy}
        speed={speed}
        setSpeed={onSpeed}
      />

      <WorldsDrawer open={worldsOpen} onClose={() => setWorldsOpen(false)} />
      <ReceiptPicker
        open={receiptsOpen}
        receipts={receipts}
        examples={examples}
        currentId={receipt?.id ?? null}
        onClose={() => setReceiptsOpen(false)}
        onSelect={(id) => void selectReceipt(id)}
      />
      <HelpModal
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        receipt={receipt}
        initialTab={helpTab}
        onStartWalkthrough={() => {
          setHelpOpen(false)
          setWalkthroughOpen(true)
        }}
      />
      <Walkthrough
        open={walkthroughOpen}
        onClose={() => setWalkthroughOpen(false)}
        onNavigate={onNav}
      />
      {shortcutsOpen ? (
        <div
          className="drawer-backdrop"
          onClick={() => setShortcutsOpen(false)}
          role="presentation"
        >
          <div
            className="drawer"
            role="dialog"
            aria-label="Keyboard shortcuts"
            onClick={(e) => e.stopPropagation()}
          >
            <header>
              <h3>Keyboard shortcuts</h3>
              <button type="button" className="btn" onClick={() => setShortcutsOpen(false)}>
                Close
              </button>
            </header>
            <ul className="prose-list">
              <li>
                <span className="mono">Esc</span>, close walkthrough, drawers, or Help
              </li>
              <li>
                <span className="mono">←</span> / <span className="mono">→</span>, walkthrough
                Back / Next (Enter = Next / Finish)
              </li>
              <li>
                <span className="mono">☰</span>, open top menu (mouse / tap)
              </li>
            </ul>
            <p className="empty-hint">Transport controls live in the Experiment rail and footer.</p>
          </div>
        </div>
      ) : null}
    </div>
  )
}
