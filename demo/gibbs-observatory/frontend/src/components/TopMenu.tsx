import type { LangMode } from '../types'

interface Props {
  lang: LangMode
  setLang: (m: LangMode) => void
  onOpenReceipts: () => void
  onOpenExamples: () => void
  onOpenWorlds: () => void
  onOpenHelp: () => void
  onOpenClaimHygiene: () => void
  onStartWalkthrough: () => void
  onOpenShortcuts?: () => void
  onSaveSnapshot: () => void
  menuOpen: boolean
  setMenuOpen: (v: boolean) => void
  running: boolean
  connected: boolean
}

export function TopMenu({
  lang,
  setLang,
  onOpenReceipts,
  onOpenExamples,
  onOpenWorlds,
  onOpenHelp,
  onOpenClaimHygiene,
  onStartWalkthrough,
  onOpenShortcuts,
  onSaveSnapshot,
  menuOpen,
  setMenuOpen,
  running,
  connected,
}: Props) {
  const pillClass = !connected
    ? 'status-pill warn-pill'
    : running
      ? 'status-pill'
      : 'status-pill idle'
  const pillText = !connected
    ? 'THRML SIMULATOR / offline'
    : running
      ? 'THRML SIMULATOR / active'
      : 'THRML SIMULATOR / ready'

  return (
    <div className="top-menu">
      <button
        className="menu-burger"
        type="button"
        aria-label="Menu"
        data-tour="menu-burger"
        onClick={() => setMenuOpen(!menuOpen)}
      >
        ☰
      </button>
      <div className="brand-inline" data-tour="brand">
        <span className="brand-mark" aria-hidden>
          β
        </span>
        <div className="brand-copy">
          <strong>Gibbs Observatory</strong>
          <span className="brand-tagline">
            Compiled-program inspector · THRML block-Gibbs
          </span>
        </div>
      </div>
      {menuOpen && (
        <div className="menu-dropdown" role="menu">
          <div className="menu-col">
            <div className="menu-head">File</div>
            <button
              type="button"
              onClick={() => {
                onOpenReceipts()
                setMenuOpen(false)
              }}
            >
              Open receipt…
            </button>
            <button
              type="button"
              onClick={() => {
                onOpenExamples()
                setMenuOpen(false)
              }}
            >
              Open Extropic example…
            </button>
            <button
              type="button"
              onClick={() => {
                onSaveSnapshot()
                setMenuOpen(false)
              }}
            >
              Save snapshot
            </button>
          </div>
          <div className="menu-col">
            <div className="menu-head">View</div>
            <button
              type="button"
              onClick={() => {
                setLang(lang === 'sci' ? 'prog' : 'sci')
                setMenuOpen(false)
              }}
            >
              Language: {lang === 'sci' ? 'Sci → Prog' : 'Prog → Sci'}
            </button>
            <button type="button" disabled>
              Theme (dark locked)
            </button>
          </div>
          <div className="menu-col">
            <div className="menu-head">Worlds</div>
            <button
              type="button"
              onClick={() => {
                onOpenWorlds()
                setMenuOpen(false)
              }}
            >
              World Studio… (demoted)
            </button>
            <button type="button" disabled>
              Saved worlds…
            </button>
          </div>
          <div className="menu-col">
            <div className="menu-head">Help</div>
            <button
              type="button"
              onClick={() => {
                onStartWalkthrough()
                setMenuOpen(false)
              }}
            >
              Start walkthrough…
            </button>
            <button
              type="button"
              onClick={() => {
                onOpenHelp()
                setMenuOpen(false)
              }}
            >
              User guide…
            </button>
            <button
              type="button"
              onClick={() => {
                onOpenClaimHygiene()
                setMenuOpen(false)
              }}
            >
              Claim hygiene…
            </button>
            <button
              type="button"
              onClick={() => {
                onOpenShortcuts?.()
                setMenuOpen(false)
              }}
            >
              Keyboard shortcuts
            </button>
          </div>
        </div>
      )}
      <div className="top-menu-spacer" />
      <div className="lang-toggle" data-tour="lang-toggle">
        <button
          type="button"
          className={lang === 'sci' ? 'on' : ''}
          onClick={() => setLang('sci')}
        >
          Sci
        </button>
        <button
          type="button"
          className={lang === 'prog' ? 'on' : ''}
          onClick={() => setLang('prog')}
        >
          Prog
        </button>
      </div>
      <div className={pillClass} data-tour="status-pill" title="Software / THRML only">
        <span className="pip" />
        {pillText}
      </div>
    </div>
  )
}
