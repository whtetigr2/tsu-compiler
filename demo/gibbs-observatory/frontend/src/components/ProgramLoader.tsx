import { useRef, useState } from 'react'

/**
 * Getting a program into the Workbench, and saying out loud what happened.
 *
 * Three ways in, because people arrive with their work in three states: they
 * have a file and want to pick it, they know the path and want to type it, or
 * they are dragging it out of a folder. Pasting into the editor is the fourth
 * and needs no control.
 *
 * The status line underneath is the part that matters. A tool that silently
 * fills a text box has told the reader nothing about what it decided: which
 * format it read, whether it converted anything, and whether the thing now on
 * screen is what will actually be compiled. So each step announces itself, and
 * the announcement includes the reasoning when the loader had to make a call.
 */

export type Stage =
  | { kind: 'idle' }
  | { kind: 'loaded'; file: string; format: string; why: string; converted: boolean }
  | { kind: 'compiling' }
  | { kind: 'compiled'; spins?: number; couplings?: number }
  | { kind: 'refused'; short: string }
  /** The compiler itself failed on this program. Not the same as a file that
   *  would not load, and saying "could not load" after a successful load sends
   *  the reader to check their disk instead of their program. */
  | { kind: 'compileFailed'; message: string }
  | { kind: 'error'; message: string }

export interface LoadedProgram {
  format: string
  why: string
  executed: boolean
  yaml: string
  message: string
  filename: string
}

interface Props {
  busy: boolean
  onLoaded: (loaded: LoadedProgram) => void
  onError: (message: string) => void
}

async function post(url: string, body: unknown): Promise<LoadedProgram> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = data?.detail
    throw new Error(typeof d === 'string' ? d : d?.message ?? 'load failed')
  }
  return data as LoadedProgram
}

export function LoadBar({ busy, onLoaded, onError }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [path, setPath] = useState('')
  const [dragging, setDragging] = useState(false)

  const fromFile = async (file: File) => {
    try {
      onLoaded(await post('/api/program/load', {
        filename: file.name,
        text: await file.text(),
      }))
    } catch (e) {
      onError(String(e instanceof Error ? e.message : e))
    }
  }

  const fromPath = async () => {
    if (!path.trim()) return
    try {
      onLoaded(await post('/api/program/load-path', { path: path.trim() }))
    } catch (e) {
      onError(String(e instanceof Error ? e.message : e))
    }
  }

  return (
    <div
      className={`load-bar ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files?.[0]
        if (f) void fromFile(f)
      }}
    >
      <button
        type="button"
        className="btn"
        disabled={busy}
        onClick={() => fileRef.current?.click()}
      >
        Open program
      </button>
      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".yaml,.yml,.json,.py,.txt"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void fromFile(f)
          e.target.value = ''
        }}
      />

      <span className="load-or">or</span>

      <input
        className="load-path mono"
        type="text"
        value={path}
        placeholder="path to a program on this machine"
        spellCheck={false}
        disabled={busy}
        onChange={(e) => setPath(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') void fromPath()
        }}
        aria-label="path to a program file"
      />
      <button type="button" className="btn" disabled={busy || !path.trim()} onClick={() => void fromPath()}>
        Load
      </button>

      <span className="load-hint">
        YAML, JSON, or a .py holding a literal SPEC. Drop a file anywhere here.
      </span>
    </div>
  )
}

/**
 * The running commentary: loaded, compiling, compile okay, press Run.
 *
 * Deliberately one line and always present once something has happened, so the
 * reader never has to wonder whether the program on screen is the one that was
 * compiled. A refusal is summarised here and explained in full by the refusal
 * panel underneath; this line's job is only to say which state we are in.
 */
export function StatusLine({ stage }: { stage: Stage }) {
  if (stage.kind === 'idle') return null

  if (stage.kind === 'loaded') {
    return (
      <div className="status-line loaded">
        <span className="status-dot" />
        <strong>loaded {stage.file}</strong>
        <span className="status-detail">
          read as {stage.format.toUpperCase()}
          {stage.converted ? ', converted to YAML' : ''} &middot; {stage.why}
        </span>
        <span className="status-next">Press Compile.</span>
      </div>
    )
  }

  if (stage.kind === 'compiling') {
    return (
      <div className="status-line working">
        <span className="status-dot pulse" />
        <strong>compiling</strong>
        <span className="status-detail">
          gates, then placement. Placement is the slow half.
        </span>
      </div>
    )
  }

  if (stage.kind === 'compiled') {
    return (
      <div className="status-line ok">
        <span className="status-dot" />
        <strong>compile okay</strong>
        <span className="status-detail">
          {stage.spins != null ? `${stage.spins.toLocaleString()} spins` : ''}
          {stage.couplings != null ? `, ${stage.couplings.toLocaleString()} couplings` : ''}
        </span>
        <span className="status-next">Press Run.</span>
      </div>
    )
  }

  if (stage.kind === 'refused') {
    return (
      <div className="status-line bad">
        <span className="status-dot" />
        <strong>compile refused</strong>
        <span className="status-detail">{stage.short}</span>
      </div>
    )
  }

  if (stage.kind === 'compileFailed') {
    return (
      <div className="status-line bad">
        <span className="status-dot" />
        <strong>compile failed</strong>
        <span className="status-detail">{stage.message}</span>
        <span className="status-next">The file loaded. The program did not compile.</span>
      </div>
    )
  }

  return (
    <div className="status-line bad">
      <span className="status-dot" />
      <strong>could not load</strong>
      <span className="status-detail">{stage.message}</span>
    </div>
  )
}
