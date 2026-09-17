import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { NavView } from '../types'

/**
 * The frame: workspaces made of panes, each pane showing any editor.
 *
 * Blender's arrangement, and for Blender's reason. A workspace is not a mode or
 * a separate screen; it is a saved arrangement of editors, and every editor is
 * available in every one of them. Switching workspaces rearranges what you are
 * looking at without changing what the program can do.
 *
 * Deliberately NOT here, and the spec says so: arbitrary splitting and joining,
 * floating windows, and user-saved layouts. Three well-chosen arrangements with
 * resizable splits get most of the feel at a fraction of the cost, and none of
 * the work below has to be redone to add the rest later.
 */

export interface EditorChoice {
  id: NavView
  label: string
}

export interface PaneSpec {
  /** Which editor this pane starts on. The reader can change it. */
  editor: NavView
  /** Relative size of this pane within its split. */
  weight?: number
}

export interface WorkspaceSpec {
  id: string
  label: string
  blurb: string
  /** Left-to-right columns. The first column may itself stack vertically. */
  columns: PaneSpec[][]
}

export const WORKSPACES: WorkspaceSpec[] = [
  {
    id: 'build',
    label: 'Build',
    blurb: 'Write a program and see what the compiler makes of it.',
    columns: [
      [{ editor: 'notepad' }],
      [{ editor: 'connectivity' }, { editor: 'schedule' }],
    ],
  },
  {
    id: 'observe',
    label: 'Observe',
    blurb: 'Instruments. What the sampler is doing, while it does it.',
    columns: [
      [{ editor: 'scope' }],
      [{ editor: 'structurefactor' }, { editor: 'ablation' }],
    ],
  },
  {
    id: 'render',
    label: 'Render',
    blurb: 'The object itself, large.',
    columns: [[{ editor: 'lattice3d' }]],
  },
]

interface PaneProps {
  editor: NavView
  editors: EditorChoice[]
  onChange: (next: NavView) => void
  render: (which: NavView) => ReactNode
}

/** One editor, with the dropdown that turns it into a different one. */
export function Pane({ editor, editors, onChange, render }: PaneProps) {
  const current = editors.find((e) => e.id === editor)
  return (
    <section className="pane">
      <header className="pane-head">
        <select
          className="pane-editor"
          value={editor}
          onChange={(e) => onChange(e.target.value as NavView)}
          aria-label="editor type"
        >
          {editors.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label}
            </option>
          ))}
        </select>
        <span className="pane-title">{current?.label ?? editor}</span>
      </header>
      <div className="pane-body">{render(editor)}</div>
    </section>
  )
}

interface SplitProps {
  direction: 'row' | 'column'
  sizes: number[]
  onResize: (sizes: number[]) => void
  children: ReactNode[]
}

/**
 * Resizable splits. Sizes are fractions summing to one, so a window resize
 * keeps the proportions instead of stranding a pane at a fixed pixel width.
 */
export function Split({ direction, sizes, onResize, children }: SplitProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const dragging = useRef<number | null>(null)

  const onMove = useCallback((e: PointerEvent) => {
    const at = dragging.current
    const el = ref.current
    if (at === null || !el) return
    const box = el.getBoundingClientRect()
    const total = direction === 'row' ? box.width : box.height
    if (total <= 0) return
    const pos = direction === 'row' ? e.clientX - box.left : e.clientY - box.top

    // Everything before this handle, plus however much of the pair the pointer
    // now sits past. Clamped so neither side can be dragged to nothing.
    const before = sizes.slice(0, at).reduce((a, b) => a + b, 0)
    const pair = sizes[at] + sizes[at + 1]
    let first = pos / total - before
    first = Math.max(0.12, Math.min(pair - 0.12, first))
    const next = [...sizes]
    next[at] = first
    next[at + 1] = pair - first
    onResize(next)
  }, [direction, sizes, onResize])

  useEffect(() => {
    const up = () => { dragging.current = null }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', up)
    }
  }, [onMove])

  return (
    <div ref={ref} className={`split split-${direction}`}>
      {children.map((child, i) => (
        <div key={i} className="split-cell" style={{ flexGrow: sizes[i] ?? 1 }}>
          {child}
          {i < children.length - 1 ? (
            <div
              className={`split-handle handle-${direction}`}
              onPointerDown={(e) => {
                e.preventDefault()
                dragging.current = i
              }}
              role="separator"
              aria-orientation={direction === 'row' ? 'vertical' : 'horizontal'}
            />
          ) : null}
        </div>
      ))}
    </div>
  )
}

interface TabsProps {
  active: string
  onChange: (id: string) => void
}

export function WorkspaceTabs({ active, onChange }: TabsProps) {
  const current = WORKSPACES.find((w) => w.id === active)
  return (
    <div className="workspace-tabs" role="tablist">
      {WORKSPACES.map((w) => (
        <button
          key={w.id}
          type="button"
          role="tab"
          aria-selected={w.id === active}
          className={w.id === active ? 'active' : ''}
          onClick={() => onChange(w.id)}
          title={w.blurb}
        >
          {w.label}
        </button>
      ))}
      <span className="workspace-blurb">{current?.blurb}</span>
    </div>
  )
}

/** Pane layout for one workspace, with sizes the reader can drag. */
export function useWorkspaceLayout(spec: WorkspaceSpec) {
  const [editors, setEditors] = useState<NavView[][]>(
    () => spec.columns.map((col) => col.map((p) => p.editor)))
  const [colSizes, setColSizes] = useState<number[]>(
    () => spec.columns.map(() => 1 / spec.columns.length))
  const [rowSizes, setRowSizes] = useState<number[][]>(
    () => spec.columns.map((col) => col.map(() => 1 / col.length)))

  useEffect(() => {
    setEditors(spec.columns.map((col) => col.map((p) => p.editor)))
    setColSizes(spec.columns.map(() => 1 / spec.columns.length))
    setRowSizes(spec.columns.map((col) => col.map(() => 1 / col.length)))
  }, [spec])

  const setEditor = (col: number, row: number, next: NavView) =>
    setEditors((prev) => prev.map((c, i) =>
      i === col ? c.map((e, j) => (j === row ? next : e)) : c))

  const setRowSizesAt = (col: number, sizes: number[]) =>
    setRowSizes((prev) => prev.map((r, i) => (i === col ? sizes : r)))

  return { editors, colSizes, rowSizes, setEditor, setColSizes, setRowSizesAt }
}
