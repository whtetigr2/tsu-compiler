import { useEffect, useRef, type MouseEvent } from 'react'
import type { GraphPayload } from '../types'

interface Props {
  graph: GraphPayload | null
  state: number[] | null
  activeBlock: number
  showEdges?: boolean
  edgeMode?: 'none' | 'coupling'
  height?: number
  selectedSpin?: number | null
  onSelectSpin?: (i: number | null) => void
  /** ThermoLith-class labeled nodes for small n */
  graphAesthetic?: boolean
}

const COL0 = 'rgba(110, 184, 212, 0.75)'
const COL1 = 'rgba(212, 122, 168, 0.75)'
const UP = '#e8f0ef'
const DOWN = '#14181c'
const CLAMP = 'rgba(212, 168, 75, 0.9)'
const MED_UP = '#7ec8c0'
const MED_DOWN = '#1a2a2c'
const FERRO = 'rgba(126, 200, 192, 0.55)'
const ANTI = 'rgba(224, 122, 122, 0.55)'
const EDGE = 'rgba(126, 200, 192, 0.28)'
const BG = '#08090a'
const SELECT = '#7ec8c0'

const LABEL_N = 32

export function SpinField({
  graph,
  state,
  activeBlock,
  showEdges = false,
  edgeMode = 'none',
  height = 520,
  selectedSpin = null,
  onSelectSpin,
  graphAesthetic = false,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const hitRef = useRef<{ i: number; x: number; y: number; r: number }[]>([])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !graph) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const cssW = 560
    const cssH = height
    canvas.width = cssW * dpr
    canvas.height = cssH * dpr
    canvas.style.width = `${cssW}px`
    canvas.style.height = `${cssH}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    ctx.fillStyle = BG
    ctx.fillRect(0, 0, cssW, cssH)

    const spins = state ?? Array(graph.n_nodes).fill(0)
    const clampSet = new Set(graph.clamp_indices)
    const c0 = new Set(graph.color0)
    const medSet = new Set(graph.mediator_idx ?? [])
    const worldIdx = graph.world_idx?.length
      ? graph.world_idx
      : Array.from({ length: graph.n_nodes }, (_, i) => i).filter((i) => !medSet.has(i))

    hitRef.current = []

    const drawCouplingEdges = (
      indices: number[],
      ox: number,
      oy: number,
      W: number,
      H: number,
      posOf: (i: number) => [number, number],
    ) => {
      if (edgeMode !== 'coupling') return
      const weights = graph.edge_weights
      const idxSet = new Set(indices)
      const absMax = weights?.length
        ? Math.max(...weights.map((w) => Math.abs(w)), 1e-6)
        : 1
      for (let e = 0; e < graph.edges.length; e++) {
        const [a, b] = graph.edges[e]
        if (!idxSet.has(a) || !idxSet.has(b)) continue
        const [ax, ay] = posOf(a)
        const [bx, by] = posOf(b)
        const w = weights?.[e] ?? 1
        const mag = Math.abs(w) / absMax
        ctx.strokeStyle = w >= 0 ? FERRO : ANTI
        ctx.lineWidth = 0.6 + mag * 3.2
        ctx.globalAlpha = 0.35 + mag * 0.5
        ctx.beginPath()
        ctx.moveTo(ox + ax * W, oy + ay * H)
        ctx.lineTo(ox + bx * W, oy + by * H)
        ctx.stroke()
      }
      ctx.globalAlpha = 1
    }

    const useLabeledGraph =
      graphAesthetic &&
      graph.n_nodes <= LABEL_N &&
      graph.layout !== 'grid'

    if (useLabeledGraph) {
      // ThermoLith-class: labeled circular nodes + phosphor edges
      const pad = 36
      const W = cssW - pad * 2
      const H = cssH - pad * 2 - 12
      const positions = graph.positions

      ctx.strokeStyle = EDGE
      ctx.lineWidth = 1.2
      for (const [a, b] of graph.edges) {
        const [ax, ay] = positions[a] ?? [0.5, 0.5]
        const [bx, by] = positions[b] ?? [0.5, 0.5]
        ctx.beginPath()
        ctx.moveTo(pad + ax * W, pad + ay * H)
        ctx.lineTo(pad + bx * W, pad + by * H)
        ctx.stroke()
      }

      const r = Math.max(14, Math.min(22, 180 / Math.sqrt(graph.n_nodes)))
      for (let i = 0; i < graph.n_nodes; i++) {
        const [px, py] = positions[i] ?? [0.5, 0.5]
        const x = pad + px * W
        const y = pad + py * H
        const up = spins[i] === 1
        const inActive =
          (activeBlock === 0 && c0.has(i)) || (activeBlock === 1 && !c0.has(i))
        const selected = selectedSpin === i

        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = up ? UP : DOWN
        ctx.fill()

        if (selected) {
          ctx.lineWidth = 3
          ctx.strokeStyle = SELECT
          ctx.stroke()
          ctx.beginPath()
          ctx.arc(x, y, r + 5, 0, Math.PI * 2)
          ctx.strokeStyle = 'rgba(126, 200, 192, 0.35)'
          ctx.lineWidth = 2
          ctx.stroke()
        } else {
          ctx.lineWidth = inActive ? 2.4 : 1.2
          ctx.strokeStyle = inActive
            ? activeBlock === 0
              ? COL0
              : COL1
            : '#3a424c'
          if (clampSet.has(i)) ctx.strokeStyle = CLAMP
          ctx.stroke()
        }

        ctx.fillStyle = up ? '#0a0b0c' : '#c5cdd4'
        ctx.font = `600 ${Math.max(10, r * 0.7)}px IBM Plex Mono, monospace`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(up ? '+1' : '−1', x, y)

        const name = graph.node_names?.[i] ?? `s${i}`
        ctx.fillStyle = '#8b939c'
        ctx.font = '10px IBM Plex Mono, monospace'
        ctx.textBaseline = 'top'
        ctx.fillText(name, x, y + r + 4)

        hitRef.current.push({ i, x, y, r: r + 6 })
      }
    } else if (graph.layout === 'grid' && graph.shape.length === 2) {
      const pad = 28
      const W = cssW - pad * 2
      const [rows, cols] = graph.shape
      const cell = Math.min(W / cols, (cssH - pad * 2) / rows)
      const ox = pad + (W - cell * cols) / 2
      const oy = pad
      if (showEdges || edgeMode === 'coupling') {
        drawCouplingEdges(
          Array.from({ length: graph.n_nodes }, (_, i) => i),
          ox,
          oy,
          cell * (cols - 1) || 1,
          cell * (rows - 1) || 1,
          (i) => {
            const r = Math.floor(i / cols)
            const c = i % cols
            return [c / Math.max(1, cols - 1), r / Math.max(1, rows - 1)]
          },
        )
      }
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c
          const x = ox + c * cell
          const y = oy + r * cell
          const up = spins[i] === 1
          ctx.fillStyle = up ? UP : DOWN
          const rad = Math.max(1, cell * 0.12)
          roundRect(ctx, x + 0.5, y + 0.5, cell - 1, cell - 1, rad)
          ctx.fill()

          const inActive =
            (activeBlock === 0 && c0.has(i)) || (activeBlock === 1 && !c0.has(i))
          if (selectedSpin === i) {
            ctx.strokeStyle = SELECT
            ctx.lineWidth = Math.max(1.5, cell * 0.1)
            ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2)
          } else if (inActive) {
            ctx.strokeStyle = activeBlock === 0 ? COL0 : COL1
            ctx.lineWidth = Math.max(1, cell * 0.08)
            ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2)
          }
          if (clampSet.has(i)) {
            ctx.strokeStyle = CLAMP
            ctx.lineWidth = Math.max(1.5, cell * 0.1)
            ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4)
          }
          hitRef.current.push({
            i,
            x: x + cell / 2,
            y: y + cell / 2,
            r: cell / 2,
          })
        }
      }
    } else if (graph.layout === 'receipt' || (graph.mediator_idx?.length ?? 0) > 0) {
      // Dual-panel: World (left) · Fabric Tax / mediators (right) — equal dignity
      const pad = 14
      const gap = 10
      const hasMeds = medSet.size > 0
      // Extra header room for Fabric Tax subtitle when mediators present
      const labelH = hasMeds ? 36 : 22
      const splitX = hasMeds ? cssW / 2 : cssW
      const worldBox = {
        x: pad,
        y: pad + labelH,
        w: (hasMeds ? splitX - gap / 2 : cssW) - pad * 2,
        h: cssH - pad * 2 - labelH,
      }
      const medBox = {
        x: splitX + gap / 2,
        y: pad + labelH,
        w: cssW - splitX - gap / 2 - pad,
        h: cssH - pad * 2 - labelH,
      }

      const worldPos = new Map<number, [number, number]>()
      for (const i of worldIdx) {
        const p = graph.positions[i]
        worldPos.set(i, p ? [p[0], p[1]] : [0.5, 0.5])
      }

      const meds = [...medSet].sort((a, b) => a - b)
      const medPos = new Map<number, [number, number]>()
      if (meds.length) {
        const cols = Math.max(1, Math.ceil(Math.sqrt(meds.length)))
        const rows = Math.max(1, Math.ceil(meds.length / cols))
        meds.forEach((i, k) => {
          const c = k % cols
          const r = Math.floor(k / cols)
          const nx = cols === 1 ? 0.5 : c / (cols - 1)
          const ny = rows === 1 ? 0.5 : r / (rows - 1)
          // slight inset so nodes don't kiss panel edges
          medPos.set(i, [0.08 + nx * 0.84, 0.08 + ny * 0.84])
        })
      }

      const screenOf = (i: number): [number, number] | null => {
        if (worldPos.has(i)) {
          const [px, py] = worldPos.get(i)!
          return [worldBox.x + px * worldBox.w, worldBox.y + py * worldBox.h]
        }
        if (medPos.has(i)) {
          const [px, py] = medPos.get(i)!
          return [medBox.x + px * medBox.w, medBox.y + py * medBox.h]
        }
        return null
      }

      // Panel chrome
      ctx.fillStyle = '#0a0c0e'
      ctx.fillRect(worldBox.x - 4, worldBox.y - labelH, worldBox.w + 8, worldBox.h + labelH + 4)
      ctx.strokeStyle = '#2a2f36'
      ctx.lineWidth = 1
      ctx.strokeRect(worldBox.x - 4, worldBox.y - 4, worldBox.w + 8, worldBox.h + 8)
      ctx.fillStyle = '#8b939c'
      ctx.font = '11px IBM Plex Mono, monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'alphabetic'
      ctx.fillText(`World · ${worldIdx.length} spins`, worldBox.x, worldBox.y - 8)

      if (hasMeds) {
        ctx.fillStyle = '#0a0c0e'
        ctx.fillRect(medBox.x - 4, medBox.y - labelH, medBox.w + 8, medBox.h + labelH + 4)
        ctx.strokeStyle = 'rgba(126, 200, 192, 0.45)'
        ctx.lineWidth = 1.2
        ctx.strokeRect(medBox.x - 4, medBox.y - 4, medBox.w + 8, medBox.h + 8)
        ctx.fillStyle = '#7ec8c0'
        ctx.font = '10px IBM Plex Mono, monospace'
        ctx.fillText(
          `Fabric Tax · mediators ×${meds.length}`,
          medBox.x,
          medBox.y - 20,
        )
        ctx.fillStyle = '#6a727a'
        ctx.font = '9px IBM Plex Mono, monospace'
        ctx.fillText(
          'auxiliary spins for bipartite Z1 fabric',
          medBox.x,
          medBox.y - 8,
        )
      }

      // Edges: world–world + any edge that touches a mediator
      const drawReceiptEdges = () => {
        const weights = graph.edge_weights
        const absMax = weights?.length
          ? Math.max(...weights.map((w) => Math.abs(w)), 1e-6)
          : 1
        const coupling = edgeMode === 'coupling'
        if (!coupling && !(showEdges && edgeMode === 'none')) return

        for (let e = 0; e < graph.edges.length; e++) {
          const [a, b] = graph.edges[e]
          const aMed = medSet.has(a)
          const bMed = medSet.has(b)
          const aWorld = worldPos.has(a)
          const bWorld = worldPos.has(b)
          // Skip edges that touch neither panel mapping
          if ((!aMed && !aWorld) || (!bMed && !bWorld)) continue
          // When no mediators, keep all world–world; with mediators draw:
          // world–world, world–mediator, mediator–mediator
          if (!hasMeds && (!aWorld || !bWorld)) continue

          const pa = screenOf(a)
          const pb = screenOf(b)
          if (!pa || !pb) continue

          const cross = (aMed && bWorld) || (bMed && aWorld)
          const medOnly = aMed && bMed
          const w = weights?.[e] ?? 1
          const mag = Math.abs(w) / absMax

          if (coupling) {
            ctx.strokeStyle = w >= 0 ? FERRO : ANTI
            ctx.lineWidth = 0.5 + mag * (cross ? 2.4 : 3.0)
            ctx.globalAlpha = cross ? 0.45 + mag * 0.4 : 0.3 + mag * 0.5
            if (cross) ctx.setLineDash([4, 3])
            else if (medOnly) ctx.setLineDash([2, 2])
            else ctx.setLineDash([])
          } else {
            ctx.strokeStyle = cross ? 'rgba(126, 200, 192, 0.55)' : EDGE
            ctx.lineWidth = cross ? 1.1 : 0.9
            ctx.globalAlpha = 1
            ctx.setLineDash(cross ? [4, 3] : [])
          }
          ctx.beginPath()
          ctx.moveTo(pa[0], pa[1])
          ctx.lineTo(pb[0], pb[1])
          ctx.stroke()
          ctx.setLineDash([])
        }
        ctx.globalAlpha = 1
        ctx.setLineDash([])
      }
      drawReceiptEdges()

      const rWorld = Math.max(
        4,
        Math.min(10, 140 / Math.sqrt(Math.max(1, worldIdx.length))),
      )
      for (const i of worldIdx) {
        const scr = screenOf(i)
        if (!scr) continue
        const [x, y] = scr
        const up = spins[i] === 1
        ctx.beginPath()
        ctx.arc(x, y, rWorld, 0, Math.PI * 2)
        ctx.fillStyle = up ? UP : DOWN
        ctx.fill()
        const inActive =
          (activeBlock === 0 && c0.has(i)) || (activeBlock === 1 && !c0.has(i))
        const selected = selectedSpin === i
        ctx.lineWidth = selected ? 2.8 : inActive ? 2.4 : 1
        ctx.strokeStyle = selected
          ? SELECT
          : inActive
            ? activeBlock === 0
              ? COL0
              : COL1
            : '#3a424c'
        if (clampSet.has(i) && !selected) ctx.strokeStyle = CLAMP
        ctx.stroke()
        hitRef.current.push({ i, x, y, r: rWorld + 4 })
      }

      if (hasMeds) {
        const rMed = Math.max(
          5,
          Math.min(11, 120 / Math.sqrt(Math.max(1, meds.length))),
        )
        for (const i of meds) {
          const scr = screenOf(i)
          if (!scr) continue
          const [x, y] = scr
          const up = spins[i] === 1
          ctx.beginPath()
          ctx.arc(x, y, rMed, 0, Math.PI * 2)
          // Phosphor family fill — mediators are first-class fabric tax nodes
          ctx.fillStyle = up ? MED_UP : MED_DOWN
          ctx.fill()
          const inActive =
            (activeBlock === 0 && c0.has(i)) || (activeBlock === 1 && !c0.has(i))
          const selected = selectedSpin === i
          ctx.lineWidth = selected ? 2.8 : inActive ? 2.4 : 1.2
          ctx.strokeStyle = selected
            ? SELECT
            : inActive
              ? activeBlock === 0
                ? COL0
                : COL1
              : 'rgba(126, 200, 192, 0.55)'
          ctx.stroke()
          if (selected) {
            ctx.beginPath()
            ctx.arc(x, y, rMed + 4, 0, Math.PI * 2)
            ctx.strokeStyle = 'rgba(126, 200, 192, 0.35)'
            ctx.lineWidth = 2
            ctx.stroke()
          }
          hitRef.current.push({ i, x, y, r: rMed + 5 })
        }
      }
    } else {
      const pad = 28
      const W = cssW - pad * 2
      const H = cssH - pad * 2
      const positions = graph.positions
      if (edgeMode === 'coupling') {
        drawCouplingEdges(
          Array.from({ length: graph.n_nodes }, (_, i) => i),
          pad,
          pad,
          W,
          H,
          (i) => positions[i] ?? [0.5, 0.5],
        )
      } else {
        ctx.strokeStyle = EDGE
        ctx.lineWidth = 1
        for (const [a, b] of graph.edges) {
          const [ax, ay] = positions[a]
          const [bx, by] = positions[b]
          ctx.beginPath()
          ctx.moveTo(pad + ax * W, pad + ay * H)
          ctx.lineTo(pad + bx * W, pad + by * H)
          ctx.stroke()
        }
      }
      const r =
        graph.layout === 'chain'
          ? 7
          : Math.max(5, Math.min(12, 140 / Math.sqrt(graph.n_nodes)))
      for (let i = 0; i < graph.n_nodes; i++) {
        const [px, py] = positions[i]
        const x = pad + px * W
        const y = pad + py * H
        const up = spins[i] === 1
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = up ? UP : DOWN
        ctx.fill()
        const inActive =
          (activeBlock === 0 && c0.has(i)) || (activeBlock === 1 && !c0.has(i))
        const selected = selectedSpin === i
        ctx.lineWidth = selected ? 2.8 : inActive ? 2.5 : 1
        ctx.strokeStyle = selected
          ? SELECT
          : inActive
            ? activeBlock === 0
              ? COL0
              : COL1
            : '#3a424c'
        if (clampSet.has(i) && !selected) ctx.strokeStyle = CLAMP
        ctx.stroke()
        hitRef.current.push({ i, x, y, r: r + 4 })
      }
    }
  }, [
    graph,
    state,
    activeBlock,
    showEdges,
    edgeMode,
    height,
    selectedSpin,
    graphAesthetic,
  ])

  const onClick = (ev: MouseEvent<HTMLCanvasElement>) => {
    if (!onSelectSpin || !ref.current) return
    const rect = ref.current.getBoundingClientRect()
    const x = ((ev.clientX - rect.left) / rect.width) * 560
    const y = ((ev.clientY - rect.top) / rect.height) * height
    let best: { i: number; d: number } | null = null
    for (const h of hitRef.current) {
      const d = Math.hypot(h.x - x, h.y - y)
      if (d <= h.r && (!best || d < best.d)) best = { i: h.i, d }
    }
    onSelectSpin(best ? best.i : null)
  }

  return (
    <canvas
      ref={ref}
      aria-label="Spin field visualization"
      onClick={onClick}
      style={{ cursor: onSelectSpin ? 'pointer' : 'default' }}
    />
  )
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}
