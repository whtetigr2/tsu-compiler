import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import type { BatchPayload, GraphPayload, LangMode, ReceiptInspect } from '../../types'
import { pcaProject } from '../../lib/pca'

/** Exact 3D embedding only claimed for small models (n ≤ this). */
export const STATE_SPACE_3D_MAX_SPINS = 16

interface Props {
  graph: GraphPayload | null
  batch: BatchPayload | null
  receipt: ReceiptInspect | null
  lang: LangMode
}

function collectSamples(batch: BatchPayload | null, cap = 256): number[][] {
  if (!batch) return []
  const out: number[][] = []
  if (Array.isArray(batch.states)) {
    for (const s of batch.states) {
      if (Array.isArray(s) && s.length) out.push(s.map(Number))
    }
  }
  if (batch.last_state?.length) {
    out.push(batch.last_state.map(Number))
  }
  // Dedup identical consecutive vectors lightly
  const uniq: number[][] = []
  let prev = ''
  for (const row of out) {
    const key = row.join(',')
    if (key !== prev) {
      uniq.push(row)
      prev = key
    }
  }
  return uniq.slice(-cap)
}

function normalizeCoords(coords: number[][], dims: number): number[][] {
  if (!coords.length) return coords
  const mins = new Array(dims).fill(Infinity)
  const maxs = new Array(dims).fill(-Infinity)
  for (const c of coords) {
    for (let i = 0; i < dims; i++) {
      mins[i] = Math.min(mins[i], c[i] ?? 0)
      maxs[i] = Math.max(maxs[i], c[i] ?? 0)
    }
  }
  return coords.map((c) =>
    c.slice(0, dims).map((v, i) => {
      const span = maxs[i] - mins[i] || 1
      return ((v - mins[i]) / span) * 2 - 1
    }),
  )
}

export function StateSpaceView({ graph, batch, receipt, lang: _lang }: Props) {
  const n =
    receipt?.spins.n_nodes ??
    graph?.n_nodes ??
    batch?.last_state?.length ??
    0
  const large = n > STATE_SPACE_3D_MAX_SPINS
  const samples = useMemo(() => collectSamples(batch), [batch])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [yaw, setYaw] = useState(0.6)
  const [pitch, setPitch] = useState(0.35)
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(
    null,
  )

  const projection = useMemo(() => {
    if (samples.length < 2) {
      return { coords: [] as number[][], hint: 'need ≥2 distinct recent samples, run/step the sampler' }
    }
    const k = large ? 2 : 3
    const { coords, explainedHint } = pcaProject(samples, k as 2 | 3)
    return { coords: normalizeCoords(coords, k), hint: explainedHint }
  }, [samples, large])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const w = canvas.width
    const h = canvas.height
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#0e1420'
    ctx.fillRect(0, 0, w, h)

    // axes
    ctx.strokeStyle = '#243044'
    ctx.beginPath()
    ctx.moveTo(40, h / 2)
    ctx.lineTo(w - 20, h / 2)
    ctx.moveTo(w / 2, 20)
    ctx.lineTo(w / 2, h - 20)
    ctx.stroke()

    const cx = w / 2
    const cy = h / 2
    const scale = Math.min(w, h) * 0.38

    const pts = projection.coords
    if (!pts.length) {
      ctx.fillStyle = '#8b9bb4'
      ctx.font = '13px IBM Plex Sans, sans-serif'
      ctx.fillText(projection.hint, 24, 32)
      return
    }

    if (large) {
      // Honest 2D sample cloud only
      ctx.fillStyle = '#8b9bb4'
      ctx.font = '12px IBM Plex Mono, monospace'
      ctx.fillText('2D PCA sample cloud (3D refused)', 24, 28)
      for (let i = 0; i < pts.length; i++) {
        const [x, y] = pts[i]
        const px = cx + x * scale
        const py = cy - y * scale
        const t = i / Math.max(1, pts.length - 1)
        ctx.fillStyle = `rgba(62, 224, 176, ${0.35 + 0.55 * t})`
        ctx.beginPath()
        ctx.arc(px, py, 3.2, 0, Math.PI * 2)
        ctx.fill()
      }
      return
    }

    // 3D rotate + project
    const cyA = Math.cos(yaw)
    const syA = Math.sin(yaw)
    const cp = Math.cos(pitch)
    const sp = Math.sin(pitch)
    const projected = pts.map(([x, y, z], i) => {
      const x1 = x * cyA + (z ?? 0) * syA
      const z1 = -x * syA + (z ?? 0) * cyA
      const y1 = y * cp - z1 * sp
      const z2 = y * sp + z1 * cp
      const depth = z2
      return { x: cx + x1 * scale, y: cy - y1 * scale, depth, i }
    })
    projected.sort((a, b) => a.depth - b.depth)

    ctx.fillStyle = '#8b9bb4'
    ctx.font = '12px IBM Plex Mono, monospace'
    ctx.fillText('3D PCA embedding · drag to orbit', 24, 28)

    // simple axes in 3D
    const axisLen = 0.9
    const axes3: [number, number, number, string][] = [
      [axisLen, 0, 0, '#f72585'],
      [0, axisLen, 0, '#3ee0b0'],
      [0, 0, axisLen, '#4cc9f0'],
    ]
    for (const [ax, ay, az, color] of axes3) {
      const x1 = ax * cyA + az * syA
      const z1 = -ax * syA + az * cyA
      const y1 = ay * cp - z1 * sp
      ctx.strokeStyle = color
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(cx + x1 * scale, cy - y1 * scale)
      ctx.stroke()
    }

    for (const p of projected) {
      const t = (p.depth + 1) / 2
      const r = 2.5 + 2.5 * Math.max(0, Math.min(1, t))
      ctx.fillStyle = `rgba(76, 201, 240, ${0.35 + 0.55 * Math.max(0, Math.min(1, t))})`
      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [projection, yaw, pitch, large])

  const onPointerDown = (e: ReactPointerEvent) => {
    if (large) return
    drag.current = { x: e.clientX, y: e.clientY, yaw, pitch }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e: ReactPointerEvent) => {
    if (!drag.current || large) return
    const dx = e.clientX - drag.current.x
    const dy = e.clientY - drag.current.y
    setYaw(drag.current.yaw + dx * 0.01)
    setPitch(Math.max(-1.2, Math.min(1.2, drag.current.pitch + dy * 0.01)))
  }
  const onPointerUp = () => {
    drag.current = null
  }

  return (
    <div className="view statespace-view">
      <div className="view-caption">
        State space · n={n || ', '} spins
        {large ? (
          <>
            {' '}
            · <span className="status-error">3D refused</span>
          </>
        ) : (
          <> · 3D embedding (small exact models, n≤{STATE_SPACE_3D_MAX_SPINS})</>
        )}
      </div>

      {large ? (
        <div className="refuse-banner">
          <strong>Full state-space 3D unavailable for this model.</strong>
          <p>
            n = {n} exceeds the honest limit (n≤{STATE_SPACE_3D_MAX_SPINS}). A complete
            configuration space has size 2^{n}, we will not fake a full embedding.
            Showing a <em>2D PCA projection of recent sample vectors only</em> (live
            THRML draws), not the Boltzmann landscape over all states.
          </p>
        </div>
      ) : (
        <div className="ok-banner">
          Small model (n≤{STATE_SPACE_3D_MAX_SPINS}): 3D PCA / sample embedding of recent
          spin vectors. This is an embedding of observed samples, not an exhaustive
          enumeration of 2^{n} unless n is tiny and you have enough draws.
        </div>
      )}

      <div className="canvas-wrap statespace-canvas">
        <canvas
          ref={canvasRef}
          width={900}
          height={480}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          style={{ width: '100%', height: 'auto', touchAction: 'none', cursor: large ? 'default' : 'grab' }}
        />
      </div>
      <p className="empty-hint mono">{projection.hint} · samples={samples.length}</p>
      <p className="claim-chip">JAX/THRML simulation, not Extropic silicon</p>
    </div>
  )
}
