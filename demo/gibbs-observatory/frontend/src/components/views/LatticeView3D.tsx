import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import type { BatchPayload, GraphPayload } from '../../types'

interface Props {
  graph: GraphPayload | null
  batch: BatchPayload | null
}

type Showing = 'draw' | 'mean'
/**
 * Two legitimate ways to place a spin, and they answer different questions.
 *
 * 'chip' uses the compiler's placement coordinates: where the model was laid
 * out on the die. Truthful about the hardware and useless for seeing shape,
 * because a 4-cube laid flat on silicon looks like a tangle.
 *
 * 'structure' ignores placement and lets the graph find its own arrangement,
 * which is how a hypercube looks like a hypercube. Neither is a claim about
 * physical position; the caption says which you are looking at.
 */
type LayoutMode = 'structure' | 'chip' | 'hypercube'

/**
 * The lattice in three dimensions, live.
 *
 * Two honesty rules govern everything here, and both are on screen rather than
 * in a comment.
 *
 * A DRAW IS NOT A DISTRIBUTION. One sample and a mean over many look similar and
 * mean entirely different things, so the viewport says which it is showing.
 *
 * A LAYOUT IS NOT A POSITION. Where a receipt carries placement coordinates
 * those are used and labelled as the compiler's placement. Otherwise positions
 * come from a spring layout computed here, which is a drawing convenience and
 * says so. Neither is a claim about where anything sits on silicon.
 */
export function LatticeView3D({ graph, batch }: Props) {
  const mount = useRef<HTMLDivElement | null>(null)
  const spinsRef = useRef<THREE.InstancedMesh | null>(null)
  const [showing, setShowing] = useState<Showing>('draw')
  const dim = graph ? hypercubeDimension(graph) : 0
  const [layoutMode, setLayoutMode] = useState<LayoutMode>(
    dim >= 4 ? 'hypercube' : 'structure')
  const [layoutKind, setLayoutKind] = useState<string>('unavailable')

  // Scene setup. Runs once per graph, because node count changes the geometry.
  useEffect(() => {
    const el = mount.current
    if (!el || !graph) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0b0c)

    const camera = new THREE.PerspectiveCamera(
      50, el.clientWidth / Math.max(el.clientHeight, 1), 0.1, 2000)
    camera.position.set(0, 0, 90)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(el.clientWidth, el.clientHeight)
    el.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.55))
    const key = new THREE.DirectionalLight(0xffffff, 0.9)
    key.position.set(40, 60, 80)
    scene.add(key)

    const n = graph.n_nodes
    const n0 = n
    const spinning = layoutMode === 'hypercube' && dim >= 3
    const pos = spinning
      ? { xyz: hypercubeLayout(n0, dim, 0),
          kind: `a ${dim}-cube, projected from ${dim} dimensions into three` }
      : layoutFor(graph, layoutMode)
    setLayoutKind(pos.kind)

    // Spins as instanced spheres: one draw call regardless of node count.
    const geo = new THREE.SphereGeometry(1.1, 16, 12)
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.45, metalness: 0.1,
    })
    const mesh = new THREE.InstancedMesh(geo, mat, n)
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    const m4 = new THREE.Matrix4()
    for (let i = 0; i < n; i++) {
      m4.setPosition(pos.xyz[i][0], pos.xyz[i][1], pos.xyz[i][2])
      mesh.setMatrixAt(i, m4)
      mesh.setColorAt(i, new THREE.Color(0x4a8f89))
    }
    mesh.instanceMatrix.needsUpdate = true
    scene.add(mesh)
    spinsRef.current = mesh

    // Edges as one line segment buffer.
    let edgePos: Float32Array | null = null
    let edgeGeo: THREE.BufferGeometry | null = null
    if (graph.edges.length) {
      const pts = new Float32Array(graph.edges.length * 6)
      graph.edges.forEach(([a, b], k) => {
        pts.set(pos.xyz[a], k * 6)
        pts.set(pos.xyz[b], k * 6 + 3)
      })
      const lg = new THREE.BufferGeometry()
      lg.setAttribute('position', new THREE.BufferAttribute(pts, 3))
      edgePos = pts
      edgeGeo = lg
      scene.add(new THREE.LineSegments(
        lg, new THREE.LineBasicMaterial({
          color: 0x2a3f3c, transparent: true, opacity: 0.55,
        })))
    }

    // Orbit by drag. Written here rather than pulled from examples/, which is
    // not part of the three package's typed entry points.
    let dragging = false
    let px = 0
    let py = 0
    let yaw = 0.5
    let pitch = 0.25
    // Fit the camera to what is actually there. A 16-spin hypercube and an
    // 11,200-spin visibility lattice need very different distances, and a fixed
    // one leaves the small model as a speck and clips the large one.
    let radius = 1
    for (const [x, y, z] of pos.xyz) {
      radius = Math.max(radius, Math.hypot(x, y, z))
    }
    let dist = Math.max(24, radius * 2.6)
    const onDown = (e: PointerEvent) => {
      dragging = true
      px = e.clientX
      py = e.clientY
    }
    const onUp = () => { dragging = false }
    const onMove = (e: PointerEvent) => {
      if (!dragging) return
      yaw += (e.clientX - px) * 0.008
      pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - py) * 0.008))
      px = e.clientX
      py = e.clientY
    }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      dist = Math.max(6, Math.min(radius * 12, dist + e.deltaY * radius * 0.004))
    }
    const cv = renderer.domElement
    cv.addEventListener('pointerdown', onDown)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointermove', onMove)
    cv.addEventListener('wheel', onWheel, { passive: false })

    // Edge buffer kept so the 4D rotation can move the lines with the points.
    let raf = 0
    const t0 = performance.now()
    const m4b = new THREE.Matrix4()
    const tick = () => {
      if (spinning) {
        const t = ((performance.now() - t0) / 1000) * 0.35
        const xyz = hypercubeLayout(n, dim, t)
        for (let i = 0; i < n; i++) {
          m4b.setPosition(xyz[i][0], xyz[i][1], xyz[i][2])
          mesh.setMatrixAt(i, m4b)
        }
        mesh.instanceMatrix.needsUpdate = true
        if (edgePos && edgeGeo) {
          graph.edges.forEach(([a, b], k) => {
            edgePos.set(xyz[a], k * 6)
            edgePos.set(xyz[b], k * 6 + 3)
          })
          edgeGeo.attributes.position.needsUpdate = true
        }
      }
      camera.position.set(
        dist * Math.cos(pitch) * Math.sin(yaw),
        dist * Math.sin(pitch),
        dist * Math.cos(pitch) * Math.cos(yaw))
      camera.lookAt(0, 0, 0)
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    const ro = new ResizeObserver(() => {
      if (!el.clientWidth) return
      camera.aspect = el.clientWidth / Math.max(el.clientHeight, 1)
      camera.updateProjectionMatrix()
      renderer.setSize(el.clientWidth, el.clientHeight)
    })
    ro.observe(el)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      cv.removeEventListener('pointerdown', onDown)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointermove', onMove)
      cv.removeEventListener('wheel', onWheel)
      renderer.dispose()
      geo.dispose()
      mat.dispose()
      if (cv.parentNode === el) el.removeChild(cv)
      spinsRef.current = null
    }
  }, [graph, layoutMode, dim])

  // Colour by state, every batch.
  useEffect(() => {
    const mesh = spinsRef.current
    if (!mesh || !batch) return
    const values = showing === 'draw'
      ? batch.last_state
      : columnMeans(batch.states)
    if (!values?.length) return

    const up = new THREE.Color(0x7ec8c0)
    const down = new THREE.Color(0x1d2b33)
    const c = new THREE.Color()
    const count = Math.min(mesh.count, values.length)
    for (let i = 0; i < count; i++) {
      // last_state is ±1; a column mean is in [-1, 1]. Both map the same way.
      const t = (Number(values[i]) + 1) / 2
      c.copy(down).lerp(up, Math.max(0, Math.min(1, t)))
      mesh.setColorAt(i, c)
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [batch, showing])

  if (!graph) {
    return <p className="empty-hint">Open an example to see its lattice.</p>
  }

  return (
    <div className="view lattice3d-view">
      <div className="lattice3d-bar">
        <div className="seg">
          <button
            type="button"
            className={showing === 'draw' ? 'active' : ''}
            onClick={() => setShowing('draw')}
          >
            One draw
          </button>
          <button
            type="button"
            className={showing === 'mean' ? 'active' : ''}
            onClick={() => setShowing('mean')}
          >
            Mean over batch
          </button>
        </div>
        <div className="seg">
          {dim >= 4 ? (
            <button
              type="button"
              className={layoutMode === 'hypercube' ? 'active' : ''}
              onClick={() => setLayoutMode('hypercube')}
            >
              {dim}-cube
            </button>
          ) : null}
          <button
            type="button"
            className={layoutMode === 'structure' ? 'active' : ''}
            onClick={() => setLayoutMode('structure')}
          >
            Structure
          </button>
          <button
            type="button"
            className={layoutMode === 'chip' ? 'active' : ''}
            onClick={() => setLayoutMode('chip')}
          >
            Chip placement
          </button>
        </div>
        <span className="lattice3d-note">
          {showing === 'draw'
            ? 'A single sample. Not a distribution.'
            : `Mean over ${batch?.states?.length ?? 0} draws. A spin near the middle of the scale is uncertain, not half-way up.`}
        </span>
      </div>

      <div className="lattice3d-canvas" ref={mount} />

      <p className="view-footnote">
        Layout: {layoutKind}. Positions are a drawing, not where anything sits on
        silicon. Drag to orbit, scroll to zoom. Simulation on CPU through THRML.
      </p>
    </div>
  )
}

function columnMeans(states: number[][] | undefined): number[] {
  if (!states?.length) return []
  const n = states[0].length
  const out = new Array(n).fill(0)
  for (const row of states) {
    for (let i = 0; i < n; i++) out[i] += Number(row[i])
  }
  return out.map((v) => v / states.length)
}

/**
 * Where to draw each spin.
 *
 * The compiler's placement coordinates when the receipt has them, lifted into a
 * plane. Otherwise a deterministic spring layout, seeded by node index so the
 * same graph always draws the same way: a picture that rearranges itself on
 * every reload is unreadable.
 */
function layoutFor(graph: GraphPayload, mode: LayoutMode):
    { xyz: [number, number, number][]; kind: string } {
  const n = graph.n_nodes
  if (mode === 'chip' && graph.positions?.length === n) {
    const xs = graph.positions.map((p) => p[0])
    const ys = graph.positions.map((p) => p[1])
    const sx = spread(xs)
    const sy = spread(ys)
    return {
      kind: "the compiler's placement, drawn flat",
      xyz: graph.positions.map(([x, y], i) => [
        (x - sx.mid) * sx.scale,
        (y - sy.mid) * sy.scale,
        // A slight lift by colour block, so the two halves of the chromatic
        // schedule are separable by eye without implying a third coordinate.
        graph.color1?.includes(i) ? 4 : -4,
      ]),
    }
  }

  // Spring layout in 3D. Deterministic: no Math.random anywhere.
  const pts: [number, number, number][] = []
  for (let i = 0; i < n; i++) {
    const golden = Math.PI * (3 - Math.sqrt(5))
    const y = 1 - (i / Math.max(n - 1, 1)) * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const th = golden * i
    pts.push([Math.cos(th) * r * 30, y * 30, Math.sin(th) * r * 30])
  }
  for (let pass = 0; pass < 120; pass++) {
    for (const [a, b] of graph.edges) {
      for (let d = 0; d < 3; d++) {
        const delta = (pts[b][d] - pts[a][d]) * 0.012
        pts[a][d] += delta
        pts[b][d] -= delta
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d2 = 0
        for (let d = 0; d < 3; d++) {
          const dd = pts[i][d] - pts[j][d]
          d2 += dd * dd
        }
        const push = 22 / Math.max(d2, 4)
        for (let d = 0; d < 3; d++) {
          const dd = pts[i][d] - pts[j][d]
          pts[i][d] += dd * push * 0.02
          pts[j][d] -= dd * push * 0.02
        }
      }
    }
  }
  return { kind: 'a spring layout computed here, not from the receipt', xyz: pts }
}

function spread(v: number[]) {
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  const range = hi - lo || 1
  return { mid: (lo + hi) / 2, scale: 60 / range }
}

/**
 * Is this graph an n-dimensional hypercube?
 *
 * Checked, not assumed. A Q(d) has 2^d vertices, every vertex has degree d, and
 * an edge joins exactly the pairs whose binary indices differ in one bit. If all
 * three hold, the vertex index IS a coordinate in d dimensions, and the model
 * can be drawn as the object it describes rather than as an abstract graph.
 *
 * Returns the dimension, or 0 when it is not a hypercube. Offering a 4D view for
 * a graph that is not a 4-cube would be a decoration pretending to be a
 * measurement.
 */
export function hypercubeDimension(graph: GraphPayload): number {
  const n = graph.n_nodes
  if (n < 2 || (n & (n - 1)) !== 0) return 0
  const d = Math.log2(n)
  if (graph.edges.length !== (d * n) / 2) return 0
  const degree = new Array<number>(n).fill(0)
  for (const [a, b] of graph.edges) {
    if (popcount(a ^ b) !== 1) return 0
    degree[a]++
    degree[b]++
  }
  return degree.every((k) => k === d) ? d : 0
}

function popcount(x: number): number {
  let c = 0
  for (let v = x; v; v >>= 1) c += v & 1
  return c
}

/**
 * A hypercube vertex, projected from d dimensions into three.
 *
 * The coordinate is the vertex's own binary index, mapped to -1 or +1 per bit.
 * Dimensions beyond the third are removed by perspective division, one at a
 * time, the same way a 3D scene is projected onto a 2D screen. That is what
 * gives a tumbling tesseract its turning-inside-out look: the inner cube is not
 * smaller, it is further away along the fourth axis.
 *
 * `t` rotates the object in the two planes that mix the fourth axis with the
 * others, which is the rotation a 3D viewer cannot perform and the reason the
 * shape appears to move through itself.
 */
function hypercubeLayout(n: number, d: number, t: number):
    [number, number, number][] {
  const out: [number, number, number][] = []
  for (let i = 0; i < n; i++) {
    const c: number[] = []
    for (let bit = 0; bit < d; bit++) c.push((i >> bit) & 1 ? 1 : -1)

    // Rotate in the x-w and y-z planes. The first is the genuinely
    // four-dimensional one.
    if (d >= 4) {
      const ca = Math.cos(t)
      const sa = Math.sin(t)
      const x = c[0]
      const w = c[3]
      c[0] = x * ca - w * sa
      c[3] = x * sa + w * ca

      const cb = Math.cos(t * 0.6)
      const sb = Math.sin(t * 0.6)
      const y = c[1]
      const z = c[2]
      c[1] = y * cb - z * sb
      c[2] = y * sb + z * cb
    }

    // Collapse the extra axes by perspective, highest first.
    let [x, y, z] = [c[0], c[1] ?? 0, c[2] ?? 0]
    for (let k = d - 1; k >= 3; k--) {
      const wDist = 3.2
      const k4 = wDist / (wDist - c[k])
      x *= k4
      y *= k4
      z *= k4
    }
    out.push([x * 16, y * 16, z * 16])
  }
  return out
}
