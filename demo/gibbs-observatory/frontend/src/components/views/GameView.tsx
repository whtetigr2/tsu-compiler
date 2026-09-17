import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * The vertical slice: a level you walk around, drawn from sampled spins.
 *
 * Every frame sends three numbers -- where the player is standing and which
 * way they face -- and gets back one visible depth per column, decoded from a
 * 1536-spin Ising model that a block-Gibbs sampler is continuously relaxing.
 * Nothing here raycasts to decide where a column stops. The ray march in the
 * program only says which cells lie along each column; the sampler decides
 * where each one ACTUALLY stops, and because neighbouring columns are coupled
 * it can overrule a cell it was told wrong about. A raycaster cannot: it stops
 * at the first thing it is told, right or wrong.
 *
 * Three things on screen are load-bearing rather than decorative.
 *
 * The CONSENT gate. Opening this runs a Python file on the reader's machine.
 * The page says so, names the file, and does nothing until they agree. The
 * file is described by reading it, never by importing it, so the description
 * itself costs nothing.
 *
 * The TRACE COUNT. A recompiled sampler costs ~300ms a frame and is otherwise
 * invisible, because the picture stays correct and merely crawls (R32). It is
 * on screen so it cannot hide.
 *
 * The SETTLE counter. The chain carries forward between frames, so the picture
 * keeps sharpening for a while after the player stops moving. That is a
 * property of the sampler, not a loading state, and calling it "loading" would
 * misdescribe what the reader is watching.
 */

const PROGRAM = 'visibility_world.py'

interface Opened {
  session: string
  name: string
  decoder: string
  n_spins: number
  n_couplings: number
  bipartite: boolean
  ports: { name: string; shape: number[]; mode: string; doc: string }[]
  sweeps: number
  /** The level, as 0/1 rows, when the program exposes one. Needed here
   *  because movement has to resolve on the frame the key is held; a
   *  round trip per keystroke would make the controls lag the picture. */
  world: number[][] | null
}

interface Stepped {
  frames: number
  traces: number
  decoded: number[] | null
}

interface Inspected {
  filename: string
  path: string
  kind: string
  why: string
  needs_consent: boolean
}

/** Movement, in world cells per second and radians per second. */
const SPEED = 3.2
const TURN = 2.4
/** Depth steps the program marches; the decoder returns an index into these. */
const N_DEPTH = 32

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = (data as { detail?: unknown })?.detail
    throw new Error(typeof d === 'string' ? d : JSON.stringify(d ?? 'failed'))
  }
  return data as T
}

export function GameView({ programsDir }: { programsDir?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [inspected, setInspected] = useState<Inspected | null>(null)
  const [opened, setOpened] = useState<Opened | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [hud, setHud] = useState({ frames: 0, traces: 0, fps: 0, still: 0 })

  // Everything the frame loop touches lives in refs. State would re-render the
  // component on every frame and the loop would restart with it.
  const pose = useRef({ x: 3.5, y: 3.5, a: 0.6 })
  const keys = useRef<Record<string, boolean>>({})
  const depths = useRef<number[]>([])
  const running = useRef(false)
  const still = useRef(0)

  const path = programsDir ? `${programsDir}/${PROGRAM}` : PROGRAM

  // -- the consent gate ------------------------------------------------

  const inspect = useCallback(async () => {
    setError(null)
    try {
      setInspected(await post<Inspected>('/api/program/inspect', { path }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [path])

  useEffect(() => { void inspect() }, [inspect])

  const open = async () => {
    setBusy(true)
    setError(null)
    try {
      setOpened(await post<Opened>('/api/program/open',
        { path, consent: true, sweeps: 8 }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // -- the frame loop ---------------------------------------------------

  useEffect(() => {
    if (!opened) return
    running.current = true
    let raf = 0
    let last = performance.now()
    let inFlight = false
    let fpsAt = performance.now()
    let fpsCount = 0

    const tick = async () => {
      if (!running.current) return
      const now = performance.now()
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now

      const k = keys.current
      const p = pose.current
      let moved = false
      if (k['arrowleft'] || k['q']) { p.a -= TURN * dt; moved = true }
      if (k['arrowright'] || k['e']) { p.a += TURN * dt; moved = true }
      let dx = 0
      let dy = 0
      if (k['w']) { dx += Math.cos(p.a); dy += Math.sin(p.a) }
      if (k['s']) { dx -= Math.cos(p.a); dy -= Math.sin(p.a) }
      if (k['a']) { dx += Math.sin(p.a); dy -= Math.cos(p.a) }
      if (k['d']) { dx -= Math.sin(p.a); dy += Math.cos(p.a) }
      if (dx || dy) {
        const n = Math.hypot(dx, dy) || 1
        const stepX = (dx / n) * SPEED * dt
        const stepY = (dy / n) * SPEED * dt
        // Resolved per axis so that walking into a wall at an angle slides
        // along it instead of stopping dead. Without this the player walks
        // INSIDE geometry, every ray hits at depth zero, and the screen fills
        // with a single flat colour -- which is exactly what happened the
        // first time this ran.
        if (!solid(opened.world, p.x + stepX, p.y)) p.x += stepX
        if (!solid(opened.world, p.x, p.y + stepY)) p.y += stepY
        moved = true
      }
      still.current = moved ? 0 : still.current + 1

      // One request in flight at a time. Queueing them would let input run
      // ahead of the sampler and the view would lag behind the keys.
      if (!inFlight) {
        inFlight = true
        try {
          const out = await post<Stepped>('/api/program/step', {
            session: opened.session,
            inputs: { pose: [p.x, p.y, p.a] },
          })
          if (out.decoded) depths.current = out.decoded
          fpsCount += 1
          const since = performance.now() - fpsAt
          if (since > 500) {
            setHud({
              frames: out.frames,
              traces: out.traces,
              fps: Math.round((fpsCount * 1000) / since),
              still: still.current,
            })
            fpsAt = performance.now()
            fpsCount = 0
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e))
          running.current = false
        } finally {
          inFlight = false
        }
      }

      draw(canvas.current, depths.current)
      raf = requestAnimationFrame(() => void tick())
    }

    raf = requestAnimationFrame(() => void tick())
    return () => {
      running.current = false
      cancelAnimationFrame(raf)
    }
  }, [opened])

  // Close the session when the pane goes away, so a walked-away-from game does
  // not keep a compiled sampler alive.
  useEffect(() => {
    if (!opened) return
    return () => {
      void fetch('/api/program/close', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: opened.session }),
      })
    }
  }, [opened])

  // -- input ------------------------------------------------------------

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      keys.current[e.key.toLowerCase()] = true
      if (['w', 'a', 's', 'd'].includes(e.key.toLowerCase())) e.preventDefault()
    }
    const up = (e: KeyboardEvent) => { keys.current[e.key.toLowerCase()] = false }
    const blur = () => { keys.current = {} }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [])

  const onMouseMove = (e: React.MouseEvent) => {
    if (document.pointerLockElement === canvas.current) {
      pose.current.a += e.nativeEvent.movementX * 0.0035
    }
  }

  // -- what the reader sees ---------------------------------------------

  if (!opened) {
    return (
      <div className="view game-consent">
        <h4>Walk around a level solved by sampling</h4>
        <p className="view-lead">
          This one is a <strong>program</strong>, not a model file: it builds a
          new lattice from wherever you are standing, every frame. Opening it
          runs the file on this machine.
        </p>
        {inspected ? (
          <dl className="consent-facts">
            <div><dt>file</dt><dd className="mono">{inspected.filename}</dd></div>
            <div><dt>read as</dt><dd>{inspected.kind}</dd></div>
            <div><dt>why</dt><dd>{inspected.why}</dd></div>
            <div><dt>run so far</dt><dd>nothing</dd></div>
          </dl>
        ) : null}
        <p className="view-footnote">
          The description above came from reading the file, not from running
          it. There is no sandbox and none is claimed; this is the same deal a
          spreadsheet with macros makes. Open it only if you trust where it
          came from.
        </p>
        {error ? <p className="status-error">{error}</p> : null}
        <button type="button" className="btn primary" disabled={busy}
                onClick={() => void open()}>
          {busy ? 'Starting…' : 'Run it'}
        </button>
      </div>
    )
  }

  return (
    <div className="view game-view">
      <canvas
        ref={canvas}
        className="game-canvas"
        width={720}
        height={420}
        tabIndex={0}
        onMouseMove={onMouseMove}
        onClick={() => canvas.current?.requestPointerLock()}
      />

      <div className="game-hud mono">
        <span><b>{hud.fps}</b> fps</span>
        <span><b>{opened.n_spins.toLocaleString()}</b> spins</span>
        <span><b>{opened.n_couplings.toLocaleString()}</b> couplings</span>
        <span><b>{opened.sweeps}</b> sweeps/frame</span>
        <span className={hud.traces > 1 ? 'bad' : ''}>
          <b>{hud.traces}</b> compile{hud.traces === 1 ? '' : 's'}
        </span>
        <span><b>{hud.frames}</b> frames</span>
        {hud.still > 20 ? <span className="settling">settling</span> : null}
      </div>

      <p className="view-footnote">
        WASD to move, Q/E or the arrow keys to turn, click the view for mouse
        look. Every frame sends three numbers and receives one visible depth
        per column, decoded from {opened.n_spins.toLocaleString()} spins.
        Nothing here decides where a column stops by raycasting: the sampler
        does, and because neighbouring columns are coupled it can overrule a
        cell it was told wrong about.
      </p>
      <p className="view-footnote">
        <b>compiles</b> should stay at 1. More means the sampler recompiled,
        which costs about 300ms a frame and would otherwise be invisible,
        because the picture stays correct and merely crawls (R32).
        {' '}<b>settling</b> means the chain is still sharpening the answer
        after you stopped moving, which is what a continuing chain does rather
        than a loading state. Simulation on CPU through THRML.
      </p>
      {error ? <p className="status-error">{error}</p> : null}
    </div>
  )
}

/**
 * Columns to a picture. Depth index to wall height by 1/d, the standard
 * projection, with the far plane drawn as horizon rather than as a wall so a
 * column that hit nothing does not read as a surface at maximum distance.
 */
/**
 * Whether a world cell is solid, with a margin so the camera never sits flush
 * inside a wall face. A camera exactly on the boundary sees depth zero in
 * every column, which draws as a flat fill rather than as a wall.
 */
const RADIUS = 0.25

function solid(world: number[][] | null, x: number, y: number): boolean {
  if (!world) return false
  const h = world.length
  const w = world[0]?.length ?? 0
  for (const [ox, oy] of [[-RADIUS, 0], [RADIUS, 0], [0, -RADIUS], [0, RADIUS]]) {
    const ix = Math.floor(x + ox)
    const iy = Math.floor(y + oy)
    if (ix < 0 || iy < 0 || ix >= w || iy >= h) return true
    if (world[iy][ix]) return true
  }
  return false
}

function draw(canvas: HTMLCanvasElement | null, depths: number[]) {
  if (!canvas || !depths.length) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const W = canvas.width
  const H = canvas.height

  const sky = ctx.createLinearGradient(0, 0, 0, H / 2)
  sky.addColorStop(0, '#0b0f14')
  sky.addColorStop(1, '#16202a')
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, W, H / 2)

  const floor = ctx.createLinearGradient(0, H / 2, 0, H)
  floor.addColorStop(0, '#11181f')
  floor.addColorStop(1, '#070a0d')
  ctx.fillStyle = floor
  ctx.fillRect(0, H / 2, W, H / 2)

  const colW = W / depths.length
  for (let i = 0; i < depths.length; i++) {
    const d = depths[i]
    if (d >= N_DEPTH - 1) continue          // hit nothing: leave the horizon
    // +1 so the nearest column is not divided by zero.
    const h = Math.min(H, (H * 1.15) / (d + 1))
    const y = (H - h) / 2
    // Nearer is brighter, which is the only depth cue a column view has.
    const t = 1 - d / N_DEPTH
    const r = Math.round(30 + 150 * t)
    const g = Math.round(70 + 180 * t)
    const b = Math.round(70 + 160 * t)
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(i * colW, y, Math.ceil(colW) + 1, h)
  }
}
