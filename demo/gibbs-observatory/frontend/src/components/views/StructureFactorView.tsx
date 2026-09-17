import { useEffect, useRef } from 'react'
import type { BatchPayload } from '../../types'

interface Props {
  batch: BatchPayload | null
}

/**
 * S(k), the lattice in reciprocal space.
 *
 * The squared magnitude of the spatial Fourier transform of the spin field,
 * averaged over draws. It is how physics asks what kind of order a system has,
 * and it answers with a picture: diffuse haze when hot, a ring blooming near a
 * critical point, sharp Bragg peaks once order sets in. Where the peak sits
 * says WHICH order, so the peak is labelled rather than left to be read off.
 *
 * Intensity is drawn on a log scale. Bragg peaks are orders of magnitude above
 * the background, and on a linear scale everything except the peak is black.
 */
export function StructureFactorView({ batch }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const sf = batch?.structure_factor ?? null

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !sf?.available || !sf.values) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = sf.width
    const H = sf.height
    const cell = Math.max(2, Math.floor(Math.min(520 / W, 520 / H)))
    canvas.width = W * cell
    canvas.height = H * cell
    canvas.style.width = `${W * cell}px`
    canvas.style.height = `${H * cell}px`

    // Log scale: peaks sit orders of magnitude above the haze, and linear
    // scaling renders everything but the peak as black.
    const top = Math.log1p(sf.max || 1)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const t = top > 0 ? Math.log1p(sf.values[y][x]) / top : 0
        ctx.fillStyle = ramp(Math.max(0, Math.min(1, t)))
        ctx.fillRect(x * cell, y * cell, cell, cell)
      }
    }

    // Mark the origin. k = 0 is excluded from the peak search, so saying where
    // it is stops the empty centre reading as a hole in the data.
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'
    ctx.lineWidth = 1
    const cx = Math.floor(W / 2) * cell + cell / 2
    const cy = Math.floor(H / 2) * cell + cell / 2
    ctx.beginPath()
    ctx.moveTo(cx - 5, cy)
    ctx.lineTo(cx + 5, cy)
    ctx.moveTo(cx, cy - 5)
    ctx.lineTo(cx, cy + 5)
    ctx.stroke()
  }, [sf])

  if (!batch) {
    return <p className="empty-hint">Run the sampler to see S(k).</p>
  }

  if (!sf) {
    return (
      <div className="view">
        <p className="view-lead">
          This model&rsquo;s spins do not sit on a lattice, so it has no
          structure factor.
        </p>
        <p className="view-footnote">
          The transform needs to know where each spin sits in the model&rsquo;s
          own space. That comes from the site names a grid program writes. Using
          placement coordinates instead would picture where the compiler laid
          spins out on the die, which is a different thing and not the physics.
          Open a grid program, such as &ldquo;Two metals settling into a
          pattern&rdquo;, to see this view work.
        </p>
      </div>
    )
  }

  if (!sf.available) {
    return (
      <div className="view">
        <p className="empty-hint">{sf.reason}</p>
      </div>
    )
  }

  const [kx, ky] = sf.peak_k_over_pi ?? [0, 0]

  return (
    <div className="view sk-view">
      <p className="view-lead">
        The spin field in reciprocal space, averaged over {sf.n_draws} draws.
        Diffuse haze means disorder. A sharp peak means order, and where the
        peak sits says which kind.
      </p>

      <div className="sk-body">
        <canvas ref={ref} className="sk-canvas" />
        <div className="sk-side">
          <h5>Peak</h5>
          <p className="sk-peak">
            k = ({fmtPi(kx)}, {fmtPi(ky)})
          </p>
          <p className="sk-label">{sf.peak_label}</p>

          <h5>Axes</h5>
          <p>
            k = 0 at the centre, marked with a cross. Edges are ±π, so the
            corners are (±π, ±π).
          </p>

          <h5>Scale</h5>
          <p>
            Logarithmic. A Bragg peak is orders of magnitude above the
            background and a linear scale would render everything else black.
          </p>

          <h5>Where the coordinates come from</h5>
          <p className="sk-source">{sf.source}</p>
        </div>
      </div>

      <p className="view-footnote">{sf.note} Simulation on CPU through THRML.</p>
    </div>
  )
}

function fmtPi(v: number): string {
  if (Math.abs(v) < 1e-9) return '0'
  if (Math.abs(Math.abs(v) - 1) < 1e-9) return v < 0 ? '-π' : 'π'
  return `${v.toFixed(2)}π`
}

/** Dark ground to phosphor, so intensity reads as brightness. */
function ramp(t: number): string {
  const stops: [number, number, number][] = [
    [10, 12, 14],
    [26, 43, 51],
    [74, 143, 137],
    [126, 200, 192],
    [220, 252, 248],
  ]
  const pos = t * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(pos))
  const f = pos - i
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}
