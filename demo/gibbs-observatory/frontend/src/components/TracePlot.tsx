import { useEffect, useRef } from 'react'

interface Props {
  data: number[]
  label: string
  unit?: string
  color?: string
  height?: number
  /** Sweep number of the most recent sample, so the x axis means something. */
  step?: number
}

/**
 * A live trace with an axis, which is the whole point.
 *
 * The sparkline this replaces drew a bare line auto-scaled to the visible
 * window, with no labels and no scale. That is actively misleading once a model
 * equilibrates: the range collapses, the auto-scale expands to fill the box, and
 * pure noise is drawn as dramatic oscillation. A reader has no way to tell a
 * trace swinging across half its value from one wobbling in the fourth decimal.
 *
 * So this draws the y range, and says how large the visible variation is
 * relative to the value. A caption reading "range is 0.03% of the value" is the
 * difference between "still equilibrating" and "settled".
 */
export function TracePlot({
  data, label, unit, color = '#7ec8c0', height = 120, step,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth || 320
    const h = height
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.height = `${h}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    const padL = 52
    const padR = 8
    const padT = 8
    const padB = 18
    const plotW = Math.max(1, w - padL - padR)
    const plotH = Math.max(1, h - padT - padB)

    const css = getComputedStyle(document.documentElement)
    const mute = css.getPropertyValue('--text-mute').trim() || '#5c656e'
    const grid = css.getPropertyValue('--border-subtle').trim() || '#1e2228'

    if (!data.length) {
      ctx.fillStyle = mute
      ctx.font = '11px ui-monospace, monospace'
      ctx.fillText('no samples yet', padL, padT + plotH / 2)
      return
    }

    const min = Math.min(...data)
    const max = Math.max(...data)
    const span = max - min || Math.abs(max) * 1e-6 || 1

    // Axis box and two gridlines, so the eye has a reference.
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padL, padT)
    ctx.lineTo(padL, padT + plotH)
    ctx.lineTo(padL + plotW, padT + plotH)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(padL, padT + plotH / 2)
    ctx.lineTo(padL + plotW, padT + plotH / 2)
    ctx.stroke()

    // Y labels: the actual range, so a flat line reads as flat.
    ctx.fillStyle = mute
    ctx.font = '10px ui-monospace, monospace'
    ctx.textAlign = 'right'
    ctx.fillText(fmtTick(max), padL - 6, padT + 8)
    ctx.fillText(fmtTick(min), padL - 6, padT + plotH)
    ctx.textAlign = 'left'
    ctx.fillText(
      step != null ? `sweep ${step}` : `${data.length} samples`,
      padL, h - 5)

    const x = (i: number) =>
      padL + (i / Math.max(1, data.length - 1)) * plotW
    const y = (v: number) =>
      padT + plotH - ((v - min) / span) * plotH

    // Fill under the curve, then the curve.
    ctx.beginPath()
    ctx.moveTo(x(0), padT + plotH)
    data.forEach((v, i) => ctx.lineTo(x(i), y(v)))
    ctx.lineTo(x(data.length - 1), padT + plotH)
    ctx.closePath()
    ctx.fillStyle = color + '1f'
    ctx.fill()

    ctx.beginPath()
    data.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))))
    ctx.strokeStyle = color
    ctx.lineWidth = 1.4
    ctx.stroke()

    // The newest sample, marked. Without it the eye cannot find "now".
    const lastX = x(data.length - 1)
    const lastY = y(data[data.length - 1])
    ctx.beginPath()
    ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
  }, [data, color, height, step])

  const last = data.length ? data[data.length - 1] : null
  const min = data.length ? Math.min(...data) : null
  const max = data.length ? Math.max(...data) : null
  const rel =
    last != null && min != null && max != null && Math.abs(last) > 1e-12
      ? ((max - min) / Math.abs(last)) * 100
      : null

  return (
    <div className="trace-plot">
      <div className="trace-head">
        <span className="trace-label">{label}</span>
        <span className="trace-last">
          {last == null ? (
            <span className="stat-unavailable">unavailable</span>
          ) : (
            <>
              {last.toPrecision(6)}
              {unit ? <span className="stat-unit">{unit}</span> : null}
            </>
          )}
        </span>
      </div>
      <canvas ref={ref} style={{ width: '100%', height }} />
      <div className="trace-foot">
        {min != null && max != null ? (
          <>
            range {fmtTick(min)} to {fmtTick(max)}
            {rel != null ? (
              <span className={rel < 1 ? 'trace-settled' : ''}>
                {'  ·  '}
                {rel < 0.01 ? 'under 0.01' : rel.toPrecision(2)}% of the value
                {rel < 1 ? ', so this is settled and the wiggle is noise' : ''}
              </span>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  )
}

function fmtTick(v: number): string {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2)
  return v.toPrecision(5)
}
