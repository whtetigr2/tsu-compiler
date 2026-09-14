import { useEffect, useRef } from 'react'

interface Props {
  data: number[]
  color?: string
  height?: number
}

export function Sparkline({ data, color = '#3ee0b0', height = 72 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth || 280
    const h = height
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.height = `${h}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    if (!data.length) return
    const min = Math.min(...data)
    const max = Math.max(...data)
    const span = max - min || 1
    const pad = 4

    ctx.beginPath()
    data.forEach((v, i) => {
      const x = pad + (i / Math.max(1, data.length - 1)) * (w - pad * 2)
      const y = h - pad - ((v - min) / span) * (h - pad * 2)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.stroke()

    ctx.lineTo(w - pad, h - pad)
    ctx.lineTo(pad, h - pad)
    ctx.closePath()
    ctx.fillStyle = color + '22'
    ctx.fill()
  }, [data, color, height])

  return <canvas ref={ref} style={{ width: '100%', height }} />
}
