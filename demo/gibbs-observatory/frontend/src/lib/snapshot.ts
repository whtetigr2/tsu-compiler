import type { BatchPayload, ReceiptInspect } from '../types'

export const STANDING_PROHIBITIONS: { id: string; text: string }[] = [
  {
    id: 'no_silicon',
    text: 'No silicon execution claims, software / THRML+JAX only',
  },
  {
    id: 'no_energy',
    text: 'No energy-savings or joule claims',
  },
  {
    id: 'timing',
    text: 'THRML wall-clock ≠ TSU / silicon timing',
  },
  {
    id: 'thermalizers',
    text: 'Never claim “Thermalizers-complete”',
  },
  {
    id: 'beta_fixed',
    text: 'Mediated β is FIXED (not a free slider)',
  },
  {
    id: 'ess',
    text: 'ESS unavailable when the receipt contract is broken',
  },
  {
    id: 'invalid_draws',
    text: 'Invalid draws never rendered as worlds',
  },
  {
    id: 'state_space',
    text: 'No fake full 2^N state space for large models',
  },
]

export const FALLBACK_CLAIM_BADGES = [
  'software / THRML+JAX',
  'documented Z1 caps (assumed fields marked)',
  'no silicon',
  'β FIXED when mediated',
  'no energy claims',
  'ESS only when honest',
]

export interface SnapshotClientMeta {
  view: string
  step?: number | null
  active_block?: number | null
}

function stamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

/** Capture first canvas under stage, or rasterize stage DOM fallback. */
export function captureStagePng(stageEl: HTMLElement | null): {
  blob: Blob | null
  filename: string
  method: string
} {
  const filename = `gibbs-observatory-${stamp()}.png`
  if (!stageEl) {
    return { blob: null, filename, method: 'none' }
  }
  const canvas = stageEl.querySelector('canvas') as HTMLCanvasElement | null
  if (canvas && canvas.width > 0 && canvas.height > 0) {
    try {
      const dataUrl = canvas.toDataURL('image/png')
      const bin = atob(dataUrl.split(',')[1] ?? '')
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      return {
        blob: new Blob([bytes], { type: 'image/png' }),
        filename,
        method: 'canvas',
      }
    } catch {
      /* fall through */
    }
  }
  // Fallback: compose a simple PNG-sized canvas with stage text summary
  const w = 720
  const h = 420
  const off = document.createElement('canvas')
  off.width = w
  off.height = h
  const ctx = off.getContext('2d')
  if (!ctx) return { blob: null, filename, method: 'none' }
  ctx.fillStyle = '#070a10'
  ctx.fillRect(0, 0, w, h)
  ctx.fillStyle = '#3ee0b0'
  ctx.font = '16px monospace'
  ctx.fillText('Gibbs Observatory, stage snapshot', 24, 40)
  ctx.fillStyle = '#9aa7bd'
  ctx.font = '13px monospace'
  const caption =
    stageEl.querySelector('.view-caption, .stage-header')?.textContent?.trim() ||
    'No canvas in active view, metadata JSON still exported'
  const lines = caption.slice(0, 240).match(/.{1,70}/g) ?? [caption]
  lines.forEach((line, i) => ctx.fillText(line, 24, 80 + i * 20))
  ctx.fillStyle = '#f0b429'
  ctx.fillText('JAX/THRML simulation, not Extropic silicon', 24, h - 28)
  try {
    const dataUrl = off.toDataURL('image/png')
    const bin = atob(dataUrl.split(',')[1] ?? '')
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    return {
      blob: new Blob([bytes], { type: 'image/png' }),
      filename,
      method: 'fallback',
    }
  } catch {
    return { blob: null, filename, method: 'none' }
  }
}

export function buildLocalSnapshotSlice(
  receipt: ReceiptInspect | null,
  batch: BatchPayload | null,
  meta: SnapshotClientMeta,
  pngFilename: string | null,
) {
  const badges = receipt?.claim_badges?.length
    ? receipt.claim_badges
    : FALLBACK_CLAIM_BADGES
  return {
    schema: 'gibbs-observatory.snapshot.v1',
    timestamp: new Date().toISOString(),
    version: '0.5.0',
    receipt_id: receipt?.id ?? null,
    verdict: receipt?.verdict ?? null,
    encoding: receipt?.encoding ?? null,
    kernel: receipt?.kernel ?? null,
    beta: receipt?.beta ?? null,
    beta_fixed: Boolean(receipt?.beta_fixed),
    spins: {
      n_nodes: receipt?.spins.n_nodes ?? null,
      world: receipt?.spins.world ?? null,
      mediators: receipt?.spins.mediators ?? null,
    },
    gates: (receipt?.gates ?? []).map((g) => ({
      gate: g.gate,
      passed: g.passed,
      measured: g.measured,
      limit: g.limit,
      assumed: g.assumed ?? false,
    })),
    claim_badges: badges,
    standing_prohibitions: STANDING_PROHIBITIONS.map((p) => p.text),
    step: batch?.step ?? meta.step ?? null,
    active_block: batch?.active_block ?? meta.active_block ?? null,
    view: meta.view,
    png_filename: pngFilename,
    label: 'JAX/THRML simulation, not Extropic silicon',
    notes: [
      'PNG is client-captured from the main stage canvas/view.',
      'This JSON is a receipt slice only, no sim sample dumps.',
      'Do not commit snapshot downloads into git receipts/.',
    ],
  }
}

export async function exportSnapshot(opts: {
  stageEl: HTMLElement | null
  receipt: ReceiptInspect | null
  batch: BatchPayload | null
  view: string
}): Promise<{ ok: boolean; png: string | null; json: string; message: string }> {
  const { blob, filename, method } = captureStagePng(opts.stageEl)
  const local = buildLocalSnapshotSlice(
    opts.receipt,
    opts.batch,
    {
      view: opts.view,
      step: opts.batch?.step,
      active_block: opts.batch?.active_block,
    },
    blob ? filename : null,
  )

  // Optional backend enrichment (metadata only)
  let slice = local
  try {
    const res = await fetch('/api/snapshot', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        receipt_id: opts.receipt?.id ?? null,
        step: opts.batch?.step ?? null,
        active_block: opts.batch?.active_block ?? null,
        view: opts.view,
        png_filename: blob ? filename : null,
      }),
    })
    if (res.ok) {
      const data = (await res.json()) as { snapshot?: typeof local }
      if (data.snapshot) slice = { ...local, ...data.snapshot, png_filename: local.png_filename }
    }
  } catch {
    /* local slice is enough */
  }

  const jsonName = `gibbs-observatory-${stamp()}.snapshot.json`
  const jsonBlob = new Blob([JSON.stringify(slice, null, 2)], {
    type: 'application/json',
  })
  downloadBlob(jsonBlob, jsonName)
  if (blob) downloadBlob(blob, filename)

  return {
    ok: true,
    png: blob ? filename : null,
    json: jsonName,
    message: blob
      ? `Saved ${filename} (${method}) + ${jsonName}`
      : `Saved ${jsonName} (no PNG, open a canvas view)`,
  }
}
