/** Lightweight PCA for sample-cloud projections (no fake 2^N). */

export type Vec = number[]

function meanVec(rows: Vec[]): Vec {
  const d = rows[0]?.length ?? 0
  const m = new Array(d).fill(0)
  if (!rows.length) return m
  for (const r of rows) {
    for (let i = 0; i < d; i++) m[i] += r[i]
  }
  for (let i = 0; i < d; i++) m[i] /= rows.length
  return m
}

function center(rows: Vec[], m: Vec): Vec[] {
  return rows.map((r) => r.map((v, i) => v - m[i]))
}

/** Power-iteration top eigenvector of X^T X (rows = samples). */
function topEigen(centered: Vec[], dim: number): Vec {
  let v = new Array(dim).fill(0).map((_, i) => (i % 2 === 0 ? 1 : -1) / Math.sqrt(dim))
  for (let iter = 0; iter < 40; iter++) {
    const Av = new Array(dim).fill(0)
    for (const row of centered) {
      let dot = 0
      for (let i = 0; i < dim; i++) dot += row[i] * v[i]
      for (let i = 0; i < dim; i++) Av[i] += row[i] * dot
    }
    let norm = Math.hypot(...Av) || 1
    v = Av.map((x) => x / norm)
  }
  return v
}

function project(centered: Vec[], axes: Vec[]): number[][] {
  return centered.map((row) =>
    axes.map((axis) => {
      let s = 0
      for (let i = 0; i < row.length; i++) s += row[i] * axis[i]
      return s
    }),
  )
}

function deflate(centered: Vec[], axis: Vec): Vec[] {
  return centered.map((row) => {
    let dot = 0
    for (let i = 0; i < row.length; i++) dot += row[i] * axis[i]
    return row.map((x, i) => x - dot * axis[i])
  })
}

/**
 * Project sample rows onto the first `k` principal components.
 * Spins expected as ±1 or 0/1, values are used as-is (no invented energies).
 */
export function pcaProject(rows: Vec[], k: 2 | 3 = 2): {
  coords: number[][]
  explainedHint: string
} {
  if (!rows.length || !rows[0]?.length) {
    return { coords: [], explainedHint: 'no samples' }
  }
  const dim = rows[0].length
  const m = meanVec(rows)
  let centered = center(rows, m)
  const axes: Vec[] = []
  const kk = Math.min(k, dim, rows.length)
  for (let a = 0; a < kk; a++) {
    const axis = topEigen(centered, dim)
    axes.push(axis)
    centered = deflate(centered, axis)
  }
  // Pad to k dims if needed
  while (axes.length < k) {
    axes.push(new Array(dim).fill(0))
  }
  const coords = project(center(rows, m), axes.slice(0, k))
  return {
    coords,
    explainedHint: `PCA on ${rows.length} recent sample vectors (dim ${dim}) → ${k}D`,
  }
}
