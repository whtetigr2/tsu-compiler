import type { LangMode } from '../types'

const GLOSSARY: Record<string, { sci: string; prog: string; tip: string }> = {
  beta: {
    sci: 'β',
    prog: 'temperature (1/T scale)',
    tip: 'Inverse temperature. Mediated receipts lock β FIXED.',
  },
  J: {
    sci: 'J',
    prog: 'coupling',
    tip: 'Pairwise spin–spin coupling weight.',
  },
  b: {
    sci: 'b',
    prog: 'bias',
    tip: 'Local field / linear bias on a spin.',
  },
  chi: {
    sci: 'χ',
    prog: 'colour count',
    tip: 'Chromatic colouring / independent-set block count.',
  },
  ess: {
    sci: 'ESS',
    prog: 'effective samples',
    tip: 'Effective sample size — only shown when the receipt contract allows.',
  },
  tv: {
    sci: 'TV',
    prog: 'total variation',
    tip: 'Total variation distance to a reference (often unavailable for large N).',
  },
  mediators: {
    sci: 'mediators',
    prog: 'Fabric Tax spins',
    tip: 'Fabric Tax: auxiliary spins that bipartite the Z1 physical fabric (not cold/hidden).',
  },
  blocks: {
    sci: 'chromatic blocks',
    prog: 'schedule phase',
    tip: 'Independent sets updated together in block-Gibbs.',
  },
  energy: {
    sci: 'energy',
    prog: 'energy',
    tip: 'Ising energy under the compiled program.',
  },
  mag: {
    sci: 'magnetization',
    prog: 'mean spin',
    tip: 'Mean ±1 magnetization of the current sample.',
  },
}

export function label(key: keyof typeof GLOSSARY, mode: LangMode): string {
  const g = GLOSSARY[key]
  return mode === 'sci' ? g.sci : g.prog
}

export function tip(key: keyof typeof GLOSSARY): string {
  return GLOSSARY[key].tip
}

export { GLOSSARY }
