# "The Diagnostic That Lied" — page source

The interactive page built overnight on 2026-09-15 by Claude and Grok, published
as a **private** claude.ai Artifact. Paul decides whether it is ever shared.

    https://claude.ai/artifact/4cBUtXaPLQq17HyDCLBQU2

`the_diagnostic_that_lied.html` is the exact source of the published page, kept
here for the same reason every other number in this project has a receipt: **the
page makes claims, so the page has to be auditable.** Its four exhibits are each
an error made during the audit of 2026-09-14/15, rebuilt as something a visitor
can operate.

| | Exhibit | Built by | Found by | Record |
|---|---|---|---|---|
| I | A convergence diagnostic reading 1.0000 on chains frozen in two basins | Claude | the claims-vs-evidence audit | `audit/findings/R20.md` |
| II | A worst-case statistic that excludes the cases that failed | Claude | Claude, while fixing I | `audit/findings/R20.md` addendum |
| III | Three-sigma outliers arriving on schedule in pure noise | Claude | Grok (Check 4 design) | `demo/extropic-pack/verification/CHECK4_*` |
| IV | A z-score built on a standard error that assumed independence | Grok | Grok; Claude insisted apply-not-annotate | `demo/extropic-pack/verification/exhibits/` |

## What has been verified, and what has not

`audit/verify_artifact_sampler.py` covers two of the page's four computations:

- **The Gibbs update rule** — transcribed back into Python and run against
  `audit/oracles/exact.py`, which does not import this package. Agreement to
  0.0047 at worst, that worst case sitting on the critical point where a finite
  sample is least precise. Signs match the toggle's labels.
- **The R-hat accumulator** — the page accumulates Gelman-Rubin from streaming
  Welford moments; this checks it against `tsu_compiler`'s two-pass form across
  six series, including the identical-chains edge case (where the published
  answer is `sqrt((n-1)/n)`, strictly below 1) and a large-offset series that
  breaks naive sum-of-squares accumulators.

**NOT verified: the free energy histogram** — its binning, its `min_count` mask,
and whether the surface it paints is supported by the draws behind it. Stated
here rather than left for a reader to infer.

Grok independently re-tested Exhibit I's physics with Metropolis rather than
Gibbs, separate code, eight seeds: at `beta*J = 0.8` antiferromagnetic, uniform
`|m| = 0.00275` while staggered `|m_s| = 0.99603`. The lattice is as ordered as
a magnet gets and the scalar reads 0.00275. Their ferromagnetic `|m| = 0.99603`
sits against Onsager's infinite-volume `0.99602`.

## Two corrections made while building it

Both are in the commit history rather than smoothed over, because the page's own
argument is that unverified numbers look exactly like verified ones.

1. **The page was published before its sampler was checked.** The verification
   above was written an hour after the first publish. It came back clean; the
   ordering was still wrong, and on this page of all pages.
2. **Exhibit II's tau distribution was fitted by eye and was wrong.** At
   `sigma = 0.62` it produces a worst tau of 33.8 against a measured 21.4, and
   leaves a spin uncertified at 128,000 draws — contradicting our own run, where
   everything certifies there. `sigma = 0.50` fits. The failure was reaching for
   a parameter that looked right instead of checking it against the thing it was
   meant to reproduce.

## Claim hygiene

No hardware claims. No watts, joules, or sampling rates. The defect in Exhibit I
was in **this project's own diagnostic**, not in thrml and not in any vendor's
published work; it is demonstrated on a control lattice whose exact answer has
been known since Onsager solved it in 1944.
