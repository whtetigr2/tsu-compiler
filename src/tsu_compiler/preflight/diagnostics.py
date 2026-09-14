"""R-hat and the ESS-based standard error -- the two estimators tsu_compiler.ess does
not provide.

Integrated autocorrelation time and effective sample size for a set of MCMC
chains live in `tsu_compiler.ess`, NOT here. `tsu_compiler.ess` implements the same Sokal
automatic-windowing estimator this module used to duplicate, but does it
better: it averages per-chain autocovariance with each chain first centred on
its OWN mean (so a between-chain mean difference, e.g. incomplete burn-in,
cannot masquerade as within-chain autocorrelation -- a plain mean of per-chain
taus, which is what this module used to compute, has no such protection), it
refuses to report `ess` at all (returning `None` with a `reason`) below a
reliability floor derived from its own AR(1) sweep, and it is validated by an
AR(1) known-answer check, cross-implementation agreement with `arviz` and
`statsmodels`, and a must-fail potency check (`tests/test_ess.py`). Rebuilding
a cruder version of that machinery here was a planning error -- this module no
longer does so. Anything needing tau or ESS should import
`tsu_compiler.ess.integrated_autocorrelation_time` / `tsu_compiler.ess.effective_sample_size`
directly.

This module keeps only:
  - `r_hat` / `RHAT_THRESHOLD`: Gelman-Rubin potential scale reduction;
    `tsu_compiler.ess` has no analogue.
  - `stderr_from_ess`: the standard-error correction, taking ESS directly
    (not tau) so no tau-convention ambiguity can leak into a call site.
    `tsu_compiler.ess`'s tau is tau_A = 1 + 2*sum_k rho_k (matching Sokal/emcee), NOT
    the tau_A/2 convention this module used before this rework -- the two
    give the same ESS, but a tau read off directly would be wrong by 2x. See
    `tests/test_preflight_diagnostics.py` for the test pinning that
    convention.
"""
from __future__ import annotations

import numpy as np

RHAT_THRESHOLD = 1.01
"""Above this, chains disagree enough that their pooled numbers are reported as
provisional. The conventional threshold; stated as a constant so a reader can
see which value was used rather than infer it."""


def stderr_from_ess(x: np.ndarray, ess: float) -> float:
    """Standard error using the effective sample size, not the raw count.

    Takes ESS directly (from `tsu_compiler.ess.effective_sample_size`), not tau, so a
    caller can never accidentally apply the wrong tau convention: this is
    exactly `std(x, ddof=1) / sqrt(ess)`.
    """
    x = np.asarray(x, dtype=float).ravel()
    return float(np.std(x, ddof=1) / np.sqrt(ess))


def r_hat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction on shape (n_chains, n_samples).

    Compares between-chain variance to within-chain variance. Approaches 1 as
    chains become indistinguishable (for chains with literally zero
    between-chain disagreement it is exactly sqrt((n-1)/n), slightly below 1 --
    the BDA3 estimator's published finite-sample behaviour) and grows when
    chains are each internally settled but disagree with one another -- the
    failure a per-chain average hides completely.
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2:
        raise ValueError("r_hat needs at least two chains, shaped "
                         "(n_chains, n_samples)")
    m, n = c.shape
    if n < 2:
        raise ValueError("r_hat needs at least two samples per chain")
    means = c.mean(axis=1)
    w = float(c.var(axis=1, ddof=1).mean())
    b = float(n * means.var(ddof=1))
    if w <= 0.0:
        return 1.0
    # BDA3's var_plus = (n-1)/n*w + b/n. For chains with zero between-chain
    # disagreement (b=0) this gives R-hat = sqrt((n-1)/n), a value strictly
    # below 1 -- the PUBLISHED behaviour of this estimator, not a defect.
    # Dropping the (n-1)/n discount would produce a statistic that is not
    # R-hat, and RHAT_THRESHOLD = 1.01 is calibrated for the one that is.
    var_plus = ((n - 1) / n) * w + b / n
    return float(np.sqrt(var_plus / w))
