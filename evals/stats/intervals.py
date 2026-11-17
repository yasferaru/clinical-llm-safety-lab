"""Confidence intervals for a binomial proportion k/n.

Two named methods, so a reported interval always says which one it is:

- ``wilson``: Wilson score interval. Default for red-team rates.
- ``clopper_pearson``: exact (conservative) interval from the beta distribution. Default for critical_pass.

Both return ``(lower, upper)`` as floats in [0, 1].
"""
from __future__ import annotations

import math

from scipy.stats import beta, norm


def _check(k: int, n: int, alpha: float) -> None:
    """Raise ValueError unless 0 <= k <= n, n >= 1 and 0 < alpha < 1."""
    if isinstance(k, bool) or isinstance(n, bool) or int(k) != k or int(n) != n:
        raise ValueError("k and n must be whole numbers")
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k successes out of n.

    With p = k/n and z = Phi^-1(1 - alpha/2):
        centre = (p + z^2 / (2n)) / (1 + z^2 / n)
        half   = z * sqrt(p(1-p)/n + z^2 / (4 n^2)) / (1 + z^2 / n)
        CI     = (centre - half, centre + half), clipped to [0, 1]
    Example: k=1, n=4, alpha=0.05 -> (0.0456, 0.6994).
    """
    _check(k, n, alpha)
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval for k successes out of n.

        lower = Beta(alpha/2; k, n-k+1)       (0 when k = 0)
        upper = Beta(1-alpha/2; k+1, n-k)     (1 when k = n)
    where Beta(q; a, b) is the q-quantile of the beta distribution.
    Example: k=1, n=4, alpha=0.05 -> (0.0063, 0.8059).
    """
    _check(k, n, alpha)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


METHODS = {"wilson": wilson, "clopper_pearson": clopper_pearson}
