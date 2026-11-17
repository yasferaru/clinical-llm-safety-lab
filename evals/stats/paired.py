"""Paired binary comparisons."""
from __future__ import annotations

from scipy.stats import binomtest


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of b successes in b + c discordant pairs at p = 0.5.

    b, c = discordant pair counts (e.g. YES->NO and NO->YES). p = 1 when b + c = 0.
    Example: b=0, c=5 -> 2 * 0.5^5 = 0.0625; b=1, c=7 -> 2 * (C(8,0) + C(8,1)) / 2^8 = 0.0703125.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be >= 0")
    if b + c == 0:
        return 1.0
    return float(binomtest(int(b), int(b + c), 0.5).pvalue)
