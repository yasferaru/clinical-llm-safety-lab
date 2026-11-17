"""Reliability across repeated epochs: pass^k, pass@k, worst@k and epoch consistency.

Input: the sample table from evals/results.py (one row per run, case and epoch). Each scored row is
one trial of a case. Rows are grouped by (`by` columns, default "model") and case_id. Runs of the same
model are pooled, so two 1-epoch runs give n = 2 trials per case. Unscored rows (sample errored) are
dropped.

Success is defined as critical_pass == 1 (every critical rubric line behaved safely, judge errors
counting as failures). A case with n trials and c successes gives, for 1 <= k <= n:

    pass^k  = C(c, k) / C(n, k)          P(k distinct trials all succeed): reliability
    pass@k  = 1 - C(n-c, k) / C(n, k)    P(at least one of k distinct trials succeeds): capability
    worst@k = sum_i x_(i) C(n-i, k-1) / C(n, k)   E[min rubric_score of k distinct trials],
              with x_(1) <= ... <= x_(n) the case's rubric scores sorted ascending

The pass^k and pass@k estimators are unbiased (pass@k: Chen et al. 2021; pass^k: Yao et al. 2024,
tau-bench, arXiv:2406.12045). Summaries average the per-case values with equal weight per case.
Intervals come from a bootstrap over scenario clusters (scenario_id), because the base and pressure
variants of one scenario are not independent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

SUCCESS_RULE = "critical_pass == 1"


# --------------------------------------------------------------------------------------------
# Per-case estimators
# --------------------------------------------------------------------------------------------
def _check_nk(n: int, k: int) -> None:
    """Raise ValueError unless 1 <= k <= n."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k = {k} is larger than the number of trials n = {n}")


def pass_hat_k(c: int, n: int, k: int) -> float:
    """pass^k = C(c, k) / C(n, k): chance that k distinct trials of the case all succeed.

    Example: c=3, n=4, k=2 -> C(3,2)/C(4,2) = 3/6 = 0.5.
    """
    _check_nk(n, k)
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    return comb(c, k) / comb(n, k)


def pass_at_k(c: int, n: int, k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k): chance that at least one of k distinct trials succeeds.

    Example: c=1, n=4, k=2 -> 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5.
    """
    _check_nk(n, k)
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    return 1 - comb(n - c, k) / comb(n, k)


def worst_at_k(scores: Sequence[float], k: int) -> float:
    """E[min of k scores drawn without replacement] = sum_i x_(i) C(n-i, k-1) / C(n, k), i = 1..n.

    x_(i) is the i-th smallest score. Weight of x_(i) = P(it is the minimum) = number of k-subsets
    whose other k-1 members all lie above position i, over all k-subsets.
    Example: scores [1, 0, 0.5], k=2 -> sorted [0, 0.5, 1]; (0*2 + 0.5*1 + 1*0) / 3 = 1/6.
    """
    x = sorted(float(s) for s in scores)
    n = len(x)
    _check_nk(n, k)
    total = comb(n, k)
    return sum(xi * comb(n - i, k - 1) for i, xi in enumerate(x, start=1)) / total


def disagreement_rate(c: int, n: int) -> float:
    """Flip rate: P(two distinct trials of the case disagree on success) = c (n - c) / C(n, 2).

    Order-free (epochs are exchangeable, so adjacent-epoch flips would depend on an arbitrary order).
    NaN when n < 2. Example: c=1, n=2 -> 1*1/1 = 1.0; c=2, n=4 -> 2*2/6 = 2/3.
    """
    if n < 2:
        return float("nan")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    return c * (n - c) / comb(n, 2)


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------
def case_table(samples: pd.DataFrame, by: Sequence[str] = ("model",)) -> pd.DataFrame:
    """One row per (*by, case_id): scenario_id, n trials, c successes, rubric scores, disagreement.

    success = (critical_pass == 1) on scored rows. disagree = 0 < c < n (the epochs do not agree).
    """
    by = list(by)
    scored = samples[samples["scored"].astype(bool)].copy()
    scored["success"] = scored["critical_pass"].astype(float) == 1.0
    rows = []
    for keys, g in scored.groupby(by + ["case_id"], dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n, c = len(g), int(g["success"].sum())
        scen = g["scenario_id"].dropna().unique()
        rows.append({**dict(zip(by + ["case_id"], keys)),
                     "scenario_id": scen[0] if len(scen) else keys[-1],
                     "n": n, "c": c, "scores": [float(v) for v in g["rubric_score"]],
                     "disagree": bool(0 < c < n), "flip_rate": disagreement_rate(c, n)})
    return pd.DataFrame(rows, columns=by + ["case_id", "scenario_id", "n", "c", "scores", "disagree", "flip_rate"])


def add_k_metrics(cases: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add pass_hat_k, pass_at_k, worst_at_k columns for one k. Raises ValueError if any case has n < k."""
    short = cases[cases["n"] < k]
    if len(short):
        raise ValueError(f"k = {k} exceeds the trials of {len(short)} case(s), e.g. "
                         f"{short.iloc[0]['case_id']} with n = {short.iloc[0]['n']}")
    out = cases.copy()
    out["pass_hat_k"] = [pass_hat_k(c, n, k) for c, n in zip(out["c"], out["n"])]
    out["pass_at_k"] = [pass_at_k(c, n, k) for c, n in zip(out["c"], out["n"])]
    out["worst_at_k"] = [worst_at_k(s, k) for s in out["scores"]]
    return out


def max_k(cases: pd.DataFrame) -> int:
    """Largest k usable for every case in the table = min over cases of n (0 if empty)."""
    return int(cases["n"].min()) if len(cases) else 0


# --------------------------------------------------------------------------------------------
# Scenario-cluster bootstrap
# --------------------------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Same fields as evals.stats.inference.BootstrapResult (percentile only in this stub)."""
    estimate: float
    percentile_ci: tuple
    bca_ci: tuple
    se: float
    n_clusters: int
    B: int
    bca_note: str = ""

    def to_dict(self):
        return asdict(self)


def cluster_bootstrap(stat_fn: Callable[[pd.DataFrame], float], df: pd.DataFrame,
                      cluster_col: str = "scenario_id", B: int = 10000, seed: int = 0,
                      level: float = 0.95) -> BootstrapResult:
    """Resample whole clusters with replacement; percentile CI of stat_fn(df).

    TODO: replace with evals.stats.inference.cluster_bootstrap (P1, same signature, adds BCa) once
    it is merged; this local stub exists so the reliability branch does not depend on unmerged work.
    Percentile CI = (alpha/2, 1 - alpha/2) quantiles of the B replicates; bca_ci = percentile_ci.
    Example: 2 clusters with case values 0 and 1, stat = mean -> replicates take 0, 0.5, 1 with
    probabilities 1/4, 1/2, 1/4, so the 95% percentile CI is (0, 1).
    """
    labels = df[cluster_col].astype(str).to_numpy()
    uniq = np.unique(labels)
    g = len(uniq)
    if g < 2:
        raise ValueError("need at least 2 clusters to bootstrap")
    members = [np.flatnonzero(labels == u) for u in uniq]
    rng = np.random.default_rng(seed)
    theta = float(stat_fn(df))
    boot = np.empty(B)
    for b in range(B):
        idx = np.concatenate([members[c] for c in rng.integers(0, g, size=g)])
        boot[b] = stat_fn(df.iloc[idx])
    alpha = 1 - level
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    pct = (float(lo), float(hi))
    return BootstrapResult(theta, pct, pct, float(boot.std(ddof=1)), g, B,
                           "stub: percentile interval only (BCa pending P1)")


# --------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------
METRICS = ("pass_hat_k", "pass_at_k", "worst_at_k")


def _ci(col: str, cases: pd.DataFrame, B: int, seed: int, level: float) -> tuple[float, float]:
    """Cluster-bootstrap percentile CI of the case mean of `col`; NaNs if fewer than 2 scenarios."""
    if cases["scenario_id"].nunique() < 2:
        return float("nan"), float("nan")
    res = cluster_bootstrap(lambda d: float(d[col].mean()), cases, "scenario_id", B=B, seed=seed, level=level)
    return res.percentile_ci


def reliability_summary(samples: pd.DataFrame, ks: Iterable[int] = (1, 2, 4, 8), by: Sequence[str] = ("model",),
                        B: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Per (*by, k): case means of pass^k, pass@k and worst@k with scenario-cluster bootstrap CIs.

    Each group uses only the ks with k <= its max_k (min trials over its cases); larger ks are
    skipped for that group, never estimated from fewer trials.
    Columns: *by, k, n_cases, n_scenarios, n_min, n_max, and for each metric m: m, m_lo, m_hi.
    """
    by = list(by)
    cases = case_table(samples, by)
    rows = []
    for keys, g in cases.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for k in sorted(set(ks)):
            if k > max_k(g):
                continue
            gk = add_k_metrics(g, k)
            row = {**dict(zip(by, keys)), "k": k, "n_cases": len(gk), "n_scenarios": gk["scenario_id"].nunique(),
                   "n_min": int(gk["n"].min()), "n_max": int(gk["n"].max())}
            for m in METRICS:
                row[m] = float(gk[m].mean())
                row[f"{m}_lo"], row[f"{m}_hi"] = _ci(m, gk, B, seed, level)
            rows.append(row)
    metric_cols = [f"{m}{s}" for m in METRICS for s in ("", "_lo", "_hi")]
    cols = by + ["k", "n_cases", "n_scenarios", "n_min", "n_max"] + metric_cols
    return pd.DataFrame(rows, columns=cols)


def consistency_summary(samples: pd.DataFrame, by: Sequence[str] = ("model",)) -> pd.DataFrame:
    """Per group, over cases with n >= 2: mean flip rate and how many cases' epochs disagree on success.

    Columns: *by, n_cases, n_disagree, frac_disagree (= n_disagree / n_cases), mean_flip_rate.
    Example: 4 cases with (c, n) = (0,2), (2,2), (1,2), (1,2) -> 2 of 4 disagree, flip rates 0,0,1,1.
    """
    by = list(by)
    cases = case_table(samples, by)
    cases = cases[cases["n"] >= 2]
    rows = []
    for keys, g in cases.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        nd = int(g["disagree"].sum())
        rows.append({**dict(zip(by, keys)), "n_cases": len(g), "n_disagree": nd,
                     "frac_disagree": nd / len(g), "mean_flip_rate": float(g["flip_rate"].mean())})
    return pd.DataFrame(rows, columns=by + ["n_cases", "n_disagree", "frac_disagree", "mean_flip_rate"])
