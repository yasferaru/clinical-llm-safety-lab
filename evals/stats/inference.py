"""Error bars for eval scores, after Miller (2024), "Adding Error Bars to Evals", arXiv:2411.00640.

Questions in this eval are not independent: base/pressure variants of one scenario share a clinical
situation, so their scores are correlated. The naive standard error treats them as independent and
is too small. Everything here treats the scenario as the cluster unit.

Results are pandas DataFrames with one row per question and (at least) the columns
    id           question id, e.g. 'triage-001-base'
    scenario_id  cluster unit (the repo calls this `cluster_id` in cases.jsonl; load_results renames it)
    rubric_score, critical_pass   per-question scores
Functions take plain arrays (numpy / pandas Series) so they work on any column.

Normal quantiles come from statistics.NormalDist, so only numpy and pandas are required.
Intervals and p-values use the normal approximation (Miller's CLT-based approach). With very few
clusters (G < ~30) they are optimistic; cluster_bootstrap is no better at G that small, so read
`n_clusters` in every result before trusting an interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def _z(p):
    return _N.inv_cdf(p)


def _arr(x):
    return np.asarray(x, dtype=float)


def _codes(cluster, n):
    """Map arbitrary cluster labels to 0..G-1 and check the length matches the data."""
    cluster = np.asarray(cluster)
    if len(cluster) != n:
        raise ValueError(f"cluster has length {len(cluster)} but data has length {n}")
    _, codes = np.unique(cluster, return_inverse=True)
    return codes


# --------------------------------------------------------------------------------------------
# Mean with naive SE
# --------------------------------------------------------------------------------------------

@dataclass
class MeanSE:
    mean: float
    se: float
    n: int
    ci_low: float
    ci_high: float

    def to_dict(self):
        return asdict(self)


def mean_with_se(x, bernoulli=False, level=0.95):
    """Mean and *naive* (independence-assuming) CLT standard error: sd(ddof=1)/sqrt(n).

    bernoulli=True uses sqrt(p(1-p)/n) for 0/1 scores. (It differs from the sample-sd version only
    by the factor sqrt(n/(n-1)).) Use clustered_se when questions share scenarios.
    """
    x = _arr(x)
    n = len(x)
    if n == 0:
        raise ValueError("empty input")
    m = float(x.mean())
    if bernoulli:
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("bernoulli=True needs values in {0, 1}")
        se = math.sqrt(m * (1 - m) / n)
    else:
        se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    zc = _z(0.5 + level / 2)
    return MeanSE(m, se, n, m - zc * se, m + zc * se)


# --------------------------------------------------------------------------------------------
# Clustered SE, ICC, design effect
# --------------------------------------------------------------------------------------------

@dataclass
class ClusteredSE:
    mean: float
    se: float            # cluster-robust SE
    naive_se: float      # CLT SE ignoring clusters
    n: int
    n_clusters: int
    icc: float           # ANOVA estimator, clipped to [0, 1]; nan if undefined
    deff: float          # 1 + (m - 1) * icc, m = n / n_clusters
    ci_low: float
    ci_high: float

    def to_dict(self):
        return asdict(self)


def _clustered_se_from_codes(x, codes):
    n = len(x)
    if int(codes.max()) + 1 < 2:        # one cluster: residual sums cancel to 0, which is not "no error"
        return float("nan")
    resid_sums = np.bincount(codes, weights=x - x.mean())
    return float(math.sqrt(np.sum(resid_sums ** 2)) / n)


def _icc_anova(x, codes):
    """One-way ANOVA estimator of the intraclass correlation (handles unequal cluster sizes)."""
    n, g = len(x), int(codes.max()) + 1
    if g < 2 or g >= n:
        return float("nan")
    sizes = np.bincount(codes).astype(float)
    sums = np.bincount(codes, weights=x)
    means = sums / sizes
    ssb = float(np.sum(sizes * (means - x.mean()) ** 2))
    ssw = float(np.sum((x - means[codes]) ** 2))
    msb, msw = ssb / (g - 1), ssw / (n - g)
    m0 = (n - np.sum(sizes ** 2) / n) / (g - 1)
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return float("nan")
    return float(min(1.0, max(0.0, (msb - msw) / denom)))


def clustered_se(x, cluster, level=0.95):
    """Cluster-robust SE of the mean: sqrt( sum_c (sum_{i in c}(x_i - xbar))^2 ) / n.

    Also returns n, n_clusters, the ICC and DEFF = 1 + (m - 1) * ICC with m the mean cluster size.
    The CI is mean +/- z * se (normal approximation).
    """
    x = _arr(x)
    codes = _codes(cluster, len(x))
    n, g = len(x), int(codes.max()) + 1
    m = float(x.mean())
    se = _clustered_se_from_codes(x, codes)
    icc = _icc_anova(x, codes)
    deff = 1 + (n / g - 1) * icc if not math.isnan(icc) else float("nan")
    naive = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    zc = _z(0.5 + level / 2)
    return ClusteredSE(m, se, naive, n, g, icc, deff, m - zc * se, m + zc * se)


# --------------------------------------------------------------------------------------------
# Paired difference
# --------------------------------------------------------------------------------------------

@dataclass
class PairedDiff:
    mean_diff: float
    se: float             # cluster-robust SE of the mean of d_i = a_i - b_i
    ci_low: float
    ci_high: float
    p_value: float        # two-sided, normal approximation
    n: int
    n_clusters: int
    sd_d: float           # sd of d_i (feeds power_analysis as sigma_d)
    se_unpaired: float    # sqrt(se_clust(a)^2 + se_clust(b)^2): SE if a and b were independent
    pairing_gain: float   # se_unpaired / se (>1: pairing helps; <1: a and b are negatively correlated)

    def to_dict(self):
        return asdict(self)


def paired_difference(a, b, cluster, level=0.95):
    """Mean of d_i = a_i - b_i over questions answered by both models, with cluster-robust SE.

    a, b must be aligned question by question. pairing_gain compares the paired SE with the SE you
    would get treating the two models' scores as independent samples.
    """
    a, b = _arr(a), _arr(b)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (aligned per question)")
    d = a - b
    codes = _codes(cluster, len(d))
    n, g = len(d), int(codes.max()) + 1
    md = float(d.mean())
    se = _clustered_se_from_codes(d, codes)
    zc = _z(0.5 + level / 2)
    if math.isnan(se):
        p = float("nan")
    elif se > 0:
        p = 2 * (1 - _N.cdf(abs(md) / se))
    else:
        p = 1.0 if md == 0 else 0.0
    se_un = math.hypot(_clustered_se_from_codes(a, codes), _clustered_se_from_codes(b, codes))
    gain = float("nan") if math.isnan(se) else se_un / se if se > 0 else float("inf" if se_un > 0 else "nan")
    sd_d = float(d.std(ddof=1)) if n > 1 else float("nan")
    return PairedDiff(md, se, md - zc * se, md + zc * se, p, n, g, sd_d, se_un, gain)


# --------------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    estimate: float
    percentile_ci: tuple
    bca_ci: tuple
    se: float
    n_clusters: int
    B: int
    bca_note: str = ""

    def to_dict(self):
        return asdict(self)


def cluster_bootstrap(stat_fn, df, cluster_col="scenario_id", B=10000, seed=0, level=0.95):
    """Resample whole clusters with replacement; percentile and BCa intervals for stat_fn(df).

    stat_fn takes a DataFrame and returns a float. Rows of a drawn cluster are all kept, and a
    cluster drawn twice appears twice. BCa uses the bias z0 from the bootstrap distribution and the
    acceleration from a leave-one-cluster-out jackknife. If BCa is undefined (e.g. every replicate
    on one side of the estimate, or a constant jackknife) it falls back to the percentile interval
    and says so in bca_note.
    """
    codes = _codes(df[cluster_col].to_numpy(), len(df))
    g = int(codes.max()) + 1
    if g < 2:
        raise ValueError("need at least 2 clusters to bootstrap")
    members = [np.flatnonzero(codes == c) for c in range(g)]
    rng = np.random.default_rng(seed)
    theta = float(stat_fn(df))

    boot = np.empty(B)
    for b in range(B):
        draw = rng.integers(0, g, size=g)
        idx = np.concatenate([members[c] for c in draw])
        boot[b] = stat_fn(df.iloc[idx])

    alpha = 1 - level
    lo_p, hi_p = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    pct = (float(lo_p), float(hi_p))

    note = ""
    bca = pct
    frac = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)  # mid-rank: sane for discrete stats
    if 0 < frac < 1:
        z0 = _z(frac)
        jack = np.array([stat_fn(df.iloc[np.concatenate([members[k] for k in range(g) if k != c])])
                         for c in range(g)])
        dev = jack.mean() - jack
        denom = 6 * np.sum(dev ** 2) ** 1.5
        if denom > 0:
            acc = float(np.sum(dev ** 3) / denom)
            zs = [_z(alpha / 2), _z(1 - alpha / 2)]
            adj = []
            for zq in zs:
                t = z0 + zq
                adj.append(_N.cdf(z0 + t / (1 - acc * t)))
            bca = tuple(float(v) for v in np.quantile(boot, adj))
        else:
            note = "jackknife constant; BCa not defined, percentile shown"
    else:
        note = "bootstrap distribution entirely on one side of the estimate; BCa not defined, percentile shown"
    return BootstrapResult(theta, pct, bca, float(boot.std(ddof=1)), g, B, note)


# --------------------------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------------------------

def power_analysis(sigma_d, delta, alpha=0.05, power=0.8, deff=1.0):
    """Questions needed to detect a mean paired difference `delta` (two-sided test):

        n = (z_{alpha/2} + z_{power})^2 * sigma_d^2 / delta^2

    sigma_d is the sd of the per-question paired differences. `deff` (default 1 = independent
    questions) multiplies n to account for clustering; pass the design effect from clustered_se.
    Returns the rounded-up integer n.
    """
    if delta == 0:
        raise ValueError("delta must be non-zero")
    n = (_z(1 - alpha / 2) + _z(power)) ** 2 * sigma_d ** 2 / delta ** 2 * deff
    return int(math.ceil(n))


def minimum_detectable_effect(n, sigma_d, alpha=0.05, power=0.8, deff=1.0):
    """Smallest true mean difference detectable with n questions at the given alpha and power
    (inverse of power_analysis): (z_{alpha/2} + z_{power}) * sigma_d * sqrt(deff / n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return (_z(1 - alpha / 2) + _z(power)) * sigma_d * math.sqrt(deff / n)


# --------------------------------------------------------------------------------------------
# Loading results from Inspect logs
# --------------------------------------------------------------------------------------------

def load_results(log_path):
    """One Inspect .eval log -> DataFrame (id, scenario_id, variant, category, rubric_score,
    critical_pass). With --epochs k > 1, each question's scores are averaged over epochs.
    Attributes: df.attrs['model'], ['log'], ['judge']."""
    import pandas as pd
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    rows = []
    for s in log.samples or []:
        score = next(iter((s.scores or {}).values()), None)
        if score is None or not isinstance(score.value, dict):
            continue
        meta = s.metadata or {}
        rows.append({"id": s.id, "scenario_id": meta.get("cluster_id", s.id),
                     "variant": meta.get("variant"), "category": meta.get("category"),
                     "rubric_score": float(score.value["rubric_score"]),
                     "critical_pass": float(score.value["critical_pass"])})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.groupby(["id", "scenario_id", "variant", "category"], as_index=False, dropna=False)[
            ["rubric_score", "critical_pass"]].mean()
    df.attrs.update(model=log.eval.model, log=str(log_path),
                    judge=(log.eval.task_args or {}).get("judge"))
    return df
