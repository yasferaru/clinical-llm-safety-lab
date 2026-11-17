"""Checks of the error-bar statistics. Run from the project folder: python tests/test_inference.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "stats"))
from inference import (mean_with_se, clustered_se, paired_difference, cluster_bootstrap,  # noqa: E402
                       power_analysis, minimum_detectable_effect)


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


# ---- hand-computed -------------------------------------------------------------------------
def test_mean_with_se():
    r = mean_with_se([1, 2, 3, 4])
    assert close(r.mean, 2.5) and close(r.se, math.sqrt(5 / 3) / 2) and r.n == 4   # sd=1.29099, /2
    b = mean_with_se([1, 1, 0, 0, 0], bernoulli=True)                              # p=0.4, n=5
    assert close(b.mean, 0.4) and close(b.se, math.sqrt(0.4 * 0.6 / 5))            # 0.219089
    try:
        mean_with_se([0.5, 1], bernoulli=True)
        raise AssertionError("bernoulli should reject non-0/1 data")
    except ValueError:
        pass


def test_clustered_se_hand_example():
    # x=1..4, clusters {1,2} and {3,4}: xbar=2.5, residuals -1.5 -.5 .5 1.5, cluster sums -2 and 2
    # SE = sqrt(4 + 4) / 4 = 0.707107
    r = clustered_se([1, 2, 3, 4], ["A", "A", "B", "B"])
    assert close(r.se, math.sqrt(8) / 4)
    assert (r.n, r.n_clusters) == (4, 2)
    # ANOVA ICC: SSB=4, SSW=1 -> MSB=4, MSW=.5, m0=2 -> (4-.5)/(4+.5) = 0.77778; DEFF = 1+(2-1)*ICC
    assert close(r.icc, 3.5 / 4.5) and close(r.deff, 1 + 3.5 / 4.5)
    # singleton clusters reproduce the naive SE up to the (n-1)/n factor
    s = clustered_se([1, 2, 3, 4], [1, 2, 3, 4])
    assert close(s.se, math.sqrt(1.25) / 2)           # sqrt(sum resid^2)/n = sqrt(5)/4 = 0.559
    assert close(s.se, mean_with_se([1, 2, 3, 4]).se * math.sqrt(3 / 4))
    assert math.isnan(s.icc)                           # ICC undefined when every cluster has one row


def test_single_cluster_is_undefined():
    r = clustered_se([1, 2, 3], ["A", "A", "A"])
    assert math.isnan(r.se) and math.isnan(r.icc)
    p = paired_difference([1, 2, 3], [0, 0, 0], ["A", "A", "A"])
    assert math.isnan(p.se) and math.isnan(p.p_value)


def test_paired_difference_hand_example():
    # d = [2,0,1,1,0,0], clusters (AA)(BB)(CC): mean 2/3, cluster residual sums 2/3, 2/3, -4/3
    # SE = sqrt(4/9 + 4/9 + 16/9)/6 = 1.63299/6 = 0.272166 ; z = sqrt(6) ; p = 0.014306
    cl = list("AABBCC")
    b = [0, 0, 0, 0, 0, 0]
    a = [2, 0, 1, 1, 0, 0]
    r = paired_difference(a, b, cl)
    assert close(r.mean_diff, 2 / 3) and close(r.se, math.sqrt(24 / 9) / 6)
    assert close(r.p_value, 0.014306, 1e-5)
    assert close(r.ci_low, 2 / 3 - 1.959964 * r.se) and close(r.ci_high, 2 / 3 + 1.959964 * r.se)
    assert (r.n, r.n_clusters) == (6, 3)
    # identical models: zero difference, p = 1
    z = paired_difference(a, a, cl)
    assert z.mean_diff == 0 and z.p_value == 1.0


def test_pairing_gain():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(30), 3)
    base = rng.normal(size=90) + np.repeat(rng.normal(size=30), 3)      # shared question difficulty
    a, b = base + 0.3 + rng.normal(scale=0.2, size=90), base + rng.normal(scale=0.2, size=90)
    r = paired_difference(a, b, cl)
    assert r.pairing_gain > 3                                          # shared difficulty cancels
    assert close(r.pairing_gain, r.se_unpaired / r.se, 1e-9)
    # independent models: no gain expected
    c = rng.normal(size=90) + np.repeat(rng.normal(size=30), 3)
    assert paired_difference(base, c, cl).pairing_gain < 1.6


def test_power():
    # (1.959964 + 0.841621)^2 = 7.84888 ; sigma_d=.3, delta=.1 -> 7.84888 * 9 = 70.64 -> 71
    assert power_analysis(0.3, 0.1) == 71
    assert power_analysis(0.3, 0.1, deff=2.0) == 142
    assert close(minimum_detectable_effect(71, 0.3), 2.801585 * 0.3 / math.sqrt(71))   # 0.09974
    n = power_analysis(0.4, 0.15)
    assert minimum_detectable_effect(n, 0.4) <= 0.15 < minimum_detectable_effect(n - 1, 0.4)


# ---- bootstrap -----------------------------------------------------------------------------
def _clustered_frame(g=12, m=3, seed=3):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(g), m)
    y = rng.normal(size=g)[cl] + rng.normal(scale=0.7, size=g * m)
    return pd.DataFrame({"scenario_id": cl, "y": y})


def test_cluster_bootstrap():
    df = _clustered_frame()
    stat = lambda d: d["y"].mean()                                    # noqa: E731
    r = cluster_bootstrap(stat, df, B=2000, seed=7)
    assert close(r.estimate, df["y"].mean(), 1e-12) and r.n_clusters == 12
    analytic = clustered_se(df["y"], df["scenario_id"]).se
    assert abs(r.se / analytic - 1) < 0.15                            # same quantity, two routes
    for lo, hi in (r.percentile_ci, r.bca_ci):
        assert lo < r.estimate < hi
    assert abs(r.bca_ci[0] - r.percentile_ci[0]) < 0.35 and abs(r.bca_ci[1] - r.percentile_ci[1]) < 0.35
    again = cluster_bootstrap(stat, df, B=300, seed=7)
    assert again.percentile_ci == cluster_bootstrap(stat, df, B=300, seed=7).percentile_ci
    # whole clusters move together: a statistic that counts rows per draw is a multiple of cluster size
    sizes = cluster_bootstrap(lambda d: len(d), df, B=200, seed=1)
    assert sizes.percentile_ci[0] % 3 == 0 and sizes.percentile_ci[1] % 3 == 0
    # degenerate data must not crash BCa
    flat = pd.DataFrame({"scenario_id": [1, 1, 2, 2, 3, 3], "y": [1.0] * 6})
    d = cluster_bootstrap(stat, flat, B=200, seed=0)
    assert d.bca_ci == (1.0, 1.0) and d.bca_note


# ---- coverage simulation -------------------------------------------------------------------
def test_coverage_naive_undercovers_clustered_ok():
    """60 scenarios x 5 questions, ICC 0.5 (design effect ~3): the naive 95% CI misses the true mean
    far too often; the clustered one is close to 95%."""
    rng = np.random.default_rng(2024)
    G, M, SIMS = 60, 5, 4000
    u = rng.normal(size=(SIMS, G, 1))                 # scenario effect, var 1
    x = (u + rng.normal(size=(SIMS, G, M))).reshape(SIMS, G, M)   # question noise, var 1; true mean 0
    n = G * M
    xbar = x.mean(axis=(1, 2))
    naive_se = x.reshape(SIMS, n).std(axis=1, ddof=1) / math.sqrt(n)
    clust_se = np.sqrt(((x - xbar[:, None, None]).sum(axis=2) ** 2).sum(axis=1)) / n
    zc = 1.959964
    cov_naive = np.mean(np.abs(xbar) <= zc * naive_se)
    cov_clust = np.mean(np.abs(xbar) <= zc * clust_se)
    print(f"coverage of nominal 95% CI: naive {cov_naive:.3f}, clustered {cov_clust:.3f}")
    assert cov_naive < 0.85                           # expected ~0.75
    assert 0.93 < cov_clust < 0.97


# ---- current minimum detectable effect -----------------------------------------------------
def test_print_current_mde():
    lines = [x for x in (ROOT / "evals" / "cases.jsonl").read_text().splitlines() if x.strip()]
    n_cases = len(lines)
    n_clusters = len({json.loads(x)["cluster_id"] for x in lines})
    sigma_d = 0.30                                    # ASSUMED sd of per-question paired differences
    mde = minimum_detectable_effect(n_cases, sigma_d)
    print(f"repo today: {n_cases} cases in {n_clusters} scenarios; assuming sigma_d={sigma_d}, "
          f"MDE = {mde:.2f} on a 0-1 score (80% power, alpha .05, independence assumed; "
          f"{minimum_detectable_effect(n_cases, sigma_d, deff=2.0):.2f} at DEFF 2)")
    for target in (40, 150):                          # TODO.md: ~40 cases now, ~150 for v0.1
        print(f"  at n={target}: MDE = {minimum_detectable_effect(target, sigma_d):.2f}")
    # MDE = (z_.975 + z_.80) * sigma_d / sqrt(n) = 2.8016 * 0.30 / sqrt(n); independent of how many cases exist
    assert abs(mde - 2.8016 * sigma_d / n_cases ** 0.5) < 1e-3
    assert minimum_detectable_effect(4, sigma_d) > 0.25   # with 4 cases only huge gaps are detectable


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("inference statistics: all checks passed")
