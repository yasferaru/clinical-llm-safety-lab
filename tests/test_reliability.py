"""Reliability estimators against hand-computed values and brute-force enumeration."""
from itertools import combinations
from math import isnan

import pandas as pd
import pytest

from evals.results import SAMPLE_COLUMNS
from evals.stats.reliability import (case_table, cluster_bootstrap, consistency_summary, disagreement_rate,
                                     max_k, pass_at_k, pass_hat_k, reliability_summary, worst_at_k)
from tests.fixtures import make_samples


# ---- per-case estimators: hand-computed ----
def test_pass_hat_k_hand():
    assert pass_hat_k(3, 4, 2) == 0.5          # C(3,2)/C(4,2) = 3/6
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(3, 4, 4) == 0.0          # C(3,4) = 0
    assert pass_hat_k(2, 5, 1) == 0.4          # k=1 is the plain success rate


def test_pass_at_k_hand():
    assert pass_at_k(1, 4, 2) == 0.5           # 1 - C(3,2)/C(4,2) = 1 - 3/6
    assert pass_at_k(0, 4, 3) == 0.0
    assert pass_at_k(1, 4, 4) == 1.0           # C(3,4) = 0
    assert pass_at_k(2, 5, 1) == 0.4


@pytest.mark.parametrize("n", range(1, 7))
def test_pass_metrics_match_enumeration(n):
    for c in range(n + 1):
        trials = [1] * c + [0] * (n - c)
        for k in range(1, n + 1):
            subsets = list(combinations(trials, k))
            assert pass_hat_k(c, n, k) == pytest.approx(sum(all(s) for s in subsets) / len(subsets))
            assert pass_at_k(c, n, k) == pytest.approx(sum(any(s) for s in subsets) / len(subsets))


def test_worst_at_k_hand():
    # sorted [0, 0.5, 1], k=2: weights C(2,1)=2, C(1,1)=1, C(0,1)=0 over C(3,2)=3 -> (0 + 0.5 + 0) / 3
    assert worst_at_k([1, 0, 0.5], 2) == pytest.approx(1 / 6)
    assert worst_at_k([0.2, 0.8], 1) == pytest.approx(0.5)   # k=1 is the mean
    assert worst_at_k([0.2, 0.8, 0.5], 3) == pytest.approx(0.2)  # k=n is the minimum


@pytest.mark.parametrize("scores", [[0.3], [0.9, 0.1], [0.5, 0.5, 0.2], [0.0, 1.0, 0.25, 0.75, 0.4],
                                    [0.6, 0.1, 0.6, 0.9, 0.3, 0.0]])
def test_worst_at_k_matches_enumeration(scores):
    for k in range(1, len(scores) + 1):
        subsets = list(combinations(scores, k))
        assert worst_at_k(scores, k) == pytest.approx(sum(min(s) for s in subsets) / len(subsets))


def test_k_larger_than_n_is_an_error():
    for f in (lambda: pass_hat_k(1, 2, 3), lambda: pass_at_k(1, 2, 3), lambda: worst_at_k([0.1, 0.2], 3),
              lambda: pass_hat_k(1, 2, 0), lambda: pass_hat_k(3, 2, 1)):
        with pytest.raises(ValueError):
            f()


def test_disagreement_rate_hand():
    assert disagreement_rate(1, 2) == 1.0      # 1*1 / C(2,2)
    assert disagreement_rate(2, 4) == pytest.approx(2 / 3)   # 2*2 / 6
    assert disagreement_rate(3, 3) == 0.0
    assert isnan(disagreement_rate(1, 1))


# ---- tables ----
def samples(rows):
    base = {c: None for c in SAMPLE_COLUMNS}
    return pd.DataFrame([{**base, "model": "m", "run_id": "r", "scored": True, "variant": "base", **r}
                         for r in rows])[SAMPLE_COLUMNS]


# The README case: two 1-epoch runs, 2 of 4 cases disagree on critical_pass.
README_RUNS = samples(
    [{"run_id": "run1", "epoch": 1, "case_id": c, "scenario_id": c[:10], "critical_pass": p, "rubric_score": s}
     for c, p, s in [("triage-001-base", 0.0, 0.29), ("triage-001-pressure", 1.0, 1.0),
                     ("triage-002-base", 0.0, 0.0), ("triage-002-pressure", 0.0, 0.38)]]
    + [{"run_id": "run2", "epoch": 1, "case_id": c, "scenario_id": c[:10], "critical_pass": p, "rubric_score": s}
       for c, p, s in [("triage-001-base", 0.0, 0.0), ("triage-001-pressure", 1.0, 1.0),
                       ("triage-002-base", 1.0, 1.0), ("triage-002-pressure", 1.0, 1.0)]])


def test_readme_disagreement_two_of_four():
    out = consistency_summary(README_RUNS).iloc[0]
    assert (out["n_cases"], out["n_disagree"], out["frac_disagree"]) == (4, 2, 0.5)
    assert out["mean_flip_rate"] == 0.5          # flip rates 0, 0, 1, 1


def test_readme_pass_hat_k():
    s = reliability_summary(README_RUNS, ks=(1, 2, 4, 8), B=500)
    assert list(s["k"]) == [1, 2]                # only 2 trials per case
    k1, k2 = s.iloc[0], s.iloc[1]
    assert k1["pass_hat_k"] == 0.5               # 4 successes / 8 trials
    assert k2["pass_hat_k"] == 0.25              # only triage-001-pressure passed both
    assert k2["pass_at_k"] == 0.75               # 3 of 4 cases passed at least once
    # worst@2 = mean of per-case minima: (0, 1, 0, 0.38) / 4
    assert k2["worst_at_k"] == pytest.approx((0 + 1 + 0 + 0.38) / 4)
    assert k2["n_scenarios"] == 2
    assert k2["pass_hat_k_lo"] <= k2["pass_hat_k"] <= k2["pass_hat_k_hi"]


def test_unequal_n_uses_per_case_n():
    rows = [{"epoch": e, "case_id": "a", "scenario_id": "A", "critical_pass": 1.0, "rubric_score": 1.0}
            for e in (1, 2, 3)]
    rows += [{"epoch": e, "case_id": "b", "scenario_id": "B", "critical_pass": v, "rubric_score": v}
             for e, v in ((1, 1.0), (2, 0.0))]
    cases = case_table(samples(rows))
    assert list(cases["n"]) == [3, 2] and max_k(cases) == 2
    s = reliability_summary(samples(rows), ks=(1, 2, 3), B=200)
    assert list(s["k"]) == [1, 2]                # k=3 skipped: case b has only 2 trials
    # k=1: case a 3/3, case b 1/2 -> (1 + 0.5) / 2
    assert s.iloc[0]["pass_hat_k"] == 0.75
    assert (s.iloc[0]["n_min"], s.iloc[0]["n_max"]) == (2, 3)


def test_unscored_rows_are_dropped_and_success_is_explicit():
    rows = [{"epoch": 1, "case_id": "a", "scenario_id": "A", "critical_pass": 1.0, "rubric_score": 1.0},
            {"epoch": 2, "case_id": "a", "scenario_id": "A", "critical_pass": float("nan"),
             "rubric_score": float("nan"), "scored": False},
            {"epoch": 1, "case_id": "b", "scenario_id": "B", "critical_pass": 0.5, "rubric_score": 0.5}]
    cases = case_table(samples(rows))
    assert list(cases["n"]) == [1, 1]
    assert list(cases["c"]) == [1, 0]            # 0.5 is not == 1, so not a success


# ---- bootstrap stub ----
def test_bootstrap_two_clusters_hand():
    df = pd.DataFrame({"scenario_id": ["A", "B"], "v": [0.0, 1.0]})
    r = cluster_bootstrap(lambda d: d["v"].mean(), df, B=4000, seed=0)
    assert r.estimate == 0.5 and r.n_clusters == 2
    assert r.percentile_ci == (0.0, 1.0)         # replicates are 0, .5, 1 with probs 1/4, 1/2, 1/4


def test_bootstrap_keeps_clusters_whole_and_is_seeded():
    df = pd.DataFrame({"scenario_id": ["A", "A", "B", "C", "C", "C"], "v": range(6)})
    r1 = cluster_bootstrap(lambda d: len(d), df, B=300, seed=3)
    assert cluster_bootstrap(lambda d: len(d), df, B=300, seed=3) == r1
    # 3 draws of clusters sized 2, 1, 3: a replicate has 3 to 9 rows, never a partial cluster
    assert r1.percentile_ci[0] >= 3 and r1.percentile_ci[1] <= 9
    assert r1.estimate == 6
    with pytest.raises(ValueError):
        cluster_bootstrap(lambda d: 0.0, df[df["scenario_id"] == "A"], B=10)


def test_single_scenario_gives_nan_ci():
    rows = [{"epoch": e, "case_id": c, "scenario_id": "A", "critical_pass": 1.0, "rubric_score": 1.0}
            for e in (1, 2) for c in ("a-base", "a-pressure")]
    s = reliability_summary(samples(rows), ks=(1,), B=50)
    assert isnan(s.iloc[0]["pass_hat_k_lo"]) and s.iloc[0]["pass_hat_k"] == 1.0


# ---- fixture: realistic ordering ----
def test_fixture_orderings():
    s = reliability_summary(make_samples(seed=0, epochs=4), ks=(1, 2, 4), B=300)
    for _, g in s.groupby("model"):
        g = g.set_index("k")
        assert g.loc[1, "pass_hat_k"] == pytest.approx(g.loc[1, "pass_at_k"])   # equal at k=1
        assert g.loc[4, "pass_hat_k"] <= g.loc[2, "pass_hat_k"] <= g.loc[1, "pass_hat_k"]
        assert g.loc[4, "pass_at_k"] >= g.loc[2, "pass_at_k"] >= g.loc[1, "pass_at_k"]
        assert g.loc[4, "worst_at_k"] <= g.loc[1, "worst_at_k"]
    assert (s["n_scenarios"] == 8 + 4).all()
