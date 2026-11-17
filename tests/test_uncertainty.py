"""Tests of evals/uncertainty/ with mocked sampler and entailment judge. No model, network or logs."""
import math

import numpy as np
import pandas as pd
import pytest

from evals import uncertainty as un
from evals.uncertainty import report as rp
from evals.uncertainty.semantic_entropy import Sample


def same_meaning(meaning):
    """Fake bidirectional-entailment judge: replies entail each other iff ``meaning`` maps them to the same value."""
    return lambda q, a, b: meaning[a] == meaning[b]


# ---- semantic entropy ------------------------------------------------------------------------
def test_discrete_semantic_entropy_hand_values():
    # counts [5, 3, 2] of N=10: -(0.5 ln .5 + 0.3 ln .3 + 0.2 ln .2) = 0.346574 + 0.361192 + 0.321888 = 1.029653
    assert un.discrete_semantic_entropy([0] * 5 + [1] * 3 + [2] * 2) == pytest.approx(1.029653, abs=1e-6)
    assert un.discrete_semantic_entropy([7] * 10) == 0.0                        # one meaning: no entropy
    assert un.discrete_semantic_entropy(list(range(10))) == pytest.approx(math.log(10))   # all different: ln N
    with pytest.raises(ValueError):
        un.discrete_semantic_entropy([])


def test_clustering_is_bidirectional():
    meaning = {"go now": "A", "you should go now": "A", "wait": "B"}
    res = un.cluster_by_entailment("q", ["go now", "you should go now", "wait"], same_meaning(meaning))
    assert res.labels == [0, 0, 1] and res.n_clusters == 2 and res.n_unknown == 0
    # one-directional entailment ("short" entails "long" but not the reverse) must NOT merge
    one_way = lambda q, a, b: (a, b) == ("short", "long")                       # noqa: E731
    assert un.cluster_by_entailment("q", ["short", "long"], one_way).labels == [0, 1]


def test_unreadable_entailment_splits_clusters_and_is_counted():
    res = un.cluster_by_entailment("q", ["a", "b", "c"], lambda q, a, b: None)
    # b vs a: 2 calls; c vs a: 2 calls; c vs b: 2 calls = 6 unreadable
    assert res.labels == [0, 1, 2] and res.n_unknown == 6


def test_sequence_logprob_score_hand_values():
    # per-token mean of -logp: [-0.5, -1.5] -> 1.0 ; [-1.0] -> 1.0 ; mean 1.0.  Un-normalised sums 2.0 and 1.0 -> 1.5
    s = [Sample("a", (-0.5, -1.5)), Sample("b", (-1.0,))]
    assert un.sequence_logprob_score(s) == pytest.approx(1.0)
    assert un.sequence_logprob_score(s, length_normalize=False) == pytest.approx(1.5)
    assert math.isnan(un.sequence_logprob_score([Sample("a", None)]))


def test_score_case_uses_sampler_with_n_and_temperature():
    calls = []
    meaning = {f"r{i}": ("A" if i < 7 else "B") for i in range(10)}

    def sampler(prompt, n, temperature):
        calls.append((prompt, n, temperature))
        return [Sample(f"r{i}", (-1.0, -2.0)) for i in range(n)]

    cs = un.score_case("What now?", sampler, same_meaning(meaning))
    assert calls == [("What now?", 10, 1.0)]                                    # defaults: N=10, T=1
    # counts [7, 3]: -(0.7 ln .7 + 0.3 ln .3) = 0.249672 + 0.361192 = 0.610864
    assert cs.semantic_entropy == pytest.approx(0.610864, abs=1e-6) and cs.n_clusters == 2
    assert cs.logprob_score == pytest.approx(1.5) and cs.label == "exploratory" and len(cs.responses) == 10


def test_score_case_validates():
    with pytest.raises(ValueError, match="expected 10"):
        un.score_case("q", lambda p, n, t: [Sample("x")] * 3, lambda q, a, b: True)
    with pytest.raises(ValueError):
        un.score_case("q", lambda p, n, t: [], lambda q, a, b: True, n=1)


# ---- AUROC -----------------------------------------------------------------------------------
def test_auroc_hand_values():
    assert un.auroc([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 0]) == 1.0
    assert un.auroc([0.3, 0.4, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert un.auroc([0.5, 0.5], [1, 0]) == 0.5                                  # tie counts half
    # scores [1,2,3,4], failure [0,1,0,1]: failure scores {2,4}, pass scores {1,3}; pairs 2>1, 2<3, 4>1, 4>3 -> 3/4
    assert un.auroc([1, 2, 3, 4], [0, 1, 0, 1]) == 0.75
    assert math.isnan(un.auroc([1, 2], [0, 0])) and math.isnan(un.auroc([1, 2], [1, 1]))


def _signal_table(n=40, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    fail = (rng.random(n) < 0.3).astype(int)
    se = rng.normal(0, 1, n) + sep * fail
    lp = rng.normal(0, 1, n) + 0.3 * sep * fail
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "scenario_id": [f"s{i // 2}" for i in range(n)],
                         "semantic_entropy": se, "logprob_score": lp, "failure": fail})


def test_auroc_bootstrap_is_seeded_clustered_and_prints_interval_first():
    t = _signal_table()
    a = un.auroc_bootstrap(t.semantic_entropy, t.failure, t.scenario_id, B=300, seed=2)
    b = un.auroc_bootstrap(t.semantic_entropy, t.failure, t.scenario_id, B=300, seed=2)
    assert a == b and list(a)[0] == "ci_low" and a["label"] == "exploratory"
    assert a["ci_low"] <= a["auroc"] <= a["ci_high"] and a["n_clusters"] == 20 and 0 < a["n_boot_valid"] <= 300
    one = un.auroc_bootstrap(t.semantic_entropy, t.failure, np.zeros(len(t)), B=50)
    assert math.isnan(one["ci_low"]) and one["n_boot_valid"] == 0                # one cluster: no interval


def test_small_n_gives_wide_interval():
    small = _signal_table(n=12, seed=3)
    big = _signal_table(n=400, seed=3)
    w = lambda d: d["ci_high"] - d["ci_low"]                                      # noqa: E731
    ws = w(un.auroc_bootstrap(small.semantic_entropy, small.failure, small.scenario_id, B=400))
    wb = w(un.auroc_bootstrap(big.semantic_entropy, big.failure, big.scenario_id, B=400))
    assert ws > 2 * wb


def test_compare_predictors_shares_resamples():
    t = _signal_table()
    r = un.compare_predictors({"se": t.semantic_entropy, "lp": t.logprob_score}, t.failure, t.scenario_id, B=300)
    assert set(r["scores"]) == {"se", "lp"} and "se - lp" in r["difference"]
    d = r["difference"]["se - lp"]
    assert d["auroc_difference"] == pytest.approx(r["scores"]["se"]["auroc"] - r["scores"]["lp"]["auroc"])
    same = un.compare_predictors({"a": t.semantic_entropy, "b": t.semantic_entropy}, t.failure, t.scenario_id, B=100)
    assert same["difference"]["a - b"]["ci_low"] == 0 == same["difference"]["a - b"]["ci_high"]


# ---- risk-coverage ---------------------------------------------------------------------------
def test_uncertainty_risk_coverage_hand_example():
    # least uncertain first: scores [.1,.2,.3,.4], failure [0,0,1,0] -> risks 0, 0, 1/3, 1/4 ; AURC = 7/48
    rc = un.uncertainty_risk_coverage([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 0])
    assert rc["risk"].tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    s = un.selective_summary([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 0])
    assert s["aurc"] == pytest.approx(7 / 48) and s["random_baseline"] == pytest.approx(0.25)
    assert s["label"] == "exploratory"
    # constant score: all tied, failures first -> risks 1/1... = [1, 1/2, 1/3, 1/4]
    flat = un.uncertainty_risk_coverage([1, 1, 1, 1], [0, 0, 1, 0])
    assert flat["risk"].tolist() == pytest.approx([1, 1 / 2, 1 / 3, 1 / 4])


# ---- joining scores to outcomes --------------------------------------------------------------
def test_build_eval_table():
    scores = pd.DataFrame({"case_id": ["c1", "c2", "c3", "c4"], "semantic_entropy": [0.5, 0.1, 0.9, np.nan],
                           "logprob_score": [1.0, 0.5, 2.0, 1.0]})
    samples = pd.DataFrame({"case_id": ["c1", "c1", "c2", "c2", "c3", "c4"], "epoch": [1, 2, 1, 2, 1, 1],
                            "critical_pass": [1.0, 0.0, 1.0, 1.0, np.nan, 1.0],
                            "scored": [True, True, True, True, False, True]})
    t = un.build_eval_table(scores, samples)
    # c1: mean critical_pass 0.5 < 1 -> failure ; c2: 1.0 -> pass ; c3 unscored and c4 NaN score are dropped
    assert t["case_id"].tolist() == ["c1", "c2"] and t["failure"].tolist() == [1, 0]
    assert t.attrs["n_dropped"] == 2 and t["scenario_id"].tolist() == ["c1", "c2"]


# ---- conformal -------------------------------------------------------------------------------
def test_calibrate_refuses_tiny_calibration_sets():
    nine = np.arange(1, 10, dtype=float)
    with pytest.raises(un.CalibrationTooSmall, match=r"1/alpha = 10"):
        un.calibrate(nine, np.zeros(9), alpha=0.1)
    # failing cases do not count: 9 correct + 5 failing is still too small
    with pytest.raises(un.CalibrationTooSmall):
        un.calibrate(np.r_[nine, np.full(5, 99.0)], np.r_[np.zeros(9), np.ones(5)], alpha=0.1)
    # exactly 10 correct: k = ceil(11 * 0.9) = 10 -> q_hat is the largest (10)
    assert un.calibrate(np.arange(1, 11, dtype=float), np.zeros(10), alpha=0.1) == 10
    # alpha = .2 needs 5: k = ceil(6 * 0.8) = 5
    assert un.calibrate([1, 2, 3, 4, 5], np.zeros(5), alpha=0.2) == 5
    with pytest.raises(un.CalibrationTooSmall):
        un.calibrate([1, 2, 3, 4], np.zeros(4), alpha=0.2)
    with pytest.raises(ValueError):
        un.calibrate([1.0], [0], alpha=0)


def test_split_by_scenario():
    t = _signal_table(n=40)
    cal, test = un.split_by_scenario(t, 0.5, seed=1)
    assert len(cal) + len(test) == 40 and not set(cal.scenario_id) & set(test.scenario_id)
    again, _ = un.split_by_scenario(t, 0.5, seed=1)
    pd.testing.assert_frame_equal(cal, again)
    with pytest.raises(ValueError):
        un.split_by_scenario(t.assign(scenario_id="one"))
    with pytest.raises(ValueError):
        un.split_by_scenario(t, 1.5)


def test_conformal_abstention_hand_example():
    cal = pd.DataFrame({"semantic_entropy": list(map(float, range(1, 11))) + [99.0], "failure": [0] * 10 + [1]})
    test = pd.DataFrame({"semantic_entropy": [3.0, 11.0, 5.0, 12.0, 7.0], "failure": [0, 0, 1, 1, 0]})
    # q_hat = 10 (ten correct calibration cases, alpha .1). answered = score <= 10 -> [T, F, T, F, T]
    # correct test cases: idx 0, 1, 4 ; answered among them: idx 0, 4 -> coverage 2/3
    # abstention 2/5 ; failures idx 2 (answered), 3 (abstained) -> caught 1/2 ; answered 3, failing among them 1 -> 1/3
    r = un.conformal_abstention(cal, test, alpha=0.1)
    assert r.q_hat == 10 and r.n_cal == 10 and r.label == "exploratory"
    assert r.coverage == pytest.approx(2 / 3) and r.abstention_rate == pytest.approx(0.4)
    assert r.failures_caught == pytest.approx(0.5) and r.risk_answered == pytest.approx(1 / 3)
    assert r.risk_all == pytest.approx(0.4)
    with pytest.raises(un.CalibrationTooSmall):
        un.conformal_abstention(cal.iloc[5:], test, alpha=0.1)


def test_conformal_coverage_on_synthetic_data():
    # correct cases score Exp(1), failing cases Exp(mean 2.5). Averaged over random splits, the share of correct
    # test cases that are answered must be about 1 - alpha = 0.9 (the guarantee).
    rng = np.random.default_rng(0)
    n = 1500
    fail = (rng.random(n) < 0.3).astype(int)
    se = np.where(fail == 1, rng.exponential(2.5, n), rng.exponential(1.0, n))
    t = pd.DataFrame({"case_id": range(n), "scenario_id": [f"s{i}" for i in range(n)], "semantic_entropy": se,
                      "failure": fail})
    cov, caught = [], []
    for seed in range(150):
        cal, test = un.split_by_scenario(t, 0.5, seed=seed)
        r = un.conformal_abstention(cal, test, alpha=0.1)
        cov.append(r.coverage)
        caught.append(r.failures_caught)
    assert 0.88 < np.mean(cov) < 0.93, np.mean(cov)
    # q_hat ~ ln 10 (the 90th percentile of Exp(1)); P(Exp(mean 2.5) > ln 10) = exp(-ln 10 / 2.5) = 0.398
    assert abs(np.mean(caught) - math.exp(-math.log(10) / 2.5)) < 0.04, np.mean(caught)


# ---- report ----------------------------------------------------------------------------------
def test_report_is_labelled_exploratory_and_prints_ci_first():
    rep = un.uncertainty_report(_signal_table(n=40), alpha=0.1, B=200)
    text = un.format_report(rep)
    assert text.startswith(rp.BANNER) and rep["label"] == "exploratory"
    assert all("exploratory" in line for line in text.splitlines() if line.startswith("###"))
    assert text.index("95% CI for AUROC") < text.index("AUROC (point estimate)")
    row = next(line for line in text.splitlines() if line.startswith("| semantic_entropy | ["))
    assert row.index("[") < row.rindex("|")                                       # interval cell precedes the estimate
    assert "coverage of correctly handled test cases" in text


def test_report_degrades_gracefully_on_a_tiny_set():
    rep = un.uncertainty_report(_signal_table(n=12, seed=5), alpha=0.1, B=100)
    text = un.format_report(rep)
    assert "not run: calibration set too small" in text and "Caution" in text


def test_end_to_end_with_mocked_models():
    # three cases; the sampler gives different spread per case, the judge groups by a meaning table
    meanings = {}

    def sampler(prompt, n, temperature):
        kinds = {"stable": 1, "mixed": 3, "chaotic": 10}[prompt]
        out = []
        for i in range(n):
            text = f"{prompt}-{i % kinds}"
            meanings[text] = text
            out.append(Sample(text, (-0.1 * (i % kinds + 1),)))
        return out

    res = {p: un.score_case(p, sampler, lambda q, a, b: meanings[a] == meanings[b])
           for p in ("stable", "mixed", "chaotic")}
    ses = [res[p].semantic_entropy for p in ("stable", "mixed", "chaotic")]
    assert ses[0] == 0.0 and ses[0] < ses[1] < ses[2] and ses[2] == pytest.approx(math.log(10))
