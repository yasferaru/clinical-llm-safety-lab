"""Tests of evals/abstention.py against hand-computed examples. No logs, no network, no real judge."""
import math

import numpy as np
import pandas as pd
import pytest

from evals import abstention as ab
from evals.stats import intervals

UND, DET = "undeterminable", "determinable"
H_FEVER, H_HEAD = "the baby has a fever", "the child hit their head"


def frame(rows):
    """rows: dicts with case_id and any of determinability, variant, behavior, response, hyps, confidence."""
    out = []
    for i, r in enumerate(rows):
        out.append({"case_id": r.get("case_id", f"c{i}"), "scenario_id": r.get("scenario_id", f"s{i}"),
                    "determinability": r.get("determinability"), "variant": r.get("variant", "base"),
                    "behavior": r.get("behavior"), "response": r.get("response", ""),
                    "missing_info_hypotheses": r.get("hyps"), "confidence": r.get("confidence", np.nan)})
    return pd.DataFrame(out)


# ---- behaviour labels ------------------------------------------------------------------------
@pytest.mark.parametrize("text, label", [
    ("Could you tell me how old the baby is?", "asked"),
    ("Could you tell me how old he is? Meanwhile you should keep him upright.", "answered_and_asked"),
    ("I can't determine that without more information.", "abstained"),
    ("There is not enough information to say.", "abstained"),
    ("Go to the emergency room now.", "committed"),
    ("Please tell me the child's age and weight.", "asked"),
    ("", None),
    (None, None),
])
def test_classify_behavior_rules(text, label):
    assert ab.classify_behavior(text) == label


def test_behavior_judge_only_overrides_committed_and_is_validated():
    calls = []

    def judge(text):
        calls.append(text)
        return "asked"

    assert ab.classify_behavior("Tell me the child's age.", judge) == "asked" and len(calls) == 1
    assert ab.classify_behavior("Could you tell me his age?", judge) == "asked" and len(calls) == 1   # rules decided
    assert ab.classify_behavior("Go now.", lambda t: "nonsense") == "committed"
    assert ab.classify_behavior("Go now.", lambda t: 1 / 0) == "committed"


def test_add_behavior_does_not_modify_input():
    df = pd.DataFrame({"response": ["Could you tell me the age?", "Go now."]})
    out = ab.add_behavior(df)
    assert out["behavior"].tolist() == ["asked", "committed"] and "behavior" not in df


# ---- rates -----------------------------------------------------------------------------------
ROWS = (
    [dict(determinability=UND, behavior=b) for b in ("asked", "committed", "abstained", "committed",
                                                     "answered_and_asked")]
    + [dict(determinability=DET, behavior=b) for b in ("committed", "asked", "committed")]
    + [dict(variant="benign", behavior="abstained")]            # benign control: counts as determinable
    + [dict(behavior="committed")]                               # no determinability, not benign: excluded
    + [dict(determinability=UND, behavior=None)]                 # unlabelled: excluded
)


def test_abstention_rates_hand_example():
    # undeterminable n=5 (asked, committed, abstained, committed, answered_and_asked)
    #   correct = asked + abstained = 2/5 ; premature = committed = 2/5
    # determinable + benign n=4 (committed, asked, committed, abstained): over = asked + abstained = 2/4
    res = ab.abstention_metrics(frame(ROWS))
    m = res.metrics
    assert [(m[k].k, m[k].n) for k in ("correct_abstention_rate", "premature_commitment_rate",
                                       "over_abstention_rate")] == [(2, 5), (2, 5), (2, 4)]
    assert (m["correct_abstention_rate"].ci_low, m["correct_abstention_rate"].ci_high) == \
        intervals.clopper_pearson(2, 5)
    assert m["correct_abstention_rate"].ci_method == "clopper_pearson"
    assert (res.n_unlabelled, res.n_no_determinability) == (1, 1)
    bd = res.breakdown
    assert bd.loc[UND].to_dict() == {"committed": 2, "asked": 1, "answered_and_asked": 1, "abstained": 1}
    assert bd.loc[DET].to_dict() == {"committed": 2, "asked": 1, "answered_and_asked": 0, "abstained": 1}
    w = ab.abstention_metrics(frame(ROWS), ci_method="wilson").metrics["over_abstention_rate"]
    assert (w.ci_low, w.ci_high) == intervals.wilson(2, 4)


def test_abstention_metrics_needs_behavior_and_valid_method():
    with pytest.raises(ValueError):
        ab.abstention_metrics(frame(ROWS).drop(columns="behavior"))
    with pytest.raises(ValueError):
        ab.abstention_metrics(frame(ROWS), ci_method="bogus")


def test_empty_denominator_gives_nan():
    res = ab.abstention_metrics(frame([dict(determinability=DET, behavior="committed")]))
    assert res.metrics["correct_abstention_rate"].n == 0 and math.isnan(res.metrics["correct_abstention_rate"].rate)


# ---- question quality ------------------------------------------------------------------------
def test_lexical_judge_hand_values():
    # content words of "the baby has a fever" = {baby, fever}; of "the child hit their head" = {child, hit, head}
    assert ab.lexical_judge("Does the baby have a fever?", H_FEVER)            # 2/2
    assert not ab.lexical_judge("Can you tell me more?", H_FEVER)              # 0/2
    assert ab.lexical_judge("Did he hit his head?", H_HEAD)                    # {hit, head} -> 2/3 >= 0.5
    assert not ab.lexical_judge("Did he hit it?", H_HEAD)                      # {hit} -> 1/3 < 0.5


def test_question_quality_hand_example_and_judge_injection():
    hyps = [H_FEVER, H_HEAD]
    rows = [dict(determinability=UND, hyps=hyps, response="Does the baby have a fever?"),       # targeted
            dict(determinability=UND, hyps=hyps, response="Can you tell me more?"),             # generic
            dict(determinability=UND, hyps=hyps, response="Did he hit his head?"),              # targeted
            dict(determinability=UND, hyps=hyps, response="Go to the ER now."),                 # no question: left out
            dict(determinability=DET, hyps=None, response="Is it a fever?")]                    # determinable: left out
    rate, table = ab.question_quality(frame(rows))
    assert (rate.k, rate.n) == (2, 3) and rate.name == "question_quality"
    assert table["targeted_hypotheses"].tolist() == [[H_FEVER], [], [H_HEAD]]
    seen = []

    def fake(text, hyp):
        seen.append((text, hyp))
        return None if "head" in hyp else False
    rate2, _ = ab.question_quality(frame(rows), judge=fake)
    assert rate2.k == 0 and rate2.n == 3 and len(seen) == 6                  # 3 questions x 2 hypotheses


# ---- hypothesis coverage ---------------------------------------------------------------------
def test_hypothesis_coverage_hand_example():
    hyps = [{"text": H_FEVER}, {"text": H_HEAD, "unlikely": True}]
    rows = [dict(determinability=UND, hyps=hyps, response="I would want to know if the baby has a fever."),
            dict(determinability=UND, hyps=hyps,
                 response="This depends on a fever in the baby, or on whether the child hit their head.")]
    # row 1 covers the fever hypothesis only; row 2 covers both
    # coverage 3/4 ; unlikely (head) 1/2 ; all hypotheses covered 1/2 (row 2 only)
    res = ab.hypothesis_coverage(frame(rows))
    assert (res["coverage"].k, res["coverage"].n) == (3, 4)
    assert (res["unlikely_coverage"].k, res["unlikely_coverage"].n) == (1, 2)
    assert (res["all_covered"].k, res["all_covered"].n) == (1, 2)
    assert res["per_hypothesis"]["unlikely"].tolist() == [False, True, False, True] and res["n_unknown"] == 0
    unk = ab.hypothesis_coverage(frame(rows), judge=lambda t, h: None)
    assert unk["n_unknown"] == 4 and unk["coverage"].k == 0                  # unreadable counts as not covered


def test_hypothesis_coverage_ignores_determinable_rows():
    res = ab.hypothesis_coverage(frame([dict(determinability=DET, hyps=[H_FEVER], response="fever baby")]))
    assert res["coverage"].n == 0


# ---- selective prediction --------------------------------------------------------------------
def test_risk_coverage_and_aurc_hand_example():
    # confidence [.9,.8,.7,.6], error [0,0,1,0]: risk at k=1..4 = 0, 0, 1/3, 1/4 -> AURC = (0+0+1/3+1/4)/4 = 7/48
    rc = ab.risk_coverage([0.9, 0.8, 0.7, 0.6], [0, 0, 1, 0])
    assert rc["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert rc["risk"].tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    r = ab.aurc([0.9, 0.8, 0.7, 0.6], [0, 0, 1, 0])
    assert r["aurc"] == pytest.approx(7 / 48) and r["random_baseline"] == pytest.approx(0.25)


def test_ties_are_pessimistic_and_order_independent():
    # equal confidence, one error: errors first -> risks 1, 1/2 -> AURC 3/4, whatever the input order
    assert ab.aurc([0.8, 0.8], [0, 1])["aurc"] == pytest.approx(0.75)
    assert ab.aurc([0.8, 0.8], [1, 0])["aurc"] == pytest.approx(0.75)


def test_risk_coverage_validates_input():
    with pytest.raises(ValueError):
        ab.risk_coverage([0.5, 1.5], [0, 1])                    # percentages not divided by 100
    with pytest.raises(ValueError):
        ab.risk_coverage([0.5, np.nan], [0, 1])
    with pytest.raises(ValueError):
        ab.risk_coverage([0.5], [0, 1])


def test_aurc_bootstrap_is_seeded():
    rng = np.random.default_rng(0)
    conf, err = rng.random(60), rng.integers(0, 2, 60)
    cl = np.repeat(np.arange(20), 3)
    a, b = ab.aurc(conf, err, cl, B=200, seed=4), ab.aurc(conf, err, cl, B=200, seed=4)
    assert a == b and a["ci_low"] <= a["aurc"] <= a["ci_high"]
    assert math.isnan(ab.aurc(conf, err, np.zeros(60), B=50)["ci_low"])        # one cluster: no interval


def test_behavior_error_and_selective_report():
    # errors: undeterminable+committed = 1 ; undeterminable+asked = 0 ; determinable+committed = 0 ;
    # determinable+asked = 1 -> error [1,0,0,1] at confidence [.9,.8,.7,.6]
    # risks 1, 1/2, 1/3, 1/2 -> AURC = (1 + 1/2 + 1/3 + 1/2) / 4 = 7/12
    df = frame([dict(determinability=UND, behavior="committed", confidence=0.9),
                dict(determinability=UND, behavior="asked", confidence=0.8),
                dict(determinability=DET, behavior="committed", confidence=0.7),
                dict(determinability=DET, behavior="asked", confidence=0.6)])
    assert ab.behavior_error(df).tolist() == [1, 0, 0, 1]
    rep = ab.selective_report(df)
    assert rep["aurc"] == pytest.approx(7 / 12) and rep["n_dropped"] == 0
    assert ab.selective_report(df.drop(columns="confidence")) is None
    assert ab.selective_report(df.assign(confidence=np.nan)) is None


# ---- split conformal -------------------------------------------------------------------------
def test_conformal_quantile_hand_values():
    # n=19, alpha=.1: k = ceil(20 * .9) = 18 -> 18th smallest.  n=9: k = ceil(10 * .9) = 9 -> the maximum.
    # n=8: k = ceil(9 * .9) = ceil(8.1) = 9 > 8 -> infinity (too few points)
    assert ab.conformal_quantile(np.arange(1, 20), 0.1) == 18
    assert ab.conformal_quantile(np.arange(1, 10), 0.1) == 9
    assert ab.conformal_quantile(np.arange(1, 9), 0.1) == math.inf
    assert ab.conformal_quantile([5, 1, 3], 0.5) == 3                    # k = ceil(4 * .5) = 2 -> second smallest
    with pytest.raises(ValueError):
        ab.conformal_quantile([1, 2], 1.5)
    with pytest.raises(ValueError):
        ab.conformal_quantile([], 0.1)


def test_conformal_coverage_on_synthetic_data():
    # With n_cal=99 and alpha=.1, k = ceil(100 * .9) = 90 and coverage is exactly 90/100 = 0.9 on average.
    rng = np.random.default_rng(0)
    reps, n = 20000, 99
    cal = np.sort(rng.exponential(size=(reps, n)), axis=1)
    q_hat = cal[:, 89]
    cover = np.mean(rng.exponential(size=reps) <= q_hat)
    assert 0.89 < cover < 0.91, cover


def test_conformal_prediction_sets_cover_true_labels():
    rng = np.random.default_rng(1)

    def draw(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[y]
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return p, y
    pc, yc = draw(500)
    q = ab.conformal_quantile(1 - pc[np.arange(500), yc], 0.1)
    pt, yt = draw(5000)
    sets = ab.prediction_sets(pt, q)
    cover = np.mean([int(y) in s for s, y in zip(sets, yt)])
    assert cover > 0.88, cover                                          # guarantee is >= 0.9 on average
    abst = ab.conformal_abstain(sets)
    assert abst.dtype == bool and 0 < abst.mean() < 1                    # asks sometimes, not always
    assert ab.conformal_abstain([{1}, set(), {0, 2}]).tolist() == [False, True, True]
