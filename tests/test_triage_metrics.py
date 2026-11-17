"""Tests of evals/triage_metrics.py against hand-computed examples. No network, no logs, no real judge."""
import math

import numpy as np
import pandas as pd
import pytest

from evals.stats import intervals
from evals import triage_metrics as tm


def frame(rows):
    """rows: (case_id, scenario_id, gold, pred) -> DataFrame; None becomes NaN."""
    return pd.DataFrame(rows, columns=["case_id", "scenario_id", "acuity_gold", "acuity_pred"]).astype(
        {"acuity_gold": float, "acuity_pred": float})


# Ten scored rows, one row with no gold (c11) and one with gold but no prediction (c12).
ROWS = [
    ("c1", "s1", 5, 5), ("c2", "s1", 5, 4), ("c3", "s2", 5, 2), ("c4", "s2", 4, 4), ("c5", "s3", 4, 2),
    ("c6", "s3", 3, 3), ("c7", "s4", 3, 5), ("c8", "s4", 2, 3), ("c9", "s5", 1, 1), ("c10", "s5", 1, 2),
    ("c11", "s6", None, 3), ("c12", "s6", 3, None),
]


# ---- extraction ----------------------------------------------------------------------------
@pytest.mark.parametrize("text, level", [
    ("Call 911 now.", 5),
    ("Go to the emergency room immediately.", 5),
    ("Take him to the emergency room right away, especially if he keeps vomiting.", 5),
    ("Book an urgent care visit today.", 4),
    ("See your GP within 2 days.", 3),
    ("Make a routine appointment next week.", 2),
    ("You can manage this at home with rest and fluids.", 1),
    ("ACUITY: 4", 4),
    ("3", 3),
    # conditional escalation is ignored; negated phrase is ignored
    ("See your GP within 2 days. If he vomits again, call 911.", 3),
    ("You do not need to go to the ER. See a doctor within 2 days.", 3),
])
def test_rule_layer_maps_text(text, level):
    got = tm.extract_acuity(text)
    assert got.acuity == level and got.source == "rule"


@pytest.mark.parametrize("text, why", [
    ("See your doctor.", "no recognised"),
    ("Go to the ER now. Then see your GP within 2 days.", "multiple levels"),
    ("If it gets worse, call 911.", "conditional"),
    ("acuity 2, but later acuity 4", "conflicting"),
    ("", "no recognised"),
    (None, "no recognised"),
])
def test_ambiguous_returns_none_with_reason(text, why):
    got = tm.extract_acuity(text)
    assert got.acuity is None and why in got.reason


def test_judge_fallback_only_when_rules_fail_and_is_validated():
    calls = []

    def judge(text):
        calls.append(text)
        return 4

    assert tm.extract_acuity("Call 911 now.", judge).acuity == 5 and calls == []   # rules decided, no call
    got = tm.extract_acuity("See your doctor.", judge)
    assert (got.acuity, got.source, len(calls)) == (4, "judge", 1)
    assert tm.extract_acuity("See your doctor.", lambda t: 7).acuity is None       # outside 1-5
    assert tm.extract_acuity("See your doctor.", lambda t: True).acuity is None    # bool is not a level
    assert tm.extract_acuity("See your doctor.", lambda t: None).acuity is None
    boom = tm.extract_acuity("See your doctor.", lambda t: 1 / 0)
    assert boom.acuity is None and "judge error" in boom.reason


def test_add_predictions_adds_columns():
    df = pd.DataFrame({"response": ["Call 911 now.", "See your doctor."]})
    out = tm.add_predictions(df)
    assert out["acuity_pred"].tolist()[0] == 5 and math.isnan(out["acuity_pred"].tolist()[1])
    assert out["extraction_source"].tolist() == ["rule", "none"]
    assert "response" in df and "acuity_pred" not in df                          # input unchanged


# ---- rate metrics --------------------------------------------------------------------------
def test_rate_metrics_hand_example():
    # scored rows c1..c10 (n=10); c11 skipped (no gold); c12 excluded (no prediction)
    # under (p<g): c2, c3, c5 -> 3/10      severe (g-p>=2): c3 (3), c5 (2) -> 2/10
    # over  (p>g): c7, c8, c10 -> 3/10     emergency recall: gold-5 rows c1,c2,c3, only c1 has p=5 -> 1/3
    # high-acuity FN: gold>=4 rows c1..c5 (n=5), p<4 for c3 (2) and c5 (2) -> 2/5
    res = tm.triage_metrics(frame(ROWS))
    m = res.metrics
    assert res.skipped_no_gold == ["c11"] and res.n_unextracted == 1 and res.n_rows == 10
    assert [(m[k].k, m[k].n) for k in ("under_triage_rate", "severe_under_triage_rate", "over_triage_rate",
                                       "emergency_recall", "high_acuity_false_negatives")] == \
        [(3, 10), (2, 10), (3, 10), (1, 3), (2, 5)]
    assert m["under_triage_rate"].rate == pytest.approx(0.3)
    assert m["emergency_recall"].ci_method == "clopper_pearson"
    assert (m["emergency_recall"].ci_low, m["emergency_recall"].ci_high) == intervals.clopper_pearson(1, 3)
    assert m["under_triage_rate"].n_scenarios == 5


def test_wilson_option_and_worst_case_policy():
    w = tm.triage_metrics(frame(ROWS), ci_method="wilson").metrics["under_triage_rate"]
    assert (w.ci_low, w.ci_high) == intervals.wilson(3, 10) and w.ci_method == "wilson"
    # worst_case: c12 (gold 3, no prediction) scored as acuity 1 -> under by 2 -> under 4/11, severe 3/11
    res = tm.triage_metrics(frame(ROWS), missing="worst_case")
    assert (res.metrics["under_triage_rate"].k, res.metrics["under_triage_rate"].n) == (4, 11)
    assert res.metrics["severe_under_triage_rate"].k == 3
    with pytest.raises(ValueError):
        tm.triage_metrics(frame(ROWS), missing="bogus")


def test_no_rows_gives_nan_not_error():
    res = tm.triage_metrics(frame([("a", "s", None, 3)]))
    assert res.metrics["under_triage_rate"].n == 0 and math.isnan(res.metrics["under_triage_rate"].rate)
    assert res.skipped_no_gold == ["a"]


# ---- weighted kappa, confusion, cost -------------------------------------------------------
def test_quadratic_kappa_hand_example():
    # gold [1,2,3,3], pred [1,3,3,2], K=3. Each pair has weight 1/4 when one level apart.
    # sum(wO) = (1/4)(1/4 + 1/4) = 1/8. Marginals are both [1/4,1/4,1/2], sum(wE) = 11/32.
    # kappa = 1 - (1/8)/(11/32) = 1 - 4/11 = 7/11
    k = tm.quadratic_weighted_kappa([1, 2, 3, 3], [1, 3, 3, 2], levels=(1, 2, 3))
    assert k == pytest.approx(7 / 11)
    assert tm.quadratic_weighted_kappa([1, 2, 3], [1, 2, 3], levels=(1, 2, 3)) == pytest.approx(1.0)
    assert math.isnan(tm.quadratic_weighted_kappa([3, 3], [3, 3]))               # one level only: undefined


def test_quadratic_kappa_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    gold = rng.integers(1, 6, 200)
    pred = np.clip(gold + rng.integers(-2, 3, 200), 1, 5)
    ours = tm.quadratic_weighted_kappa(gold, pred)
    assert ours == pytest.approx(sk.cohen_kappa_score(gold, pred, labels=[1, 2, 3, 4, 5], weights="quadratic"))


def test_confusion_and_cost_view():
    cm = tm.confusion_matrix(frame(ROWS))
    assert cm.to_numpy().sum() == 10 and cm.loc[5, 5] == 1 and cm.loc[5, 2] == 1 and cm.loc[3, 5] == 1
    # under cost 3 per level: c2 (1 level) 3 + c3 (3 levels) 9 + c5 (2 levels) 6 = 18
    # over cost 1 per level:  c7 (2) 2 + c8 (1) 1 + c10 (1) 1 = 4   -> total 22, per case 2.2
    cv = tm.cost_view(cm)
    assert cv.total == pytest.approx(22.0) and cv.per_case == pytest.approx(2.2)
    assert cv.cost_matrix.loc[5, 2] == 9 and cv.cost_matrix.loc[2, 5] == 3
    flat = tm.cost_view(cm, tm.default_cost_matrix(under_weight=1.0))            # symmetric: 6 + 4 = 10
    assert flat.total == pytest.approx(10.0)
    with pytest.raises(ValueError):
        tm.cost_view(cm, cm.iloc[:3, :3])


# ---- format sensitivity --------------------------------------------------------------------
def ft_frame(free_under, forced_under, epochs=1):
    """Pairs f<i> (free text, links to g<i>) and g<i> (forced choice). Gold 5; pred 3 if 'under' else 5."""
    rows = []
    for i, (f, g) in enumerate(zip(free_under, forced_under)):
        for case, flag, link in ((f"f{i}", f, f"g{i}"), (f"g{i}", g, True)):
            flags = flag if isinstance(flag, list) else [flag] * epochs
            for e, fl in enumerate(flags):
                rows.append({"case_id": case, "scenario_id": f"s{i}", "epoch": e + 1, "acuity_gold": 5.0,
                             "acuity_pred": 3.0 if fl else 5.0, "forced_choice_variant": link})
    return pd.DataFrame(rows)


def test_mcnemar_exact_hand_values():
    assert tm.mcnemar_exact(5, 0) == pytest.approx(2 / 32)                        # 2 * P(X<=0 | n=5)
    assert tm.mcnemar_exact(3, 1) == pytest.approx(0.625)                         # 2 * (1+4)/16
    assert tm.mcnemar_exact(2, 2) == pytest.approx(1.0)
    assert tm.mcnemar_exact(0, 0) == 1.0


def test_format_sensitivity_pairs_and_discordance():
    # 6 pairs, under-triage (free, forced): (1,1) (1,0) (1,0) (0,0) (0,0) (0,1)
    # both=1, free_only b=2, forced_only c=1, neither=2; p = 2 * P(X<=1 | n=3) = 2 * 4/8 -> capped at 1
    res = tm.format_sensitivity(ft_frame([1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]))
    assert (res.n_pairs, res.both, res.free_only, res.forced_only, res.neither) == (6, 1, 2, 1, 2)
    assert res.free_rate == pytest.approx(3 / 6) and res.forced_rate == pytest.approx(2 / 6)
    assert res.p_value == pytest.approx(1.0)
    assert sorted((d["free_case"], d["direction"]) for d in res.discordant) == \
        [("f1", "free_only"), ("f2", "free_only"), ("f5", "forced_only")]
    assert res.discordant[0]["scenario_id"].startswith("s")


def test_format_sensitivity_significant_and_epoch_majority():
    res = tm.format_sensitivity(ft_frame([1] * 5, [0] * 5))                        # b=5, c=0 -> p = 0.0625
    assert res.free_only == 5 and res.p_value == pytest.approx(0.0625)
    # three epochs [under, under, ok] -> majority under (mean 2/3 > 0.5); [under, ok, ok] -> not under
    res = tm.format_sensitivity(ft_frame([[1, 1, 0], [1, 0, 0]], [0, 0], epochs=3))
    assert (res.free_only, res.neither) == (1, 1)


def test_format_sensitivity_drops_unpairable():
    df = ft_frame([1, 1], [0, 0])
    df.loc[df.case_id == "g1", "acuity_pred"] = np.nan                            # twin unscored -> pair dropped
    res = tm.format_sensitivity(df)
    assert (res.n_pairs, res.n_pairs_dropped) == (1, 1)
