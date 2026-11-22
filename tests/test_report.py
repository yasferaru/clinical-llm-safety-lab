"""critical_pass_summary: unit = case, pass^k over epochs, named interval."""
import pandas as pd
import pytest

from evals.report import critical_pass_summary
from evals.results import SAMPLE_COLUMNS
from tests.fixtures import make_samples


def table(rows):
    base = {c: None for c in SAMPLE_COLUMNS}
    return pd.DataFrame([{**base, "run_id": "r", "model": "m", "scored": True, **r} for r in rows])[SAMPLE_COLUMNS]


def test_one_in_four_clopper_pearson_default():
    s = table([{"case_id": c, "epoch": 1, "critical_pass": v} for c, v in zip("abcd", [1.0, 0.0, 0.0, 0.0])])
    r = critical_pass_summary(s).iloc[0]
    assert (r["k"], r["n"], r["method"]) == (1, 4, "clopper_pearson")
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((0.0063, 0.8059), abs=1e-4)
    w = critical_pass_summary(s, method="wilson").iloc[0]
    assert (w["ci_low"], w["ci_high"]) == pytest.approx((0.0456, 0.6994), abs=1e-4)


def test_epochs_use_pass_hat_k_and_unscored_excluded():
    rows = [{"case_id": "a", "epoch": e, "critical_pass": 1.0} for e in (1, 2)]
    rows += [{"case_id": "b", "epoch": 1, "critical_pass": 1.0}, {"case_id": "b", "epoch": 2, "critical_pass": 0.0}]
    rows += [{"case_id": "c", "epoch": 1, "critical_pass": float("nan"), "scored": False}]
    r = critical_pass_summary(table(rows)).iloc[0]
    assert (r["k"], r["n"], r["epochs"], r["unscored_samples"]) == (1, 2, 2, 1)


def test_on_fixture_groups_by_run():
    out = critical_pass_summary(make_samples(seed=0))
    assert len(out) == 2 and (out["n"] == 8 * 2 + 4).all()
    assert ((out["ci_low"] <= out["rate"]) & (out["rate"] <= out["ci_high"])).all()
