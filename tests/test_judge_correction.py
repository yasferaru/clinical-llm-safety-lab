"""Tests of evals/stats/judge_correction.py: hand-computed examples, then simulations. No logs, no network."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from evals.stats import judge_correction as jc
from evals.stats.inference import clustered_se

# Hand examples use tiny calibration sets on purpose; their banner warning is expected.
pytestmark = pytest.mark.filterwarnings("ignore:JUDGE NOT VALIDATED")


def cal_frame(y, f, clusters=None):
    return pd.DataFrame({"human": y, "judge": f, "scenario_id": clusters if clusters is not None
                         else [f"c{i}" for i in range(len(y))]})


def unl_frame(f, clusters=None):
    return pd.DataFrame({"judge": f, "scenario_id": clusters if clusters is not None
                         else [f"u{i}" for i in range(len(f))]})


# Y = [1,1,1,1,0,0,0,0,0,0], f = [1,1,1,0,0,0,0,0,1,0]
Y10 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
F10 = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]


# ---- hand-computed ---------------------------------------------------------------------------
def test_estimate_judge_error_hand_example():
    # q1 = 3/4 (3 of 4 human positives flagged), q0 = 5/6 (5 of 6 human negatives not flagged), J = 3/4 + 5/6 - 1 = 7/12
    e = jc.estimate_judge_error(Y10, F10)
    assert e["q1"] == pytest.approx(0.75) and e["q0"] == pytest.approx(5 / 6) and e["J"] == pytest.approx(7 / 12)
    assert (e["n_pos"], e["n_neg"]) == (4, 6)
    assert math.isnan(jc.estimate_judge_error([1, 1], [1, 0])["q0"])               # no human negatives


def test_rogan_gladen_hand_values_and_refusal():
    # p_hat .45, q0 .85, q1 .9: J = .75, theta = (.45 + .85 - 1)/.75 = .4
    theta, j, raw = jc.rogan_gladen(0.45, 0.85, 0.9)
    assert theta == pytest.approx(0.4) and j == pytest.approx(0.75) and raw == pytest.approx(0.4)
    # p_hat .05: raw = (.05 + .85 - 1)/.75 = -0.1333 -> clipped to 0 ; p_hat .98: raw = 0.83/.75 = 1.1067 -> 1
    assert jc.rogan_gladen(0.05, 0.85, 0.9)[0] == 0.0 and jc.rogan_gladen(0.05, 0.85, 0.9)[2] == pytest.approx(-2 / 15)
    assert jc.rogan_gladen(0.98, 0.85, 0.9)[0] == 1.0
    # q0 = .5, q1 = .52: J = .02 < .1 -> refuse (NaN)
    assert math.isnan(jc.rogan_gladen(0.5, 0.5, 0.52)[0])
    assert math.isnan(jc.rogan_gladen(0.5, math.nan, 0.9)[0])


def test_corrected_rate_rogan_gladen_hand_example():
    # 10 unlabeled items with 3 judge positives: p_hat = .3. q1 = 3/4, q0 = 5/6, J = 7/12.
    # theta = (.3 + 5/6 - 1) / (7/12) = (2/15)/(7/12) = 8/35 = 0.228571
    r = jc.corrected_rate(unl_frame([1] * 3 + [0] * 7), cal_frame(Y10, F10), ci="delta")
    assert r.estimate == pytest.approx(8 / 35) and r.diagnostics["q1"] == pytest.approx(0.75)
    # delta SE with singleton clusters: V_p = .3*.7/10, V_q1 = .75*.25/4, V_q0 = (5/6)(1/6)/6, covariance 0
    # SE = sqrt(V_p + (1-theta)^2 V_q0 + theta^2 V_q1) / J
    th, jj = 8 / 35, 7 / 12
    var = (0.3 * 0.7 / 10 + (1 - th) ** 2 * (5 / 6) * (1 / 6) / 6 + th ** 2 * 0.75 * 0.25 / 4) / jj ** 2
    assert r.se == pytest.approx(math.sqrt(var), rel=1e-9)
    assert r.ci_low == pytest.approx(max(0, th - 1.959964 * r.se), abs=1e-5) and r.ci_high <= 1.0


def test_ppi_hand_example():
    # unlabeled f = [1,0,1,0] (mean .5). calibration Y = [1,0,0,1], f = [1,0,1,1]: Y - f = [0,0,-1,0], mean -.25
    # theta = .5 - .25 = .25. V(mean f_u) = sqrt(4 * .25)/4 squared = 1/16 ; V(mean(Y-f)) = (sqrt(.75)/4)^2 = 3/64
    # SE = sqrt(1/16 + 3/64) = sqrt(7/64) = 0.33072
    r = jc.corrected_rate(unl_frame([1, 0, 1, 0]), cal_frame([1, 0, 0, 1], [1, 0, 1, 1]), method="ppi")
    assert r.estimate == pytest.approx(0.25) and r.se == pytest.approx(math.sqrt(7 / 64))
    assert r.diagnostics["lambda"] == 1.0 and not r.validated


def test_ppi_plus_plus_lambda_hand_example():
    # Cov(Y, f) over calibration = 1/6 ; pooled Var(f) (8 values: five 1s, three 0s) = 15/56 ; n/N = 4/4 = 1
    # lam = (1/6) / ((15/56) * 2) = 28/90 = 0.31111
    f_c, y_c, f_u = np.array([1, 0, 1, 1.0]), np.array([1, 0, 0, 1.0]), np.array([1, 0, 1, 0.0])
    assert jc._ppi_lambda(y_c, f_c, f_u) == pytest.approx(28 / 90)
    # estimate = lam * .5 + mean(Y - lam f) = lam * .5 + (.5 - lam * .75) = .5 - .25 lam
    r = jc.corrected_rate(unl_frame(f_u), cal_frame(y_c, f_c), method="ppi++")
    assert r.diagnostics["lambda"] == pytest.approx(28 / 90) and r.estimate == pytest.approx(0.5 - 0.25 * 28 / 90)


def test_ppi_lambda_edges():
    assert jc._ppi_lambda(np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0])) <= 1.0   # clipped
    assert jc._ppi_lambda(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0])) == 0.0  # Var(f)=0
    assert jc._ppi_lambda(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])) == 0.0      # negative cov


def test_naive_and_human_only_use_clustered_se():
    u, c = unl_frame([1, 0, 1, 1], ["a", "a", "b", "b"]), cal_frame(Y10, F10, list("aabbccddee"))
    n = jc.corrected_rate(u, c, method="naive")
    assert n.estimate == 0.75 and n.se == pytest.approx(clustered_se([1, 0, 1, 1], ["a", "a", "b", "b"]).se)
    h = jc.corrected_rate(u, c, method="human_only")
    assert h.estimate == pytest.approx(0.4) and h.se == pytest.approx(clustered_se(Y10, list("aabbccddee")).se)


# ---- refusal, banner, validation -------------------------------------------------------------
def test_uninformative_judge_is_refused_with_a_warning():
    y = [1] * 20 + [0] * 20
    f = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10              # judge ignores Y: q1 = q0 = .5, J = 0
    with pytest.warns(UserWarning, match="JUDGE NOT VALIDATED"):
        r = jc.corrected_rate(unl_frame([1, 0] * 20), cal_frame(y, f))
    assert math.isnan(r.estimate) and math.isnan(r.ci_low) and not r.validated
    assert any("not identifiable" in w for w in r.warnings)
    with pytest.raises(jc.NonIdentifiable):
        jc.corrected_rate(unl_frame([1, 0] * 20), cal_frame(y, f), strict=True)


def test_banner_without_calibration_and_with_too_few_labels():
    with pytest.warns(UserWarning, match="no calibration labels"):
        r = jc.corrected_rate(unl_frame([1, 0, 1, 1]), None)
    assert not r.validated and r.method == "rogan_gladen" and r.method_used == "naive"
    assert r.estimate == 0.75 and r.report().startswith("*** JUDGE NOT VALIDATED")
    assert "NOT corrected" in r.diagnostics["banner_reason"]
    with pytest.warns(UserWarning):
        assert not jc.corrected_rate(unl_frame([1, 0]), cal_frame([], [])).validated      # empty calibration frame
    with pytest.warns(UserWarning, match="only 10 human labels"):
        small = jc.corrected_rate(unl_frame([1, 0, 1, 1]), cal_frame(Y10, F10))
    assert not small.validated and small.banner.startswith("*** JUDGE NOT VALIDATED")


def test_no_banner_when_calibration_is_large_enough():
    rng = np.random.default_rng(0)
    y = (rng.random(80) < 0.4).astype(int)
    f = np.where(rng.random(80) < 0.9, y, 1 - y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                          # a banner would raise here
        r = jc.corrected_rate(unl_frame(rng.integers(0, 2, 200)), cal_frame(y, f), B=200)
    assert r.validated and r.banner == "" and "JUDGE NOT VALIDATED" not in r.report()


def test_input_validation():
    with pytest.raises(ValueError):
        jc.corrected_rate(unl_frame([1, 0]), cal_frame(Y10, F10), method="bogus")
    with pytest.raises(ValueError, match="0/1"):
        jc.corrected_rate(unl_frame([1, 2]), cal_frame(Y10, F10))
    with pytest.raises(ValueError):
        jc.corrected_rate(unl_frame([]), cal_frame(Y10, F10))
    with pytest.raises(ValueError, match="missing column"):
        jc.corrected_rate(unl_frame([1, 0]).drop(columns="scenario_id"), cal_frame(Y10, F10))
    with pytest.raises(ValueError):
        jc.corrected_rate(unl_frame([1, 0]), cal_frame(Y10, F10), alpha=2)


def test_bootstrap_is_seeded_and_reports_unusable_draws():
    rng = np.random.default_rng(1)
    y = (rng.random(60) < 0.5).astype(int)
    f = np.where(rng.random(60) < 0.85, y, 1 - y)
    u, c = unl_frame(rng.integers(0, 2, 100)), cal_frame(y, f)
    a, b = jc.corrected_rate(u, c, B=300, seed=5), jc.corrected_rate(u, c, B=300, seed=5)
    assert (a.estimate, a.ci_low, a.ci_high, a.se) == (b.estimate, b.ci_low, b.ci_high, b.se)
    assert a.diagnostics["n_boot_valid"] > 250 and a.ci_low <= a.estimate <= a.ci_high
    # a weak judge makes many resamples unusable (J < min_j) and the result says so
    weak_f = np.where(rng.random(60) < 0.55, y, 1 - y)
    w = jc.corrected_rate(u, cal_frame(y, weak_f), B=300, seed=5)
    assert w.diagnostics["J"] < 0.2


def test_clipped_estimate_is_flagged():
    y = [1] * 20 + [0] * 20
    f = [1] * 18 + [0] * 2 + [0] * 18 + [1] * 2                               # q1 = q0 = .9
    r = jc.corrected_rate(unl_frame([0] * 50), cal_frame(y, f), ci="delta")   # p_hat 0 -> raw (0 + .9 - 1)/.8 < 0
    assert r.estimate == 0.0 and any("clipped" in w for w in r.warnings)


# ---- how many labels -------------------------------------------------------------------------
def test_required_calibration_size_hand_values():
    # human only: z^2 * .25 / .05^2 = 1.959964^2 * 100 = 384.15
    assert jc.required_calibration_size("human_only", 0.05) == pytest.approx(384.146, abs=1e-2)
    # ppi, theta .5, q1 = q0 = .9: p = .5, d = .1, Var(Y-f) = .1 ; N -> inf: n = .1 z^2 / h^2 = 153.66
    assert jc.required_calibration_size("ppi", 0.05, 0.5, 0.9, 0.9) == pytest.approx(153.658, abs=1e-2)
    # rogan-gladen, same setting: J = .8 ; numerator = .5*.09 + .5*.09 = .09 ; n = .09 z^2 / (.64 h^2) = 216.09
    assert jc.required_calibration_size("rogan_gladen", 0.05, 0.5, 0.9, 0.9) == pytest.approx(216.08, abs=1e-1)
    assert jc.required_calibration_size("human_only", 0.05, deff=2.0) == pytest.approx(2 * 384.146, abs=2e-2)
    # a small unlabeled set can make the target unreachable: N = 100, p(1-p)/N = .0025 > (.05/1.96)^2 = .00065
    assert jc.required_calibration_size("ppi", 0.05, 0.5, 0.9, 0.9, n_unlabeled=100) == math.inf
    assert jc.required_calibration_size("rogan_gladen", 0.05, 0.5, 0.5, 0.5) == math.inf        # J = 0
    with pytest.raises(ValueError):
        jc.required_calibration_size("bogus", 0.05)


# ---- simulations -----------------------------------------------------------------------------
Q1, Q0, THETA = 0.90, 0.85, 0.40


def _sim_clustered(rng, n_unl_scen, n_cal_scen, m=5):
    """Scenarios with their own prevalence ~ Beta(2, 3) (mean .4); judge error rates are the same for every item.

    True population rate = .4, p = .4 * .9 + .6 * .15 = .45, so the raw judge rate is biased by +.05.
    """
    def block(g, tag):
        prev = rng.beta(2, 3, g)
        y = (rng.random((g, m)) < prev[:, None]).astype(int)
        f = np.where(y == 1, rng.random((g, m)) < Q1, rng.random((g, m)) >= Q0).astype(int)
        return y.ravel(), f.ravel(), np.repeat([f"{tag}{i}" for i in range(g)], m)
    yu, fu, cu = block(n_unl_scen, "u")
    yc, fc, cc = block(n_cal_scen, "c")
    return unl_frame(fu, cu), cal_frame(yc, fc, cc)


def test_rogan_gladen_interval_coverage_delta_and_bootstrap():
    rng = np.random.default_rng(2024)
    R = 300
    hit = {"delta": 0, "bootstrap": 0}
    width = {"delta": [], "bootstrap": []}
    for r in range(R):
        u, c = _sim_clustered(rng, n_unl_scen=80, n_cal_scen=50)
        for ci in hit:
            res = jc.corrected_rate(u, c, ci=ci, B=300, seed=r)
            hit[ci] += res.ci_low <= THETA <= res.ci_high
            width[ci].append(res.ci_high - res.ci_low)
    cov = {k: v / R for k, v in hit.items()}
    print(f"Rogan-Gladen coverage of the true rate over {R} simulations: {cov}")
    for k, v in cov.items():
        assert 0.91 < v < 0.99, (k, v)
    assert 0.8 < np.mean(width["delta"]) / np.mean(width["bootstrap"]) < 1.25    # the two intervals agree


def test_ignoring_the_judges_errors_undercovers():
    rng = np.random.default_rng(7)
    R, hit_naive, hit_rg = 200, 0, 0
    for r in range(R):
        u, c = _sim_clustered(rng, n_unl_scen=200, n_cal_scen=50)
        n = jc.corrected_rate(u, c, method="naive")
        g = jc.corrected_rate(u, c, ci="delta")
        hit_naive += n.ci_low <= THETA <= n.ci_high
        hit_rg += g.ci_low <= THETA <= g.ci_high
    print(f"naive judge-only coverage {hit_naive / R:.2f} vs corrected {hit_rg / R:.2f}")
    assert hit_naive / R < 0.5 and hit_rg / R > 0.90


def test_ppi_widths_and_coverage_against_human_only_and_naive():
    # N = 3000 judge-only items, n = 150 human-labelled. Unclustered items (each its own scenario).
    # Var(Y - f) = d - (theta - p)^2 = .13 - .05^2 = .1275 vs Var(Y) = .24, so PPI should be about
    # sqrt(.1275/.24) = 0.73 times as wide as the human-only interval.
    rng = np.random.default_rng(11)
    R = 300
    res = {m: {"hit": 0, "w": []} for m in ("human_only", "ppi", "ppi++", "naive")}
    for _ in range(R):
        def draw(n):
            y = (rng.random(n) < THETA).astype(int)
            return y, np.where(y == 1, rng.random(n) < Q1, rng.random(n) >= Q0).astype(int)
        _, fu = draw(3000)
        yc, fc = draw(150)
        for m in res:
            r = jc.corrected_rate(unl_frame(fu), cal_frame(yc, fc), method=m)
            res[m]["hit"] += r.ci_low <= THETA <= r.ci_high
            res[m]["w"].append(r.ci_high - r.ci_low)
    cov = {m: v["hit"] / R for m, v in res.items()}
    w = {m: float(np.mean(v["w"])) for m, v in res.items()}
    print(f"coverage {cov}\nmean width {w}")
    for m in ("human_only", "ppi", "ppi++"):
        assert 0.91 < cov[m] < 0.99, (m, cov[m])
    assert cov["naive"] < 0.3                                            # tiny interval around the wrong value
    assert w["ppi"] < 0.85 * w["human_only"]                              # the judge's labels buy a narrower interval
    assert w["ppi++"] <= w["ppi"] * 1.01                                  # tuning never hurts materially
    assert w["naive"] < w["ppi"]                                          # naive looks most precise, but is wrong
