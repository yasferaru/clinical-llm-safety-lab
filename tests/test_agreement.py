"""Checks of the agreement statistics against hand-computed values."""
from evals.judge_validation.analyze_agreement import agreement, bootstrap_ci, cohen_kappa, gwet_ac1

A, B = [True, True, False, False], [True, False, False, False]


def test_hand_computed():
    # po = 3/4; pa = 1/2, pb = 1/4 -> pe = 1/8 + 3/8 = 1/2 -> kappa = (3/4 - 1/2) / (1/2) = 0.5
    assert abs(agreement(A, B) - 0.75) < 1e-9
    assert abs(cohen_kappa(A, B) - 0.5) < 1e-9
    # pi = 3/8 -> pe = 2 * 3/8 * 5/8 = 15/32 -> AC1 = (3/4 - 15/32) / (17/32) = 9/17
    assert abs(gwet_ac1(A, B) - 9 / 17) < 1e-9


def test_degenerate():
    assert cohen_kappa([True] * 4, [True] * 4) is None          # no variation: undefined
    assert abs(cohen_kappa(A, A) - 1.0) < 1e-9


def test_bootstrap_ci_is_seeded_and_sane():
    x = [True, False] * 30
    y = x[:]
    y[0] = not y[0]
    lo, hi = bootstrap_ci(x, y)
    assert 0.6 < lo < hi <= 1.0
    assert bootstrap_ci(x, y) == (lo, hi)


# ---- edge cases (regressions for bugs found 2026-11-17) ----
import pytest  # noqa: E402

from evals.judge_validation.analyze_agreement import (block, kappa_paradox, pabak,  # noqa: E402
                                                      prevalence_bias)

T, F = True, False


def test_empty_input_is_undefined_not_a_crash():
    # was: ZeroDivisionError
    assert agreement([], []) is None and cohen_kappa([], []) is None and gwet_ac1([], []) is None
    assert bootstrap_ci([], []) is None


def test_length_mismatch_raises():
    # was: silently returned kappa 0.40 for [T,F,T] vs [T,F] (zip truncated, n taken from the first list)
    with pytest.raises(ValueError, match="differ in length"):
        cohen_kappa([T, F, T], [T, F])


def test_unlabelled_items_raise_clearly():
    # was: TypeError from sum()
    with pytest.raises(ValueError, match="True/False"):
        cohen_kappa([T, None], [T, F])


def test_constant_columns():
    assert cohen_kappa([T] * 5, [T] * 5) is None          # pe = 1: undefined
    assert gwet_ac1([T] * 5, [T] * 5) == 1.0              # pi = 1 -> pe = 0 -> AC1 = po = 1
    assert cohen_kappa([T] * 5, [F] * 5) == 0.0           # pe = 1*0 + 0*1 = 0, po = 0
    assert cohen_kappa([T] * 5, [T, T, F, T, T]) == 0.0   # one rater constant: po = pe = 0.8


def test_tiny_n():
    assert cohen_kappa([T], [T]) is None and gwet_ac1([T], [T]) == 1.0
    assert cohen_kappa([T], [F]) == 0.0                    # pe = 0
    assert gwet_ac1([T], [F]) == -1.0                      # pi = 1/2 -> pe = 1/2, po = 0 -> -1
    assert block("x", [T, F], [T, F])[1].endswith("bias index 0.00")  # no CI under n = 10


def test_kappa_paradox_hand():
    # 100 items: 90 both YES, 5 YES/NO, 5 NO/YES, 0 both NO
    a = [T] * 90 + [T] * 5 + [F] * 5
    b = [T] * 90 + [F] * 5 + [T] * 5
    assert agreement(a, b) == pytest.approx(0.90)
    # pa = pb = 0.95 -> pe = 0.9025 + 0.0025 = 0.905 -> kappa = (0.90 - 0.905) / 0.095 = -0.05263
    assert cohen_kappa(a, b) == pytest.approx(-0.0526316, abs=1e-6)
    # pi = 0.95 -> pe = 2 * 0.95 * 0.05 = 0.095 -> AC1 = 0.805 / 0.905 = 0.88950
    assert gwet_ac1(a, b) == pytest.approx(0.8895028, abs=1e-6)
    assert pabak(a, b) == pytest.approx(0.80)
    assert prevalence_bias(a, b) == pytest.approx((0.90, 0.0))  # |90 - 0| / 100, |5 - 5| / 100
    assert kappa_paradox(a, b) is True
    assert any("kappa paradox" in ln for ln in block("x", a, b))
    assert kappa_paradox(A, B) is False                     # po = 0.75 < 0.8


def test_cluster_bootstrap():
    # one cluster: every resample equals the data, so the CI collapses to the point estimate
    x = [T, F] * 10
    y = x[:]
    y[0] = F
    k = cohen_kappa(x, y)
    lo, hi = bootstrap_ci(x, y, groups=["r1"] * 20, reps=200)
    assert lo == pytest.approx(k) and hi == pytest.approx(k)
    # clustering by reply widens the CI when lines of a reply are correlated (here: duplicated)
    xs, ys = x * 3, y * 3
    groups = [f"r{i % 20}" for i in range(60)]           # item i and its two copies share a reply
    w_items = bootstrap_ci(xs, ys, reps=500)
    w_clust = bootstrap_ci(xs, ys, reps=500, groups=groups)
    assert (w_clust[1] - w_clust[0]) > (w_items[1] - w_items[0])
    assert bootstrap_ci(xs, ys, reps=300, groups=groups) == bootstrap_ci(xs, ys, reps=300, groups=groups)
