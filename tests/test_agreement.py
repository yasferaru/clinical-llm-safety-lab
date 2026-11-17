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
