"""Interval functions against hand-computed values (exact binomial tail inversion / Wilson formula)."""
import pytest

from evals.stats.intervals import clopper_pearson, wilson

TOL = 1e-4

# (k, n): (wilson, clopper_pearson), alpha = 0.05
KNOWN = {
    (1, 4): ((0.0456, 0.6994), (0.0063, 0.8059)),
    (3, 4): ((0.3006, 0.9544), (0.1941, 0.9937)),
    (2, 8): ((0.0715, 0.5907), (0.0319, 0.6509)),
    (8, 8): ((0.6756, 1.0), (0.6306, 1.0)),
    (0, 4): ((0.0, 0.4899), (0.0, 0.6024)),
    (4, 4): ((0.5101, 1.0), (0.3976, 1.0)),
    (0, 1): ((0.0, 0.7935), (0.0, 0.975)),
    (1, 1): ((0.2065, 1.0), (0.025, 1.0)),
}


@pytest.mark.parametrize("kn", sorted(KNOWN))
def test_known_values(kn):
    w, cp = KNOWN[kn]
    assert wilson(*kn) == pytest.approx(w, abs=TOL)
    assert clopper_pearson(*kn) == pytest.approx(cp, abs=TOL)


def test_boundaries_are_exact():
    assert wilson(0, 10)[0] == 0.0 and clopper_pearson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0 and clopper_pearson(10, 10)[1] == 1.0


@pytest.mark.parametrize("k,n", [(k, n) for n in (1, 2, 5, 13, 40) for k in range(n + 1)])
def test_contains_point_estimate_and_cp_is_wider(k, n):
    p = k / n
    for f in (wilson, clopper_pearson):
        lo, hi = f(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0
    wlo, whi = wilson(k, n)
    clo, chi = clopper_pearson(k, n)
    assert clo <= wlo + 1e-12 and chi >= whi - 1e-12  # exact interval is the conservative one


def test_alpha_widens():
    assert clopper_pearson(3, 10, 0.01)[1] > clopper_pearson(3, 10, 0.05)[1]
    assert wilson(3, 10, 0.01)[1] > wilson(3, 10, 0.05)[1]


@pytest.mark.parametrize("args", [(1, 0), (-1, 4), (5, 4), (1, 4, 0.0), (1, 4, 1.0), (True, 4), (1.5, 4)])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        wilson(*args)
    with pytest.raises(ValueError):
        clopper_pearson(*args)
