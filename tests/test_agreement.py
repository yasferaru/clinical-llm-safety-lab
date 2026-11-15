"""Checks of the agreement statistics. Run from the project folder: python tests/test_agreement.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "judge_validation"))
from analyze_agreement import cohen_kappa, gwet_ac1, bootstrap_ci, agreement  # noqa: E402

a, b = [True, True, False, False], [True, False, False, False]
assert abs(agreement(a, b) - 0.75) < 1e-9
assert abs(cohen_kappa(a, b) - 0.5) < 1e-9
assert abs(gwet_ac1(a, b) - 0.5294117647) < 1e-6
assert cohen_kappa([True] * 4, [True] * 4) is None          # no variation: undefined
assert abs(cohen_kappa(a, a) - 1.0) < 1e-9
x = [True, False] * 30
y = x[:]
y[0] = not y[0]
lo, hi = bootstrap_ci(x, y)
assert 0.6 < lo < hi <= 1.0
print("agreement statistics: all checks passed")
