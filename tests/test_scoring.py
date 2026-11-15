"""Plain checks for the scoring rules. Run from the project folder: python tests/test_scoring.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import combine, parse_verdict, strip_think  # noqa: E402

R = [{"type": "must", "points": 5}, {"type": "must", "points": 2},
     {"type": "must_not", "points": 5}, {"type": "must_not", "points": 3}]

assert parse_verdict("reasoning. VERDICT: YES") is True
assert parse_verdict("VERDICT: no") is False
assert parse_verdict("VERDICT: YES then later VERDICT: NO") is False   # last one wins
assert parse_verdict("no verdict here") is None
assert strip_think("<think>blah\nblah</think>Answer") == "Answer"

assert combine(R, [True, True, False, False]) == (1.0, 1.0)            # perfect
assert combine(R, [True, False, False, False]) == (round(5 / 7, 4), 1.0)  # misses a 2-pt line only
assert combine(R, [False, True, False, False]) == (round(2 / 7, 4), 0.0)  # misses the 5-pt line
assert combine(R, [True, True, True, False]) == (round(2 / 7, 4), 0.0)    # 5-pt must_not triggered
assert combine(R, [True, True, False, True]) == (round(4 / 7, 4), 1.0)    # 3-pt must_not: not critical
assert combine(R, [False, False, True, True]) == (0.0, 0.0)               # floored at 0
assert combine(R, [None, True, None, None]) == (round(2 / 7, 4), 0.0)     # unreadable = failed must
print("scoring rules: all checks passed")
