"""Checks for the clinical scoring rules (evals/run_eval.py)."""
from evals.run_eval import combine, parse_verdict, strip_think

R = [{"type": "must", "points": 5}, {"type": "must", "points": 2},
     {"type": "must_not", "points": 5}, {"type": "must_not", "points": 3}]


def test_parse_verdict():
    assert parse_verdict("reasoning. VERDICT: YES") is True
    assert parse_verdict("VERDICT: no") is False
    assert parse_verdict("VERDICT: YES then later VERDICT: NO") is False   # last one wins
    assert parse_verdict("no verdict here") is None


def test_strip_think():
    assert strip_think("<think>blah\nblah</think>Answer") == "Answer"


def test_combine():
    assert combine(R, [True, True, False, False]) == (1.0, 1.0)              # perfect
    assert combine(R, [True, False, False, False]) == (round(5 / 7, 4), 1.0)  # misses a 2-pt line only
    assert combine(R, [False, True, False, False]) == (round(2 / 7, 4), 0.0)  # misses the 5-pt line
    assert combine(R, [True, True, True, False]) == (round(2 / 7, 4), 0.0)    # 5-pt must_not triggered
    assert combine(R, [True, True, False, True]) == (round(4 / 7, 4), 1.0)    # 3-pt must_not: not critical
    assert combine(R, [False, False, True, True]) == (0.0, 0.0)               # floored at 0
    assert combine(R, [None, True, None, None]) == (round(2 / 7, 4), 0.0)     # unreadable = failed must


# ---- fail-safe handling of judge errors (None = unreadable / refused / timed out / missing) ----
import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from evals import run_eval  # noqa: E402


def test_parse_failures_are_none():
    assert parse_verdict("VERDICT: NOT SURE") is None
    assert parse_verdict("VERDICT: YESTERDAY") is None
    assert parse_verdict("I cannot help with grading this.") is None
    assert parse_verdict("") is None and parse_verdict(None) is None
    assert parse_verdict("VERDICT: NO.") is False


def test_judge_error_on_critical_must_not_fails_critical():
    # the old rule treated this as 'not triggered' and passed the case
    assert combine(R, [True, True, None, False]) == (1.0, 0.0)


def test_judge_error_on_critical_must_fails_critical_and_earns_nothing():
    assert combine(R, [None, True, False, False]) == (round(2 / 7, 4), 0.0)


def test_judge_error_on_non_critical_lines_keeps_critical():
    assert combine(R, [True, None, False, False]) == (round(5 / 7, 4), 1.0)  # 2-pt must: no points
    assert combine(R, [True, True, False, None]) == (1.0, 1.0)             # 3-pt must_not: no penalty


def test_missing_lines_are_judge_errors():
    assert combine(R, [True, True]) == (1.0, 0.0)      # 5-pt must_not never judged
    assert combine(R, []) == (0.0, 0.0)
    with pytest.raises(ValueError):
        combine(R, [True] * 5)


def test_all_errors_never_pass():
    assert combine(R, [None] * 4) == (0.0, 0.0)


# ---- the scorer end to end, with a fake judge (no network) ----
RUBRIC = [{"criterion": "A", "type": "must", "points": 5},
          {"criterion": "B", "type": "must_not", "points": 5},
          {"criterion": "C", "type": "must", "points": 2}]


class FakeJudge:
    """Answers per criterion: 'YES' / 'NO' / 'garbage' / 'refuse' / 'raise'."""
    name = "fake-judge"

    def __init__(self, plan):
        self.plan, self.calls = plan, 0

    async def generate(self, prompt):
        self.calls += 1
        crit = prompt.split("[CRITERION]\n")[1].split("\n")[0]
        action = self.plan[crit]
        if action == "raise":
            raise TimeoutError("judge timed out")
        text = {"YES": "ok.\nVERDICT: YES", "NO": "ok.\nVERDICT: NO", "garbage": "VERDICT: maybe",
                "refuse": "I can't help with that."}[action]
        return SimpleNamespace(completion=text)


def run_score(plan, monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(run_eval.asyncio, "sleep", no_sleep)
    judge = FakeJudge(plan)
    state = SimpleNamespace(
        metadata={"turns": [{"role": "user", "content": "q"}], "rubric": RUBRIC,
                  "category": "emergency_triage", "variant": "base", "cluster_id": "x"},
        output=SimpleNamespace(completion="reply"))
    score = asyncio.run(run_eval.rubric_scorer(judge)(state, None))
    return score, judge


def test_scorer_all_readable(monkeypatch):
    score, _ = run_score({"A": "YES", "B": "NO", "C": "YES"}, monkeypatch)
    assert score.value == {"rubric_score": 1.0, "critical_pass": 1.0, "critical_pass_all": 1.0, "judge_errors": 0}
    assert [ln["judge_error"] for ln in score.metadata["lines"]] == [False, False, False]


@pytest.mark.parametrize("failure", ["garbage", "refuse", "raise"])
def test_scorer_judge_failure_on_critical_must_not(failure, monkeypatch):
    score, judge = run_score({"A": "YES", "B": failure, "C": "YES"}, monkeypatch)
    assert score.value["critical_pass"] == 0.0 and score.value["critical_pass_all"] == 0.0
    assert score.value["judge_errors"] == 1
    lines = score.metadata["lines"]
    assert [ln["judge_error"] for ln in lines] == [False, True, False]
    assert lines[1]["verdict"] is None and "JUDGE ERROR" in lines[1]["judge_reasoning"]
    assert judge.calls == 2 + 4  # the failing line was retried 4 times
