"""Judge drift: fingerprints, the run log, and test-retest agreement with a clustered bootstrap."""
import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from evals.judge_validation.drift import (JudgeFingerprint, prompt_sha, read_judge_runs, record_judge_run,
                                          rejudge, retest_agreement)
from evals.run_eval import JUDGE_PROMPT

FP = JudgeFingerprint("groq/openai/gpt-oss-120b", "openai/gpt-oss-120b", prompt_sha())


def run_table(verdicts):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(len(verdicts))],
                         "response_id": [f"r{k // 2}" for k in range(len(verdicts))], "verdict": verdicts})


def test_prompt_sha_tracks_the_template():
    assert prompt_sha() == prompt_sha(JUDGE_PROMPT)
    assert prompt_sha(JUDGE_PROMPT + " ") != prompt_sha()
    assert len(prompt_sha()) == 64


def test_run_log(tmp_path):
    log = tmp_path / "runs.jsonl"
    t = datetime(2026, 11, 17, tzinfo=timezone.utc)
    record_judge_run(FP, "run-1", 40, log, now=t)
    record_judge_run(FP, "run-2", 40, log, now=t)
    runs = read_judge_runs(log)
    assert [r["run_id"] for r in runs] == ["run-1", "run-2"] and runs[0]["prompt_sha"] == FP.prompt_sha
    assert read_judge_runs(tmp_path / "missing.jsonl") == []


def test_identical_runs():
    v = [True, False] * 10
    r = retest_agreement(run_table(v), run_table(v), FP, FP, reps=200)
    assert (r["kind"], r["n"], r["flips"], r["agreement"], r["kappa"]) == ("test-retest", 20, 0, 1.0, 1.0)
    assert r["agreement_ci"] == (1.0, 1.0)


def test_one_flip_hand():
    a = [True, False] * 10
    b = a[:]
    b[3] = True                                              # one NO -> YES
    r = retest_agreement(run_table(a), run_table(b), FP, FP, reps=500, seed=0)
    assert r["flips"] == 1 and r["agreement"] == pytest.approx(19 / 20)
    # pa = 10/20, pb = 11/20 -> pe = 0.5 * 0.55 + 0.5 * 0.45 = 0.5 -> kappa = (0.95 - 0.5) / 0.5 = 0.9
    assert r["kappa"] == pytest.approx(0.9)
    lo, hi = r["agreement_ci"]
    assert lo < 0.95 <= hi <= 1.0
    assert retest_agreement(run_table(a), run_table(b), FP, FP, reps=500, seed=0) == r


def test_unreadable_and_drift_label():
    a = run_table([True, False, None, True])
    b = run_table([True, False, True, None])
    other = JudgeFingerprint(FP.judge_model, "openai/gpt-oss-120b-2026-12", FP.prompt_sha)
    r = retest_agreement(a, b, FP, other, reps=50)
    assert (r["kind"], r["n"], r["unreadable"]) == ("drift", 2, 2)


def test_rejudge_with_fake_judge():
    items = pd.DataFrame({"item_id": ["a", "b"], "response_id": ["r", "r"], "conversation": ["c", "c"],
                          "reply": ["Go to the ER", "wait"], "criterion": ["x", "x"]})

    async def judge(conversation, reply, criterion):
        return "ER" in reply
    out = asyncio.run(rejudge(items, judge))
    assert list(out["verdict"]) == [True, False] and list(out.columns) == ["item_id", "response_id", "verdict"]
