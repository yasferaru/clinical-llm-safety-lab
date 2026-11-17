"""clinical_safety() startup: explicit judge, independence check, default epochs, logged metadata."""
from types import SimpleNamespace

import inspect_ai.model
import pytest

from evals import run_eval
from evals.judge_validation.independence import JudgeIndependenceError
from evals.results import CASE_FIELDS, SAMPLE_COLUMNS, case_table, merge_samples, rows_from_log, samples_from_lines


@pytest.fixture
def model_under_test(monkeypatch):
    def use(name):
        monkeypatch.setattr(inspect_ai.model, "get_model", lambda *a, **k: name)          # model under test
        # the judge client: never construct a real provider client in tests (no keys, no network)
        monkeypatch.setattr(run_eval, "get_model", lambda m, **k: SimpleNamespace(name=m))
    return use


def test_judge_is_required(model_under_test):
    model_under_test("ollama/qwen3:8b")
    with pytest.raises(ValueError, match="give the judge explicitly"):
        run_eval.clinical_safety()


def test_same_family_judge_is_refused(model_under_test):
    model_under_test("ollama/qwen3:8b")
    with pytest.raises(JudgeIndependenceError):
        run_eval.clinical_safety(judge="ollama/qwen3:32b")


def test_independent_judge_defaults(model_under_test):
    model_under_test("mistral/mistral-small-latest")
    t = run_eval.clinical_safety(judge="groq/openai/gpt-oss-120b")
    assert t.epochs == 5 == run_eval.DEFAULT_EPOCHS
    assert t.metadata["judge_family"] == "openai" and t.metadata["independence"] == "enforce"
    assert t.metadata["judge_prompt_sha256"] == run_eval.JUDGE_PROMPT_SHA and len(run_eval.JUDGE_PROMPT_SHA) == 64


def test_skip_is_explicit_and_warns(model_under_test):
    model_under_test("ollama/qwen3:8b")
    with pytest.warns(UserWarning, match="NOT checked"):
        t = run_eval.clinical_safety(judge="ollama/qwen3:8b", independence="skip", epochs=1)
    assert t.metadata["independence"] == "skip" and t.epochs == 1
    with pytest.raises(ValueError):
        run_eval.clinical_safety(judge="groq/openai/gpt-oss-120b", independence="maybe")


def test_case_fields_and_reply_reach_the_sample_table():
    lines = [{"type": "must", "points": 5, "verdict": True, "criterion": "c"}]
    md = {"cluster_id": "c1", "variant": "base", "category": "emergency_triage",
          "turns": [{"role": "user", "content": "q"}], "acuity_gold": 5, "determinability": "determinable"}
    sample = SimpleNamespace(id="c1-base", epoch=1, output=SimpleNamespace(model="m"), metadata=md,
                             scores={"s": SimpleNamespace(metadata={"lines": lines}, value={}, answer="Call 999 now.")})
    log = SimpleNamespace(eval=SimpleNamespace(task="clinical_safety", eval_id="E", model="p/m",
                                               task_args={"judge": "j"}), samples=[sample])
    lr, sr = rows_from_log(log)
    import pandas as pd
    from evals.results import LINE_COLUMNS
    s = merge_samples(samples_from_lines(pd.DataFrame(lr, columns=LINE_COLUMNS)), sr)
    assert list(s.columns) == SAMPLE_COLUMNS
    row = s.iloc[0]
    assert (row["response"], row["acuity_gold"], row["determinability"]) == ("Call 999 now.", 5, "determinable")
    ct = case_table(s)
    assert list(ct.columns) == ["case_id", "scenario_id", "variant", "category"] + CASE_FIELDS and len(ct) == 1
