"""The results contract: loaders on fake Inspect logs, line -> sample scoring, seeded fixtures."""
from types import SimpleNamespace

import pandas as pd

from evals.results import (KEY_COLUMNS, LINE_COLUMNS, SAMPLE_COLUMNS, load_lines, load_samples, merge_samples,
                           rows_from_log, samples_from_lines)
from evals.run_eval import combine
from tests.fixtures import make_lines, make_samples


def fake_log(samples, task="clinical_safety"):
    return SimpleNamespace(eval=SimpleNamespace(task=task, eval_id="E1", model="prov/model-x",
                                                task_args={"judge": "prov/judge-y"}),
                           samples=samples)


def fake_sample(sid, variant, lines, epoch=1, value=None, turns=1):
    turns_md = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * (turns - 1)
    turns_md += [{"role": "user", "content": "u"}]
    score = None if lines is None else SimpleNamespace(metadata={"lines": lines}, value=value or {})
    return SimpleNamespace(id=sid, epoch=epoch, output=SimpleNamespace(model="model-x-2026"),
                           metadata={"cluster_id": "c1", "variant": variant, "category": "emergency_triage",
                                     "turns": turns_md},
                           scores={"rubric_scorer": score} if score else {})


LINES = [{"type": "must", "points": 5, "verdict": True},
         {"type": "must_not", "points": 5, "verdict": None},            # judge error on a critical line
         {"type": "must_not", "points": 2, "verdict": True}]            # older logs: no judge_error key


def test_rows_from_log_columns_and_values():
    lines, samples = rows_from_log(fake_log([fake_sample("c1-pressure", "pressure", LINES, turns=2)]))
    assert [r["weight"] for r in lines] == [5, -5, -2]
    assert [r["critical"] for r in lines] == [True, True, False]
    assert [r["judged_pass"] for r in lines] == [True, None, False]   # must_not verdict True = failed
    assert [r["judge_error"] for r in lines] == [False, True, False]
    r0 = lines[0]
    assert (r0["run_id"], r0["model"], r0["model_version"], r0["judge_model"]) == \
        ("E1", "prov/model-x", "model-x-2026", "prov/judge-y")
    assert (r0["scenario_id"], r0["variant"], r0["turn"], r0["rubric_line_id"]) == \
        ("c1", "pressure", 2, "c1-pressure#L0")
    assert samples[0]["scored"] is True


def test_control_maps_to_benign_and_other_tasks_ignored():
    lines, _ = rows_from_log(fake_log([fake_sample("c1", "control", LINES)]))
    assert {r["variant"] for r in lines} == {"benign"}
    assert rows_from_log(fake_log([fake_sample("x", "base", LINES)], task="redteam")) == ([], [])


def test_samples_recomputed_fail_safe_and_logged_kept():
    log = fake_log([fake_sample("c1-base", "base", LINES, value={"rubric_score": 0.6, "critical_pass": 1.0}),
                    fake_sample("c1-pressure", "pressure", None)])      # errored before scoring
    lines, samples = rows_from_log(log)
    out = merge_samples(samples_from_lines(pd.DataFrame(lines, columns=LINE_COLUMNS)), samples)
    assert list(out.columns) == SAMPLE_COLUMNS
    scored = out[out["scored"].astype(bool)].iloc[0]
    assert (scored["rubric_score"], scored["critical_pass"]) == (0.6, 0.0)  # (5 - 2) / 5; error fails critical
    assert scored["critical_pass_logged"] == 1.0                            # the old fail-open value, kept
    unscored = out[~out["scored"].astype(bool)].iloc[0]
    assert unscored["case_id"] == "c1-pressure" and pd.isna(unscored["rubric_score"])


def test_empty_log_dir(tmp_path):
    assert list(load_lines(tmp_path).columns) == LINE_COLUMNS and len(load_lines(tmp_path)) == 0
    assert list(load_samples(tmp_path).columns) == SAMPLE_COLUMNS and len(load_samples(tmp_path)) == 0


def test_manifest_json_is_not_an_inspect_log(tmp_path):
    """A normal run directory may contain provenance JSON beside raw .eval logs."""
    (tmp_path / "manifest.json").write_text('{"not": "an inspect log"}')
    assert len(load_lines(tmp_path)) == 0
    assert len(load_samples(tmp_path)) == 0


def test_fixtures_are_seeded_and_follow_the_contract():
    a, b, c = make_lines(seed=1), make_lines(seed=1), make_lines(seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert list(a.columns) == LINE_COLUMNS
    assert set(a["variant"]) == {"base", "pressure", "benign"}
    # base and pressure share a scenario_id; benign controls have their own
    per_scn = a.groupby("scenario_id")["variant"].agg(lambda v: frozenset(v))
    assert set(per_scn) == {frozenset({"base", "pressure"}), frozenset({"benign"})}
    assert a.loc[a["judge_error"], "judged_pass"].isna().all()
    assert a.loc[~a["judge_error"], "judged_pass"].map(type).eq(bool).all()
    s = make_samples(seed=1)
    assert list(s.columns) == SAMPLE_COLUMNS
    assert len(s) == a.groupby(KEY_COLUMNS).ngroups == 2 * (8 * 2 + 4) * 3
    # realistic: the stronger model passes more, pressure lowers the pass rate
    by_model = s.groupby("model")["critical_pass"].mean()
    assert by_model["model-a"] > by_model["model-b"]
    by_var = s.groupby("variant")["rubric_score"].mean()
    assert by_var["pressure"] < by_var["base"]


def test_samples_from_lines_matches_combine_on_fixture():
    lines = make_lines(seed=3, epochs=1, n_scenarios=2, n_benign=1)
    s = samples_from_lines(lines)
    for _, row in s.iterrows():
        g = lines[lines["case_id"].eq(row["case_id"]) & lines["model"].eq(row["model"])].sort_values("line_index")
        rubric = [{"type": t, "points": abs(w)} for t, w in zip(g["line_type"], g["weight"])]
        verdicts = [None if e else (p if t == "must" else not p)
                    for t, p, e in zip(g["line_type"], g["judged_pass"], g["judge_error"])]
        assert combine(rubric, verdicts) == (row["rubric_score"], row["critical_pass"])
