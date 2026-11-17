"""Epoch reduction: per-epoch Scores -> one Score per case -> task metrics, as Inspect does at --epochs k."""
from inspect_ai.scorer import SampleScore, Score, mean, mean_score, stderr

from evals.run_eval import reduce_epochs, rubric_epochs


def S(rubric, critical, errors, epoch):
    return Score(value={"rubric_score": rubric, "critical_pass": critical,
                        "critical_pass_all": critical, "judge_errors": errors},
                 answer=f"reply {epoch}",
                 metadata={"category": "emergency_triage", "variant": "base", "cluster_id": "c1",
                           "unreadable": errors, "lines": [f"epoch {epoch} lines"]})


# two cases, three epochs each
CASE_1 = [S(1.0, 1.0, 0, 1), S(0.5, 0.0, 2, 2), S(1.0, 1.0, 1, 3)]
CASE_2 = [S(0.0, 0.0, 0, 1), S(0.0, 0.0, 0, 2), S(0.3, 0.0, 0, 3)]


def apply_metric(metric, reduced, key):
    """What Inspect's {'*': [...]} metric spec does for one key of a dict-valued score."""
    return metric([SampleScore(score=Score(value=s.value[key]), sample_id=i) for i, s in enumerate(reduced)])


def test_reduce_one_case():
    r = reduce_epochs(CASE_1)
    assert r.value["rubric_score"] == round(2.5 / 3, 4)
    assert r.value["critical_pass"] == 2 / 3               # pass@1 estimate
    assert r.value["critical_pass_all"] == 0.0             # worst-of-k: one epoch failed
    assert r.value["judge_errors"] == 1.0                  # (0 + 2 + 1) / 3
    assert r.metadata["judge_errors_total"] == 3
    assert [m["lines"] for m in r.metadata["epochs"]] == [["epoch 1 lines"], ["epoch 2 lines"], ["epoch 3 lines"]]
    assert r.metadata["cluster_id"] == "c1" and r.answer is None  # answers differ across epochs


def test_registered_reducer_matches_pure_function():
    assert rubric_epochs()(CASE_1) == reduce_epochs(CASE_1)


def test_metrics_after_reduction():
    reduced = [reduce_epochs(CASE_1), reduce_epochs(CASE_2)]
    assert apply_metric(mean(), reduced, "critical_pass") == (2 / 3 + 0) / 2
    assert apply_metric(mean(), reduced, "critical_pass_all") == 0.0       # pass^3
    assert apply_metric(mean(), reduced, "judge_errors") == 0.5
    assert apply_metric(stderr(), reduced, "rubric_score") > 0


def test_single_epoch_is_identity_on_values():
    r = reduce_epochs([CASE_1[0]])
    assert r.value == {"rubric_score": 1.0, "critical_pass": 1.0, "critical_pass_all": 1.0, "judge_errors": 0.0}


def test_default_reducer_loses_metadata():
    """Why judge_errors moved into Score.value: Inspect's default keeps only epoch 1's metadata."""
    default = mean_score()(CASE_1)
    assert default.metadata["unreadable"] == 0           # epochs 2 and 3 had 3 errors between them
    assert reduce_epochs(CASE_1).metadata["judge_errors_total"] == 3
