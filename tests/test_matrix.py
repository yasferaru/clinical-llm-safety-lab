# ruff: noqa: E702
"""Offline coverage for the resumable response/judgment artifact layer."""
import asyncio
import json
from pathlib import Path

import pytest

from evals.matrix import MatrixConfig, MatrixExecutionHalted, MatrixRun, mock_judge, mock_target, read_artifact
from evals.judge_validation.remediation import classify_invalid


def config(tmp_path, **changes):
    base = dict(targets=("target-a",), judges=("judge-a", "judge-b"), epochs=2,
                case_ids=("triage-001-base",), output=str(tmp_path / "run"))
    base.update(changes)
    return MatrixConfig(**base)


def run_all(run, target=mock_target, judge=mock_judge):
    asyncio.run(run.run_targets(target))
    asyncio.run(run.run_judges(judge))


def test_response_reuse_multiple_judges_and_complete_matrix(tmp_path):
    calls = {"target": 0, "judge": 0}
    async def target(case, meta):
        calls["target"] += 1
        return await mock_target(case, meta)
    async def judge(*args):
        calls["judge"] += 1
        return await mock_judge(*args)
    run = MatrixRun(config(tmp_path)); run_all(run, target, judge)
    # triage-001-base has four rubric lines: 2 epochs × 2 judges × 4 lines.
    assert calls == {"target": 2, "judge": 16}
    assert run.validate() == {"completed": 16, "failed": 0, "skipped": 0, "pending": 0, "invalid": 0,
                              "responses_completed": 2, "responses_failed": 0, "responses_invalid": 0}


def test_resume_is_idempotent_and_inflight_is_not_recalled(tmp_path):
    calls = 0
    async def target(case, meta):
        nonlocal calls; calls += 1
        return await mock_target(case, meta)
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    asyncio.run(run.run_targets(target)); asyncio.run(run.run_targets(target))
    assert calls == 1
    response = next((run.root / "responses").glob("*.json"))
    doc = json.loads(response.read_text()); doc["status"] = "in_flight"; response.write_text(json.dumps(doc))
    asyncio.run(run.run_targets(target))
    assert calls == 1  # uncertain provider outcome needs explicit reconciliation, never a duplicate call


def test_bounded_target_run_stops_after_one_pending_cell(tmp_path):
    calls = 0
    async def target(case, meta):
        nonlocal calls; calls += 1
        return await mock_target(case, meta)
    run = MatrixRun(config(tmp_path, epochs=2))
    assert asyncio.run(run.run_targets(target, targets={"target-a"}, max_cells=1)) == 1
    assert calls == 1 and run.validate()["responses_completed"] == 1


def test_failed_target_and_judge_are_reported_and_resumeable(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    async def bad_target(case, meta): raise RuntimeError("offline target failure")
    asyncio.run(run.run_targets(bad_target))
    assert run.validate()["failed"] == 4 and run.validate()["responses_failed"] == 1
    asyncio.run(run.run_targets(mock_target))
    async def bad_judge(*args): raise RuntimeError("offline judge failure")
    asyncio.run(run.run_judges(bad_judge))
    assert run.validate()["failed"] == 4
    asyncio.run(run.run_judges(mock_judge))
    assert run.validate()["completed"] == 4


def test_empty_target_response_is_invalid_and_not_automatically_retried(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    async def empty_target(case, meta):
        return {"response": "", "provider_model": "mock/empty"}
    asyncio.run(run.run_targets(empty_target))
    asyncio.run(run.run_targets(mock_target))
    counts = run.validate()
    assert counts["responses_invalid"] == 1 and counts["invalid"] == 4


def test_corrupt_and_configuration_drift_are_invalid_or_rejected(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1)); run_all(run)
    next((run.root / "judgments").glob("*.json")).write_text("not json")
    assert run.validate()["invalid"] == 1
    changed = MatrixRun(config(tmp_path, judges=("other-judge",), epochs=1))
    try:
        changed.initialize()
    except ValueError as exc:
        assert "different matrix config" in str(exc)
    else:
        raise AssertionError("configuration drift was accepted")


def test_duplicate_cells_are_impossible_by_content_address(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1)); run.initialize()
    case = run.cases[0]
    assert run.target_id(case, "target-a", 1) == run.target_id(case, "target-a", 1)
    assert len({run.target_id(case, "target-a", 1), run.target_id(case, "target-a", 1)}) == 1


def test_target_specific_judges_reuse_one_response_without_pooling(tmp_path):
    run = MatrixRun(config(tmp_path, targets=("target-a", "target-b"), judges=("common", "a-only", "b-only"),
                           judge_assignments={"target-a": ("common", "a-only"), "target-b": ("common", "b-only")},
                           epochs=1))
    run_all(run)
    # One four-line response per target; each has precisely its two configured judges.
    assert run.validate()["responses_completed"] == 2
    assert run.validate()["completed"] == 16
    artifacts = [json.loads(path.read_text()) for path in (run.root / "judgments").glob("*.json")]
    by_response = {}
    for artifact in artifacts:
        by_response.setdefault(artifact["response_cell_id"], set()).add(artifact["requested_judge"])
    expected = {frozenset(("common", "a-only")), frozenset(("common", "b-only"))}
    assert set(map(frozenset, by_response.values())) == expected


def test_missing_judge_verdict_is_invalid_not_completed(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    async def null_verdict(*args):
        return {"text": "unparseable", "verdict": None, "provider_model": "mock"}
    asyncio.run(run.run_targets(mock_target)); asyncio.run(run.run_judges(null_verdict))
    assert run.validate()["invalid"] == 4


def test_fail_fast_checkpoints_first_invalid_judgment(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    async def null_verdict(*args):
        return {"text": "bad", "verdict": None, "provider_model": "mock"}
    asyncio.run(run.run_targets(mock_target))
    try:
        asyncio.run(run.run_judges(null_verdict, fail_fast_invalid=True))
    except MatrixExecutionHalted:
        pass
    else:
        raise AssertionError("invalid verdict did not halt")
    assert run.validate()["invalid"] == 1 and len(list((run.root / "judgments").glob("*.json"))) == 1
    with pytest.raises(MatrixExecutionHalted, match="requires explicit recovery"):
        asyncio.run(run.run_judges(mock_judge, fail_fast_invalid=True))
    assert len(list((run.root / "judgments").glob("*.json"))) == 1


def test_fail_fast_checkpoints_provider_failure_without_running_other_judgments(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    async def failed_provider(*args):
        raise RuntimeError("provider 503")
    asyncio.run(run.run_targets(mock_target))
    with pytest.raises(MatrixExecutionHalted, match="provider failure checkpointed"):
        asyncio.run(run.run_judges(failed_provider, fail_fast_failures=True))
    artifacts = list((run.root / "judgments").glob("*.json"))
    assert len(artifacts) == 1 and read_artifact(artifacts[0])["status"] == "failed"


def test_fail_fast_checkpoints_invalid_target_without_another_provider_call(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    calls = 0

    async def empty_target(*_args):
        nonlocal calls
        calls += 1
        return {"response": "", "provider_model": "deepseek-flash", "usage": {"reasoning_tokens": 1024}}

    with pytest.raises(MatrixExecutionHalted, match="invalid target response checkpointed"):
        asyncio.run(run.run_targets(empty_target, fail_fast_failures=True))
    artifacts = list((run.root / "responses").glob("*.json"))
    assert calls == 1 and len(artifacts) == 1 and read_artifact(artifacts[0])["status"] == "invalid"


def test_fail_fast_refuses_uncertain_in_flight_judgment_without_another_call(tmp_path):
    """Reproduce v5's persisted pre-response checkpoint safely and offline."""
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    asyncio.run(run.run_targets(mock_target))
    response = read_artifact(next((run.root / "responses").glob("*.json")))
    cid = run.judge_id(response, "judge-a", 0)
    (run.root / "judgments").mkdir(exist_ok=True)
    run.judge_path(cid).write_text(json.dumps({"schema_version": 1, "cell_id": cid, "status": "in_flight"}))
    calls = 0

    async def must_not_call(*_args):
        nonlocal calls
        calls += 1
        return {"verdict": True}

    with pytest.raises(MatrixExecutionHalted, match="uncertain provider outcome"):
        asyncio.run(run.run_judges(must_not_call, fail_fast_failures=True))
    assert calls == 0


def test_lifecycle_log_records_dispatch_receipt_and_persistence_without_content(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    asyncio.run(run.run_targets(mock_target))
    asyncio.run(run.run_judges(mock_judge))
    events = [json.loads(line) for line in (run.root / "meta" / "request_lifecycle.jsonl").read_text().splitlines()]
    expected = {"checkpoint_created", "request_dispatched", "provider_response_received",
                "terminal_artifact_persisted"}
    assert expected <= {event["event"] for event in events}
    assert all("prompt" not in event and "response" not in event and "credential" not in event for event in events)


def test_remediation_classifier_rejects_cap_truncated_structured_output():
    result = classify_invalid({"requested_judge": "xai/grok-4.7", "request_max_output_tokens": 64,
                               "usage": {"completion_tokens": 64},
                               "raw_output": '{"verdict":"NO","reason":"cut'})
    assert result["cause"] == "truncation_suspected_malformed_json"
    assert result["recoverable_prefix_verdict"] == "NO"


def test_v3_persisted_invalid_judgments_remain_rejected():
    root = Path("matrix-runs/smoke-three-target-xai-35-lite-v3/judgments")
    if not root.exists():
        pytest.skip("v3 live artifact fixture unavailable")
    invalid = [read_artifact(path) for path in root.glob("*.json")]
    invalid = [item for item in invalid if item and item.get("status") == "invalid"]
    assert len(invalid) == 22
    classifications = [classify_invalid(item) for item in invalid]
    assert all(item["cause"].startswith("truncation_suspected_") for item in classifications)


def test_judge_allowlist_runs_only_requested_cells(tmp_path):
    run = MatrixRun(config(tmp_path, judges=("judge-a",), epochs=1))
    asyncio.run(run.run_targets(mock_target))
    response = next(read_artifact(path) for path in (run.root / "responses").glob("*.json"))
    allowed = {(response["cell_id"], "judge-a", 1)}
    asyncio.run(run.run_judges(mock_judge, allowed_cells=allowed))
    assert len(list((run.root / "judgments").glob("*.json"))) == 1
