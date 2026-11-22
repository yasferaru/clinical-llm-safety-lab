"""Offline tests for append-only provider-attempt evidence."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from evals.attempt_artifacts import AttemptStore, read_attempt_or_legacy, read_provider_attempt
from evals.matrix import MatrixConfig, MatrixRun, mock_target


def test_multiple_attempts_have_independent_ids_and_numbers(tmp_path):
    store = AttemptStore(tmp_path)
    first = store.begin(logical_cell_id="logical-cell", kind="judgment", provider="mock", model="mock/judge",
                        request_identity="one", request_parameters={"temperature": 0})
    second = store.begin(logical_cell_id="logical-cell", kind="judgment", provider="mock", model="mock/judge",
                         request_identity="two", request_parameters={"temperature": 0})
    store.persist(first, status="failed", error="provider 503")
    store.persist(second, status="completed", response={"verdict": True}, provider_request_id="provider-2",
                  usage={"total_tokens": 3})
    assert first["attempt_id"] != second["attempt_id"]
    assert [first["attempt_number"], second["attempt_number"]] == [1, 2]
    failed = read_provider_attempt(store.path(first["attempt_id"]))
    completed = read_provider_attempt(store.path(second["attempt_id"]))
    assert failed["logical_cell_id"] == completed["logical_cell_id"] == "logical-cell"
    assert failed["lifecycle_status"] == "failed" and completed["provider_request_id"] == "provider-2"


def test_attempt_files_are_immutable(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = store.begin(logical_cell_id="cell", kind="response", provider="mock", model="mock/model",
                          request_identity="request", request_parameters={})
    store.persist(attempt, status="failed", error="first failure")
    with pytest.raises(FileExistsError, match="immutable"):
        store.persist(attempt, status="completed", response="replacement")
    assert read_provider_attempt(store.path(attempt["attempt_id"]))["error"] == "first failure"


def test_concurrent_creation_cannot_collide(tmp_path):
    store = AttemptStore(tmp_path)

    def create(_: int):
        attempt = store.begin(logical_cell_id="shared", kind="judgment", provider="mock", model="mock/judge",
                              request_identity="same", request_parameters={})
        store.persist(attempt, status="completed", response={"verdict": True})
        return attempt

    with ThreadPoolExecutor(max_workers=12) as pool:
        attempts = list(pool.map(create, range(40)))
    assert len({item["attempt_id"] for item in attempts}) == 40
    assert {item["attempt_number"] for item in attempts} == set(range(1, 41))
    assert len(list((tmp_path / "attempts" / "records").glob("*.json"))) == 40


def test_legacy_artifacts_are_read_without_migration(tmp_path):
    legacy = tmp_path / "old.json"
    legacy.write_text(
        '{"schema_version": 1, "kind": "judgment", "cell_id": "old-cell", '
        '"status": "failed", "request_id": "old-request", "error": "historical"}'
    )
    evidence = read_attempt_or_legacy(legacy)
    assert evidence["legacy_artifact"] is True
    assert evidence["logical_cell_id"] == "old-cell"
    assert evidence["lifecycle_status"] == "failed"
    assert not (tmp_path / "attempts").exists()


def test_historical_failure_is_preserved_when_logical_cell_is_revisited(tmp_path):
    config = MatrixConfig(targets=("target-a",), judges=("judge-a",), epochs=1,
                          case_ids=("triage-001-base",), output=str(tmp_path / "run"))
    run = MatrixRun(config)

    async def fail(*_args):
        raise RuntimeError("offline failure")

    asyncio.run(run.run_targets(fail))
    asyncio.run(run.run_targets(mock_target))
    records = [read_provider_attempt(path) for path in (run.root / "attempts" / "records").glob("*.json")]
    records = [record for record in records if record and record["kind"] == "response"]
    assert len(records) == 2
    assert {record["lifecycle_status"] for record in records} == {"failed", "completed"}
    assert len({record["attempt_id"] for record in records}) == 2
