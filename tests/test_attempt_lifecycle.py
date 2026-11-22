"""Offline crash-recovery coverage for immutable provider-attempt lifecycles."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from evals.attempt_artifacts import (
    AMBIGUOUS,
    COMPLETED,
    DISPATCH_INTENT,
    DISPATCH_STARTED,
    AttemptStore,
    read_provider_attempt,
)


def make_attempt(store, cell="cell"):
    return store.begin(logical_cell_id=cell, kind="judgment", provider="mock", model="mock/judge",
                       request_identity="request", request_parameters={"temperature": 0})


def completed(store, cell="cell"):
    attempt = make_attempt(store, cell)
    store.transition(attempt["attempt_id"], DISPATCH_INTENT)
    store.transition(attempt["attempt_id"], DISPATCH_STARTED)
    store.record_response(attempt["attempt_id"], response={"verdict": True}, provider_request_id="req-1")
    store.validate_response(attempt["attempt_id"], valid=True)
    store.finalize(attempt, status=COMPLETED, response={"verdict": True}, provider_request_id="req-1")
    return attempt


def test_valid_lifecycle_is_durable_and_selectable(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = completed(store)
    assert [event["state"] for event in store.lifecycle(attempt["attempt_id"])] == [
        "reserved", DISPATCH_INTENT, DISPATCH_STARTED, "provider_response_received", "validated", COMPLETED,
    ]
    selection = store.select("cell", attempt["attempt_id"])
    assert selection["selected_attempt_id"] == attempt["attempt_id"]
    assert store.recover(attempt["attempt_id"]) == "completed"


def test_invalid_lifecycle_transition_is_rejected(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        store.transition(attempt["attempt_id"], COMPLETED)


def test_crash_before_dispatch_is_safely_abandonable(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    assert store.recover(attempt["attempt_id"]) == "safe_to_abandon_before_dispatch"


def test_crash_after_dispatch_intent_is_ambiguous_and_blocked(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    store.transition(attempt["attempt_id"], DISPATCH_INTENT)
    assert store.recover(attempt["attempt_id"]) == "ambiguous_blocked"
    store.transition(attempt["attempt_id"], AMBIGUOUS, reason="restart without receipt")
    assert store.recover(attempt["attempt_id"]) == "ambiguous_blocked"


def test_crash_after_response_receipt_recovers_without_provider(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    store.transition(attempt["attempt_id"], DISPATCH_INTENT)
    store.transition(attempt["attempt_id"], DISPATCH_STARTED)
    store.record_response(attempt["attempt_id"], response={"verdict": True})
    assert store.recover(attempt["attempt_id"]) == "recover_validation_without_provider"


def test_crash_after_validation_completes_selection_idempotently(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    store.transition(attempt["attempt_id"], DISPATCH_INTENT)
    store.transition(attempt["attempt_id"], DISPATCH_STARTED)
    store.record_response(attempt["attempt_id"], response={"verdict": True})
    store.validate_response(attempt["attempt_id"], valid=True)
    assert store.recover(attempt["attempt_id"]) == "recover_selection_without_provider"
    first = store.recover_and_select(attempt)
    assert store.recover_and_select(attempt) == first


def test_selection_is_atomic_under_concurrency_and_never_replaces(tmp_path):
    store = AttemptStore(tmp_path)
    first, second = completed(store), completed(store)

    def select(attempt):
        try:
            return store.select("cell", attempt["attempt_id"])["selected_attempt_id"]
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(select, (first, second)))
    selection = store.read_selection("cell")
    assert selection["selected_attempt_id"] in {first["attempt_id"], second["attempt_id"]}
    assert "lost" in outcomes
    assert read_provider_attempt(store.path(first["attempt_id"])) is not None
    assert read_provider_attempt(store.path(second["attempt_id"])) is not None


def test_invalid_and_ambiguous_attempts_cannot_be_selected(tmp_path):
    store = AttemptStore(tmp_path)
    attempt = make_attempt(store)
    store.transition(attempt["attempt_id"], DISPATCH_INTENT)
    store.transition(attempt["attempt_id"], AMBIGUOUS)
    with pytest.raises(ValueError):
        store.select("cell", attempt["attempt_id"])


def test_legacy_completed_judgment_is_recognized_without_selection_write(tmp_path):
    store = AttemptStore(tmp_path)
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"schema_version":1,"kind":"judgment","cell_id":"old","status":"completed",'
                      '"verdict":true,"raw_output":"VERDICT: YES"}')
    result = store.selection_or_legacy("old", legacy)
    assert result["legacy_selection"] is True and result["selected_attempt_id"] is None
    assert not (tmp_path / "attempts" / "selections").exists()
