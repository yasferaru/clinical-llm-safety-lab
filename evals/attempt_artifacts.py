"""Immutable evidence for individual provider API attempts.

Logical matrix artifacts remain deliberately readable for compatibility.  This
module is an append-only companion: an attempt is never addressed by, or allowed
to replace, its logical cell.  Number reservations are separate small files so a
crash may leave a gap, but never permits two attempts for one cell to share a
number.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ATTEMPT_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1

RESERVED = "reserved"
DISPATCH_INTENT = "dispatch_intent"
DISPATCH_STARTED = "dispatch_started"
POSSIBLY_STARTED = "possibly_started"
RESPONSE_RECEIVED = "provider_response_received"
VALIDATED = "validated"
COMPLETED = "completed"
CONFIRMED_REJECTION = "confirmed_rejection"
AMBIGUOUS = "ambiguous"

_TRANSITIONS = {
    RESERVED: {DISPATCH_INTENT, CONFIRMED_REJECTION},
    DISPATCH_INTENT: {DISPATCH_STARTED, POSSIBLY_STARTED, AMBIGUOUS},
    DISPATCH_STARTED: {RESPONSE_RECEIVED, AMBIGUOUS},
    POSSIBLY_STARTED: {RESPONSE_RECEIVED, AMBIGUOUS},
    RESPONSE_RECEIVED: {VALIDATED},
    VALIDATED: {COMPLETED, CONFIRMED_REJECTION},
    COMPLETED: set(), CONFIRMED_REJECTION: set(), AMBIGUOUS: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    """Create *path* atomically, failing rather than replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    try:
        # link(2) is an atomic create-if-absent operation on the same filesystem.
        os.link(temp, path)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable attempt artifact already exists: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


class AttemptStore:
    """Append-only attempt artifacts rooted at ``<matrix-root>/attempts``."""

    def __init__(self, root: Path):
        self.root = root / "attempts"

    def path(self, attempt_id: str) -> Path:
        return self.root / "records" / f"{attempt_id}.json"

    def lifecycle_path(self, attempt_id: str) -> Path:
        return self.root / "lifecycle" / f"{attempt_id}.jsonl"

    def selection_path(self, logical_cell_id: str) -> Path:
        return self.root / "selections" / f"{logical_cell_id}.json"

    def _append_lifecycle(self, attempt_id: str, state: str, **evidence: Any) -> dict[str, Any]:
        path = self.lifecycle_path(attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {"attempt_id": attempt_id, "state": state, "timestamp": _now(), **evidence}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def lifecycle(self, attempt_id: str) -> list[dict[str, Any]]:
        try:
            lines = self.lifecycle_path(attempt_id).read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []

    def state(self, attempt_id: str) -> str | None:
        events = self.lifecycle(attempt_id)
        return events[-1]["state"] if events else None

    def transition(self, attempt_id: str, state: str, **evidence: Any) -> dict[str, Any]:
        """Durably advance a lifecycle state; illegal or terminal rewrites fail."""
        current = self.state(attempt_id)
        if current not in _TRANSITIONS or state not in _TRANSITIONS[current]:
            raise ValueError(f"invalid lifecycle transition: {current!r} -> {state!r}")
        return self._append_lifecycle(attempt_id, state, **evidence)

    def _reserve_number(self, logical_cell_id: str) -> int:
        directory = self.root / "numbers" / logical_cell_id
        directory.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            reservation = directory / str(number)
            try:
                fd = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                number += 1
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("reserved\n")
                handle.flush()
                os.fsync(handle.fileno())
            return number

    def begin(self, *, logical_cell_id: str, kind: str, provider: str | None,
              model: str | None, request_identity: str, request_parameters: dict[str, Any]) -> dict[str, Any]:
        """Allocate identity before dispatch; no provider request is made here."""
        attempt = {
            "attempt_schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": uuid.uuid4().hex,
            "logical_cell_id": logical_cell_id,
            "attempt_number": self._reserve_number(logical_cell_id),
            "kind": kind,
            "provider": provider,
            "model": model,
            "request_identity": request_identity,
            "request_parameters": request_parameters,
            "started_at": _now(),
        }
        self._append_lifecycle(attempt["attempt_id"], RESERVED, logical_cell_id=logical_cell_id)
        return attempt

    def recover(self, attempt_id: str) -> str:
        """Classify restart handling without ever dispatching another provider call."""
        state = self.state(attempt_id)
        if state == RESERVED:
            return "safe_to_abandon_before_dispatch"
        if state in {DISPATCH_INTENT, DISPATCH_STARTED, POSSIBLY_STARTED, AMBIGUOUS}:
            return "ambiguous_blocked"
        if state == RESPONSE_RECEIVED:
            return "recover_validation_without_provider"
        if state == VALIDATED:
            return "recover_selection_without_provider"
        if state == COMPLETED:
            return "completed"
        if state == CONFIRMED_REJECTION:
            return "confirmed_rejection"
        return "missing_lifecycle_blocked"

    def record_response(self, attempt_id: str, *, response: Any, provider_request_id: str | None = None,
                        usage: dict[str, Any] | None = None, billing: dict[str, Any] | None = None,
                        provider_model: str | None = None) -> dict[str, Any]:
        """Durably retain a received response before validation or selection."""
        return self.transition(attempt_id, RESPONSE_RECEIVED, response=response,
                               provider_request_id=provider_request_id, usage=usage, billing=billing,
                               provider_model=provider_model)

    def validate_response(self, attempt_id: str, *, valid: bool, error: str | None = None) -> dict[str, Any]:
        return self.transition(attempt_id, VALIDATED, valid=valid, error=error)

    def finalize(self, attempt: dict[str, Any], *, status: str, response: Any = None,
                 error: str | None = None, provider_request_id: str | None = None,
                 usage: dict[str, Any] | None = None, billing: dict[str, Any] | None = None,
                 provider_model: str | None = None) -> Path:
        """Persist terminal evidence only after validation, then close the lifecycle."""
        valid = next((event for event in reversed(self.lifecycle(attempt["attempt_id"]))
                      if event["state"] == VALIDATED), None)
        if not valid:
            raise ValueError("attempt must be validated before finalization")
        if bool(valid.get("valid")) != (status == COMPLETED):
            raise ValueError("terminal status contradicts validation")
        path = self.persist(attempt, status=status, response=response, error=error,
                            provider_request_id=provider_request_id, usage=usage, billing=billing,
                            provider_model=provider_model)
        self.transition(attempt["attempt_id"], COMPLETED if status == COMPLETED else CONFIRMED_REJECTION)
        return path

    def recover_and_select(self, attempt: dict[str, Any]) -> dict[str, Any]:
        """Finish a validated interrupted attempt from durable evidence only."""
        attempt_id = attempt["attempt_id"]
        if self.state(attempt_id) == COMPLETED:
            return self.select(attempt["logical_cell_id"], attempt_id)
        events = self.lifecycle(attempt_id)
        validation = next((event for event in reversed(events) if event["state"] == VALIDATED), None)
        receipt = next((event for event in reversed(events) if event["state"] == RESPONSE_RECEIVED), None)
        if not validation or not validation.get("valid") or not receipt:
            raise ValueError("validated response evidence is required for recovery selection")
        self.finalize(attempt, status=COMPLETED, response=receipt.get("response"),
                      provider_request_id=receipt.get("provider_request_id"), usage=receipt.get("usage"),
                      billing=receipt.get("billing"), provider_model=receipt.get("provider_model"))
        return self.select(attempt["logical_cell_id"], attempt_id)

    def persist(self, attempt: dict[str, Any], *, status: str, response: Any = None,
                error: str | None = None, provider_request_id: str | None = None,
                usage: dict[str, Any] | None = None, billing: dict[str, Any] | None = None,
                provider_model: str | None = None, completed_at: str | None = None) -> Path:
        """Write one terminal attempt record. Existing records are never modified."""
        required = {"attempt_id", "logical_cell_id", "attempt_number", "kind", "request_identity", "started_at"}
        missing = required - set(attempt)
        if missing:
            raise ValueError(f"attempt is missing required fields: {sorted(missing)}")
        record = {**attempt, "lifecycle_status": status, "completed_at": completed_at or _now(),
                  "provider_model": provider_model, "provider_request_id": provider_request_id,
                  "usage": usage, "billing": billing, "response": response, "error": error}
        _atomic_create_json(self.path(record["attempt_id"]), record)
        return self.path(record["attempt_id"])

    def read(self, attempt_id: str) -> dict[str, Any] | None:
        return read_provider_attempt(self.path(attempt_id))

    def select(self, logical_cell_id: str, attempt_id: str) -> dict[str, Any]:
        """Atomically select one validated completed attempt for a logical cell."""
        prior = self.read_selection(logical_cell_id)
        if prior:
            if prior.get("selected_attempt_id") == attempt_id:
                return prior
            raise FileExistsError(f"logical result already selected: {logical_cell_id}")
        attempt = self.read(attempt_id)
        events = self.lifecycle(attempt_id)
        validated = next((event for event in reversed(events) if event["state"] == VALIDATED), None)
        if not attempt or attempt.get("logical_cell_id") != logical_cell_id:
            raise ValueError("attempt does not belong to logical cell")
        if attempt.get("lifecycle_status") != COMPLETED or not validated or not validated.get("valid"):
            raise ValueError("only validated completed attempts may be selected")
        selection = {"selection_schema_version": SELECTION_SCHEMA_VERSION, "logical_cell_id": logical_cell_id,
                     "selected_attempt_id": attempt_id, "selected_at": _now(),
                     "result": attempt.get("response"), "provider_request_id": attempt.get("provider_request_id")}
        try:
            _atomic_create_json(self.selection_path(logical_cell_id), selection)
        except FileExistsError:
            prior = self.read_selection(logical_cell_id)
            if prior and prior.get("selected_attempt_id") == attempt_id:
                return prior
            raise
        return selection

    def read_selection(self, logical_cell_id: str) -> dict[str, Any] | None:
        try:
            value = json.loads(self.selection_path(logical_cell_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        valid = isinstance(value, dict) and value.get("selection_schema_version") == SELECTION_SCHEMA_VERSION
        return value if valid else None

    def selection_or_legacy(self, logical_cell_id: str, legacy_path: Path | None = None) -> dict[str, Any] | None:
        """Return immutable selection, or recognize a historical completed artifact read-only."""
        selection = self.read_selection(logical_cell_id)
        if selection:
            return selection
        if legacy_path:
            legacy = read_attempt_or_legacy(legacy_path)
            is_completed = legacy and legacy.get("lifecycle_status") == COMPLETED
            if legacy and legacy.get("logical_cell_id") == logical_cell_id and is_completed:
                return {"selection_schema_version": 0, "logical_cell_id": logical_cell_id,
                        "selected_attempt_id": None, "result": legacy.get("response"), "legacy_selection": True}
        return None


def read_provider_attempt(path: Path) -> dict[str, Any] | None:
    """Read a new immutable attempt record, returning ``None`` for any legacy file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("attempt_schema_version") != ATTEMPT_SCHEMA_VERSION:
        return None
    return value


def read_attempt_or_legacy(path: Path) -> dict[str, Any] | None:
    """Read an attempt, or expose legacy logical artifacts as attempt-like evidence.

    This is intentionally read-only compatibility: historical artifacts are not
    rewritten or assigned synthetic files.
    """
    attempt = read_provider_attempt(path)
    if attempt:
        return attempt
    try:
        legacy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(legacy, dict) or legacy.get("schema_version") != 1 or not legacy.get("cell_id"):
        return None
    return {"attempt_schema_version": 0, "attempt_id": None, "logical_cell_id": legacy["cell_id"],
            "attempt_number": legacy.get("attempt"), "kind": legacy.get("kind"),
            "lifecycle_status": legacy.get("status"), "provider": None,
            "model": legacy.get("provider_model") or legacy.get("requested_model") or legacy.get("requested_judge"),
            "provider_request_id": legacy.get("request_id"), "usage": legacy.get("usage"),
            "billing": None, "response": legacy.get("response", legacy.get("raw_output")),
            "error": legacy.get("error"), "legacy_artifact": True}
