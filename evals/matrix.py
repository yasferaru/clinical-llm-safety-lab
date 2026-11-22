# ruff: noqa: E501, E701, E702
"""Resumable, provider-agnostic clinical evaluation matrix.

This module deliberately does not call a provider.  Production code supplies async
target and judge callables; the CLI's ``--mock`` mode is the only built-in runner.
Artifacts are immutable JSON objects written atomically, so target responses can be
rejudged without regenerating them.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from evals.attempt_artifacts import AttemptStore
from evals.run_eval import JUDGE_PROMPT, conversation_text, parse_verdict, strip_think

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

TargetCall = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
JudgeCall = Callable[[str, str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class MatrixExecutionHalted(RuntimeError):
    """A checkpoint has been written but live execution must not continue."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        temp = Path(f.name)
    os.replace(temp, path)


def read_artifact(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION else None


def safe_name(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


@dataclass(frozen=True)
class MatrixConfig:
    dataset: str = "evals/cases.jsonl"
    targets: tuple[str, ...] = ()
    judges: tuple[str, ...] = ()
    # Optional target-specific subset. Omitting it retains the historical all-to-all
    # behavior; specifying it prevents invalid family-conflict combinations.
    judge_assignments: dict[str, tuple[str, ...]] = field(default_factory=dict)
    epochs: int = 1
    case_ids: tuple[str, ...] = ()
    decoding: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.7, "seed": 0})
    judge_prompt: str = JUDGE_PROMPT
    rubric_version: str = "dataset"
    judge_output_schema_versions: dict[str, str] = field(default_factory=dict)
    target_attempt_namespace: str = ""
    judge_attempt_namespace: str = ""
    concurrency: int = 1
    retry_limit: int = 0
    output: str = "matrix-runs/offline-mock"

    @classmethod
    def from_file(cls, path: str | Path) -> "MatrixConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        raw["targets"] = tuple(raw.get("targets", ()))
        raw["judges"] = tuple(raw.get("judges", ()))
        raw["judge_assignments"] = {k: tuple(v) for k, v in raw.get("judge_assignments", {}).items()}
        raw["case_ids"] = tuple(raw.get("case_ids", ()))
        return cls(**raw)


def load_cases(config: MatrixConfig) -> list[dict[str, Any]]:
    path = ROOT / config.dataset
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(config.case_ids)
    if wanted:
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise ValueError(f"case_ids absent from dataset: {sorted(missing)}")
        cases = [c for c in cases if c["id"] in wanted]
    if not cases or not config.targets or not config.judges or config.epochs < 1:
        raise ValueError("need cases, at least one target and judge, and epochs >= 1")
    return cases


class MatrixRun:
    def __init__(self, config: MatrixConfig):
        self.config, self.cases = config, load_cases(config)
        self.root = Path(config.output).resolve()
        self.dataset_hash = hashlib.sha256((ROOT / config.dataset).read_bytes()).hexdigest()
        self.config_hash = digest(asdict(config))
        self.prompt_hash = hashlib.sha256(config.judge_prompt.encode()).hexdigest()
        self.attempts = AttemptStore(self.root)

    def initialize(self) -> None:
        snapshot = self.root / "meta" / "dataset.jsonl"
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != self.dataset_hash:
            raise ValueError("existing run has a different dataset snapshot; choose a new output directory")
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=snapshot.parent)
            with os.fdopen(fd, "wb") as out:
                out.write((ROOT / self.config.dataset).read_bytes())
            os.replace(tmp, snapshot)
        manifest = self.root / "meta" / "matrix_manifest.json"
        prior = read_artifact(manifest)
        if prior and prior.get("config_hash") != self.config_hash:
            raise ValueError("existing run has a different matrix config; choose a new output directory")
        if not prior:
            try:
                inspect_version = version("inspect-ai")
            except PackageNotFoundError:
                inspect_version = None
            atomic_json(manifest, {"schema_version": SCHEMA_VERSION, "created_at": now(), "config": asdict(self.config),
                "config_hash": self.config_hash, "dataset_hash": self.dataset_hash, "dataset_snapshot": "meta/dataset.jsonl",
                "judge_prompt_hash": self.prompt_hash, "rubric_hash": digest([{k: c[k] for k in ("id", "rubric")} for c in self.cases]),
                "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "python": platform.python_version(), "inspect_ai": inspect_version,
                "requirements_lock_sha256": hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()})

    def target_id(self, case: dict[str, Any], target: str, epoch: int) -> str:
        identity = {"dataset": self.dataset_hash, "case": case["id"], "target": target, "epoch": epoch,
                    "decoding": self.config.decoding}
        if self.config.target_attempt_namespace:
            identity["target_attempt_namespace"] = self.config.target_attempt_namespace
        return digest(identity)

    def judge_id(self, response: dict[str, Any], judge: str, line_index: int) -> str:
        identity = {"response_id": response["cell_id"], "judge": judge, "line": line_index,
                       "judge_prompt_hash": self.prompt_hash, "rubric_version": self.config.rubric_version,
                       "rubric_line": response["case"]["rubric"][line_index]}
        # Keep legacy artifact IDs stable; explicit output-schema versions create a
        # new cell identity because their response contract is scientifically distinct.
        if judge in self.config.judge_output_schema_versions:
            identity["judge_output_schema_version"] = self.config.judge_output_schema_versions[judge]
        if self.config.judge_attempt_namespace:
            identity["judge_attempt_namespace"] = self.config.judge_attempt_namespace
        return digest(identity)

    def lifecycle_event(self, event: str, *, cell_id: str, kind: str,
                        requested_model: str, **metadata: Any) -> None:
        """Durably log non-sensitive request lifecycle evidence.

        Deliberately excludes prompts, completions, credentials, and headers.
        """
        path = self.root / "meta" / "request_lifecycle.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": now(), "event": event, "cell_id": cell_id,
                  "kind": kind, "requested_model": requested_model, **metadata}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def target_path(self, cell_id: str) -> Path:
        return self.root / "responses" / f"{cell_id}.json"

    def judge_path(self, cell_id: str) -> Path:
        return self.root / "judgments" / f"{cell_id}.json"

    def provider_attempt(self, *, cell_id: str, kind: str, model: str,
                         request_parameters: dict[str, Any]) -> dict[str, Any]:
        """Allocate an immutable provider-attempt identity for one logical cell."""
        provider = model.split("/", 1)[0] if "/" in model else None
        return self.attempts.begin(logical_cell_id=cell_id, kind=kind, provider=provider, model=model,
                                   request_identity=digest({"cell_id": cell_id, "kind": kind,
                                                            "model": model, "parameters": request_parameters}),
                                   request_parameters=request_parameters)

    def persist_provider_attempts(self, first: dict[str, Any], *, status: str, response: Any,
                                  error: str | None, result: dict[str, Any] | None = None) -> None:
        """Persist one record per adapter-reported HTTP attempt (or one legacy call)."""
        evidence = (result or {}).get("provider_attempts") or [{}]
        for position, item in enumerate(evidence):
            attempt = first if position == 0 else self.provider_attempt(
                cell_id=first["logical_cell_id"], kind=first["kind"], model=first["model"],
                request_parameters=first["request_parameters"])
            self.attempts.persist(attempt, status=item.get("lifecycle_status", status),
                                  response=item.get("response", response), error=item.get("error", error),
                                  provider_request_id=item.get("provider_request_id") or (result or {}).get("request_id"),
                                  usage=item.get("usage") or (result or {}).get("usage"),
                                  billing=item.get("billing") or (result or {}).get("billing"),
                                  provider_model=item.get("provider_model") or (result or {}).get("provider_model"))

    def assigned_judges(self, target: str) -> tuple[str, ...]:
        """The explicitly assigned judge set for a target, or legacy all-to-all."""
        assigned = self.config.judge_assignments.get(target, self.config.judges)
        unknown = set(assigned) - set(self.config.judges)
        if unknown:
            raise ValueError(f"assigned judge(s) absent from judges: {sorted(unknown)}")
        return assigned

    async def run_targets(self, call: TargetCall, targets: set[str] | None = None,
                          max_cells: int | None = None,
                          after_artifact: Callable[[dict[str, Any]], None] | None = None,
                          fail_fast_failures: bool = False) -> int:
        """Run unfinished target cells, optionally a bounded target subset for a gated smoke stage."""
        self.initialize()
        invoked = 0
        for target in self.config.targets:
            if targets is not None and target not in targets:
                continue
            for epoch in range(1, self.config.epochs + 1):
                for case in self.cases:
                    if max_cells is not None and invoked >= max_cells:
                        return invoked
                    cid, path = self.target_id(case, target, epoch), None
                    path = self.target_path(cid)
                    existing = read_artifact(path)
                    if existing and existing.get("status") in {"completed", "invalid"}:
                        continue
                    if existing and existing.get("status") == "in_flight":
                        # Never retry an outcome that might already have reached a provider.
                        if fail_fast_failures:
                            raise MatrixExecutionHalted(
                                f"uncertain provider outcome requires explicit reconciliation: {cid}")
                        continue
                    atomic_json(path, {"schema_version": SCHEMA_VERSION, "kind": "response", "cell_id": cid,
                        "status": "in_flight", "started_at": now(), "case": case, "requested_model": target,
                        "epoch": epoch, "decoding": self.config.decoding, "dataset_hash": self.dataset_hash,
                        "attempt": 1})
                    self.lifecycle_event("checkpoint_created", cell_id=cid, kind="response", requested_model=target)
                    attempt = self.provider_attempt(cell_id=cid, kind="response", model=target,
                                                    request_parameters={"epoch": epoch, "decoding": self.config.decoding,
                                                                        "dataset_hash": self.dataset_hash})
                    try:
                        invoked += 1
                        # This fsynced intent is deliberately before the call: a crash
                        # from here onward is conservative/ambiguous until reconciled.
                        self.attempts.transition(attempt["attempt_id"], "dispatch_intent")
                        self.lifecycle_event("request_dispatched", cell_id=cid, kind="response", requested_model=target,
                                             attempt_id=attempt["attempt_id"], attempt_number=attempt["attempt_number"])
                        result = await call(case, {"model": target, "epoch": epoch, "decoding": self.config.decoding})
                        self.lifecycle_event("provider_response_received", cell_id=cid, kind="response",
                                             requested_model=target, provider_model=result.get("provider_model"),
                                             request_id_present=bool(result.get("request_id")),
                                             finish_reason=result.get("finish_reason"),
                                             usage_fields=sorted((result.get("usage") or {}).keys()))
                        for event in result.get("attempt_events", []):
                            self.lifecycle_event(event.pop("event"), cell_id=cid, kind="response",
                                                 requested_model=target, **event)
                        response_text = result.get("response", "")
                        valid_response = isinstance(response_text, str) and bool(response_text.strip())
                        artifact = {"schema_version": SCHEMA_VERSION, "kind": "response", "cell_id": cid,
                            "status": "completed" if valid_response else "invalid",
                            "started_at": existing.get("started_at") if existing else now(), "completed_at": now(),
                            "case": case, "requested_model": target, "provider_model": result.get("provider_model"), "epoch": epoch,
                            "decoding": self.config.decoding, "dataset_hash": self.dataset_hash, "response": response_text,
                            "usage": result.get("usage"), "request_id": result.get("request_id"),
                            "request_max_output_tokens": result.get("request_max_output_tokens"),
                            "generation_policy": result.get("generation_policy"),
                            "finish_reason": result.get("finish_reason"),
                            "attempt": result.get("attempts", 1),
                            "error": None if valid_response else "empty target response"}
                        atomic_json(path, artifact)
                        self.persist_provider_attempts(attempt, status=artifact["status"], response=response_text,
                                                       error=artifact["error"], result=result)
                        self.lifecycle_event("terminal_artifact_persisted", cell_id=cid, kind="response",
                                             requested_model=target, status=artifact["status"])
                        if after_artifact:
                            after_artifact(artifact)
                        if not valid_response and fail_fast_failures:
                            raise MatrixExecutionHalted(f"invalid target response checkpointed: {cid}")
                    except MatrixExecutionHalted:
                        raise
                    except Exception as exc:
                        for event in getattr(exc, "attempt_events", []):
                            self.lifecycle_event(event.pop("event"), cell_id=cid, kind="response",
                                                 requested_model=target, **event)
                        self.lifecycle_event("provider_failure_classified", cell_id=cid, kind="response",
                                             requested_model=target,
                                             classification=getattr(exc, "classification", "unknown"),
                                             retry_after=getattr(exc, "retry_after", None))
                        artifact = {"schema_version": SCHEMA_VERSION, "kind": "response", "cell_id": cid,
                            "status": "failed", "failed_at": now(), "case": case, "requested_model": target, "epoch": epoch,
                            "decoding": self.config.decoding, "dataset_hash": self.dataset_hash, "error": f"{type(exc).__name__}: {exc}", "attempt": 1}
                        atomic_json(path, artifact)
                        self.persist_provider_attempts(attempt, status="failed", response=None, error=artifact["error"],
                                                       result={"provider_attempts": getattr(exc, "provider_attempts", []),
                                                               "request_id": getattr(exc, "request_id", None), "usage": getattr(exc, "usage", None),
                                                               "billing": getattr(exc, "billing", None), "provider_model": getattr(exc, "provider_model", None)})
                        self.lifecycle_event("terminal_artifact_persisted", cell_id=cid, kind="response",
                                             requested_model=target, status="failed", error_type=type(exc).__name__)
                        if after_artifact:
                            after_artifact(artifact)
                        if fail_fast_failures:
                            raise MatrixExecutionHalted(f"provider failure checkpointed: {cid}")
        return invoked

    async def run_judges(self, call: JudgeCall, fail_fast_invalid: bool = False,
                         after_artifact: Callable[[dict[str, Any]], None] | None = None,
                         allowed_cells: set[tuple[str, str, int]] | None = None,
                         fail_fast_failures: bool = False) -> None:
        self.initialize()
        if fail_fast_invalid:
            prior = next((path for path in (self.root / "judgments").glob("*.json")
                          if (artifact := read_artifact(path)) and artifact.get("status") == "invalid"), None)
            if prior:
                raise MatrixExecutionHalted(f"invalid verdict requires explicit recovery: {prior.name}")
        for response_path in sorted((self.root / "responses").glob("*.json")):
            response = read_artifact(response_path)
            if not response or response.get("status") != "completed":
                continue
            for judge in self.assigned_judges(response["requested_model"]):
                for index, line in enumerate(response["case"]["rubric"]):
                    if allowed_cells is not None and (response["cell_id"], judge, index) not in allowed_cells:
                        continue
                    cid, path = self.judge_id(response, judge, index), None
                    path = self.judge_path(cid)
                    existing = read_artifact(path)
                    if existing and existing.get("status") in {"completed", "invalid"}:
                        continue
                    if existing and existing.get("status") == "in_flight":
                        if fail_fast_failures:
                            raise MatrixExecutionHalted(
                                f"uncertain provider outcome requires explicit reconciliation: {cid}")
                        continue
                    schema_version = self.config.judge_output_schema_versions.get(judge, "legacy-json-object-v1")
                    atomic_json(path, {"schema_version": SCHEMA_VERSION, "kind": "judgment", "cell_id": cid, "status": "in_flight",
                        "started_at": now(), "response_cell_id": response["cell_id"], "requested_judge": judge,
                        "line_index": index, "criterion": line, "judge_prompt_hash": self.prompt_hash, "rubric_version": self.config.rubric_version,
                        "judge_output_schema_version": schema_version})
                    self.lifecycle_event("checkpoint_created", cell_id=cid, kind="judgment", requested_model=judge)
                    attempt = self.provider_attempt(cell_id=cid, kind="judgment", model=judge,
                                                    request_parameters={"response_cell_id": response["cell_id"],
                                                                        "line_index": index, "judge_prompt_hash": self.prompt_hash,
                                                                        "rubric_version": self.config.rubric_version,
                                                                        "judge_output_schema_version": schema_version})
                    try:
                        # Do not leave a dispatched request looking safely abandoned.
                        self.attempts.transition(attempt["attempt_id"], "dispatch_intent")
                        self.lifecycle_event("request_dispatched", cell_id=cid, kind="judgment", requested_model=judge,
                                             attempt_id=attempt["attempt_id"], attempt_number=attempt["attempt_number"])
                        result = await call(conversation_text(response["case"]["turns"]), response["response"], line["criterion"],
                                            {"model": judge, "judge_prompt": self.config.judge_prompt})
                        self.lifecycle_event("provider_response_received", cell_id=cid, kind="judgment",
                                             requested_model=judge, provider_model=result.get("provider_model"),
                                             request_id_present=bool(result.get("request_id")),
                                             finish_reason=result.get("finish_reason"),
                                             usage_fields=sorted((result.get("usage") or {}).keys()))
                        for event in result.get("attempt_events", []):
                            self.lifecycle_event(event.pop("event"), cell_id=cid, kind="judgment",
                                                 requested_model=judge, **event)
                        text = result.get("text", "")
                        verdict = result.get("verdict", parse_verdict(strip_think(text)))
                        status = "completed" if isinstance(verdict, bool) else "invalid"
                        artifact = {"schema_version": SCHEMA_VERSION, "kind": "judgment", "cell_id": cid, "status": status,
                            "completed_at": now(), "response_cell_id": response["cell_id"], "requested_judge": judge,
                            "provider_model": result.get("provider_model"), "line_index": index, "criterion": line,
                            "judge_prompt_hash": self.prompt_hash, "rubric_version": self.config.rubric_version,
                            "judge_output_schema_version": schema_version, "verdict": verdict, "raw_output": text,
                            "usage": result.get("usage"), "request_id": result.get("request_id"),
                            "request_max_output_tokens": result.get("request_max_output_tokens"),
                            "attempt": result.get("attempts", 1),
                            "error": None if status == "completed" else "missing or unparseable verdict"}
                        atomic_json(path, artifact)
                        self.persist_provider_attempts(attempt, status=status, response={"text": text, "verdict": verdict},
                                                       error=artifact["error"], result=result)
                        self.lifecycle_event("terminal_artifact_persisted", cell_id=cid, kind="judgment",
                                             requested_model=judge, status=status)
                        if after_artifact:
                            after_artifact(artifact)
                        if status == "invalid" and fail_fast_invalid:
                            raise MatrixExecutionHalted(f"invalid verdict checkpointed: {cid}")
                    except MatrixExecutionHalted:
                        raise
                    except Exception as exc:
                        for event in getattr(exc, "attempt_events", []):
                            self.lifecycle_event(event.pop("event"), cell_id=cid, kind="judgment",
                                                 requested_model=judge, **event)
                        self.lifecycle_event("provider_failure_classified", cell_id=cid, kind="judgment",
                                             requested_model=judge,
                                             classification=getattr(exc, "classification", "unknown"),
                                             retry_after=getattr(exc, "retry_after", None))
                        artifact = {"schema_version": SCHEMA_VERSION, "kind": "judgment", "cell_id": cid, "status": "failed",
                            "failed_at": now(), "response_cell_id": response["cell_id"], "requested_judge": judge,
                            "line_index": index, "criterion": line, "judge_prompt_hash": self.prompt_hash,
                            "rubric_version": self.config.rubric_version, "error": f"{type(exc).__name__}: {exc}", "attempt": 1}
                        atomic_json(path, artifact)
                        self.persist_provider_attempts(attempt, status="failed", response=None, error=artifact["error"],
                                                       result={"provider_attempts": getattr(exc, "provider_attempts", []),
                                                               "request_id": getattr(exc, "request_id", None), "usage": getattr(exc, "usage", None),
                                                               "billing": getattr(exc, "billing", None), "provider_model": getattr(exc, "provider_model", None)})
                        self.lifecycle_event("terminal_artifact_persisted", cell_id=cid, kind="judgment",
                                             requested_model=judge, status="failed", error_type=type(exc).__name__)
                        if after_artifact:
                            after_artifact(artifact)
                        if fail_fast_failures:
                            raise MatrixExecutionHalted(f"provider failure checkpointed: {cid}")

    def validate(self) -> dict[str, int]:
        """Return expected target/judge cells classified without treating failures as completion."""
        counts = {k: 0 for k in ("completed", "failed", "skipped", "pending", "invalid", "responses_completed", "responses_failed", "responses_invalid")}
        for target in self.config.targets:
            for epoch in range(1, self.config.epochs + 1):
                for case in self.cases:
                    rp = self.target_path(self.target_id(case, target, epoch)); response = read_artifact(rp)
                    if rp.exists() and not response:
                        counts["invalid"] += len(case["rubric"]) * len(self.assigned_judges(target)); continue
                    if not response:
                        counts["pending"] += len(case["rubric"]) * len(self.assigned_judges(target)); continue
                    if response.get("status") != "completed":
                        if response.get("status") == "invalid":
                            counts["responses_invalid"] += 1
                            counts["invalid"] += len(case["rubric"]) * len(self.assigned_judges(target))
                            continue
                        counts["responses_failed"] += 1
                        counts["failed" if response.get("status") == "failed" else "pending"] += len(case["rubric"]) * len(self.assigned_judges(target)); continue
                    counts["responses_completed"] += 1
                    for judge in self.assigned_judges(target):
                        for i, _ in enumerate(case["rubric"]):
                            artifact = read_artifact(self.judge_path(self.judge_id(response, judge, i)))
                            if self.judge_path(self.judge_id(response, judge, i)).exists() and not artifact: counts["invalid"] += 1
                            elif not artifact: counts["pending"] += 1
                            elif artifact.get("response_cell_id") != response["cell_id"]: counts["invalid"] += 1
                            elif artifact.get("status") == "completed" and artifact.get("verdict") in (True, False): counts["completed"] += 1
                            elif artifact.get("status") == "failed": counts["failed"] += 1
                            elif artifact.get("status") == "invalid" or artifact.get("verdict") not in (True, False): counts["invalid"] += 1
                            else: counts["pending"] += 1
        return counts


async def mock_target(case: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    return {"response": f"Mock response for {case['id']}", "provider_model": f"mock/{meta['model']}", "usage": {"total_tokens": 1}, "request_id": f"mock-{case['id']}"}


async def mock_judge(conversation: str, response: str, criterion: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"text": "VERDICT: YES", "verdict": True, "provider_model": f"mock/{meta['model']}", "usage": {"total_tokens": 1}, "request_id": digest([response, criterion])[:12]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="offline-safe resumable matrix runner")
    ap.add_argument("config")
    ap.add_argument("--mock", action="store_true", help="run deterministic local mock providers")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    run = MatrixRun(MatrixConfig.from_file(args.config))
    if not args.mock:
        raise SystemExit("No provider is wired here. Use --mock; production adapters require a separate approved phase.")
    asyncio.run(run.run_targets(mock_target)); asyncio.run(run.run_judges(mock_judge))
    print(json.dumps(run.validate(), sort_keys=True))


if __name__ == "__main__":
    main()
