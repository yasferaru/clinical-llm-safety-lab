#!/usr/bin/env python3
"""Create a fresh production selection root, optionally selecting one recovered judgment."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, atomic_json, read_artifact  # noqa: E402


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="matrix-runs/production-three-target-strict-resume-v3")
    parser.add_argument("--failure-source", help="root retaining the original failed judgment; defaults to --source")
    parser.add_argument("--config", default="config/matrix.production_three_target_strict_resume_v4.yaml")
    parser.add_argument("--recovery", help="completed isolated recovery root, if selecting its valid judgment")
    parser.add_argument("--destination-output", help="required only to select a recovery into a second fresh root")
    parser.add_argument("--failed-cell", default="78035ee974bf66b6310335a7fe7f15a8cee96ea5e0c4546cd63c478afcc62501")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    failure_source = Path(args.failure_source).resolve() if args.failure_source else source
    config = MatrixConfig.from_file(args.config)
    if args.destination_output:
        config = replace(config, output=args.destination_output)
    run = MatrixRun(config)
    if run.root.exists() and any(run.root.iterdir()):
        raise SystemExit(f"destination already exists and is nonempty: {run.root}")
    failed = read_artifact(failure_source / "judgments" / f"{args.failed_cell}.json")
    if not failed or failed.get("status") != "failed":
        raise SystemExit("source failed judgment artifact is absent or not failed")
    run.initialize()
    targets: dict[str, dict] = {}
    judgments: dict[str, dict] = {}
    for path in sorted((source / "responses").glob("*.json")):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed":
            continue
        logical = run.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
        if logical != artifact["cell_id"] or logical in targets:
            raise SystemExit(f"duplicate or unexpected source target identity: {path}")
        targets[logical] = {"source_artifact_cell_id": artifact["cell_id"], "source_artifact_path": rel(path),
                            "selection_kind": "imported_completed"}
        dest = run.target_path(logical)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for path in sorted((source / "judgments").glob("*.json")):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed" or artifact.get("verdict") not in (True, False):
            continue
        response = read_artifact(run.target_path(artifact["response_cell_id"]))
        if not response:
            continue
        logical = run.judge_id(response, artifact["requested_judge"], artifact["line_index"])
        if logical != artifact["cell_id"] or logical in judgments:
            raise SystemExit(f"duplicate or unexpected source judgment identity: {path}")
        judgments[logical] = {"source_artifact_cell_id": artifact["cell_id"], "source_artifact_path": rel(path),
                              "selection_kind": "imported_completed"}
        dest = run.judge_path(logical)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    if args.recovery:
        recovery = Path(args.recovery).resolve()
        candidates = [read_artifact(p) for p in (recovery / "judgments").glob("*.json")]
        candidates = [
            x for x in candidates
            if x and x.get("status") == "completed" and x.get("verdict") in (True, False)
            and x.get("requested_judge") == failed["requested_judge"]
            and x.get("line_index") == failed["line_index"]
            and x.get("response_cell_id") == failed["response_cell_id"]
        ]
        if len(candidates) != 1:
            raise SystemExit(f"expected exactly one completed recovery judgment, found {len(candidates)}")
        recovered = dict(candidates[0])
        response = read_artifact(run.target_path(recovered["response_cell_id"]))
        logical = run.judge_id(response, recovered["requested_judge"], recovered["line_index"])
        if logical != args.failed_cell or logical in judgments:
            raise SystemExit("recovery cannot be selected for the original failed logical judgment")
        recovery_path = recovery / "judgments" / f"{recovered['cell_id']}.json"
        recovered.update({
            "source_artifact_cell_id": recovered["cell_id"],
            "source_artifact_path": rel(recovery_path),
            "recovery_protocol": "production-xai-429-judgment-retry-v1",
            "recovered_from_cell_id": args.failed_cell,
            "cell_id": logical,
        })
        atomic_json(run.judge_path(logical), recovered)
        judgments[logical] = {
            "source_artifact_cell_id": recovered["source_artifact_cell_id"],
            "source_artifact_path": recovered["source_artifact_path"],
            "selection_kind": "recovered_completed",
            "recovered_from_cell_id": args.failed_cell,
        }
    manifest = {"schema_version": 1, "selection_protocol": "production-judgment-resume-selection-v1",
                "source_root": rel(source), "failure_source_root": rel(failure_source),
                "source_failed_judgment": args.failed_cell,
                "recovery_root": rel(Path(args.recovery)) if args.recovery else None,
                "selected_targets": targets, "selected_judgments": judgments,
                "validation": run.validate()}
    atomic_json(run.root / "meta" / "resolved_selection.json", manifest)
    print(json.dumps({"destination": str(run.root), "selected_targets": len(targets),
                      "selected_judgments": len(judgments), "validation": run.validate()}, sort_keys=True))


if __name__ == "__main__":
    main()
