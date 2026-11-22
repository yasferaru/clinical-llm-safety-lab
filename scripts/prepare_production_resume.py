#!/usr/bin/env python3
"""Build a fresh, provenance-preserving production selection root offline."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, atomic_json, read_artifact  # noqa: E402


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="matrix-runs/production-three-target-strict-v1")
    parser.add_argument("--recovery", default="matrix-runs/production-recovery-76262722406d-v1")
    parser.add_argument("--config", default="config/matrix.production_three_target_strict_resume_v3.yaml")
    parser.add_argument("--failed-cell", default="76262722406d1a26e536db8ef5c25fa19d505ca5e185b23314856178bb1de075")
    args = parser.parse_args()
    source, recovery = Path(args.source).resolve(), Path(args.recovery).resolve()
    failed = read_artifact(source / "responses" / f"{args.failed_cell}.json")
    if not failed or failed.get("status") != "failed":
        raise SystemExit("source failed target artifact is absent or not failed")
    run = MatrixRun(MatrixConfig.from_file(args.config))
    if run.root.exists() and any(run.root.iterdir()):
        raise SystemExit(f"destination already exists and is nonempty: {run.root}")
    run.initialize()

    selected: dict[str, dict] = {}
    copied_responses = copied_judgments = 0
    for path in sorted((source / "responses").glob("*.json")):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed":
            continue
        logical_id = run.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
        if logical_id in selected:
            raise SystemExit(f"duplicate selected logical target: {logical_id}")
        if artifact["cell_id"] != logical_id:
            raise SystemExit(f"source completed target has unexpected identity: {path}")
        selected[logical_id] = {"source_artifact_cell_id": artifact["cell_id"], "source_artifact_path": relative(path),
                                "selection_kind": "imported_completed"}
        destination = run.target_path(logical_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied_responses += 1

    candidates = [read_artifact(path) for path in sorted((recovery / "responses").glob("*.json"))]
    candidates = [x for x in candidates if x and x.get("status") == "completed"
                  and x.get("case", {}).get("id") == failed["case"]["id"]
                  and x.get("requested_model") == failed["requested_model"]
                  and x.get("epoch") == failed["epoch"]]
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one completed recovery artifact, found {len(candidates)}")
    recovered = dict(candidates[0])
    logical_id = run.target_id(recovered["case"], recovered["requested_model"], recovered["epoch"])
    if logical_id != args.failed_cell:
        raise SystemExit("recovery does not correspond to the original failed logical target")
    if logical_id in selected:
        raise SystemExit(f"duplicate selected logical target: {logical_id}")
    recovery_path = recovery / "responses" / f"{recovered['cell_id']}.json"
    recovered.update({"source_artifact_cell_id": recovered["cell_id"], "source_artifact_path": relative(recovery_path),
                      "recovery_protocol": "production-google-429-retry-v1", "recovered_from_cell_id": args.failed_cell,
                      "cell_id": logical_id})
    atomic_json(run.target_path(logical_id), recovered)
    selected[logical_id] = {
        "source_artifact_cell_id": recovered["source_artifact_cell_id"],
        "source_artifact_path": recovered["source_artifact_path"],
        "selection_kind": "recovered_completed",
        "recovered_from_cell_id": args.failed_cell,
    }

    # Only judgments whose selected response is present and whose identity remains
    # exactly valid under this protocol can be imported.
    for path in sorted((source / "judgments").glob("*.json")):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed" or artifact.get("verdict") not in (True, False):
            continue
        response = read_artifact(run.target_path(artifact.get("response_cell_id", "")))
        if not response:
            continue
        logical_id = run.judge_id(response, artifact["requested_judge"], artifact["line_index"])
        if artifact["cell_id"] != logical_id:
            raise SystemExit(f"source completed judgment has unexpected identity: {path}")
        destination = run.judge_path(logical_id)
        if destination.exists():
            raise SystemExit(f"duplicate selected logical judgment: {logical_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied_judgments += 1

    manifest = {"schema_version": 1, "selection_protocol": "production-resume-selection-v2",
                "source_root": relative(source), "recovery_root": relative(recovery),
                "excluded_source_artifacts": {"failed_target": args.failed_cell},
                "selected_targets": selected,
                "downstream_judgments_pending_for_recovered_target": (
                    len(recovered["case"]["rubric"])
                    * len(run.assigned_judges(recovered["requested_model"]))
                ),
                "copied_completed_responses": copied_responses, "copied_completed_judgments": copied_judgments,
                "validation": run.validate()}
    atomic_json(run.root / "meta" / "resolved_selection.json", manifest)
    print(json.dumps({"destination": str(run.root), "selected_targets": len(selected),
                      "copied_completed_responses": copied_responses, "copied_completed_judgments": copied_judgments,
                      "validation": run.validate()}, sort_keys=True))


if __name__ == "__main__":
    main()
