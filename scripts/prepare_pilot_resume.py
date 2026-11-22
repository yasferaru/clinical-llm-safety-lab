#!/usr/bin/env python3
"""Build a fresh pilot response-selection root without overwriting source attempts."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, atomic_json, read_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="matrix-runs/pilot-three-target-strict-v1")
    parser.add_argument("--recovery", required=True, help="completed isolated Google retry root")
    parser.add_argument("--destination-config", default="config/matrix.pilot_three_target_strict_resume_v2.yaml")
    parser.add_argument("--case-id", default="ctx-malaria-which-pill-pressure")
    parser.add_argument("--model", default="google/gemini-3.5-flash-lite")
    parser.add_argument("--destination-output", default=None)
    args = parser.parse_args()
    source, recovery = Path(args.source).resolve(), Path(args.recovery).resolve()
    config = MatrixConfig.from_file(args.destination_config)
    destination = MatrixRun(replace(config, output=args.destination_output) if args.destination_output else config)
    destination.initialize()
    copied, copied_judgments = 0, 0
    for path in (source / "responses").glob("*.json"):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed":
            continue
        target_id = destination.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
        expected = destination.target_path(target_id)
        if not expected.exists():
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, expected)
            copied += 1
    for path in (source / "judgments").glob("*.json"):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed" or artifact.get("verdict") not in (True, False):
            continue
        response = read_artifact(destination.target_path(artifact["response_cell_id"]))
        if not response:
            continue
        judgment_id = destination.judge_id(response, artifact["requested_judge"], artifact["line_index"])
        expected = destination.judge_path(judgment_id)
        if not expected.exists():
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, expected)
            copied_judgments += 1
    recovered = [read_artifact(path) for path in (recovery / "responses").glob("*.json")]
    recovered = [item for item in recovered if item and item.get("status") == "completed"
                 and item.get("requested_model") == args.model and item.get("case", {}).get("id") == args.case_id]
    if len(recovered) != 1:
        raise SystemExit(f"expected one completed isolated recovery response, found {len(recovered)}")
    artifact = dict(recovered[0])
    destination_id = destination.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
    artifact["source_artifact_cell_id"] = artifact["cell_id"]
    artifact["source_artifact_path"] = str((recovery / "responses" / f"{artifact['cell_id']}.json").relative_to(ROOT))
    artifact["recovery_protocol"] = "pilot-google-503-retry-v1"
    artifact["cell_id"] = destination_id
    atomic_json(destination.target_path(destination_id), artifact)
    print({"copied_completed_source_responses": copied, "copied_completed_judgments": copied_judgments,
           "selected_recovery_response": destination_id,
           "destination": str(destination.root)})


if __name__ == "__main__":
    main()
