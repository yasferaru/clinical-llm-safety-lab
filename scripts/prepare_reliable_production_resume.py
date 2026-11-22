#!/usr/bin/env python3
"""Create an immutable continuation root and allowlist safe pending judgments."""
from __future__ import annotations

import json
import shutil
import sys
import argparse
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evals.matrix import MatrixConfig, MatrixRun, atomic_json, read_artifact  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="matrix-runs/production-three-target-strict-resume-v5")
    parser.add_argument("--config", default="config/matrix.production_three_target_strict_resume_v6.yaml")
    parser.add_argument("--destination-output")
    parser.add_argument("--blocked-cell", action="append", required=True,
                        help="dispatched-without-response logical judgment; never include it in the live allowlist")
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    blocked_cells = set(args.blocked_cell)
    config = MatrixConfig.from_file(ROOT / args.config)
    run = MatrixRun(replace(config, output=args.destination_output) if args.destination_output else config)
    if run.root.exists() and any(run.root.iterdir()):
        raise SystemExit(f"destination already exists and is nonempty: {run.root}")
    run.initialize()
    copied = {"responses": 0, "judgments": 0}
    for folder in copied:
        for path in (source / folder).glob("*.json"):
            artifact = read_artifact(path)
            valid = artifact and artifact.get("status") == "completed"
            valid = valid and (folder == "responses" or artifact.get("verdict") in (True, False))
            if valid:
                destination = run.root / folder / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                copied[folder] += 1
    plan, blocked = [], []
    for response_path in (run.root / "responses").glob("*.json"):
        response = read_artifact(response_path)
        for judge in run.assigned_judges(response["requested_model"]):
            for index, _ in enumerate(response["case"]["rubric"]):
                logical = run.judge_id(response, judge, index)
                if read_artifact(run.judge_path(logical)):
                    continue
                row = {"response_cell_id": response["cell_id"], "judge": judge, "line_index": index}
                if logical in blocked_cells:
                    blocked.append({**row, "logical_cell_id": logical, "reason": "uncertain_dispatched_read_timeout"})
                else:
                    plan.append(row)
    document = {"schema_version": 1, "protocol": "reliable-production-continuation-v1", "retry_limit": 0,
                "cells": plan, "blocked_uncertain_cells": blocked, "source_root": str(source.relative_to(ROOT)),
                "max_rejudge_usd": 0.0}
    atomic_json(run.root / "meta" / "safe_continuation_plan.json", document)
    atomic_json(run.root / "meta" / "resolved_selection.json", {"schema_version": 1,
                "source_root": str(source.relative_to(ROOT)), "copied_valid": copied,
                "blocked_uncertain_cells": blocked, "validation": run.validate()})
    print(json.dumps({"copied": copied, "safe_pending": len(plan), "blocked_uncertain": len(blocked),
                      "validation": run.validate()}, sort_keys=True))


if __name__ == "__main__":
    main()
