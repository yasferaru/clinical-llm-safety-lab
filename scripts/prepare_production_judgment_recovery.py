#!/usr/bin/env python3
"""Prepare a one-cell, provenance-preserving production judgment recovery offline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, atomic_json, read_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="matrix-runs/production-three-target-strict-resume-v3")
    parser.add_argument("--config", default="config/matrix.production_judgment_recovery_78035ee974bf_v1.yaml")
    parser.add_argument("--failed-cell", default="78035ee974bf66b6310335a7fe7f15a8cee96ea5e0c4546cd63c478afcc62501")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    failed_path = source / "judgments" / f"{args.failed_cell}.json"
    failed = read_artifact(failed_path)
    if not failed or failed.get("status") != "failed":
        raise SystemExit("source failed judgment artifact is absent or not failed")
    response = read_artifact(source / "responses" / f"{failed['response_cell_id']}.json")
    if not response or response.get("status") != "completed":
        raise SystemExit("failed judgment does not reference a completed stored response")
    run = MatrixRun(MatrixConfig.from_file(args.config))
    if run.root.exists() and any(run.root.iterdir()):
        raise SystemExit(f"destination already exists and is nonempty: {run.root}")
    run.initialize()
    copied = dict(response)
    copied["source_artifact_cell_id"] = response["cell_id"]
    copied["source_artifact_path"] = str((source / "responses" / f"{response['cell_id']}.json").relative_to(ROOT))
    destination = run.target_path(response["cell_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, copied)
    line_index = int(failed["line_index"])
    recovery_id = run.judge_id(copied, failed["requested_judge"], line_index)
    plan = {
        "schema_version": 1,
        "protocol": "production-judgment-recovery-v1",
        "retry_limit": 0,
        "source_root": str(source.relative_to(ROOT)),
        "source_failed_artifact_path": str(failed_path.relative_to(ROOT)),
        "source_failed_cell_id": args.failed_cell,
        "response_source_cell_id": response["cell_id"],
        "recovery_judgment_cell_id": recovery_id,
        "judge": failed["requested_judge"],
        "line_index": line_index,
        "cells": [{
            "response_cell_id": response["cell_id"],
            "judge": failed["requested_judge"],
            "line_index": line_index,
        }],
        "max_rejudge_usd": 0.0052,
        "warning": "Run with --skip-targets. No automatic retry is permitted; source failure remains immutable.",
    }
    atomic_json(run.root / "meta" / "judgment_selection_plan.json", plan)
    atomic_json(source / "meta" / "judgment_recovery_plan_78035ee974bf.json", plan)
    print(json.dumps({"destination": str(run.root), "recovery_judgment_cell_id": recovery_id,
                      "plan": str(run.root / 'meta' / 'judgment_selection_plan.json')}, sort_keys=True))


if __name__ == "__main__":
    main()
