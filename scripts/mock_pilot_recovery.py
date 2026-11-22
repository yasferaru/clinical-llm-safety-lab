#!/usr/bin/env python3
"""Offline mock: one Google recovery response, then a provenance-preserving resume selection."""
from __future__ import annotations

import asyncio
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, atomic_json, mock_target, read_artifact  # noqa: E402


def main() -> None:
    root = Path("/tmp/clinical-llm-pilot-recovery-mock")
    if root.exists():
        shutil.rmtree(root)
    retry = MatrixRun(replace(MatrixConfig.from_file(ROOT / "config/matrix.pilot_google_503_retry_v1.yaml"),
                              output=str(root / "retry")))
    asyncio.run(retry.run_targets(mock_target, fail_fast_failures=True))
    recovered = read_artifact(next((retry.root / "responses").glob("*.json")))
    resume = MatrixRun(replace(MatrixConfig.from_file(ROOT / "config/matrix.pilot_three_target_strict_resume_v2.yaml"),
                               output=str(root / "resume")))
    resume.initialize()
    source = ROOT / "matrix-runs/pilot-three-target-strict-v1/responses"
    copied = 0
    for path in source.glob("*.json"):
        artifact = read_artifact(path)
        if artifact and artifact.get("status") == "completed":
            target_id = resume.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
            target = resume.target_path(target_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    selected = dict(recovered)
    selected["source_artifact_cell_id"] = recovered["cell_id"]
    selected["recovery_protocol"] = "pilot-google-503-retry-v1"
    selected["cell_id"] = resume.target_id(selected["case"], selected["requested_model"], selected["epoch"])
    atomic_json(resume.target_path(selected["cell_id"]), selected)
    print({"retry": retry.validate(), "copied": copied, "resume": resume.validate(),
           "fresh_recovery_id": recovered["cell_id"], "selected_resume_id": selected["cell_id"]})


if __name__ == "__main__":
    main()
