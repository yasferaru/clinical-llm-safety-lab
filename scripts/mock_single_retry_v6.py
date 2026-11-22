#!/usr/bin/env python3
"""Offline mock of the isolated v6 attempt, written only under /tmp."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, mock_judge  # noqa: E402


def main() -> None:
    plan = json.loads((ROOT / "matrix-runs/smoke-single-retry-v5/meta/single_retry_plan_v6.json").read_text())
    config = MatrixConfig.from_file(ROOT / "config/matrix.smoke_single_retry_v6.yaml")
    mock_root = Path("/tmp/clinical-llm-smoke-single-retry-v6-mock")
    if mock_root.exists():
        shutil.rmtree(mock_root)
    run = MatrixRun(replace(config, output=str(mock_root)))
    run.initialize()
    source_responses = ROOT / "matrix-runs/smoke-three-target-rejudge-v4/responses"
    shutil.copytree(source_responses, run.root / "responses", dirs_exist_ok=True)
    allowed = {(row["response_cell_id"], row["judge"], int(row["line_index"])) for row in plan["cells"]}
    asyncio.run(run.run_judges(mock_judge, allowed_cells=allowed, fail_fast_invalid=True, fail_fast_failures=True))
    events = [json.loads(line) for line in (run.root / "meta/request_lifecycle.jsonl").read_text().splitlines()]
    print(json.dumps({"mock_root": str(run.root), "new_judgments": len(list((run.root / "judgments").glob("*.json"))),
                      "events": [event["event"] for event in events], "validation": run.validate()}, sort_keys=True))


if __name__ == "__main__":
    main()
