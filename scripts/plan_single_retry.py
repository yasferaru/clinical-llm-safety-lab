#!/usr/bin/env python3
"""Build an isolated no-retry plan from exactly one preserved failed v4 artifact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    root = ROOT / "matrix-runs/smoke-three-target-rejudge-v4"
    failed = [read_artifact(path) for path in (root / "judgments").glob("*.json")]
    failed = [item for item in failed if item and item.get("status") == "failed"]
    if len(failed) != 1:
        raise SystemExit(f"expected exactly one failed v4 judgment, found {len(failed)}")
    artifact = failed[0]
    # The approved plan computed a conservative per-cell cap from the old failed source.
    cap = 0.000535375
    plan = {"protocol": "single-cell-retry-v5", "retry_limit": 0, "max_rejudge_usd": cap,
            "preserved_failed_v4_cell_id": artifact["cell_id"], "cells": [{
                "response_cell_id": artifact["response_cell_id"], "judge": artifact["requested_judge"],
                "line_index": artifact["line_index"], "source_failed_v4_cell_id": artifact["cell_id"],
            }]}
    out = root / "meta" / "single_retry_plan_v5.json"
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "max_rejudge_usd": cap, "cells": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
