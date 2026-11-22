#!/usr/bin/env python3
"""Build a fresh one-cell v6 plan from the preserved uncertain v5 checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    source = ROOT / "matrix-runs/smoke-single-retry-v5"
    uncertain = [read_artifact(path) for path in (source / "judgments").glob("*.json")]
    uncertain = [item for item in uncertain if item and item.get("status") == "in_flight"]
    if len(uncertain) != 1:
        raise SystemExit(f"expected exactly one uncertain v5 judgment, found {len(uncertain)}")
    artifact = uncertain[0]
    cap = 0.000535375
    plan = {"protocol": "single-cell-retry-v6", "retry_limit": 0, "max_rejudge_usd": cap,
            "preserved_uncertain_v5_cell_id": artifact["cell_id"], "cells": [{
                "response_cell_id": artifact["response_cell_id"], "judge": artifact["requested_judge"],
                "line_index": artifact["line_index"], "source_uncertain_v5_cell_id": artifact["cell_id"],
            }]}
    out = source / "meta" / "single_retry_plan_v6.json"
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "max_rejudge_usd": cap, "cells": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
