#!/usr/bin/env python3
"""Write a redacted, offline remediation audit; source artifacts are never changed."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.remediation import classify_invalid  # noqa: E402
from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", default="matrix-runs/smoke-three-target-xai-35-lite-v3", nargs="?")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.run)
    rows = []
    for path in sorted((root / "judgments").glob("*.json")):
        artifact = read_artifact(path)
        if artifact and artifact.get("status") == "invalid":
            rows.append({"artifact": str(path.relative_to(root)), "cell_id": artifact["cell_id"],
                         "judge": artifact["requested_judge"], "response_cell_id": artifact["response_cell_id"],
                         "line_index": artifact["line_index"], **classify_invalid(artifact)})
    report = {"source_run": str(root), "invalid_count": len(rows), "by_judge": dict(Counter(x["judge"] for x in rows)),
              "by_cause": dict(Counter(x["cause"] for x in rows)), "cells": rows,
              "note": "Raw outputs remain only in source artifacts; this report stores hashes and classifications."}
    output = Path(args.output) if args.output else root / "meta" / "invalid_judgment_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "invalid_count": len(rows),
                      "by_cause": report["by_cause"]}, sort_keys=True))


if __name__ == "__main__":
    main()
