#!/usr/bin/env python3
"""Offline-only costed plan for rejudging v3 invalid cells; makes no provider calls."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.remediation import classify_invalid  # noqa: E402
from evals.matrix import read_artifact  # noqa: E402


def estimate(artifact: dict, config: dict) -> float:
    model, usage = artifact["requested_judge"], artifact.get("usage") or {}
    prices, cap = config["pricing_usd_per_million_tokens"], config["request_limits"]["judge_max_output_tokens"]
    if model.startswith("anthropic/"):
        raw = usage.get("input_tokens", 0) * prices[model]["input"] / 1e6 + cap * prices[model]["output"] / 1e6
    elif model.startswith("google/"):
        raw = usage.get("promptTokenCount", 0) * prices[model]["input"] / 1e6 + cap * prices[model]["output"] / 1e6
    else:
        prior = usage.get("cost_in_usd_ticks", 0) / 10_000_000_000
        raw = prior + (cap - usage.get("completion_tokens", 0)) * prices[model]["output"] / 1e6
    return raw * config["cost_calibration"]["safety_margin"]


def main() -> None:
    run = ROOT / "matrix-runs/smoke-three-target-xai-35-lite-v3"
    config = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    cells = []
    for path in sorted((run / "judgments").glob("*.json")):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "invalid":
            continue
        audit = classify_invalid(artifact)
        cells.append({"source_cell_id": artifact["cell_id"], "source_artifact": str(path.relative_to(run)),
                      "judge": artifact["requested_judge"], "response_cell_id": artifact["response_cell_id"],
                      "line_index": artifact["line_index"], "classification": audit["cause"],
                      "offline_recovery_candidate": audit["recoverable_prefix_verdict"] is not None,
                      "max_rejudge_usd": round(estimate(artifact, config), 9)})
    total = sum(x["max_rejudge_usd"] for x in cells)
    recovered = sum(x["max_rejudge_usd"] for x in cells if x["offline_recovery_candidate"])
    report = {"new_inference_calls_default": len(cells), "by_judge": dict(Counter(x["judge"] for x in cells)),
              "max_rejudge_usd": round(total, 9),
              "offline_recovery_candidates": sum(x["offline_recovery_candidate"] for x in cells),
              "remaining_if_owner_approves_recovery_candidates": round(total - recovered, 9), "cells": cells,
              "warning": "Malformed raw output is not automatically recovered; all 22 source artifacts remain invalid."}
    output = run / "meta" / "rejudge_plan_v4.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "cells"}, sort_keys=True))


if __name__ == "__main__":
    main()
