#!/usr/bin/env python3
"""Offline pilot topology, strict-protocol, and conservative-budget preflight."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger  # noqa: E402
from evals.judge_validation.independence import assert_judge_independent  # noqa: E402
from evals.matrix import MatrixConfig, MatrixRun  # noqa: E402
from scripts.run_live_matrix import estimate_run  # noqa: E402


def main() -> int:
    config = MatrixConfig.from_file(ROOT / "config/matrix.pilot_three_target_strict_v1.yaml")
    run = MatrixRun(config)
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    lines = sum(len(case["rubric"]) for case in run.cases)
    expected_responses = len(run.cases) * len(config.targets) * config.epochs
    expected_judgments = lines * sum(len(run.assigned_judges(target)) for target in config.targets) * config.epochs
    forecast = estimate_run(run, budget["pricing_usd_per_million_tokens"], config.retry_limit + 1)
    ledger = SpendLedger(ROOT / budget["budget"]["ledger_path"], BudgetPolicy(
        ceiling_usd=budget["budget"]["ceiling_usd"], stop_usd=budget["budget"]["stop_usd"],
        stage_limits=budget["budget"]["stage_limits_usd"]))
    current = ledger.spent()
    errors = []
    if len(run.cases) != 16 or lines != 78 or expected_responses != 48 or expected_judgments != 468:
        errors.append("unexpected pilot matrix dimensions")
    if config.retry_limit != 0:
        errors.append("pilot must have zero automatic retries")
    if set(config.judge_output_schema_versions.values()) != {"verdict-first-native-schema-v3"}:
        errors.append("pilot judges do not share the strict protocol")
    for target in config.targets:
        for judge in run.assigned_judges(target):
            try:
                assert_judge_independent(judge, [target])
            except ValueError as exc:
                errors.append(str(exc))
    variants = Counter(case["variant"] for case in run.cases)
    within_stage = forecast <= budget["budget"]["stage_limits_usd"]["pilot"]
    within_global = current + forecast <= budget["budget"]["stop_usd"]
    if not within_stage:
        errors.append("pilot forecast exceeds stage cap")
    if not within_global:
        errors.append("pilot forecast exceeds operational stop")
    print(json.dumps({"offline_only": True, "cases": len(run.cases), "rubric_lines": lines,
                      "expected_target_responses": expected_responses, "expected_judgments": expected_judgments,
                      "variants": dict(variants), "pilot_forecast_usd": round(forecast, 9),
                      "ledger_current_usd": round(current, 9), "post_reservation_usd": round(current + forecast, 9),
                      "pilot_stage_cap_usd": budget["budget"]["stage_limits_usd"]["pilot"],
                      "within_stage_cap": within_stage, "within_operational_stop": within_global,
                      "errors": errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
