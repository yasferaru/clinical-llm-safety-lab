#!/usr/bin/env python3
"""Offline pricing/topology preflight for the one-cell Google recovery attempt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger  # noqa: E402
from evals.matrix import MatrixConfig, MatrixRun  # noqa: E402
from scripts.run_live_matrix import estimate_run  # noqa: E402


def main() -> int:
    run = MatrixRun(MatrixConfig.from_file(ROOT / "config/matrix.pilot_google_503_retry_v1.yaml"))
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    # The authorized first-pending-target command returns before judging, so do not
    # reserve downstream judgments in this one-target recovery forecast.
    forecast = estimate_run(run, budget["pricing_usd_per_million_tokens"], 1, allowed_cells=set())
    ledger = SpendLedger(ROOT / budget["budget"]["ledger_path"], BudgetPolicy(
        ceiling_usd=budget["budget"]["ceiling_usd"], stop_usd=budget["budget"]["stop_usd"],
        stage_limits=budget["budget"]["stage_limits_usd"]))
    current, pilot_reserved = ledger.spent(), sum(float(x.get("usd", 0)) for x in ledger.entries()
                                                   if x.get("stage") == "pilot" and x.get("state") != "released")
    errors = []
    if len(run.cases) != 1 or run.config.retry_limit != 0 or len(run.config.targets) != 1:
        errors.append("retry topology is not one target cell with zero retries")
    if current + forecast > budget["budget"]["stop_usd"]:
        errors.append("retry exceeds operational stop")
    if pilot_reserved + forecast > budget["budget"]["stage_limits_usd"]["pilot"]:
        errors.append("retry exceeds pilot stage cap")
    print(json.dumps({"offline_only": True, "retry_target": "google/gemini-3.5-flash-lite",
                      "case": run.cases[0]["id"], "forecast_usd": round(forecast, 9),
                      "existing_pilot_reservation_usd": round(pilot_reserved, 9),
                      "post_pilot_commitment_usd": round(pilot_reserved + forecast, 9),
                      "post_total_commitment_usd": round(current + forecast, 9), "errors": errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
