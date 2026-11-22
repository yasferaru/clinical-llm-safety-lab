#!/usr/bin/env python3
"""Offline pilot cost reconciliation retaining two uncertain Google requests."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger  # noqa: E402


def main() -> None:
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    ledger = SpendLedger(ROOT / budget["budget"]["ledger_path"], BudgetPolicy(
        ceiling_usd=budget["budget"]["ceiling_usd"], stop_usd=budget["budget"]["stop_usd"],
        stage_limits=budget["budget"]["stage_limits_usd"]))
    # Source-level accounting: final valid artifacts, three billed invalid DeepSeek
    # calls, and two Google calls with dispatch but no usage remain conservative.
    final_valid, invalid_deepseek = 1.1116915, 0.0037749
    google_recoveries, deepseek_recoveries = 0.0006895 + 0.0011992, 0.0014871
    primary_low = final_valid - google_recoveries - deepseek_recoveries + invalid_deepseek
    uncertain_google = 2 * 0.00262
    primary_key = "7643f1e0b6cb2c32fe51e0967ae327b967f5aa86b573c89f570445cfc1a7b51b:pilot:forecast"
    ledger.reconcile_range(primary_key, primary_low, primary_low + uncertain_google,
                           {"basis": "source-level pilot reconciliation", "uncertain_google_calls": 2,
                            "uncertain_google_max_usd_each": 0.00262})
    deep_keys = ["998fe7b3e484a3a7bf3a79f74c82dd327aa85ce1dbd789314f411c54f2eab1c4:pilot:forecast",
                 "b055de7dc0091c0e78065c7d6baffd9893438aefc851884bf26ef314d1977625:pilot:forecast",
                 "ed60d0f0381bdb94b444a218d39e79973aaef6eb0be81239957dc93fd3ed9d7b:pilot:forecast"]
    for key, value in zip(deep_keys, (0.0004731, 0.0003981, 0.0006159), strict=True):
        ledger.reconcile(key, value, {"basis": "reported DeepSeek token usage; non-thinking recovery"})
    ledger.reconcile("52c4aff051fb87d52fc1f6b5b23903e5e06f08b7f5755d3869773bc77e9e5944:pilot:forecast",
                     0.0011992, {"basis": "reported Google target token usage; provider did not report USD"})
    print({"pilot_low_usd": round(primary_low + google_recoveries + deepseek_recoveries, 9),
           "pilot_high_usd": round(primary_low + uncertain_google + google_recoveries + deepseek_recoveries, 9)})


if __name__ == "__main__":
    main()
