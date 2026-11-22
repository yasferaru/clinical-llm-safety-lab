#!/usr/bin/env python3
"""Offline reconciliation of the observed v6 Google usage; preserves v5 reservation."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger, estimate_usd  # noqa: E402
from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    policy = BudgetPolicy(ceiling_usd=budget["budget"]["ceiling_usd"], stop_usd=budget["budget"]["stop_usd"],
                          stage_limits=budget["budget"]["stage_limits_usd"])
    ledger = SpendLedger(ROOT / budget["budget"]["ledger_path"], policy)
    key = "d12fb1c6759afcd022f3588fc4a6407040f5316216bb8cae283e7781b9c023c0:smoke:forecast"
    artifact = read_artifact(ROOT / "matrix-runs/smoke-single-retry-v6/judgments/"
                             "a583d7e2e936eb14ac3e45dc5e41ee26e816e23aaa413b77445fd5bbaa9673ce.json")
    if not artifact or artifact.get("status") != "completed":
        raise SystemExit("v6 terminal artifact is unavailable")
    usage = artifact.get("usage") or {}
    price = budget["pricing_usd_per_million_tokens"][artifact["requested_judge"]]
    output = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
    estimate = estimate_usd(int(usage.get("promptTokenCount", 0)), output, price["input"], price["output"])
    provenance = {"basis": "local published-price estimate; provider did not report USD",
                  "request_id": artifact.get("request_id"), "provider_model": artifact.get("provider_model"),
                  "usage": usage, "pricing": price}
    ledger.reconcile(key, estimate, provenance)
    print(f"reconciled {key} to ${estimate:.9f}; v5 reservation unchanged")


if __name__ == "__main__":
    main()
