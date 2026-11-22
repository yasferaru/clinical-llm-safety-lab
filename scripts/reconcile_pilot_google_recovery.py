#!/usr/bin/env python3
"""Offline reconciliation for the completed one-cell Google pilot recovery."""
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
    key = "6b3f45aa810baab5492ccaa11c74de6653c34acfc4e6a8a26a3991f0b907ad89:pilot:forecast"
    artifact = read_artifact(ROOT / "matrix-runs/pilot-google-503-retry-v1/responses/"
                             "d317fe8158a117df2050a741fea6eced7b0f4d124016c4c2b2f0f628b2cc1335.json")
    if not artifact or artifact.get("status") != "completed":
        raise SystemExit("completed recovery artifact unavailable")
    usage = artifact.get("usage") or {}
    price = budget["pricing_usd_per_million_tokens"][artifact["requested_model"]]
    output = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
    estimate = estimate_usd(int(usage.get("promptTokenCount", 0)), output, price["input"], price["output"])
    provenance = {"basis": "local published-price estimate; provider did not report USD",
                  "request_id": artifact.get("request_id"), "provider_model": artifact.get("provider_model"),
                  "usage": usage, "pricing": price}
    ledger.reconcile(key, estimate, provenance)
    print(f"reconciled {key} to ${estimate:.9f}; original pilot reservation unchanged")


if __name__ == "__main__":
    main()
