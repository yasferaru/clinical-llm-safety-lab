#!/usr/bin/env python3
"""One explicitly authorized, non-clinical Flash-Lite generation-access probe."""
from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger, estimate_usd  # noqa: E402
from evals.providers import ProviderAdapter, ProviderError  # noqa: E402

MODEL = "gemini-3.5-flash-lite"
KEY = "flash-lite-access-v2-gemini-35"
RESERVATION_USD = 0.01


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-one-billable", action="store_true")
    args = parser.parse_args()
    if not args.execute_one_billable:
        raise SystemExit("refusing probe: pass --execute-one-billable after explicit owner authorization")
    config = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    budget = config["budget"]
    ledger = SpendLedger(ROOT / budget["ledger_path"], BudgetPolicy(
        ceiling_usd=budget["ceiling_usd"], stop_usd=budget["stop_usd"], stage_limits=budget["stage_limits_usd"]))
    ledger.reserve(KEY, "smoke", RESERVATION_USD, "google")
    try:
        result = await ProviderAdapter("google", retries=0).request(
            MODEL, [{"role": "user", "content": "Reply exactly: OK"}], max_output_tokens=1)
    except ProviderError as exc:
        if exc.status is None:
            ledger.mark_unconfirmed_transport(
                KEY, RESERVATION_USD,
                "Probe failed before an HTTP response; provider receipt/billing is unconfirmed.")
        print(json.dumps({"success": False, "http_status": exc.status, "error": str(exc),
                          "provider_model": None, "usage": None, "reservation_usd": RESERVATION_USD}, sort_keys=True))
        return 1
    usage = result.get("usage") or {}
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
    price = config["pricing_usd_per_million_tokens"]["google/gemini-3.5-flash-lite"]
    estimated = estimate_usd(input_tokens, output_tokens, price["input"], price["output"])
    response_text = result.get("text", "")
    ledger.reconcile(KEY, estimated, {"usage": usage, "request_id": result.get("request_id"),
                                      "provider_model": result.get("provider_model"),
                                      "response_text_exact_ok": response_text == "OK"})
    print(json.dumps({"success": True, "http_status": 200, "provider_model": result.get("provider_model"),
                      "usage": usage, "estimated_usd": estimated, "reservation_usd": RESERVATION_USD,
                      "request_id_present": bool(result.get("request_id")),
                      "response_text_exact_ok": response_text == "OK"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
