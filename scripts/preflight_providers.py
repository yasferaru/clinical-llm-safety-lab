#!/usr/bin/env python3
"""Offline-only provider configuration preflight.

This deliberately does not list models or send a request. A present key does not
establish account access, pricing tier, rate limit, or model availability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.independence import assert_judge_independent  # noqa: E402
from evals.providers import SPECS, credential_status, split_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="offline-only benchmark provider preflight")
    parser.add_argument("config", nargs="?", default="config/benchmark_budget_v1.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    targets, judges = config["targets"], config["judges"]
    model_rows = targets + judges
    configured = []
    errors = []
    for row in model_rows:
        provider, _ = split_model(row["id"])
        if provider not in SPECS:
            errors.append(f"unknown provider '{provider}' for {row['id']}")
        elif row.get("provider") != provider:
            errors.append(f"provider prefix/config mismatch for {row['id']}")
        configured.append({"model": row["id"], "provider": provider,
                           "credential_env": SPECS[provider].key_env if provider in SPECS else None})
    for target in targets:
        assigned = config["judge_assignments"].get(target["id"], [])
        if len(assigned) != 2:
            errors.append(f"target does not have exactly two assigned judges: {target['id']}")
        unknown = set(assigned) - {judge["id"] for judge in judges}
        if unknown:
            errors.append(f"target has unknown assigned judge(s): {sorted(unknown)}")
        for judge in assigned:
            try:
                assert_judge_independent(judge, [target["id"]])
            except ValueError as exc:
                errors.append(str(exc))
    active_providers = {row["provider"] for row in model_rows}
    print(json.dumps({"offline_only": True, "credentials_present": credential_status(active_providers),
                      "models": configured, "independence_errors": errors,
                      "live_access_unverified": True}, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
