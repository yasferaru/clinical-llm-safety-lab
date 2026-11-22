#!/usr/bin/env python3
"""Guarded billable matrix runner. It refuses to call providers without an explicit flag."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.budget import BudgetPolicy, SpendLedger, estimate_usd  # noqa: E402
from evals.matrix import MatrixConfig, MatrixRun, read_artifact  # noqa: E402
from evals.providers import ProviderAdapter, ProviderExecutionPolicy, judge_router, split_model, target_router  # noqa: E402


def estimate_run(run: MatrixRun, prices: dict, retry_multiplier: float = 3.0,
                 allowed_cells: set[tuple[str, str, int]] | None = None) -> float:
    """Conservative estimate of only unfinished target/judge cells."""
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    assumptions = budget["token_assumptions"]
    calibration = budget.get("cost_calibration", {})
    observed_judge = calibration.get("observed_judge_usd_per_call", {})
    margin = float(calibration.get("safety_margin", 1.0))
    total = 0.0
    for target in run.config.targets:
        for epoch in range(1, run.config.epochs + 1):
            for case in run.cases:
                response = run.target_path(run.target_id(case, target, epoch))
                artifact = None if not response.exists() else read_artifact(response)
                response_complete = bool(artifact and artifact.get("status") == "completed")
                if not response_complete:
                    price = prices[target]
                    total += estimate_usd(assumptions["target_input_per_call"], assumptions["target_output_per_call"],
                                          price["input"], price["output"])
                for judge in run.assigned_judges(target):
                    for index, _ in enumerate(case["rubric"]):
                        selected = response_complete and (artifact["cell_id"], judge, index) in (allowed_cells or set())
                        if allowed_cells is not None and not selected:
                            continue
                        complete = False
                        if response_complete:
                            judgment = run.judge_path(run.judge_id(artifact, judge, index))
                            prior = read_artifact(judgment)
                            complete = bool(prior and prior.get("status") == "completed"
                                            and prior.get("verdict") in (True, False))
                        if not complete:
                            if judge in observed_judge:
                                total += float(observed_judge[judge]) * margin
                            else:
                                price = prices[judge]
                                total += estimate_usd(assumptions["judge_input_per_call"],
                                                      assumptions["judge_output_per_call"],
                                                      price["input"], price["output"])
    return total * retry_multiplier  # initial request plus every configured retry


def artifact_snapshot(run: MatrixRun) -> dict[Path, str]:
    """Fingerprint artifacts that predate this invocation.

    An isolated retry imports completed artifacts solely for response reuse.  Their
    historical spend belongs to an earlier ledger entry, even if those artifacts
    carry output-cap metadata.  Content hashes also make an overwritten in-flight
    checkpoint count as new work, while an untouched import stays excluded.
    """
    snapshot: dict[Path, str] = {}
    for folder in ("responses", "judgments"):
        for path in (run.root / folder).glob("*.json"):
            snapshot[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def observed_request_cost(run: MatrixRun, prices: dict,
                          baseline: dict[Path, str] | None = None) -> float:
    """Price persisted live artifacts; xAI's provider-reported ticks take precedence.

    Only artifacts with a request cap were created by the guarded runner. Imported
    historical artifacts are deliberately excluded to avoid double-counting.
    """
    total = 0.0
    for folder, key in (("responses", "requested_model"), ("judgments", "requested_judge")):
        for path in (run.root / folder).glob("*.json"):
            if baseline is not None and baseline.get(path) == hashlib.sha256(path.read_bytes()).hexdigest():
                continue
            artifact = read_artifact(path)
            if not artifact or artifact.get("request_max_output_tokens") is None:
                continue
            model, usage = artifact.get(key), artifact.get("usage") or {}
            if not model or model not in prices:
                continue
            if model.startswith("xai/") and usage.get("cost_in_usd_ticks") is not None:
                total += float(usage["cost_in_usd_ticks"]) / 10_000_000_000
            elif model.startswith("anthropic/"):
                total += estimate_usd(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                                      prices[model]["input"], prices[model]["output"])
            elif model.startswith("google/"):
                output = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
                total += estimate_usd(int(usage.get("promptTokenCount", 0)), output,
                                      prices[model]["input"], prices[model]["output"])
    return total


def assert_reservation(run: MatrixRun, ledger: SpendLedger, key: str, prices: dict,
                       baseline: dict[Path, str] | None = None) -> None:
    entry = next((item for item in ledger.entries() if item.get("key") == key), None)
    if not entry:
        raise RuntimeError("missing live reservation")
    reserved = float(entry.get("reserved_usd", entry.get("usd", 0)))
    observed = observed_request_cost(run, prices, baseline)
    if observed > reserved + 1e-9:
        raise RuntimeError(f"budget-reservation discrepancy: observed ${observed:.6f} exceeds reserved ${reserved:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="budget-guarded live evaluation runner")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "production"), required=True)
    parser.add_argument("--first-pending-target",
                        help="execute at most one pending target cell for this exact model, then stop")
    parser.add_argument("--judge-selection-plan", help="offline rejudge-plan JSON; limits judge cells exactly")
    parser.add_argument("--skip-targets", action="store_true",
                        help="do not invoke target generation; only valid with a finite judge selection plan")
    parser.add_argument("--existing-reservation-key", help="charge this invocation to a prior retained reservation")
    parser.add_argument("--execute-billable", action="store_true", help="required before any provider request")
    args = parser.parse_args()
    if not args.execute_billable:
        raise SystemExit("refusing provider calls: pass --execute-billable only after owner approval")
    budget = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    run = MatrixRun(MatrixConfig.from_file(args.matrix))
    # Snapshot after imports but before this invocation: accounting must charge
    # only requests made by this process, never response-reuse artifacts.
    baseline = artifact_snapshot(run)
    selection = None
    plan_cost = 0.0
    if args.judge_selection_plan:
        plan = json.loads(Path(args.judge_selection_plan).read_text())
        selection = {(row["response_cell_id"], row["judge"], int(row["line_index"])) for row in plan["cells"]}
        plan_cost = float(plan.get("max_rejudge_usd", 0.0))
    if args.skip_targets and selection is None:
        raise ValueError("--skip-targets requires --judge-selection-plan")
    estimated = estimate_run(run, budget["pricing_usd_per_million_tokens"], run.config.retry_limit + 1, selection)
    if args.first_pending_target:
        # This guarded mode returns immediately after one target response, before
        # judging.  Its reservation must not include downstream judge cells.
        assumptions = budget["token_assumptions"]
        price = budget["pricing_usd_per_million_tokens"][args.first_pending_target]
        estimated = estimate_usd(assumptions["target_input_per_call"], assumptions["target_output_per_call"],
                                 price["input"], price["output"]) * (run.config.retry_limit + 1)
    forecast = max(estimated, plan_cost)
    policy = BudgetPolicy(ceiling_usd=budget["budget"]["ceiling_usd"], stop_usd=budget["budget"]["stop_usd"],
                          stage_limits=budget["budget"]["stage_limits_usd"])
    ledger_path = ROOT / budget["budget"].get("ledger_path", str(run.root / "meta" / "spend_ledger.json"))
    ledger = SpendLedger(ledger_path, policy)
    reservation_key = f"{run.config_hash}:{args.stage}:forecast"
    if args.existing_reservation_key:
        reservation_key = args.existing_reservation_key
        entry = next((item for item in ledger.entries() if item.get("key") == reservation_key), None)
        if not entry or entry.get("stage") != args.stage or entry.get("state") == "released":
            raise ValueError("existing reservation is absent, released, or belongs to another stage")
    else:
        ledger.reserve(reservation_key, args.stage, forecast, "multi-provider")
    providers = {split_model(model)[0] for model in (*run.config.targets, *run.config.judges)}
    limits = budget.get("request_limits", {})
    target_limit = int(limits.get("target_max_output_tokens", 512))
    judge_limit = int(limits.get("judge_max_output_tokens", 128))
    configured_limits = limits.get("judge_max_output_tokens_by_provider", {})
    judge_limits = {provider: int(value) for provider, value in configured_limits.items()}
    if target_limit < 1 or judge_limit < 1 or any(value < 1 for value in judge_limits.values()):
        raise ValueError("request output-token limits must be positive")
    policies = budget.get("provider_execution_policy", {})
    if any(int((policies.get(provider) or {}).get("max_retries", 0)) > run.config.retry_limit
           for provider in providers):
        raise ValueError(
            "provider automatic retries exceed matrix retry_limit; update the approved matrix and budget forecast")
    adapters = {
        provider: ProviderAdapter(provider, policy=ProviderExecutionPolicy.from_mapping(
            policies.get(provider), run.config.retry_limit))
        for provider in providers
    }
    if args.first_pending_target:
        if args.first_pending_target not in run.config.targets:
            raise ValueError("first-pending target is not in the matrix")
        before = {path.name for path in (run.root / "responses").glob("*.json")}
        invoked = asyncio.run(run.run_targets(target_router(adapters, max_output_tokens=target_limit),
                                                targets={args.first_pending_target}, max_cells=1,
                                                after_artifact=lambda _: assert_reservation(
                                                    run, ledger, reservation_key,
                                                    budget["pricing_usd_per_million_tokens"], baseline),
                                                fail_fast_failures=True))
        assert_reservation(run, ledger, reservation_key, budget["pricing_usd_per_million_tokens"], baseline)
        created = [read_artifact(path) for path in (run.root / "responses").glob("*.json") if path.name not in before]
        artifact = next((item for item in created if item), None)
        raw = args.first_pending_target.split("/", 1)[1]
        response = artifact.get("response", "") if artifact else ""
        usage = artifact.get("usage") or {} if artifact else {}
        output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
        acceptable = bool(artifact and artifact.get("status") == "completed" and response.strip()
                          and artifact.get("provider_model") == raw and artifact.get("request_id")
                          and artifact.get("finish_reason") == "STOP" and output_tokens < target_limit)
        summary = {"first_pending_target": args.first_pending_target, "target_calls_attempted": invoked,
                   "validation": run.validate(), "first_response": None if not artifact else {
                       "status": artifact.get("status"), "provider_model": artifact.get("provider_model"),
                       "request_id_present": bool(artifact.get("request_id")), "response_chars": len(response),
                       "finish_reason": artifact.get("finish_reason"), "output_tokens": output_tokens,
                       "requested_max_output_tokens": artifact.get("request_max_output_tokens"),
                       "acceptable": acceptable}}
        print(json.dumps(summary, sort_keys=True))
        return 0 if acceptable else 2
    if not args.skip_targets:
        asyncio.run(run.run_targets(target_router(adapters, max_output_tokens=target_limit),
                                    after_artifact=lambda _: assert_reservation(
                                        run, ledger, reservation_key,
                                        budget["pricing_usd_per_million_tokens"], baseline),
                                    fail_fast_failures=True))
        assert_reservation(run, ledger, reservation_key, budget["pricing_usd_per_million_tokens"], baseline)
    asyncio.run(run.run_judges(judge_router(adapters, max_output_tokens=judge_limits or judge_limit),
                                fail_fast_invalid=True,
                                after_artifact=lambda _: assert_reservation(
                                    run, ledger, reservation_key, budget["pricing_usd_per_million_tokens"], baseline),
                                allowed_cells=selection, fail_fast_failures=True))
    if args.skip_targets and selection is not None:
        selected_statuses: dict[str, int] = {}
        for response_id, judge, line_index in selection:
            response = read_artifact(run.target_path(response_id))
            artifact = None
            if response:
                judgment_path = run.judge_path(run.judge_id(response, judge, line_index))
                artifact = read_artifact(judgment_path)
            status = artifact.get("status", "missing") if artifact else "missing"
            selected_statuses[status] = selected_statuses.get(status, 0) + 1
        print({"selected_judgment_cells": len(selection), "selected_statuses": selected_statuses,
               "general_matrix_validation": run.validate()})
    else:
        print(run.validate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
