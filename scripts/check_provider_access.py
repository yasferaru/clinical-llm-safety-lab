#!/usr/bin/env python3
"""Authenticated, non-generative model-catalog readiness check.

Only GET model-list endpoints are used. It never sends a prompt or calls a generation
endpoint, and prints only model availability/status—not keys or raw provider payloads.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.providers import credential_diagnostics, credential_value, split_model  # noqa: E402


async def fetch(client: httpx.AsyncClient, provider: str) -> tuple[int, dict]:
    key = credential_value(provider)
    if not key:
        return 0, {}
    if provider == "anthropic":
        response = await client.get("https://api.anthropic.com/v1/models", headers={
            "x-api-key": key, "anthropic-version": "2023-06-01"})
    elif provider == "google":
        response = await client.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key})
    elif provider == "deepseek":
        response = await client.get("https://api.deepseek.com/models", headers={"Authorization": f"Bearer {key}"})
    elif provider == "groq":
        response = await client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"})
    elif provider == "xai":
        response = await client.get("https://api.x.ai/v1/models", headers={"Authorization": f"Bearer {key}"})
    else:
        raise ValueError(provider)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {}


def ids(provider: str, payload: dict) -> set[str]:
    rows = payload.get("data", []) if provider != "google" else payload.get("models", [])
    return {str(row.get("id") or row.get("name", "")).removeprefix("models/") for row in rows if isinstance(row, dict)}


def google_generate_content_support(payload: dict, wanted: set[str]) -> dict[str, bool]:
    """Return a redacted capability check for configured Google models only."""
    by_id = {str(row.get("name", "")).removeprefix("models/"): row
             for row in payload.get("models", []) if isinstance(row, dict)}
    return {model: "generateContent" in by_id.get(split_model(model)[1], {}).get("supportedGenerationMethods", [])
            for model in sorted(wanted) if split_model(model)[0] == "google"}


async def main() -> int:
    config = yaml.safe_load((ROOT / "config/benchmark_budget_v1.yaml").read_text())
    rows = config["targets"] + config["judges"]
    wanted = {row["id"] for row in rows}
    providers = sorted({split_model(model)[0] for model in wanted})
    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        for provider in providers:
            try:
                status, payload = await fetch(client, provider)
                error = None
            except httpx.HTTPError as exc:
                status, payload, error = 0, {}, type(exc).__name__
            available = ids(provider, payload)
            model_status = {model: (split_model(model)[1] in available if status else None)
                            for model in sorted(wanted) if split_model(model)[0] == provider}
            generation_status = (google_generate_content_support(payload, wanted) if provider == "google" and status
                                 else {model: None for model in model_status} if provider == "google" else None)
            results[provider] = {**credential_diagnostics(provider),
                                 "http_status": status or None,
                                 "transport_error": error,
                                 "models": model_status,
                                 **({"generate_content": generation_status} if provider == "google" else {})}
    print(json.dumps({"non_generative_get_only": True, "results": results}, indent=2, sort_keys=True))
    healthy = all(result["http_status"] == 200 and all(value is True for value in result["models"].values())
                  for result in results.values())
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
