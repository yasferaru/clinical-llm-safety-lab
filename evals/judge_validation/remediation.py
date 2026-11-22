"""Offline classification for malformed persisted judge artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def output_tokens(artifact: dict[str, Any]) -> int:
    usage = artifact.get("usage") or {}
    judge = artifact.get("requested_judge", "")
    if judge.startswith("anthropic/"):
        return int(usage.get("output_tokens", 0))
    if judge.startswith("google/"):
        return int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
    return int(usage.get("completion_tokens", 0))


def classify_invalid(artifact: dict[str, Any]) -> dict[str, Any]:
    """Classify without repairing or changing an invalid artifact."""
    raw = str(artifact.get("raw_output", ""))
    cap = artifact.get("request_max_output_tokens")
    tokens = output_tokens(artifact)
    near_cap = bool(cap and tokens >= int(cap) * 0.9)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
        json_valid = False
    else:
        json_valid = isinstance(value, dict)
    verdict = value.get("verdict") if json_valid else None
    if json_valid and verdict not in {"YES", "NO"}:
        cause = "schema_violation_missing_or_invalid_verdict"
    elif not json_valid and raw.lstrip().startswith("{") and near_cap:
        cause = "truncation_suspected_malformed_json"
    elif not json_valid and near_cap:
        cause = "truncation_suspected_missing_verdict"
    elif not json_valid:
        cause = "parsing_error_missing_verdict"
    else:
        cause = "missing_verdict"
    prefix = re.match(r'^\s*\{\s*"verdict"\s*:\s*"(YES|NO)"', raw)
    return {"cause": cause, "output_tokens": tokens, "requested_cap": cap,
            "near_cap": near_cap, "json_valid": json_valid,
            "recoverable_prefix_verdict": prefix.group(1) if prefix else None,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(), "raw_length": len(raw)}
