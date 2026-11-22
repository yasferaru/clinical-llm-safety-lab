#!/usr/bin/env python3
"""Create a one-cell, zero-retry recovery config from a preserved failed target."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cell-id", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    artifact = read_artifact(source / "responses" / f"{args.cell_id}.json")
    if not artifact or artifact.get("status") not in {"failed", "invalid"}:
        raise SystemExit("source cell must be a preserved failed or invalid target artifact")
    case_id, model = artifact["case"]["id"], artifact["requested_model"]
    tag = args.cell_id[:12]
    output = f"matrix-runs/pilot-recovery-{tag}-v1"
    config_path = ROOT / f"config/matrix.pilot_recovery_{tag}_v1.yaml"
    config = {"dataset": "evals/cases.jsonl", "case_ids": [case_id], "targets": [model],
              "judges": ["xai/grok-4.7", "anthropic/claude-haiku-4-5-20251001"],
              "judge_assignments": {model: ["xai/grok-4.7", "anthropic/claude-haiku-4-5-20251001"]},
              "epochs": 1, "decoding": artifact["decoding"], "retry_limit": 0,
              "target_attempt_namespace": f"recovery-{tag}-v1", "output": output}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {"protocol": "transient-target-recovery-v1", "retry_limit": 0,
            "source_root": str(source.relative_to(ROOT)), "source_terminal_cell_id": args.cell_id,
            "case_id": case_id, "target": model, "maximum_target_reservation_usd": 0.00262,
            "warning": "Explicit authorization is required; this plan never retries automatically."}
    plan_path = source / "meta" / f"recovery_plan_{tag}.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"config": str(config_path), "plan": str(plan_path), "output": output}, sort_keys=True))


if __name__ == "__main__":
    main()
