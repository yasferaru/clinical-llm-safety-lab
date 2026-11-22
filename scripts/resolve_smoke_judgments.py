#!/usr/bin/env python3
"""Offline-only provenance selector; it never averages or merges duplicate judgments."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, read_artifact  # noqa: E402


def valid(artifact: dict | None) -> bool:
    return bool(artifact and artifact.get("status") == "completed" and isinstance(artifact.get("verdict"), bool))


def by_logical_cell(root: Path) -> dict[tuple[str, str, int], tuple[dict, Path]]:
    rows = {}
    for path in (root / "judgments").glob("*.json"):
        artifact = read_artifact(path)
        if artifact:
            rows[(artifact["response_cell_id"], artifact["requested_judge"], artifact["line_index"])] = (artifact, path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-root", default=None)
    parser.add_argument("--output", default="matrix-runs/smoke-three-target-rejudge-v4/meta/resolved_selection.json")
    args = parser.parse_args()
    v3 = MatrixRun(MatrixConfig.from_file("config/matrix.smoke_four_provider.yaml"))
    v4 = MatrixRun(MatrixConfig.from_file("config/matrix.smoke_rejudge_v4.yaml"))
    old, new = by_logical_cell(v3.root), by_logical_cell(v4.root)
    retry_root = Path(args.retry_root).resolve() if args.retry_root else None
    retry = by_logical_cell(retry_root) if retry_root else {}
    selected, unresolved = [], []
    for key, source in sorted(old.items()):
        source_artifact, _ = source
        chosen, provenance = (source, "v3") if valid(source_artifact) else (new.get(key), "v4")
        if not chosen or not valid(chosen[0]):
            if key in retry:
                chosen, provenance = retry[key], retry_root.name
        if not chosen or not valid(chosen[0]):
            unresolved.append({"response_cell_id": key[0], "judge": key[1], "line_index": key[2],
                               "v3_status": source_artifact.get("status"),
                               "v4_status": (new.get(key) or ({}, None))[0].get("status"),
                               "retry_status": (retry.get(key) or ({}, None))[0].get("status")})
            continue
        artifact, path = chosen
        selected.append({"response_cell_id": key[0], "judge": key[1], "line_index": key[2],
                         "selected_artifact_cell_id": artifact["cell_id"],
                         "source_artifact_path": str(path.relative_to(ROOT)),
                         "source": provenance, "provider": artifact["requested_judge"].split("/", 1)[0],
                         "requested_judge": artifact["requested_judge"],
                         "provider_model": artifact.get("provider_model"),
                         "judge_prompt_hash": artifact.get("judge_prompt_hash"),
                         "judge_output_schema_version": artifact.get("judge_output_schema_version"),
                         "rubric_version": artifact.get("rubric_version"), "verdict": artifact["verdict"]})
    if len({(x["response_cell_id"], x["judge"], x["line_index"]) for x in selected}) != len(selected):
        raise RuntimeError("duplicate logical cells selected")
    paired = None
    if not unresolved:
        groups = defaultdict(list)
        for row in selected:
            groups[(row["response_cell_id"], row["line_index"])].append(row)
        paired_rows = [values for values in groups.values() if len(values) == 2]
        paired = {"paired_valid_cells": len(paired_rows),
                  "disagreements": sum(values[0]["verdict"] != values[1]["verdict"] for values in paired_rows),
                  "note": "Computed only after every logical judgment cell is resolved."}
    report = {"protocol": "resolved-smoke-selection-v1", "selected_count": len(selected),
              "unresolved_count": len(unresolved), "paired_coverage": paired,
              "selected": selected, "unresolved": unresolved,
              "note": "One source artifact is selected per logical cell; duplicate judgments are never pooled."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": len(selected), "unresolved": len(unresolved), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
