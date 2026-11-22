#!/usr/bin/env python3
"""Copy only immutable, compatible completed artifacts into a new run directory.

This is an explicit provenance-preserving import: source files remain untouched and
only cells whose content-addressed IDs match the destination configuration are copied.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import MatrixConfig, MatrixRun, read_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination_config")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = MatrixRun(MatrixConfig.from_file(args.destination_config))
    destination.initialize()
    imported = {"responses": 0, "judgments": 0}
    for path in (source / "responses").glob("*.json"):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed":
            continue
        cell_id = destination.target_id(artifact["case"], artifact["requested_model"], artifact["epoch"])
        expected = destination.target_path(cell_id)
        if expected.name == path.name and not expected.exists():
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, expected)
            imported["responses"] += 1
    for path in (source / "judgments").glob("*.json"):
        artifact = read_artifact(path)
        if not artifact or artifact.get("status") != "completed" or artifact.get("verdict") not in (True, False):
            continue
        response_path = destination.target_path(artifact["response_cell_id"])
        response = read_artifact(response_path)
        if not response:
            continue
        cell_id = destination.judge_id(response, artifact["requested_judge"], artifact["line_index"])
        expected = destination.judge_path(cell_id)
        if expected.name == path.name and not expected.exists():
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, expected)
            imported["judgments"] += 1
    print(imported)


if __name__ == "__main__":
    main()
