#!/usr/bin/env bash
# Build report.md and one CSV per table for each run directory (written to RUN_DIR/report/).
# Usage:  bash scripts/make_report.sh RUN_DIR [RUN_DIR ...]
#         bash scripts/make_report.sh --synthetic OUT_DIR      # every section, on seeded synthetic fixtures
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [RUN_DIR ...] | --synthetic OUT_DIR" >&2
  exit 2
fi
if [[ "$1" == "--synthetic" ]]; then
  out="${2:?give an output folder}"
  mkdir -p "$out"
  out="$(cd "$out" && pwd)"
  cd "$ROOT" && "$PY" evals/report.py --synthetic "$out"
  exit 0
fi
for d in "$@"; do
  [[ -d "$d" ]] || { echo "not a directory: $d" >&2; exit 2; }
  abs="$(cd "$d" && pwd)"
  (cd "$ROOT" && "$PY" evals/report.py "$abs" --out "$abs/report")
done
