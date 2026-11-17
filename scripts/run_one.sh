#!/usr/bin/env bash
# Run the evaluation for ONE model into its own run directory, record what was run, and build the report.
# Usage (project folder, venv active):  bash scripts/run_one.sh <model> [judge]
#   bash scripts/run_one.sh ollama/qwen3:8b
#   TEMPERATURE=0.7 SEED=1 EPOCHS=5 bash scripts/run_one.sh mistral/mistral-small-latest
# judge defaults to groq/openai/gpt-oss-120b and must be from another model family (checked before any call).
# Red-team (local ollama models only): dev split, both prompts, judged. Set REDTEAM_TEST=1 to also run the
# held-out test split (refused by redteam/prompt_guard.py if a prompt was edited after its last dev run).
# Cost: 5 epochs x 4 cases x ~5 rubric lines = ~100 judge calls per model for the clinical eval; on Groq's free
# tier run ONE model per day.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?give a model, e.g. ollama/qwen3:8b}"
JUDGE="${2:-groq/openai/gpt-oss-120b}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-5}"
RUN_DIR="logs/runs/$(date -u +%Y%m%dT%H%M%SZ)_$(printf '%s' "$MODEL" | tr '/:' '__')"
mkdir -p "$RUN_DIR"

record() {  # always write the manifest, with the exit status, even if a step fails
  local status=$?
  python scripts/write_manifest.py --run-dir "$RUN_DIR" --model "$MODEL" --judge "$JUDGE" \
    --temperature "$TEMPERATURE" --seed "$SEED" --epochs "$EPOCHS" --status "$status" || true
  echo "run directory: $RUN_DIR (exit $status)"
}
trap record EXIT

python check_cases.py   # validate only; cases.jsonl is compiled and committed separately (CI checks it is current)
python -c "import sys; from evals.judge_validation.independence import assert_judge_independent as a; \
a(sys.argv[1], [sys.argv[2]])" "$JUDGE" "$MODEL"

COMMON=(--model "$MODEL" --log-dir "$RUN_DIR" --temperature "$TEMPERATURE" --seed "$SEED" --epochs "$EPOCHS")

# 1. Clinical cases, graded by the judge (independence is checked again inside the task)
inspect eval evals/run_eval.py "${COMMON[@]}" -T judge="$JUDGE"

# 2. Red-team: local open-weight models only (README scope)
if [[ "$MODEL" == ollama/* ]]; then
  for prompt in baseline hardened_v1; do
    inspect eval redteam/run_redteam.py "${COMMON[@]}" -T prompt="$prompt" -T split=dev -T judge="$JUDGE"
    if [[ "${REDTEAM_TEST:-0}" == "1" ]]; then
      inspect eval redteam/run_redteam.py "${COMMON[@]}" -T prompt="$prompt" -T split=test -T judge="$JUDGE"
    fi
  done
fi

# 3. Report for this run directory
bash scripts/make_report.sh "$RUN_DIR"
