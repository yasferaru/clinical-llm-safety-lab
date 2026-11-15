#!/usr/bin/env bash
# Run the whole evaluation for ONE model and record exactly what was run.
# Usage (project folder, venv active):  bash scripts/run_one.sh <model> [judge]
#   bash scripts/run_one.sh ollama/qwen3:8b
#   bash scripts/run_one.sh mistral/mistral-small-latest
# judge defaults to groq/openai/gpt-oss-120b. Judged runs on Groq's free tier can use about 125k of its
# 200k daily tokens per model, so run ONE model per day on the free tier.
set -euo pipefail
MODEL="${1:?give a model, e.g. ollama/qwen3:8b}"
JUDGE="${2:-groq/openai/gpt-oss-120b}"
mkdir -p logs

python check_cases.py --build

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  model=$MODEL  judge=$JUDGE"
  echo "inspect-ai: $(pip show inspect-ai 2>/dev/null | awk '/^Version/{print $2}')"
  echo "cases.jsonl sha256: $(shasum -a 256 evals/cases.jsonl | cut -d' ' -f1)  records: $(wc -l < evals/cases.jsonl | tr -d ' ')"
  if [[ "$MODEL" == ollama/* ]]; then
    echo "ollama model entry (NAME ID SIZE MODIFIED):"; ollama list | grep -F "${MODEL#ollama/}" || true
  fi
} >> logs/run_manifest.txt

# 1. Clinical cases, graded by the judge (one reply per case; epochs would multiply judge usage)
inspect eval evals/run_eval.py --model "$MODEL" -T judge="$JUDGE"

# 2. Red-team: local open-weight models only (README scope). Deterministic checks, no judge,
#    so repeating each note 5 times is cheap.
if [[ "$MODEL" == ollama/* ]]; then
  inspect eval redteam/run_redteam.py --model "$MODEL" -T prompt=baseline --epochs 5
  inspect eval redteam/run_redteam.py --model "$MODEL" -T prompt=hardened_v1 --epochs 5
fi

python redteam/summarize.py
