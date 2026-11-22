"""scripts/write_manifest.py and scripts/make_report.sh (offline)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from evals.run_eval import JUDGE_PROMPT_SHA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import write_manifest  # noqa: E402
import check_provider_access  # noqa: E402


def test_manifest_records_what_was_run(tmp_path):
    log = tmp_path / "run_manifest.txt"
    m = write_manifest.main(["--run-dir", str(tmp_path / "run"), "--model", "mistral/mistral-small-latest",
                             "--judge", "groq/openai/gpt-oss-120b", "--temperature", "0.7", "--seed", "3",
                             "--epochs", "5", "--status", "0", "--manifest-log", str(log)])
    for key in ("git_commit", "git_dirty", "inspect_ai", "model_versions_reported", "judge_prompt_sha256",
                "temperature", "seed", "epochs", "cases_jsonl_sha256", "redteam_prompt_sha256"):
        assert key in m, key
    assert m["judge_prompt_sha256"] == JUDGE_PROMPT_SHA and len(m["git_commit"]) == 40
    assert (m["temperature"], m["seed"], m["epochs"], m["exit_status"]) == ("0.7", "3", "5", 0)
    assert m["ollama_id"] is None and set(m["redteam_prompt_sha256"]) == {"baseline", "hardened_v1"}
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["judge"] == "groq/openai/gpt-oss-120b"
    text = log.read_text()
    assert text.startswith("=== ") and f'judge_prompt_sha256: "{JUDGE_PROMPT_SHA}"' in text


def test_make_report_synthetic(tmp_path):
    env = {**os.environ, "PYTHON": sys.executable}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "make_report.sh"), "--synthetic", str(tmp_path / "rep")],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "rep" / "report.md").exists() and len(list((tmp_path / "rep").glob("*.csv"))) >= 15


def test_make_report_needs_arguments():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "make_report.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr


def test_google_catalog_capability_check_is_redacted_and_model_specific():
    payload = {"models": [
        {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/unrelated", "supportedGenerationMethods": []},
    ]}
    assert check_provider_access.google_generate_content_support(
        payload, {"google/gemini-3.5-flash-lite"}) == {"google/gemini-3.5-flash-lite": True}
