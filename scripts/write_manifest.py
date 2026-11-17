#!/usr/bin/env python3
"""Record exactly what a run used: append to logs/run_manifest.txt and write RUN_DIR/manifest.json.

    python scripts/write_manifest.py --run-dir DIR --model M --judge J --temperature T --seed S --epochs E [--status N]

Records: UTC time, exit status, git commit and whether the tree was dirty, Python and Inspect versions,
model under test and the version(s) the provider reported in the logs, the Ollama model ID for local models,
judge model and the sha256 of the judge prompt, temperature, seed, epochs, sha256 of cases.jsonl, of the
red-team cases and of each red-team system prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(p: Path) -> str | None:
    """sha256 hex of a file, None if missing."""
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def git(*args: str) -> str:
    """Output of a git command in the repo, '' if git fails."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def reported_versions(run_dir: Path) -> list[str]:
    """Distinct model names the provider reported in the samples of the run's logs (may be aliases)."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return []
    seen: set[str] = set()
    for p in sorted(run_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(p))
        except Exception:  # noqa: BLE001 - a broken log should not stop the manifest
            continue
        for s in log.samples or []:
            m = getattr(getattr(s, "output", None), "model", None)
            if m:
                seen.add(str(m))
    return sorted(seen)


def ollama_id(model: str) -> str | None:
    """'NAME ID' of an ollama/<tag> model from `ollama list` (exact tag match), None otherwise."""
    if not model.startswith("ollama/"):
        return None
    tag = model.split("/", 1)[1]
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "ollama list failed"
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] == tag:
            return f"{parts[0]} {parts[1]}"
    return f"{tag} not found in ollama list"


def build_manifest(run_dir: Path, model: str, judge: str, temperature: str, seed: str, epochs: str,
                   status: int | None = None) -> dict:
    """The manifest dict (pure apart from reading files, git and ollama)."""
    from evals.run_eval import JUDGE_PROMPT_SHA
    try:
        inspect_version = version("inspect-ai")
    except PackageNotFoundError:
        inspect_version = None
    prompts = {p.stem: sha256_file(p) for p in sorted((ROOT / "redteam" / "prompts").glob("*.md"))}
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "exit_status": status,
        "run_dir": str(run_dir), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "inspect_ai": inspect_version,
        "model": model, "model_versions_reported": reported_versions(run_dir), "ollama_id": ollama_id(model),
        "judge": judge, "judge_prompt_sha256": JUDGE_PROMPT_SHA, "judge_temperature": 0,
        "temperature": temperature, "seed": seed, "epochs": epochs,
        "cases_jsonl_sha256": sha256_file(ROOT / "evals" / "cases.jsonl"),
        "redteam_cases_sha256": sha256_file(ROOT / "redteam" / "cases.yaml"), "redteam_prompt_sha256": prompts,
    }


def main(argv: list[str] | None = None) -> dict:
    """CLI: write RUN_DIR/manifest.json and append a block to logs/run_manifest.txt."""
    ap = argparse.ArgumentParser()
    for a in ("--run-dir", "--model", "--judge", "--temperature", "--seed", "--epochs"):
        ap.add_argument(a, required=True)
    ap.add_argument("--status", type=int)
    ap.add_argument("--manifest-log", default=str(ROOT / "logs" / "run_manifest.txt"))
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    m = build_manifest(run_dir, a.model, a.judge, a.temperature, a.seed, a.epochs, a.status)
    (run_dir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    log = Path(a.manifest_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"=== {m['time_utc']}  model={m['model']}  judge={m['judge']}\n")
        for k, v in m.items():
            if k not in ("time_utc",):
                f.write(f"{k}: {json.dumps(v)}\n")
    return m


if __name__ == "__main__":
    main()
