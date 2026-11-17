"""Guard against tuning system prompts on the held-out test notes.

Every red-team evaluation appends {time, prompt, sha256, split, forced} to redteam/prompt_log.jsonl
(commit this file with the results). Before a run on split 'test' (or 'all', which includes test):

  1. the prompt must have a dev run, and its MOST RECENT dev run must have used the exact current text
     (sha256). Otherwise the prompt was edited after its last dev run -> PromptGuardError.
  2. no earlier test/all run of the same prompt name may have used a different text. Otherwise the test
     notes have already been seen by an earlier version, and a new version would be tuned on them.

`force=True` bypasses both checks and is recorded as "forced": true. Entries with "event": "note"
carry caveats (e.g. a prompt written before the split existed); they do not count as runs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).parent / "prompt_log.jsonl"
TEST_SPLITS = ("test", "all")


class PromptGuardError(RuntimeError):
    """Raised when a prompt version would be evaluated on test without a matching, frozen dev run."""


def prompt_sha(text: str) -> str:
    """sha256 hex digest of the prompt text (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_log(path: Path | str = LOG_PATH) -> list[dict]:
    """All entries of the JSONL log, oldest first ([] if the file does not exist)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(prompt: str, text: str, split: str, path: Path | str = LOG_PATH) -> None:
    """Raise PromptGuardError if evaluating `prompt` with `text` on `split` breaks the rules above."""
    if split not in TEST_SPLITS:
        return
    sha = prompt_sha(text)
    runs = [e for e in read_log(path) if e.get("prompt") == prompt and e.get("event", "run") == "run"]
    devs = [e for e in runs if e["split"] == "dev"]
    if not devs:
        raise PromptGuardError(f"prompt '{prompt}' has no dev run; run it on split=dev first with this exact text")
    if devs[-1]["sha256"] != sha:
        raise PromptGuardError(
            f"prompt '{prompt}' was edited after its last dev run (dev sha {devs[-1]['sha256'][:12]}, now {sha[:12]}); "
            "re-run dev with the current text before evaluating on test")
    earlier = {e["sha256"] for e in runs if e["split"] in TEST_SPLITS} - {sha}
    if earlier:
        raise PromptGuardError(
            f"the test split was already evaluated with an earlier version of '{prompt}' "
            f"({', '.join(s[:12] for s in sorted(earlier))}); a new version would be tuned on test. "
            "Use force=True only for a documented reason")


def record(prompt: str, text: str, split: str, path: Path | str = LOG_PATH, force: bool = False,
           now: Optional[datetime] = None) -> dict:
    """check() unless force, then append a run entry and return it."""
    if not force:
        check(prompt, text, split, path)
    entry = {"event": "run", "time": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
             "prompt": prompt, "sha256": prompt_sha(text), "split": split, "forced": bool(force)}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def notes(path: Path | str = LOG_PATH) -> dict[str, list[str]]:
    """{prompt: [caveat texts]} from 'note' entries."""
    out: dict[str, list[str]] = {}
    for e in read_log(path):
        if e.get("event") == "note":
            out.setdefault(e["prompt"], []).append(e["note"])
    return out
