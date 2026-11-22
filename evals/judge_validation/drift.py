"""Judge drift: record which judge version and prompt produced each set of verdicts, and measure
test-retest agreement of the same judge on the same responses.

A judge run is identified by its fingerprint (judge_model, judge_version, prompt_sha):
    judge_version = model name the provider reported for the calls (may be an alias; record what you get)
    prompt_sha    = sha256 of the judge prompt template (evals/run_eval.JUDGE_PROMPT by default)
Two runs with the same fingerprint measure test-retest reliability; runs with different fingerprints
measure drift (version or prompt changed). Runs are appended to a JSONL log (default
evals/judge_validation/judge_runs.jsonl).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.analyze_agreement import agreement, bootstrap_ci, cohen_kappa, gwet_ac1  # noqa: E402
from evals.judge_validation.bias_audit import LineJudge  # noqa: E402

RUNS_LOG = Path(__file__).parent / "judge_runs.jsonl"


def prompt_sha(template: Optional[str] = None) -> str:
    """sha256 of the judge prompt template (run_eval.JUDGE_PROMPT if None)."""
    if template is None:
        from evals.run_eval import JUDGE_PROMPT
        template = JUDGE_PROMPT
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgeFingerprint:
    """What produced a set of verdicts."""
    judge_model: str
    judge_version: Optional[str]
    prompt_sha: str


def record_judge_run(fp: JudgeFingerprint, run_id: str, n_items: int, path: Path | str = RUNS_LOG,
                     now: Optional[datetime] = None) -> dict:
    """Append {time, run_id, n_items, judge_model, judge_version, prompt_sha} to the JSONL log."""
    entry = {"time": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "run_id": run_id,
             "n_items": int(n_items), **asdict(fp)}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_judge_runs(path: Path | str = RUNS_LOG) -> list[dict]:
    """Entries of the judge-run log, oldest first ([] if absent)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


async def rejudge(items: pd.DataFrame, judge: LineJudge) -> pd.DataFrame:
    """item_id, response_id, verdict from judging every item once more (columns of bias_audit.ITEM_COLUMNS)."""
    got = await asyncio.gather(*[judge(c, r, k) for c, r, k in zip(items["conversation"], items["reply"],
                                                                     items["criterion"])])
    return pd.DataFrame({"item_id": items["item_id"].values, "response_id": items["response_id"].values,
                         "verdict": got})


def retest_agreement(run_a: pd.DataFrame, run_b: pd.DataFrame, fp_a: JudgeFingerprint, fp_b: JudgeFingerprint,
                reps: int = 2000, seed: int = 0) -> dict:
    """Agreement of two verdict runs (item_id, response_id, verdict) on their common readable items.

    Returns n, flips, agreement, kappa, ac1, 95% bootstrap CIs for agreement and kappa (resampling whole
    responses, seeded), unreadable counts, and kind = 'test-retest' (same fingerprint) or 'drift'.
    """
    m = run_a.merge(run_b, on=["item_id", "response_id"], suffixes=("_a", "_b"))
    ok = m["verdict_a"].notna() & m["verdict_b"].notna()
    both = m[ok]
    a, b = [bool(x) for x in both["verdict_a"]], [bool(x) for x in both["verdict_b"]]
    groups = list(both["response_id"])
    return {"kind": "test-retest" if fp_a == fp_b else "drift", "fingerprint_a": asdict(fp_a),
            "fingerprint_b": asdict(fp_b), "n": len(a), "unreadable": int((~ok).sum()),
            "flips": sum(x != y for x, y in zip(a, b)), "agreement": agreement(a, b) if a else None,
            "agreement_ci": bootstrap_ci(a, b, stat=agreement, reps=reps, seed=seed, groups=groups) if a else None,
            "kappa": cohen_kappa(a, b) if a else None,
            "kappa_ci": bootstrap_ci(a, b, reps=reps, seed=seed, groups=groups) if a else None,
            "ac1": gwet_ac1(a, b) if a else None}
