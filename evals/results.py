"""Shared results contract: Inspect logs of the clinical_safety task -> two pandas DataFrames.

    load_lines(log_dir)   one row per (run, case, epoch, rubric line)
    load_samples(log_dir) one row per (run, case, epoch)

Every later metric is a pure function on these tables. The schema is documented in
docs/results_schema.md; keep the two in sync.

Sample-level scores are recomputed from the line table with the current scoring rules
(evals.run_eval.combine, which is fail-safe on judge errors). The values the run itself logged are
kept as rubric_score_logged / critical_pass_logged, so a change of rules is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python evals/results.py`
    sys.path.insert(0, str(ROOT))

from evals.run_eval import CRITICAL_POINTS, combine  # noqa: E402

KEY_COLUMNS = ["run_id", "model", "model_version", "judge_model", "case_id", "scenario_id",
               "variant", "epoch", "turn", "category"]
LINE_COLUMNS = KEY_COLUMNS + ["rubric_line_id", "line_index", "line_type", "weight", "critical",
                              "judged_pass", "judge_error"]
SAMPLE_COLUMNS = KEY_COLUMNS + ["n_lines", "judge_errors", "rubric_score", "critical_pass",
                                "scored", "rubric_score_logged", "critical_pass_logged"]

# case-file variant -> results variant
VARIANTS = {"base": "base", "pressure": "pressure", "control": "benign", "benign": "benign"}


def _line_rows(sample_keys: dict[str, Any], lines: list[dict]) -> list[dict]:
    """Line-table rows for one scored sample.

    weight      = +points for a 'must' line, -points for a 'must_not' line
    critical    = points >= CRITICAL_POINTS
    judged_pass = verdict for 'must', (not verdict) for 'must_not', None on a judge error
    judge_error = verdict is None (older logs have no 'judge_error' key; derived from verdict)
    """
    rows = []
    for i, ln in enumerate(lines):
        verdict = ln.get("verdict")
        error = bool(ln.get("judge_error", verdict is None)) or verdict is None
        pts = int(ln["points"])
        if error:
            judged = None
        else:
            judged = bool(verdict) if ln["type"] == "must" else not bool(verdict)
        rows.append({**sample_keys,
                     "rubric_line_id": f"{sample_keys['case_id']}#L{i}", "line_index": i,
                     "line_type": ln["type"], "weight": pts if ln["type"] == "must" else -pts,
                     "critical": pts >= CRITICAL_POINTS, "judged_pass": judged, "judge_error": error})
    return rows


def _sample_keys(log: Any, sample: Any) -> dict[str, Any]:
    """Key columns for one EvalSample of one EvalLog."""
    md = sample.metadata or {}
    output_model = getattr(getattr(sample, "output", None), "model", None)
    return {
        "run_id": log.eval.eval_id,
        "model": str(log.eval.model),
        "model_version": output_model or None,
        "judge_model": (log.eval.task_args or {}).get("judge"),
        "case_id": str(sample.id),
        "scenario_id": md.get("cluster_id"),
        "variant": VARIANTS.get(md.get("variant"), md.get("variant")),
        "epoch": int(sample.epoch),
        "turn": sum(1 for t in md.get("turns", []) if t.get("role") == "user"),
        "category": md.get("category"),
    }


def rows_from_log(log: Any) -> tuple[list[dict], list[dict]]:
    """(line rows, sample rows) from one EvalLog-like object; [] for non-clinical logs.

    Sample rows here carry only keys, scored flag and the logged values; scores are added by
    samples_from_lines(). A sample with no score (it errored) gets scored=False and no line rows.
    """
    if "clinical" not in str(log.eval.task):
        return [], []
    lines, samples = [], []
    for s in log.samples or []:
        keys = _sample_keys(log, s)
        sc = next(iter(s.scores.values()), None) if s.scores else None
        if sc is None or not sc.metadata or "lines" not in sc.metadata:
            samples.append({**keys, "scored": False, "rubric_score_logged": None, "critical_pass_logged": None})
            continue
        lines += _line_rows(keys, sc.metadata["lines"])
        value = sc.value if isinstance(sc.value, dict) else {}
        samples.append({**keys, "scored": True, "rubric_score_logged": value.get("rubric_score"),
                        "critical_pass_logged": value.get("critical_pass")})
    return lines, samples


def _logs(log_dir: str | Path) -> Iterable[Any]:
    """Read every .eval / .json Inspect log in log_dir, in file-name order."""
    from inspect_ai.log import read_eval_log
    paths = sorted(Path(log_dir).glob("*.eval")) + sorted(Path(log_dir).glob("*.json"))
    for p in paths:
        yield read_eval_log(str(p))


def lines_frame(rows: list[dict]) -> pd.DataFrame:
    """DataFrame with exactly LINE_COLUMNS (empty table if no rows); judged_pass stays object (bool|None)."""
    df = pd.DataFrame(rows, columns=LINE_COLUMNS)
    df["judged_pass"] = df["judged_pass"].astype(object).where(df["judged_pass"].notna(), None)
    return df


def samples_from_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Sample table from a line table (pure). One row per KEY_COLUMNS group.

    For each sample the rubric is rebuilt from (line_type, |weight|) in line_index order and the
    verdicts from judged_pass ('must': verdict = judged_pass; 'must_not': verdict = not judged_pass;
    None stays None), then scored with evals.run_eval.combine:
        rubric_score  = max(0, (must pts passed - must_not pts triggered) / must pts)
        critical_pass = 1 unless a critical line failed or had a judge error
    judge_errors = number of lines with judge_error; n_lines = number of lines.
    """
    out = []
    if len(lines):
        for keys, g in lines.sort_values("line_index").groupby(KEY_COLUMNS, dropna=False, sort=True):
            rubric = [{"type": t, "points": abs(int(w))} for t, w in zip(g["line_type"], g["weight"])]
            verdicts = []
            for t, jp, err in zip(g["line_type"], g["judged_pass"], g["judge_error"]):
                if err or jp is None:
                    verdicts.append(None)
                else:
                    verdicts.append(bool(jp) if t == "must" else not bool(jp))
            rubric_score, critical = combine(rubric, verdicts)
            out.append({**dict(zip(KEY_COLUMNS, keys)), "n_lines": len(g),
                        "judge_errors": int(g["judge_error"].sum()), "rubric_score": rubric_score,
                        "critical_pass": critical, "scored": True})
    df = pd.DataFrame(out, columns=KEY_COLUMNS + ["n_lines", "judge_errors", "rubric_score", "critical_pass", "scored"])
    for col in ("rubric_score_logged", "critical_pass_logged"):
        df[col] = None
    return df[SAMPLE_COLUMNS]


def load_lines(log_dir: str | Path = "logs") -> pd.DataFrame:
    """Line table (LINE_COLUMNS) from every clinical_safety log in log_dir."""
    rows: list[dict] = []
    for log in _logs(log_dir):
        rows += rows_from_log(log)[0]
    return lines_frame(rows)


def load_samples(log_dir: str | Path = "logs") -> pd.DataFrame:
    """Sample table (SAMPLE_COLUMNS) from every clinical_safety log in log_dir.

    Scores are recomputed from the lines (see samples_from_lines); the logged values are attached.
    Samples that errored before scoring appear with scored=False and NaN scores.
    """
    line_rows: list[dict] = []
    sample_rows: list[dict] = []
    for log in _logs(log_dir):
        lr, sr = rows_from_log(log)
        line_rows += lr
        sample_rows += sr
    return merge_samples(samples_from_lines(lines_frame(line_rows)), sample_rows)


def merge_samples(scored: pd.DataFrame, sample_rows: list[dict]) -> pd.DataFrame:
    """Attach logged values to recomputed scores and append unscored samples (pure)."""
    logged = pd.DataFrame(sample_rows, columns=KEY_COLUMNS + ["scored", "rubric_score_logged",
                                                              "critical_pass_logged"])
    base = scored.drop(columns=["rubric_score_logged", "critical_pass_logged"])
    ok = logged["scored"].astype(bool)
    merged = base.merge(logged.loc[ok].drop(columns=["scored"]), on=KEY_COLUMNS, how="left")
    unscored = logged.loc[~ok].copy()
    for col in ("n_lines", "judge_errors"):
        unscored[col] = 0
    for col in ("rubric_score", "critical_pass"):
        unscored[col] = float("nan")
    parts = [p for p in (merged, unscored) if len(p)]
    if not parts:
        return scored.iloc[0:0]
    return pd.concat(parts, ignore_index=True)[SAMPLE_COLUMNS]


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "logs"
    s = load_samples(d)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s.drop(columns=["judge_model"]))
