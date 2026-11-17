"""One-call exploratory report for the uncertainty layer. Every section is labelled EXPLORATORY."""
from __future__ import annotations

import math

import pandas as pd

from evals.uncertainty import conformal as cf
from evals.uncertainty import evaluation as ev

BANNER = ("**EXPLORATORY ANALYSIS, not a validated safety measure.** With a few dozen cases the intervals below "
          "are wide; read them before any point estimate.")


def uncertainty_report(table: pd.DataFrame, alpha: float = 0.1, cal_fraction: float = 0.5, B: int = 2000,
                       seed: int = 0) -> dict:
    """Run the whole analysis on a table from evaluation.build_eval_table.

    Returns {"label", "predictor" (compare_predictors), "selective" (AURC per score), "conformal" (ConformalResult or
    an error string), "n", "n_failures"}. The conformal step is skipped with the reason when the calibration set is
    too small (CalibrationTooSmall), so a small run still produces the rest.
    """
    scores = {"semantic_entropy": table["semantic_entropy"].to_numpy(float),
              "logprob_baseline": table["logprob_score"].to_numpy(float)}
    fail = table["failure"].to_numpy(int)
    cl = table["scenario_id"].to_numpy()
    out = {"label": "exploratory", "n": len(table), "n_failures": int(fail.sum()),
           "predictor": ev.compare_predictors(scores, fail, cl, B=B, seed=seed),
           "selective": {k: ev.selective_summary(v, fail, cl, B=min(B, 1000), seed=seed) for k, v in scores.items()}}
    try:
        cal, test = cf.split_by_scenario(table, cal_fraction, seed)
        out["conformal"] = cf.conformal_abstention(cal, test, "semantic_entropy", alpha)
    except (cf.CalibrationTooSmall, ValueError) as exc:
        out["conformal"] = f"not run: {exc}"
    return out


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def format_report(rep: dict) -> str:
    """Markdown. The banner comes first, then each AUROC as "95% CI [lo, hi]" before the point estimate."""
    p = rep["predictor"]
    lines = [BANNER, "", f"Cases: {rep['n']}, failures (critical_pass < 1): {rep['n_failures']}, "
             f"scenarios: {p['n_clusters']}, valid bootstrap draws: {p['n_boot_valid']}", "",
             "### Predicting critical_pass failure (exploratory)", "",
             "| score | 95% CI for AUROC | AUROC (point estimate) |", "|---|---|---|"]
    for k, v in p["scores"].items():
        lines.append(f"| {k} | [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}] | {_fmt(v['auroc'])} |")
    for k, v in p.get("difference", {}).items():
        lines.append(f"| difference {k} | [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}] | "
                     f"{_fmt(v['auroc_difference'])} |")
    widths = [v["ci_high"] - v["ci_low"] for v in p["scores"].values() if not math.isnan(v["ci_low"])]
    if not widths or max(widths) > 0.3 or rep["n_failures"] < 10:
        lines += ["", "> Caution: the interval is wide or there are fewer than 10 failures. This cannot show that "
                      "either score is useful."]
    lines += ["", "### Selective answering: risk-coverage (exploratory)", "",
              "| score | AURC (lower is better) | always-answer risk | 95% bootstrap CI for AURC |",
              "|---|---|---|---|"]
    for k, v in rep["selective"].items():
        lines.append(f"| {k} | {_fmt(v['aurc'], 3)} | {_fmt(v['random_baseline'], 3)} | "
                     f"[{_fmt(v['ci_low'], 3)}, {_fmt(v['ci_high'], 3)}] |")
    lines += ["", "### Split-conformal abstention on semantic entropy (exploratory)", ""]
    c = rep["conformal"]
    if isinstance(c, str):
        lines.append(f"> {c}")
    else:
        lines += [f"alpha = {c.alpha}; q_hat = {_fmt(c.q_hat, 3)} from {c.n_cal} correct calibration cases; "
                  f"{c.n_test} test cases.", "",
                  f"- coverage of correctly handled test cases (target >= {_fmt(1 - c.alpha)}): {_fmt(c.coverage)} "
                  f"(n={c.n_test_correct})",
                  f"- abstention rate: {_fmt(c.abstention_rate)}",
                  f"- failures caught by abstaining: {_fmt(c.failures_caught)}",
                  f"- failure rate among answered cases: {_fmt(c.risk_answered)} (all test cases: {_fmt(c.risk_all)})"]
    return "\n".join(lines) + "\n"

