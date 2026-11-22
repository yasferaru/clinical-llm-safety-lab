"""Markdown reliability report from clinical_safety logs.

    python scripts/reliability_report.py [logs_dir] [--ks 1,2,4,8] [--B 2000] [--seed 0] [--out FILE]

For each model, prints pass^k, pass@k and worst@k (rubric_score) for every requested k up to the
smallest number of trials any case has, with 95% scenario-cluster bootstrap intervals, then the
epoch-consistency table. Runs of the same model are pooled (see evals/stats/reliability.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.stats.reliability import SUCCESS_RULE, consistency_summary, reliability_summary  # noqa: E402

MIN_SCENARIOS = 5  # below this, cluster-bootstrap intervals are flagged as unreliable


def _cell(v: float, lo: float, hi: float) -> str:
    """'0.50 (0.25–0.75)', or '0.50 (n/a)' when the interval is undefined (fewer than 2 scenarios)."""
    if pd.isna(lo) or pd.isna(hi):
        return f"{v:.2f} (n/a)"
    return f"{v:.2f} ({lo:.2f}–{hi:.2f})"


def reliability_markdown(samples: pd.DataFrame, ks: Sequence[int] = (1, 2, 4, 8), B: int = 2000,
                         seed: int = 0, level: float = 0.95) -> str:
    """Markdown: one reliability table (rows = model x k) and one consistency table (rows = model)."""
    rel = reliability_summary(samples, ks=ks, B=B, seed=seed, level=level)
    con = consistency_summary(samples)
    pct = f"{level:.0%}"
    out = ["## Reliability across epochs", "",
           f"Success = `{SUCCESS_RULE}`. Values are means over cases; brackets are {pct} percentile intervals "
           f"from a bootstrap over scenario clusters (B = {B}, seed = {seed}). k is shown only up to the "
           "smallest number of trials any case of that model has.", "",
           "| Model | k | Cases | Scenarios | Trials per case | pass^k | pass@k | worst@k (rubric) |",
           "|---|---|---|---|---|---|---|---|"]
    if rel.empty:
        out.append("| (no scored samples) | | | | | | | |")
    for _, r in rel.iterrows():
        trials = str(r["n_min"]) if r["n_min"] == r["n_max"] else f"{r['n_min']}–{r['n_max']}"
        out.append(f"| `{r['model']}` | {r['k']} | {r['n_cases']} | {r['n_scenarios']} | {trials} | "
                   f"{_cell(r['pass_hat_k'], r['pass_hat_k_lo'], r['pass_hat_k_hi'])} | "
                   f"{_cell(r['pass_at_k'], r['pass_at_k_lo'], r['pass_at_k_hi'])} | "
                   f"{_cell(r['worst_at_k'], r['worst_at_k_lo'], r['worst_at_k_hi'])} |")
    few = rel[rel["n_scenarios"] < MIN_SCENARIOS]["model"].unique() if not rel.empty else []
    if len(few):
        out += ["", f"Warning: fewer than {MIN_SCENARIOS} scenarios for "
                + ", ".join(f"`{m}`" for m in few)
                + ". A bootstrap over so few clusters gives intervals that are too narrow or degenerate; "
                "do not read them as uncertainty."]
    skipped = sorted(set(ks) - set(rel["k"])) if not rel.empty else sorted(set(ks))
    if skipped:
        out += ["", f"Not shown: k = {', '.join(map(str, skipped))} (more than the available trials for every model)."]
    out += ["", "## Epoch consistency", "",
            "Cases with at least 2 trials. *Disagree* = the trials do not all agree on success. "
            "*Flip rate* = chance that two distinct trials of a case disagree, averaged over cases.", "",
            "| Model | Cases | Cases whose trials disagree | Mean flip rate |", "|---|---|---|---|"]
    if con.empty:
        out.append("| (no case has 2 or more trials) | | | |")
    for _, r in con.iterrows():
        out.append(f"| `{r['model']}` | {r['n_cases']} | {r['n_disagree']} of {r['n_cases']} "
                   f"({r['frac_disagree']:.0%}) | {r['mean_flip_rate']:.2f} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    """CLI: load logs with evals.results.load_samples and print (or write) the markdown report."""
    from evals.results import load_samples
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir", nargs="?", default="logs")
    ap.add_argument("--ks", default="1,2,4,8", help="comma-separated k values")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the markdown here instead of printing it")
    a = ap.parse_args(argv)
    ks = [int(k) for k in a.ks.split(",") if k.strip()]
    text = reliability_markdown(load_samples(a.logs_dir), ks=ks, B=a.B, seed=a.seed)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
