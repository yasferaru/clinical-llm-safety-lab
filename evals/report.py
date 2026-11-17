"""Critical-pass rate with an explicitly named confidence interval.

    python evals/report.py [logs_dir] [--method clopper_pearson|wilson] [--alpha 0.05]

The unit is the case. With several epochs a case counts as passed only if it passed in every
epoch (pass^k). With one epoch this is the plain pass rate. Samples that errored before scoring
are excluded and reported separately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.results import load_samples  # noqa: E402
from evals.stats.intervals import METHODS  # noqa: E402

GROUP = ["run_id", "model"]


def critical_pass_summary(samples: pd.DataFrame, alpha: float = 0.05, method: str = "clopper_pearson",
                          by: list[str] | None = None) -> pd.DataFrame:
    """Per group: k cases passed (all epochs) out of n scored cases, rate = k/n, CI by `method`.

    Columns: *by, k, n, epochs, rate, ci_low, ci_high, method, alpha, unscored_samples.
    Example: critical_pass 1,0,0,0 over 4 cases -> k=1, n=4, Clopper-Pearson 95% CI (0.0063, 0.8059).
    """
    by = by or GROUP
    ci = METHODS[method]
    rows = []
    for keys, g in samples.groupby(by, dropna=False, sort=True):
        scored = g[g["scored"].astype(bool)]
        per_case = scored.groupby("case_id")["critical_pass"].min()  # pass^k: every epoch passed
        k, n = int((per_case == 1.0).sum()), int(len(per_case))
        lo, hi = ci(k, n, alpha) if n else (float("nan"), float("nan"))
        rows.append({**dict(zip(by, keys if isinstance(keys, tuple) else (keys,))),
                     "k": k, "n": n, "epochs": int(scored["epoch"].nunique()) if n else 0,
                     "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
                     "method": method, "alpha": alpha, "unscored_samples": int((~g["scored"].astype(bool)).sum())})
    return pd.DataFrame(rows, columns=by + ["k", "n", "epochs", "rate", "ci_low", "ci_high", "method", "alpha",
                                           "unscored_samples"])


def main(argv: list[str] | None = None) -> None:
    """Print critical_pass_summary for every clinical_safety log in a folder."""
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir", nargs="?", default="logs")
    ap.add_argument("--method", default="clopper_pearson", choices=sorted(METHODS))
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    summary = critical_pass_summary(load_samples(a.logs_dir), a.alpha, a.method)
    if summary.empty:
        print("No clinical_safety logs found.")
        return
    label = "Clopper-Pearson" if a.method == "clopper_pearson" else "Wilson"
    for _, r in summary.iterrows():
        print(f"{r['model']}  run {r['run_id']}  critical pass {r['k']}/{r['n']} cases"
              f" ({r['epochs']} epoch(s), all must pass) = {r['rate']:.0%}"
              f"  {1 - a.alpha:.0%} {label} CI {r['ci_low']:.1%} to {r['ci_high']:.1%}"
              + (f"  [{r['unscored_samples']} unscored samples excluded]" if r["unscored_samples"] else ""))


if __name__ == "__main__":
    main()
