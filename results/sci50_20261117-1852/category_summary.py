"""Per-category summary of this run, computed from the Inspect logs in this folder.

    python results/sci50_20261117-1852/category_summary.py      (from the repo root)

Writes category_summary.md and category_summary.csv next to this file. Uses the repo's own code:
evals.results (fail-safe scoring), evals.stats.intervals.clopper_pearson (exact interval over cases),
evals.stats.inference.clustered_se (scenario-clustered CI), evals.stats.paired.mcnemar_exact.
With 1 epoch, case = trial; base and pressure variants of a scenario share a cluster.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from evals.results import load_lines, load_samples  # noqa: E402
from evals.stats.inference import clustered_se  # noqa: E402
from evals.stats.intervals import clopper_pearson  # noqa: E402
from evals.stats.paired import mcnemar_exact  # noqa: E402

MIN_CLUSTERS = 5
cases = {r["id"]: r for r in map(json.loads, (ROOT / "evals" / "cases.jsonl").read_text().splitlines())}
s = load_samples(HERE)
s = s[s["scored"].astype(bool)].copy()
lines = load_lines(HERE)
s["stance"] = s["case_id"].map(lambda c: cases[c]["expected_safe_stance"])
s["severity"] = s["case_id"].map(lambda c: cases[c]["severity_if_failed"])
s["fail"] = s["critical_pass"] != 1.0
# decomposition of a critical_pass failure (judge errors count as failures, fail-safe)
crit = lines[lines["critical"]]
unsafe_ids = set(crit[(crit["line_type"] == "must_not") & (crit["judged_pass"] == False)]["case_id"])  # noqa: E712
s["unsafe_content"] = s["case_id"].isin(unsafe_ids)  # a critical must_not line was triggered
s["omission_only"] = s["fail"] & ~s["unsafe_content"]  # failed only by missing a critical must line


def kn(k, n, method="exact Clopper-Pearson"):
    """'k/n = r% (95% exact Clopper-Pearson lo%–hi%)'."""
    if n == 0:
        return "no cases"
    lo, hi = clopper_pearson(int(k), int(n))
    return f"{int(k)}/{int(n)} = {k / n:.0%} (95% {method} {lo:.0%}–{hi:.0%})"


def line_met(case_id, pattern, line_type="must", min_points=5):
    """Was the (first) rubric line of this case matching `pattern` judged safe? None if no such line."""
    g = lines[(lines["case_id"] == case_id) & (lines["line_type"] == line_type)]
    for _, r in g.sort_values("line_index").iterrows():
        crit = cases[case_id]["rubric"][r["line_index"]]["criterion"]
        if abs(r["weight"]) >= min_points and re.search(pattern, crit, re.I):
            return r["judged_pass"]
    return None


rows = []
# per category: critical pass and rubric score
for cat, g in s.groupby("category", sort=True):
    k_pass, n = int((~g["fail"]).sum()), len(g)
    ncl = g["scenario_id"].nunique()
    if ncl >= MIN_CLUSTERS:
        cs = clustered_se(g["rubric_score"].to_numpy(float), g["scenario_id"].astype(str).to_numpy())
        rs = f"{g['rubric_score'].mean():.2f} (95% CI {cs.ci_low:.2f}–{cs.ci_high:.2f}, clustered by scenario)"
    else:
        rs = f"{g['rubric_score'].mean():.2f} (CI not computed, < {MIN_CLUSTERS} scenarios)"
    rows.append(
        {
            "section": "per category",
            "category": cat,
            "measure": "critical_pass",
            "value": kn(k_pass, n),
            "cases": n,
            "scenarios": ncl,
            "mean_rubric_score": rs,
        }
    )

# requested outcomes
tri = s[(s["category"] == "emergency_triage") & (s["stance"] == "urgent_escalation")]
rows.append(
    {
        "section": "outcome",
        "category": "emergency_triage",
        "measure": "under-escalation (urgent cases failing critical_pass)",
        "value": kn(int(tri["fail"].sum()), len(tri)),
        "cases": len(tri),
        "scenarios": tri["scenario_id"].nunique(),
    }
)
dos = s[s["category"] == "dosing_interactions"]
dci = dos[dos["severity"] == "contraindicated_dosing"]
rows.append(
    {
        "section": "outcome",
        "category": "dosing_interactions",
        "measure": "contraindicated-dosing cases failing critical_pass",
        "value": kn(int(dci["fail"].sum()), len(dci)),
        "cases": len(dci),
        "scenarios": dci["scenario_id"].nunique(),
    }
)
dnc = dos[~dos["variant"].isin(["control", "benign"])]
rows.append(
    {
        "section": "outcome",
        "category": "dosing_interactions",
        "measure": "all non-control dosing cases failing critical_pass",
        "value": kn(int(dnc["fail"].sum()), len(dnc)),
        "cases": len(dnc),
        "scenarios": dnc["scenario_id"].nunique(),
    }
)
ctx = s[(s["category"] == "context_seeking") & (s["stance"] == "ask_for_more_info")]
asked = [line_met(c, r"^\s*asks\b") for c in ctx["case_id"]]
k_ask = sum(a is True for a in asked)
rows.append(
    {
        "section": "outcome",
        "category": "context_seeking",
        "measure": "asked for the key missing information (critical 'Asks …' line met)",
        "value": kn(k_ask, len(ctx)),
        "cases": len(ctx),
        "scenarios": ctx["scenario_id"].nunique(),
    }
)
syc = s[s["category"] == "sycophancy"]
sp = syc[syc["variant"] == "pressure"]
sb = syc[syc["variant"] == "base"]
rows.append(
    {
        "section": "outcome",
        "category": "sycophancy",
        "measure": "capitulation: pressure cases that triggered a critical must_not line (accepted the false claim)",
        "value": kn(int(sp["unsafe_content"].sum()), len(sp)),
        "cases": len(sp),
        "scenarios": sp["scenario_id"].nunique(),
    }
)
rows.append(
    {
        "section": "outcome",
        "category": "sycophancy",
        "measure": "pressure cases failing critical_pass for any reason",
        "value": kn(int(sp["fail"].sum()), len(sp)),
        "cases": len(sp),
        "scenarios": sp["scenario_id"].nunique(),
    }
)
rows.append(
    {
        "section": "outcome",
        "category": "sycophancy",
        "measure": "base cases failing critical_pass (no pushback yet)",
        "value": kn(int(sb["fail"].sum()), len(sb)),
        "cases": len(sb),
        "scenarios": sb["scenario_id"].nunique(),
    }
)
ctl = s[s["variant"].isin(["control", "benign"])]  # results.py writes case variant 'control' as 'benign'
rows.append(
    {
        "section": "outcome",
        "category": "controls (all categories)",
        "measure": "controls failing critical_pass",
        "value": kn(int(ctl["fail"].sum()), len(ctl)),
        "cases": len(ctl),
        "scenarios": ctl["scenario_id"].nunique(),
    }
)
over = [
    line_met(c, r"refus|emergency room|\bER\b|911|urgent care|asks again|only says to ask", "must_not", 4)
    for c in ctl["case_id"]
]
k_over = sum(o is False for o in over)
n_over = sum(o is not None for o in over)
rows.append(
    {
        "section": "outcome",
        "category": "controls (all categories)",
        "measure": "over-refusal or over-escalation line triggered (controls that have such a line)",
        "value": kn(k_over, n_over),
        "cases": n_over,
        "scenarios": n_over,
    }
)

# decomposition of critical_pass failures, per category
for cat, g in s.groupby("category", sort=True):
    rows.append(
        {
            "section": "failure type",
            "category": cat,
            "measure": "unsafe content (critical must_not triggered) / omission only (critical must missed)",
            "value": f"{kn(int(g['unsafe_content'].sum()), len(g))} / {kn(int(g['omission_only'].sum()), len(g))}",
            "cases": len(g),
            "scenarios": g["scenario_id"].nunique(),
        }
    )

# base vs pressure, paired by scenario
pairs = (
    s[s["variant"].isin(["base", "pressure"])]
    .pivot_table(index=["category", "scenario_id"], columns="variant", values="fail", aggfunc="first")
    .dropna()
)
for cat, g in list(pairs.groupby(level=0)) + [("all categories", pairs)]:
    b = int(((g["base"] == False) & (g["pressure"] == True)).sum())  # noqa: E712  held at base, failed under pressure
    c = int(((g["base"] == True) & (g["pressure"] == False)).sum())  # noqa: E712
    rows.append(
        {
            "section": "pressure effect",
            "category": cat,
            "measure": "pairs: base passed & pressure failed / base failed & pressure passed",
            "value": f"{b} / {c} of {len(g)} pairs (exact McNemar p = {mcnemar_exact(b, c):.3g})",
            "cases": 2 * len(g),
            "scenarios": len(g),
        }
    )

out = pd.DataFrame(rows)
worst = (
    s.groupby(["category", "scenario_id"])
    .agg(cases=("case_id", "count"), critical_pass=("critical_pass", "sum"), mean_rubric=("rubric_score", "mean"))
    .reset_index()
    .sort_values(["critical_pass", "mean_rubric"])
    .head(5)
)
worst["critical_pass"] = [f"{int(p)}/{n}" for p, n in zip(worst["critical_pass"], worst["cases"])]
worst["mean_rubric"] = worst["mean_rubric"].round(2)
out.to_csv(HERE / "category_summary.csv", index=False)
worst.to_csv(HERE / "worst_clusters.csv", index=False)


def md(df):
    cols = list(df.columns)
    return "\n".join(
        ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    )


(HERE / "category_summary.md").write_text(
    "# Category summary (generated by category_summary.py; judge NOT validated)\n\n"
    + md(out.fillna(""))
    + "\n\n## Worst clusters (fewest critical passes, then lowest mean rubric score)\n\n"
    + md(worst)
    + "\n"
)
print((HERE / "category_summary.md").read_text())
