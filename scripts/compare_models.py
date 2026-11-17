#!/usr/bin/env python3
"""Compare two models' clinical-safety results with cluster-robust error bars (Miller 2024).

Run from the project folder, venv active:
    python scripts/compare_models.py logs_model_a/ logs_model_b/

Each log dir should hold the .eval log(s) of ONE model; the most recent successful clinical_safety
log in each dir is used (named in the output). Questions are paired by sample id and clustered by
scenario (cluster_id in cases.jsonl). Prints a markdown table for rubric_score and critical_pass.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "stats"))
from inference import (load_results, clustered_se, paired_difference,  # noqa: E402
                       minimum_detectable_effect, power_analysis)

METRICS = ["rubric_score", "critical_pass"]


def latest_log(log_dir):
    from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log  # noqa: F401
    files = sorted(Path(log_dir).glob("*clinical*safety*.eval"))
    for f in reversed(files):                       # file names start with a UTC timestamp
        try:
            if read_eval_log(str(f), header_only=True).status == "success":
                return f
        except Exception:
            continue
    sys.exit(f"no successful clinical-safety .eval log in {log_dir}")


def f(x, nd=3):
    return "n/a" if x != x else f"{x:.{nd}f}"


def ci(lo, hi):
    return f"[{f(lo)}, {f(hi)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir_a")
    ap.add_argument("log_dir_b")
    ap.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    args = ap.parse_args()

    a, b = (load_results(latest_log(d)) for d in (args.log_dir_a, args.log_dir_b))
    name_a, name_b = a.attrs["model"], b.attrs["model"]
    if name_a == name_b:
        name_a, name_b = f"{name_a} (A)", f"{name_b} (B)"
    both = a.merge(b, on="id", suffixes=("_a", "_b"))
    if both.empty:
        sys.exit("the two log dirs share no question ids")
    cl = both["scenario_id_a"]
    pct = int(round(args.level * 100))

    print(f"A = {name_a}  (judge {a.attrs['judge']}, {Path(a.attrs['log']).name})")
    print(f"B = {name_b}  (judge {b.attrs['judge']}, {Path(b.attrs['log']).name})")
    print(f"Paired on {len(both)} shared questions ({len(a) - len(both)} only in A, "
          f"{len(b) - len(both)} only in B); cluster unit = scenario.\n")
    if a.attrs["judge"] != b.attrs["judge"]:
        print("> **Warning:** different judges; differences mix model and judge effects.\n")

    notes, sigmas = [], []
    for m in METRICS:
        xa, xb = both[f"{m}_a"], both[f"{m}_b"]
        ra, rb = clustered_se(xa, cl, args.level), clustered_se(xb, cl, args.level)
        d = paired_difference(xa, xb, cl, args.level)
        sigmas.append(d.sd_d)
        print(f"**{m}**\n")
        print(f"| row | n | clusters | mean | SE | {pct}% CI | paired diff | p |")
        print("|---|---|---|---|---|---|---|---|")
        for label, r in ((f"A {name_a}", ra), (f"B {name_b}", rb)):
            print(f"| {label} | {r.n} | {r.n_clusters} | {f(r.mean)} | {f(r.se)} | {ci(r.ci_low, r.ci_high)} | | |")
        print(f"| A − B | {d.n} | {d.n_clusters} | {f(d.mean_diff)} | {f(d.se)} | {ci(d.ci_low, d.ci_high)} "
              f"| {f(d.mean_diff)} | {f(d.p_value, 4)} |\n")
        notes.append(f"- {m}: naive SE A {f(ra.naive_se)} / B {f(rb.naive_se)}; "
                     f"ICC A {f(ra.icc, 2)} / B {f(rb.icc, 2)}; DEFF A {f(ra.deff, 2)} / B {f(rb.deff, 2)}; "
                     f"pairing gain (unpaired SE / paired SE) {f(d.pairing_gain, 2)}")

    print("Clustering and pairing diagnostics:\n")
    print("\n".join(notes))

    n_cases = len(both)
    clusters = both["scenario_id_a"].nunique()
    sig = max(s for s in sigmas if s == s) if any(s == s for s in sigmas) else float("nan")
    if sig == sig and sig > 0:
        mde = minimum_detectable_effect(n_cases, sig)
        print(f"\nPower: with {n_cases} questions in {clusters} scenarios and observed sd of paired differences "
              f"{f(sig, 2)}, the minimum detectable effect is {f(mde, 2)} (80% power, alpha 0.05, questions treated as "
              f"independent; larger once clustering is counted). Detecting a 0.10 gap would need about "
              f"{power_analysis(sig, 0.10)} questions.")
    if clusters < 30:
        print(f"\n> **Caution:** only {clusters} scenarios. Normal-approximation intervals and p-values are "
              f"unreliable at this size; treat them as indicative only.")


if __name__ == "__main__":
    main()
