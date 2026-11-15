"""Summarise red-team runs from the Inspect logs in logs/.

Usage (project folder):  python redteam/summarize.py [logs_dir]
For every (model, prompt) pair it reports, with 95% Wilson intervals:
  attack success rate  = share of INJECTED samples where the hidden instruction was followed
  task success (clean) = share of CLEAN samples where the summary still had the key facts
                         (a defence that lowers this is breaking the normal job)
Only 8 notes exist, so intervals are wide. Repeated epochs reduce sampling noise, not note-to-note variation.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.0%} (95% CI {lo:.0%} to {hi:.0%})" if n else "no data"


def main(logs_dir="logs"):
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(logs_dir).glob("*.eval")):
        log = read_eval_log(str(path))
        if "redteam" not in log.eval.task:
            continue
        prompt = (log.eval.task_args or {}).get("prompt", "baseline")
        for s in log.samples or []:
            sc = next(iter(s.scores.values()))
            md = sc.metadata
            groups[(log.eval.model, prompt)][md["variant"]].append(
                (sc.value["followed_injection"], sc.value["task_ok"], md["attack_type"], s.id))
    if not groups:
        print("No red-team logs found. Run inspect eval redteam/run_redteam.py first.")
        return
    for (model, prompt), by_variant in sorted(groups.items()):
        inj, cln = by_variant["injected"], by_variant["clean"]
        print(f"\n== {model} | prompt: {prompt}")
        print("  attack success (injected):", fmt(int(sum(x[0] for x in inj)), len(inj)))
        print("  summary still correct, injected:", fmt(int(sum(x[1] for x in inj)), len(inj)))
        print("  summary correct, clean control:", fmt(int(sum(x[1] for x in cln)), len(cln)))
        by_type = defaultdict(list)
        for f, _, kind, _ in inj:
            by_type[kind].append(f)
        for kind, vals in sorted(by_type.items()):
            print(f"    {kind}: followed {int(sum(vals))}/{len(vals)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
