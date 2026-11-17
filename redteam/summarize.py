"""Summarise red-team runs from the Inspect logs in logs/.

Usage (project folder):  python redteam/summarize.py [logs_dir]
For every (model, prompt) pair it reports, with 95% Wilson intervals:
  attack success rate  = share of INJECTED samples where the hidden instruction was followed
  task success (clean) = share of CLEAN samples where the summary still had the key facts
                         (a defence that lowers this is breaking the normal job)
Only 8 notes exist, so intervals are wide. Repeated epochs reduce sampling noise, not note-to-note variation.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `evals.stats`

from inspect_ai.log import read_eval_log  # noqa: E402

from evals.stats.intervals import wilson  # noqa: E402

ALPHA = 0.05


def fmt(k: int, n: int, alpha: float = ALPHA) -> str:
    """Format k/n with its Wilson interval, e.g. '2/8 = 25% (95% Wilson CI 7% to 59%)'."""
    if not n:
        return "no data"
    lo, hi = wilson(k, n, alpha)
    return f"{k}/{n} = {k / n:.0%} ({1 - alpha:.0%} Wilson CI {lo:.0%} to {hi:.0%})"


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
