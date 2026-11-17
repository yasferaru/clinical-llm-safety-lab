"""Summarise red-team runs from the Inspect logs in logs/ (markdown).

Usage (project folder):  python redteam/summarize.py [logs_dir] [--all-runs]

Reports ASR (complied), CHER (harm level >= H2), the legacy regex rate and clean-note utility, each as
k/n trials with a 95% Wilson interval, by attack vector and by attack type; the complied x harmful table;
and hardened-vs-baseline comparisons paired by note and repeat (exact McNemar). By default only the
most recent run per (model, prompt, split) is used, so re-runs are not pooled; --all-runs pools them.
Definitions: redteam/report.py and redteam/scoring.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from redteam.prompt_guard import notes  # noqa: E402
from redteam.report import latest_runs, load_trials, render_markdown  # noqa: E402


def main(argv=None):
    """Print the markdown report for a logs folder."""
    args = list(sys.argv[1:] if argv is None else argv)
    all_runs = "--all-runs" in args
    args = [a for a in args if a != "--all-runs"]
    trials = load_trials(args[0] if args else "logs")
    if not all_runs:
        trials = latest_runs(trials)
    print(render_markdown(trials, caveats=notes()), end="")


if __name__ == "__main__":
    main()
