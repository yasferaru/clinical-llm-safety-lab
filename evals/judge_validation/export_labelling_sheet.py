#!/usr/bin/env python3
"""Make a BLIND labelling sheet from Inspect logs of the clinical_safety eval.

Every item is one rubric line judged for one reply. You label it YES or NO: is the criterion
TRUE of the reply? (Same question the judge answers; for a 'must_not' line YES means the bad
behaviour happened.) The judge's verdicts are written to a separate key file. Do not open the
key until you have finished labelling.

Usage, from the project folder:
    python evals/judge_validation/export_labelling_sheet.py logs/<run>.eval [more logs] --n 100
    python evals/judge_validation/export_labelling_sheet.py --retest evals/judge_validation/labelling/sheet.csv

Writes evals/judge_validation/labelling/sheet.csv (fill the 'label' column) and key.csv.
--retest writes sheet_retest.csv: the same items, reshuffled, labels blank (for your second pass).
"""
import argparse
import csv
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "labelling"
SHEET_COLS = ["item_id", "category", "conversation", "reply", "criterion", "label"]
KEY_COLS = ["item_id", "judge_verdict", "line_type", "points", "variant", "model", "reply_words"]


def load_items(log_paths):
    from inspect_ai.log import read_eval_log
    items = []
    for lp in log_paths:
        log = read_eval_log(lp)
        if not str(log.eval.task).startswith("clinical"):
            continue
        tag = Path(lp).stem[-6:]
        for s in log.samples or []:
            sc = next(iter(s.scores.values()), None) if s.scores else None
            if not sc or not sc.metadata or "lines" not in sc.metadata:
                continue
            conv = "\n\n".join(f"{t['role'].upper()}: {t['content']}" for t in s.metadata["turns"])
            reply = sc.answer or ""
            for i, line in enumerate(sc.metadata["lines"]):
                if line["verdict"] is None:  # unreadable judge output: nothing to compare
                    continue
                items.append({
                    "item_id": f"{tag}|{s.id}|e{s.epoch}|L{i}",
                    "category": s.metadata["category"], "conversation": conv, "reply": reply,
                    "criterion": line["criterion"], "judge_verdict": "YES" if line["verdict"] else "NO",
                    "line_type": line["type"], "points": line["points"],
                    "variant": s.metadata["variant"], "model": str(log.eval.model),
                    "reply_words": len(reply.split()),
                })
    return items


def stratified_sample(items, n, rng):
    """Spread the sample across category x judge verdict so rare outcomes are not missed."""
    strata = {}
    for it in items:
        strata.setdefault((it["category"], it["judge_verdict"]), []).append(it)
    for v in strata.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < n and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(chosen) < n:
                chosen.append(strata[k].pop())
    rng.shuffle(chosen)  # blind: order carries no information
    return chosen


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="Inspect .eval log files")
    ap.add_argument("--n", type=int, default=100, help="items to sample (plan: about 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retest", help="existing sheet.csv: write a reshuffled blank copy for the second pass")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.retest:
        with open(a.retest, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rng.shuffle(rows)
        for r in rows:
            r["label"] = ""
        out = OUT_DIR / "sheet_retest.csv"
        write_csv(out, SHEET_COLS, rows)
        print(f"Wrote {len(rows)} items to {out}. Label without looking at your first pass.")
        return

    if not a.logs:
        ap.error("give at least one log file, or --retest")
    items = load_items(a.logs)
    if not items:
        raise SystemExit("No judged rubric lines found in those logs.")
    sample = stratified_sample(items, a.n, rng)
    for r in sample:
        r["label"] = ""
    write_csv(OUT_DIR / "sheet.csv", SHEET_COLS, sample)
    write_csv(OUT_DIR / "key.csv", KEY_COLS, sample)
    print(f"Sampled {len(sample)} of {len(items)} judged lines.")
    print(f"Label the 'label' column (YES or NO) in {OUT_DIR / 'sheet.csv'} BEFORE opening key.csv.")
    if len(sample) < 100:
        print("Note: your plan wants about 100 items; fewer gives a very wide confidence interval.")


if __name__ == "__main__":
    main()
