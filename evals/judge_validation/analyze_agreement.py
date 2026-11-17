#!/usr/bin/env python3
"""Compare your blind human labels with the LLM judge.

Usage, from the project folder (after labelling sheet.csv):
    python evals/judge_validation/analyze_agreement.py
    python evals/judge_validation/analyze_agreement.py --retest evals/judge_validation/labelling/sheet_retest.csv

Reports: raw agreement, Cohen's kappa with a bootstrap 95% CI, Gwet's AC1, the confusion matrix,
the same per category and per rubric-line type, a length-bias check, and your own test-retest
kappa when a second labelled pass is given. Writes disagreements.csv: read every one.
"""
import argparse
import csv
import random
from pathlib import Path

DIR = Path(__file__).parent / "labelling"


def to_bool(x):
    x = (x or "").strip().upper()
    return True if x in ("YES", "Y", "TRUE", "1") else False if x in ("NO", "N", "FALSE", "0") else None


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a, b):
    n = len(a)
    po = agreement(a, b)
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(a, b):
    n = len(a)
    po = agreement(a, b)
    pi = (sum(a) / n + sum(b) / n) / 2
    pe = 2 * pi * (1 - pi)
    return None if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(a, b, stat=cohen_kappa, reps=2000, seed=0):
    rng = random.Random(seed)
    n, vals = len(a), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        v = stat([a[i] for i in idx], [b[i] for i in idx])
        if v is not None:
            vals.append(v)
    if len(vals) < reps // 2:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def confusion(h, j):
    c = {(x, y): 0 for x in (True, False) for y in (True, False)}
    for x, y in zip(h, j):
        c[(x, y)] += 1
    return c


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((v - mx) ** 2 for v in x) ** 0.5
    sy = sum((v - my) ** 2 for v in y) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def block(name, h, j, with_ci=True):
    n = len(h)
    lines = [f"{name} (n={n})"]
    if n == 0:
        return lines + ["  no items"]
    k = cohen_kappa(h, j)
    ci = bootstrap_ci(h, j) if with_ci and n >= 10 else None
    lines.append(f"  raw agreement {agreement(h, j):.3f} | kappa {fmt(k)}"
                 + (f" (95% CI {ci[0]:.2f} to {ci[1]:.2f})" if ci else " (CI needs n>=10)")
                 + f" | Gwet AC1 {fmt(gwet_ac1(h, j))}")
    return lines


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(DIR / "sheet.csv"))
    ap.add_argument("--key", default=str(DIR / "key.csv"))
    ap.add_argument("--retest", help="second labelled pass (sheet_retest.csv)")
    a = ap.parse_args()

    key = {r["item_id"]: r for r in read_rows(a.key)}
    rows = []
    for r in read_rows(a.sheet):
        lab, k = to_bool(r["label"]), key.get(r["item_id"])
        if lab is None or k is None:
            continue
        rows.append({"id": r["item_id"], "human": lab, "judge": to_bool(k["judge_verdict"]),
                     "category": r["category"], "type": k["line_type"], "words": float(k["reply_words"]),
                     "criterion": r["criterion"]})
    total = len(read_rows(a.sheet))
    if not rows:
        raise SystemExit("No labelled rows found. Fill the 'label' column with YES or NO first.")
    out = [f"Labelled items used: {len(rows)} of {total}", ""]
    h, j = [r["human"] for r in rows], [r["judge"] for r in rows]
    out += block("OVERALL: human vs judge", h, j)
    c = confusion(h, j)
    out += ["", "Confusion matrix (rows = you, columns = judge)",
            "            judge YES  judge NO", f"  you YES   {c[(True, True)]:9d}  {c[(True, False)]:8d}",
            f"  you NO    {c[(False, True)]:9d}  {c[(False, False)]:8d}", ""]
    out.append("PER CATEGORY")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        out += block(f"  {cat}", [r["human"] for r in sub], [r["judge"] for r in sub])
    out.append("\nPER RUBRIC-LINE TYPE")
    for t in ("must", "must_not"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            out += block(f"  {t}", [r["human"] for r in sub], [r["judge"] for r in sub])

    w = [r["words"] for r in rows]
    out += ["", "LENGTH BIAS (does reply length predict YES?)",
            f"  judge: correlation of YES with reply words = {fmt(pearson([1.0 * x for x in j], w))}",
            f"  you:   correlation of YES with reply words = {fmt(pearson([1.0 * x for x in h], w))}",
            "  A judge correlation clearly higher than yours suggests verbosity bias. Self-preference needs a",
            "  second judge from another family; this script cannot test it from one judge."]

    if a.retest:
        second = {r["item_id"]: to_bool(r["label"]) for r in read_rows(a.retest)}
        pairs = [(r["human"], second[r["id"]]) for r in rows if second.get(r["id"]) is not None]
        out.append("")
        if pairs:
            out += block("YOUR TEST-RETEST (pass 1 vs pass 2)", [p[0] for p in pairs], [p[1] for p in pairs])
        else:
            out.append("YOUR TEST-RETEST: no labelled rows in the retest sheet yet.")

    dis = [r for r in rows if r["human"] != r["judge"]]
    with open(DIR / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["item_id", "category", "criterion", "you", "judge", "why_they_differ"])
        for r in dis:
            wr.writerow([r["id"], r["category"], r["criterion"], "YES" if r["human"] else "NO",
                         "YES" if r["judge"] else "NO", ""])
    out += ["", f"Disagreements: {len(dis)}. Read each one and fill 'why_they_differ' in {DIR / 'disagreements.csv'}.",
            "Context for reading kappa: no universal cut-off; 0.6 is a common working bar, 0.8 strong,",
            "to be judged against how well humans agree with each other."]
    text = "\n".join(out)
    print(text)
    (DIR / "agreement_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
