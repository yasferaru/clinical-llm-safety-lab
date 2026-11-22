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

import numpy as np

DIR = Path(__file__).parent / "labelling"


def to_bool(x):
    x = (x or "").strip().upper()
    return True if x in ("YES", "Y", "TRUE", "1") else False if x in ("NO", "N", "FALSE", "0") else None


def _check_pair(a, b):
    """Raise ValueError unless a and b are equal-length sequences of bools (None is not a label)."""
    if len(a) != len(b):
        raise ValueError(f"rater lists differ in length: {len(a)} vs {len(b)}")
    for x in (*a, *b):
        if not isinstance(x, (bool, np.bool_)):
            raise ValueError(f"labels must be True/False, got {x!r}; drop unlabelled items first")


def agreement(a, b):
    """Raw agreement po = share of items where both raters give the same label; None if n = 0."""
    _check_pair(a, b)
    return sum(x == y for x, y in zip(a, b)) / len(a) if len(a) else None


def cohen_kappa(a, b):
    """Cohen's kappa = (po - pe) / (1 - pe), pe = pa*pb + (1-pa)(1-pb) with pa, pb each rater's YES share.

    None when undefined: n = 0, or pe = 1 (both raters constant on the same label).
    Example: [T,T,F,F] vs [T,F,F,F]: po = 3/4, pe = 1/2 * 1/4 + 1/2 * 3/4 = 1/2 -> kappa = 0.5.
    """
    po = agreement(a, b)
    if po is None:
        return None
    n = len(a)
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(a, b):
    """Gwet's AC1 = (po - pe) / (1 - pe), pe = 2*pi*(1 - pi) with pi the mean YES share of the two raters.

    pe <= 0.5, so AC1 is defined whenever n > 0 (None if n = 0). Robust to skewed prevalence.
    Example: [T,T,F,F] vs [T,F,F,F]: pi = 3/8, pe = 15/32 -> AC1 = (3/4 - 15/32) / (17/32) = 9/17.
    """
    po = agreement(a, b)
    if po is None:
        return None
    n = len(a)
    pi = (sum(a) / n + sum(b) / n) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def pabak(a, b):
    """Prevalence- and bias-adjusted kappa = 2*po - 1 (None if n = 0)."""
    po = agreement(a, b)
    return None if po is None else 2 * po - 1


def prevalence_bias(a, b):
    """(prevalence index, bias index) = (|p_YY - p_NN|, |p_YN - p_NY|) as shares of n; (None, None) if n = 0.

    A high prevalence index with high po but low kappa is the kappa paradox (Feinstein & Cicchetti 1990).
    """
    _check_pair(a, b)
    n = len(a)
    if n == 0:
        return None, None
    yy = sum(x and y for x, y in zip(a, b))
    nn = sum((not x) and (not y) for x, y in zip(a, b))
    yn = sum(x and not y for x, y in zip(a, b))
    ny = sum((not x) and y for x, y in zip(a, b))
    return abs(yy - nn) / n, abs(yn - ny) / n


def kappa_paradox(a, b, po_min=0.8, kappa_max=0.4):
    """True when raw agreement is high (>= po_min) but kappa is low (<= kappa_max) or undefined."""
    po, k = agreement(a, b), cohen_kappa(a, b)
    return po is not None and po >= po_min and (k is None or k <= kappa_max)


def bootstrap_ci(a, b, stat=cohen_kappa, reps=2000, seed=0, groups=None, level=0.95):
    """Percentile bootstrap CI of stat(a, b); None if fewer than half the replicates are defined.

    groups: optional cluster label per item (e.g. the reply a rubric line belongs to). When given, whole
    clusters are resampled with replacement, because lines of one reply are not independent. Without it,
    items are resampled. Quantiles use numpy's linear interpolation. Replicates where stat is undefined
    (e.g. kappa on a constant resample) are dropped, which biases the CI; prefer AC1 when that happens often.
    """
    _check_pair(a, b)
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(seed)
    if groups is None:
        clusters = [[i] for i in range(n)]
    else:
        if len(groups) != n:
            raise ValueError("groups must have one label per item")
        by = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        clusters = list(by.values())
    g = len(clusters)
    vals = []
    for _ in range(reps):
        idx = [i for _ in range(g) for i in clusters[rng.randrange(g)]]
        v = stat([a[i] for i in idx], [b[i] for i in idx])
        if v is not None:
            vals.append(v)
    if len(vals) < reps // 2:
        return None
    alpha = 1 - level
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


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


def block(name, h, j, with_ci=True, groups=None):
    """Report lines for one comparison: po, kappa (cluster-bootstrap CI), AC1, PABAK, prevalence, paradox flag."""
    n = len(h)
    lines = [f"{name} (n={n})"]
    if n == 0:
        return lines + ["  no items"]
    k = cohen_kappa(h, j)
    ci = bootstrap_ci(h, j, groups=groups) if with_ci and n >= 10 else None
    prev, bias = prevalence_bias(h, j)
    lines.append(f"  raw agreement {agreement(h, j):.3f} | kappa {fmt(k)}"
                 + (f" (95% CI {ci[0]:.2f} to {ci[1]:.2f}{', clustered by reply' if groups else ''})"
                    if ci else " (CI needs n>=10)")
                 + f" | Gwet AC1 {fmt(gwet_ac1(h, j))} | PABAK {fmt(pabak(h, j))}"
                 + f" | prevalence index {prev:.2f}, bias index {bias:.2f}")
    if kappa_paradox(h, j):
        lines.append("  ! kappa paradox: high raw agreement but low kappa, driven by skewed prevalence;"
                     " read AC1 / PABAK alongside kappa")
    return lines


def reply_of(item_id):
    """Cluster label of a sheet item: 'tag|sample|eN|Lk' -> 'tag|sample|eN' (the judged reply)."""
    return item_id.rsplit("|", 1)[0]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(DIR / "sheet.csv"))
    ap.add_argument("--key", default=str(DIR / "key.csv"))
    ap.add_argument("--retest", help="second labelled pass (sheet_retest.csv)")
    ap.add_argument("--out", help="folder for disagreements.csv and agreement_report.txt (default: the sheet's folder)")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else Path(a.sheet).parent
    out_dir.mkdir(parents=True, exist_ok=True)

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
    out += block("OVERALL: human vs judge", h, j, groups=[reply_of(r["id"]) for r in rows])
    c = confusion(h, j)
    out += ["", "Confusion matrix (rows = you, columns = judge)",
            "            judge YES  judge NO", f"  you YES   {c[(True, True)]:9d}  {c[(True, False)]:8d}",
            f"  you NO    {c[(False, True)]:9d}  {c[(False, False)]:8d}", ""]
    out.append("PER CATEGORY")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        out += block(f"  {cat}", [r["human"] for r in sub], [r["judge"] for r in sub],
                     groups=[reply_of(r["id"]) for r in sub])
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
    with open(out_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["item_id", "category", "criterion", "you", "judge", "why_they_differ"])
        for r in dis:
            wr.writerow([r["id"], r["category"], r["criterion"], "YES" if r["human"] else "NO",
                         "YES" if r["judge"] else "NO", ""])
    out += ["", f"Disagreements: {len(dis)}. Read each one and fill 'why_they_differ' in "
            f"{out_dir / 'disagreements.csv'}.",
            "Context for reading kappa: no universal cut-off; 0.6 is a common working bar, 0.8 strong,",
            "to be judged against how well humans agree with each other."]
    text = "\n".join(out)
    print(text)
    (out_dir / "agreement_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
