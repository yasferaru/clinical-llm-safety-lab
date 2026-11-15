#!/usr/bin/env python3
"""Check hand-written eval cases and optionally compile them to evals/cases.jsonl.

Run from the project folder, with the Python workspace active:
    python check_cases.py            # check only
    python check_cases.py --build    # check, then write evals/cases.jsonl if there are no errors
Files starting with an underscore in evals/cases/ are ignored.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

CASES_DIR = Path("evals/cases")
OUT = Path("evals/cases.jsonl")

CATEGORIES = ["emergency_triage", "dosing_interactions", "context_seeking", "sycophancy"]
VARIANTS = ["base", "pressure", "control"]
STANCES = ["urgent_escalation", "answer_directly", "ask_for_more_info",
           "hold_position_on_false_claim", "state_dose_with_limits"]
SEVERITIES = ["none", "misleading", "delays_care", "could_cause_harm", "contraindicated_dosing"]
RUBRIC_TYPES = ["must", "must_not"]
FIELDS = ["id", "category", "subcategory", "cluster_id", "variant", "turns", "source_guideline",
          "expected_safe_stance", "rubric", "severity_if_failed", "notes"]
TARGET = {"base": 4, "pressure": 4, "control": 2}  # per category
VAGUE = [r"\be\.g\.", r"\bi\.e\.", r"\betc\b", r"\bappropriate(ly)?\b", r"\bproperly\b"]


def nonempty_str(x):
    return isinstance(x, str) and x.strip() != ""


def check_record(rec, errors, warnings):
    """Append problems with one record to errors / warnings."""
    rid = rec.get("id", "?") if isinstance(rec, dict) else "?"

    def err(msg):
        errors.append(f"{rid}: {msg}")

    def warn(msg):
        warnings.append(f"{rid}: {msg}")

    if not isinstance(rec, dict):
        errors.append("a record is not a labelled block (check the leading '- ' and indentation)")
        return
    for f in FIELDS:
        if f not in rec or rec[f] in (None, "", []):
            err(f"missing or empty field '{f}'")
    for f, allowed in (("category", CATEGORIES), ("variant", VARIANTS),
                       ("expected_safe_stance", STANCES), ("severity_if_failed", SEVERITIES)):
        if rec.get(f) not in allowed and rec.get(f) not in (None, ""):
            err(f"'{f}' is '{rec.get(f)}', must be one of: {', '.join(allowed)}")
    for f in ("id", "subcategory", "cluster_id", "notes"):
        if f in rec and not nonempty_str(rec[f]):
            err(f"'{f}' must be text")

    # id convention
    cid, var = rec.get("cluster_id"), rec.get("variant")
    if nonempty_str(rec.get("id")) and nonempty_str(cid) and var in VARIANTS:
        expected = cid if var == "control" else f"{cid}-{var}"
        if rec["id"] != expected:
            warn(f"id is usually '{expected}'")

    # turns
    turns = rec.get("turns")
    if isinstance(turns, list) and turns:
        roles = []
        for i, t in enumerate(turns, 1):
            if not isinstance(t, dict) or t.get("role") not in ("user", "assistant") or not nonempty_str(t.get("content")):
                err(f"turn {i} needs 'role' (user or assistant) and non-empty 'content'")
            else:
                roles.append(t["role"])
        if len(roles) == len(turns):
            if roles[0] != "user" or roles[-1] != "user":
                err("the conversation must start and end with a user turn")
            if any(a == b for a, b in zip(roles, roles[1:])):
                err("turns must alternate user / assistant")
            if var in ("base", "control") and len(turns) != 1:
                err(f"a {var} record has exactly one user turn")
            if var == "pressure" and len(turns) < 3:
                err("a pressure record needs user turn, scripted assistant turn, user pushback (3 turns)")
    elif "turns" in rec:
        err("'turns' must be a list")

    # sources
    src = rec.get("source_guideline")
    if isinstance(src, list) and src:
        for i, s in enumerate(src, 1):
            if not isinstance(s, dict):
                err(f"source {i} is not a labelled block")
                continue
            for k in ("title", "url", "opened", "supports"):
                if not nonempty_str(str(s.get(k, "")).strip()):
                    err(f"source {i} is missing '{k}'")
            if nonempty_str(s.get("url")) and not s["url"].startswith("http"):
                err(f"source {i} url must start with http")
            if "opened" in s and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s["opened"])):
                err(f"source {i} 'opened' must look like 2026-11-14")
    elif "source_guideline" in rec:
        err("'source_guideline' must be a list with at least one source")

    # rubric
    rub = rec.get("rubric")
    if isinstance(rub, list) and rub:
        n_must = n_not = 0
        for i, r in enumerate(rub, 1):
            if not isinstance(r, dict):
                err(f"rubric line {i} is not a labelled block")
                continue
            if not nonempty_str(r.get("criterion")):
                err(f"rubric line {i} has no 'criterion' text")
            else:
                for pat in VAGUE:
                    if re.search(pat, r["criterion"], re.I):
                        warn(f"rubric line {i} contains vague wording ({pat.strip(chr(92)+'b')}); name the exact behaviour")
                        break
            p = r.get("points")
            if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 5:
                err(f"rubric line {i} 'points' must be a whole number 1-5")
            if r.get("type") not in RUBRIC_TYPES:
                err(f"rubric line {i} 'type' must be must or must_not")
            n_must += r.get("type") == "must"
            n_not += r.get("type") == "must_not"
        if n_must < 1:
            err("needs at least one 'must' line")
        if n_must + n_not > 6:
            warn("more than 6 rubric lines; keep it to what the source supports")
        if n_not < 1:
            warn("no 'must_not' lines")
    elif "rubric" in rec:
        err("'rubric' must be a list")


def main(build=False):
    errors, warnings, records = [], [], []
    files = sorted(p for p in CASES_DIR.glob("*.yaml") if not p.name.startswith("_"))
    if not files:
        print(f"No case files found in {CASES_DIR}/")
        return 1
    for path in files:
        text = path.read_text(encoding="utf-8")
        e, w = [], []
        for n, line in enumerate(text.splitlines(), 1):
            if "TODO" in line:
                w.append(f"line {n}: unresolved TODO")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            errors.append(f"{path.name}: cannot read the file ({str(exc).splitlines()[0]}); usually a quote or indentation slip")
            continue
        if not isinstance(data, list):
            errors.append(f"{path.name}: file must be a list of records, each starting with '- '")
            continue
        for rec in data:
            check_record(rec, e, w)
            records.append(rec)
        errors += [f"{path.name} / {m}" for m in e]
        warnings += [f"{path.name} / {m}" for m in w]

    ids = Counter(r.get("id") for r in records if isinstance(r, dict))
    for i, c in ids.items():
        if c > 1:
            errors.append(f"duplicate id '{i}' appears {c} times")
    clusters = {}
    for r in records:
        if isinstance(r, dict):
            clusters.setdefault(r.get("cluster_id"), set()).add(r.get("variant"))
    for cid, vs in clusters.items():
        if "pressure" in vs and "base" not in vs:
            warnings.append(f"cluster '{cid}' has a pressure record but no base record")
        if "base" in vs and "pressure" not in vs:
            warnings.append(f"cluster '{cid}' has a base record but no pressure record")

    print("\nERRORS (fix these):" if errors else "\nNo errors.")
    for m in errors:
        print("  x", m)
    print("\nWARNINGS (worth fixing):" if warnings else "\nNo warnings.")
    for m in warnings:
        print("  !", m)

    print("\nProgress toward 40 (done/target):")
    counts = Counter((r.get("category"), r.get("variant")) for r in records if isinstance(r, dict))
    total = 0
    for cat in CATEGORIES:
        parts = []
        for v in VARIANTS:
            parts.append(f"{v} {counts[(cat, v)]}/{TARGET[v]}")
            total += counts[(cat, v)]
        print(f"  {cat:20s} " + "   ".join(parts))
    print(f"  total records: {total}/40")

    if build and not errors:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False, default=str) + "\n")
        print(f"\nWrote {len(records)} records to {OUT}")
    elif build:
        print("\nNot built: fix the errors first.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(build="--build" in sys.argv))
