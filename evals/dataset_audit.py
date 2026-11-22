# ruff: noqa: E501
"""Machine-verifiable integrity audit for the compiled clinical case set.

This checker deliberately reports structural and design signals only.  It never
asserts that a clinical claim, source, or rubric is medically correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("id", "category", "subcategory", "cluster_id", "variant", "turns", "source_guideline",
            "expected_safe_stance", "rubric", "severity_if_failed", "notes")


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def load_yaml_cases() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "evals" / "cases").glob("*.yaml")):
        rows.extend(yaml.safe_load(path.read_text(encoding="utf-8")) or [])
    return rows


def finding(kind: str, severity: str, ids: list[str], evidence: str, machine_verifiable: bool = True) -> dict:
    return {"kind": kind, "severity": severity, "case_ids": sorted(ids), "evidence": evidence,
            "machine_verifiable": machine_verifiable}


def audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict] = []
    ids = Counter(c.get("id") for c in cases)
    for case_id, count in ids.items():
        if count > 1:
            findings.append(finding("duplicate_id", "P0", [str(case_id)], f"id occurs {count} times"))
    for case in cases:
        missing = [key for key in REQUIRED if not case.get(key)]
        if missing:
            findings.append(finding("missing_field", "P0", [case.get("id", "?")], f"missing: {missing}"))
        for index, line in enumerate(case.get("rubric", [])):
            if line.get("type") not in {"must", "must_not"} or not isinstance(line.get("points"), int):
                findings.append(finding("invalid_rubric_line", "P0", [case.get("id", "?")], f"line {index}"))
            if line.get("type") == "must_not" and line.get("points", 0) >= 4:
                # This is an inventory signal, not a defect: it defines unsafe-content endpoints.
                pass
        if not case.get("source_guideline"):
            findings.append(finding("missing_source", "P0", [case.get("id", "?")], "no source reference"))

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_cluster[case.get("cluster_id")].append(case)
    for cluster, rows in by_cluster.items():
        base = next((r for r in rows if r.get("variant") == "base"), None)
        pressure = next((r for r in rows if r.get("variant") == "pressure"), None)
        if not base or not pressure:
            continue
        changed = [k for k in ("category", "subcategory", "expected_safe_stance", "severity_if_failed", "source_guideline")
                   if base.get(k) != pressure.get(k)]
        if changed:
            findings.append(finding("pair_metadata_change", "P1", [base["id"], pressure["id"]],
                                    f"base/pressure differ in {changed}"))
        if base["turns"][0].get("content") != pressure["turns"][0].get("content"):
            findings.append(finding("pair_initial_fact_change", "P1", [base["id"], pressure["id"]],
                                    "first user turn differs"))
        if base.get("rubric") != pressure.get("rubric"):
            findings.append(finding("pair_rubric_change", "P1", [base["id"], pressure["id"]],
                                    "rubrics differ; paired endpoint must use shared lines only or be labelled exploratory"))
        if len(pressure.get("turns", [])) > len(base.get("turns", [])):
            findings.append(finding("pair_scripted_history", "P2", [base["id"], pressure["id"]],
                                    "pressure case contains added scripted assistant/user history", False))

    # Exact prompt duplicates are defects; same-subcategory pairs are intentional related cases, not duplicates.
    prompts: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        prompts[_norm("\n".join(t.get("content", "") for t in case.get("turns", [])))].append(case["id"])
    for case_ids in prompts.values():
        if len(case_ids) > 1:
            findings.append(finding("exact_conversation_duplicate", "P1", case_ids, "normalized full turns match"))
    subcategories = defaultdict(list)
    for case in cases:
        subcategories[case.get("subcategory")].append(case["id"])
    potential = {k: sorted(v) for k, v in subcategories.items() if len(v) > 2}
    for subcategory, case_ids in potential.items():
        findings.append(finding("potential_semantic_overlap", "P2", case_ids,
                                f"shared subcategory '{subcategory}'; requires human review", False))

    compiled = ROOT / "evals" / "cases.jsonl"
    compiled_rows = [json.loads(x) for x in compiled.read_text(encoding="utf-8").splitlines() if x]
    if [c.get("id") for c in cases] != [c.get("id") for c in compiled_rows]:
        findings.append(finding("compiled_dataset_drift", "P0", [], "YAML and JSONL id order differs"))
    return {"schema_version": 1, "dataset_path": "evals/cases.jsonl", "dataset_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
            "case_count": len(cases), "rubric_line_count": sum(len(c.get("rubric", [])) for c in cases),
            "by_category": dict(sorted(Counter(c["category"] for c in cases).items())),
            "by_severity": dict(sorted(Counter(c["severity_if_failed"] for c in cases).items())),
            "findings": findings}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report = audit(load_yaml_cases())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
