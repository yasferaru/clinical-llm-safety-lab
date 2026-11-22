#!/usr/bin/env python3
"""Generate an offline engineering-completion report from resolved smoke artifacts."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.matrix import read_artifact  # noqa: E402


def main() -> None:
    selection_path = ROOT / "matrix-runs/smoke-three-target-rejudge-v4/meta/resolved_selection.json"
    selection = json.loads(selection_path.read_text())
    rows = selection["selected"]
    response_root = ROOT / "matrix-runs/smoke-three-target-xai-35-lite-v3/responses"
    responses = {artifact["cell_id"]: artifact for path in response_root.glob("*.json")
                 if (artifact := read_artifact(path))}
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["response_cell_id"], row["line_index"])].append(row)
    disagreements = []
    for (response_id, line_index), paired in sorted(groups.items()):
        if len(paired) == 2 and paired[0]["verdict"] != paired[1]["verdict"]:
            response = responses[response_id]
            disagreements.append({"case_id": response["case"]["id"], "target": response["requested_model"],
                                  "response_cell_id": response_id, "line_index": line_index,
                                  "criterion": response["case"]["rubric"][line_index]["criterion"],
                                  "judgments": [{"judge": row["judge"], "verdict": row["verdict"],
                                                 "source": row["source"]} for row in paired]})
    ledger = json.loads((ROOT / "matrix-runs/smoke_budget_ledger.json").read_text())
    spend = sum(float(entry.get("usd", 0)) for entry in ledger["entries"] if entry.get("state") != "released")
    body = ["# Smoke acceptance report", "", "## Engineering completion", "",
            f"- Selected valid judgments: **{len(rows)}/114**",
            f"- Unresolved logical cells: **{selection['unresolved_count']}**",
            f"- Paired-valid rubric cells: **{selection['paired_coverage']['paired_valid_cells']}/57**",
            f"- Paired-judge disagreements: **{selection['paired_coverage']['disagreements']}/57**",
            f"- Conservative recorded smoke accounting: **${spend:.9f}** of **$0.60**", "",
            "The engineering matrix is complete. This is not clinical validation and does not establish",
            "real-world safety.",
            "", "## Protocol and interpretation limits", "",
            "- The selected set mixes 92 v3 judgments with 21 v4 and one v6 strict-schema judgments.",
            "- Judge disagreement is descriptive; it is not a calibrated clinical-adjudication statistic.",
            "- The uncertain v5 reservation remains in the ledger conservatively and is not selected.",
            "- Do not pool target-specific secondary judges as a calibrated common jury.", "",
            "## Paired-judge disagreements", ""]
    for item in disagreements:
        votes = "; ".join(f"{vote['judge']}={vote['verdict']} ({vote['source']})" for vote in item["judgments"])
        body.append(f"- `{item['case_id']}` / target `{item['target']}` / line {item['line_index']}: "
                    f"{item['criterion']} — {votes}")
    out = selection_path.parent / "smoke_acceptance_report.md"
    out.write_text("\n".join(body) + "\n")
    detail = selection_path.parent / "paired_judge_disagreements.json"
    detail.write_text(json.dumps(disagreements, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(out), "disagreements": str(detail), "count": len(disagreements)}, sort_keys=True))


if __name__ == "__main__":
    main()
