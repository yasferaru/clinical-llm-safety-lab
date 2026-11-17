"""evals/report.py end to end on synthetic fixtures (the CI smoke test runs the same thing via the CLI)."""
import re
from dataclasses import replace

import pandas as pd
import pytest

from evals.report import ReportInputs, build_report, load_inputs, main, write_report
from evals.results import LINE_COLUMNS, SAMPLE_COLUMNS
from tests.fixtures import make_report_inputs

SECTIONS = ["1. Overview", "2. Per-model scores", "3. Paired model comparisons", "4. Reliability across epochs",
            "5. Triage acuity", "6. Red-team", "7. Sycophancy ladders", "8. Abstention", "9. Judge bias audit",
            "10. Judge-corrected", "11. Uncertainty (EXPLORATORY)"]


@pytest.fixture(scope="module")
def synthetic():
    inp = make_report_inputs(seed=0)
    md, tables = build_report(inp, B=100, seed=0)
    return inp, md, tables


def test_every_section_renders_with_data(synthetic):
    _, md, tables = synthetic
    for s in SECTIONS:
        assert f"## {s}" in md, s
    assert "*Not available" not in md                       # the fixtures exercise every section
    assert "nan" not in md.lower().replace("nan%", "NAN!")  # no NaN leaks (and no 'nan%' either)
    assert "NAN!" not in md
    assert {"02_models", "03_comparisons", "04_reliability", "05_triage", "05_format_sensitivity", "06_redteam_rates",
            "07_flip_rate", "07_time_to_capitulation", "08_abstention", "09_bias_audit", "09_jury_agreement",
            "10_corrected_critical_pass"} <= set(tables)


def test_every_rate_has_denominator_and_named_interval(synthetic):
    _, md, _ = synthetic
    for sec in ("## 2.", "## 5.", "## 8."):
        body = md.split(sec)[1].split("\n## ")[0]
        rate_rows = [ln for ln in body.splitlines() if ln.startswith("| ") and re.search(r"\d+/\d+ = \d+%", ln)]
        assert rate_rows, sec
        for line in rate_rows:
            # every 'k/n = x%' is followed by its named 95% interval
            bare = re.sub(r"\d+/\d+ = \d+% \(95% (Clopper-Pearson|Wilson) \d+%–\d+%\)", "", line)
            assert not re.search(r"\d+/\d+ = \d+%", bare), line


def test_banner_names_unvalidated_models(synthetic):
    _, md, _ = synthetic
    overview = md.split("## 1. Overview")[1].split("\n## ")[0]
    assert "JUDGE NOT VALIDATED" in overview and "mistral/mistral-small-latest" in overview
    assert "`ollama/qwen3:8b`:" not in overview             # qwen has enough calibration labels in the fixture


def test_runs_with_a_dependent_judge_are_excluded(synthetic):
    inp, _, _ = synthetic
    bad = inp.samples.copy()
    bad.loc[bad["model"] == "ollama/qwen3:8b", "judge_model"] = "ollama/qwen3:32b"
    md, tables = build_report(replace(inp, samples=bad), B=50)
    assert "Excluded 1 run(s)" in md and "01_excluded_runs" in tables
    assert "| ollama/qwen3:8b | 38 |" not in md.split("## 2.")[1].split("\n## ")[0]


def test_empty_inputs_still_produce_a_report(tmp_path):
    empty = ReportInputs(samples=pd.DataFrame(columns=SAMPLE_COLUMNS), lines=pd.DataFrame(columns=LINE_COLUMNS))
    md, tables = build_report(empty, B=10)
    assert md.count("*Not available") >= 9 and "## 11." not in md   # uncertainty only with P8 outputs
    md2, _ = build_report(load_inputs(tmp_path), B=10)               # an empty run directory
    assert "## 2. Per-model scores" in md2


def test_few_scenarios_suppress_cluster_intervals(synthetic):
    inp, _, _ = synthetic
    s = inp.samples[inp.samples["scenario_id"].isin(["scn-000", "scn-001"])]
    md, _ = build_report(replace(inp, samples=s, lines=inp.lines[inp.lines["scenario_id"].isin(["scn-000", "scn-001"])],
                                 uncertainty=None), B=50)
    assert "CI n/a, fewer than 5 scenarios" in md and "CI n/a, < 5 scenarios" in md


def test_cli_synthetic_writes_markdown_and_csvs(tmp_path):
    main(["--synthetic", str(tmp_path / "out"), "--B", "50"])
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Clinical LLM safety report")
    assert len(list((tmp_path / "out").glob("*.csv"))) >= 15
    p = write_report("# x\n", {"a": pd.DataFrame({"x": [1]})}, tmp_path / "w")
    assert p.exists() and (tmp_path / "w" / "a.csv").exists()
