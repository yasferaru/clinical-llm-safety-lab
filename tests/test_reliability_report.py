"""scripts/reliability_report.py: markdown structure, k capped at the available trials."""
from scripts.reliability_report import reliability_markdown
from tests.fixtures import make_samples
from tests.test_reliability import README_RUNS


def test_readme_runs_report():
    md = reliability_markdown(README_RUNS, ks=(1, 2, 4, 8), B=200)
    rows = [ln for ln in md.splitlines() if ln.startswith("| `m` |")]
    assert len(rows) == 2 + 1                     # k = 1, 2 and one consistency row
    assert "| `m` | 2 | 4 | 2 | 2 | 0.25 (" in md  # pass^2 = 0.25 over 4 cases, 2 scenarios, 2 trials
    assert "Not shown: k = 4, 8" in md
    assert "2 of 4 (50%)" in md
    assert "critical_pass == 1" in md
    assert "Warning: fewer than 5 scenarios for `m`" in md   # 2 scenarios only


def test_fixture_report_has_all_ks_and_is_seeded():
    s = make_samples(seed=1, epochs=8)
    md = reliability_markdown(s, B=100, seed=0)
    for k in (1, 2, 4, 8):
        assert f"| `model-a` | {k} |" in md and f"| `model-b` | {k} |" in md
    assert "Not shown" not in md and "Warning" not in md      # 12 scenarios per model
    assert md == reliability_markdown(s, B=100, seed=0)


def test_empty_input():
    md = reliability_markdown(README_RUNS.iloc[0:0], B=10)
    assert "(no scored samples)" in md and "(no case has 2 or more trials)" in md
