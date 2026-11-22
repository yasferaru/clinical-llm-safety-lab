from pathlib import Path

import yaml

from evals.judge_validation.independence import assert_judge_independent, family_of


def test_fixed_matrix_has_six_independent_target_judge_cells():
    config = yaml.safe_load(Path("config/benchmark_budget_v1.yaml").read_text())
    targets = [row["id"] for row in config["targets"]]
    judges = [row["id"] for row in config["judges"]]
    assert len(targets) == 3 and len(judges) == 3
    assert {row["provider"] for row in config["targets"]} == {"anthropic", "google", "deepseek"}
    assert "google/gemini-3.5-flash-lite" in targets
    assert "mistral" not in {row["provider"] for row in config["targets"] + config["judges"]}
    assert "together" not in {row["provider"] for row in config["targets"] + config["judges"]}
    pairs = [(target, judge) for target in targets for judge in config["judge_assignments"][target]]
    assert len(pairs) == 6
    for target, judge in pairs:
        assert family_of(target) != family_of(judge)
        assert_judge_independent(judge, [target])
