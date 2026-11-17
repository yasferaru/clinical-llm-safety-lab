"""Seeded synthetic results tables that follow the evals/results.py contract (no logs, no network).

    make_lines(seed=0, ...)   -> line table (results.LINE_COLUMNS)
    make_samples(seed=0, ...) -> sample table (results.SAMPLE_COLUMNS), derived from make_lines

Generative model (logit scale), per model m, scenario s, variant v, epoch e, rubric line l:
    eta = skill[m] - difficulty[s] - pressure_penalty * [v == pressure] - 0.5 * critical[l] + noise
    P(judged_pass) = sigmoid(eta); a judge error replaces the verdict with probability judge_error_rate.
Scenarios have a base and a pressure variant sharing scenario_id; benign controls are separate
scenarios with variant 'benign'. Same seed -> identical tables.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from evals.results import lines_frame, samples_from_lines

CATEGORIES = ["emergency_triage", "dosing_interactions", "context_seeking", "sycophancy"]

# (line_type, points) templates
CLINICAL_RUBRIC = [("must", 5), ("must", 2), ("must_not", 5), ("must_not", 4), ("must_not", 2)]
BENIGN_RUBRIC = [("must", 5), ("must", 2), ("must_not", 4)]   # answers directly / gives detail / refuses


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_lines(seed: int = 0, models: dict[str, float] | None = None, n_scenarios: int = 8,
               n_benign: int = 4, epochs: int = 3, pressure_penalty: float = 1.0,
               judge_error_rate: float = 0.02, judge_model: str = "judge/other-family") -> pd.DataFrame:
    """Synthetic line table. `models` maps model name -> skill (logit); defaults to two models."""
    rng = np.random.default_rng(seed)
    models = models or {"model-a": 1.5, "model-b": 0.5}
    difficulty = rng.normal(0.0, 1.0, n_scenarios + n_benign)
    cases = []
    for s in range(n_scenarios):
        cat = CATEGORIES[s % len(CATEGORIES)]
        for v in ("base", "pressure"):
            cases.append((f"scn-{s:03d}", f"scn-{s:03d}-{v}", v, cat, 1 if v == "base" else 2, CLINICAL_RUBRIC, s))
    for b in range(n_benign):
        cat = CATEGORIES[b % len(CATEGORIES)]
        cases.append((f"ctl-{b:03d}", f"ctl-{b:03d}", "benign", cat, 1, BENIGN_RUBRIC, n_scenarios + b))
    rows = []
    for m_i, (model, skill) in enumerate(models.items()):
        run_id = f"run-{seed}-{m_i}"
        for scenario_id, case_id, variant, cat, turn, rubric, d_i in cases:
            for epoch in range(1, epochs + 1):
                for i, (ltype, pts) in enumerate(rubric):
                    critical = pts >= 4
                    eta = (skill - difficulty[d_i] - pressure_penalty * (variant == "pressure")
                           - 0.5 * critical + rng.normal(0.0, 0.5))
                    error = bool(rng.random() < judge_error_rate)
                    judged = None if error else bool(rng.random() < _sigmoid(eta))
                    rows.append({"run_id": run_id, "model": model, "model_version": f"{model}@fixture",
                                 "judge_model": judge_model, "case_id": case_id, "scenario_id": scenario_id,
                                 "variant": variant, "epoch": epoch, "turn": turn, "category": cat,
                                 "rubric_line_id": f"{case_id}#L{i}", "line_index": i, "line_type": ltype,
                                 "weight": pts if ltype == "must" else -pts, "critical": critical,
                                 "judged_pass": judged, "judge_error": error})
    return lines_frame(rows)


def make_samples(seed: int = 0, **kwargs) -> pd.DataFrame:
    """Synthetic sample table, scored from make_lines(seed, **kwargs) with the real scoring rules."""
    return samples_from_lines(make_lines(seed, **kwargs))
