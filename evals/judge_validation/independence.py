"""Judge independence: a judge must not grade models from its own family.

    assert_judge_independent(judge_model, models_under_test) -> judge family, or raises.

Families come from config/model_families.yaml. Exposed for run_eval (P10) and called by the audit and
jury tooling. Unknown or ambiguous models raise too: an unmapped model cannot be shown independent.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import yaml

FAMILIES_PATH = Path(__file__).resolve().parents[2] / "config" / "model_families.yaml"


class JudgeIndependenceError(ValueError):
    """The judge shares a family with a model under test, or a model's family is unknown."""


@lru_cache(maxsize=4)
def load_families(path: str = str(FAMILIES_PATH)) -> dict[str, list[str]]:
    """{family: [regex, ...]} from the YAML config."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["families"]


def model_id(model: str) -> str:
    """Model string without its first provider segment: 'groq/openai/gpt-oss-120b' -> 'openai/gpt-oss-120b'."""
    return model.split("/", 1)[1] if "/" in model else model


def family_of(model: str, families: Mapping[str, list[str]] | None = None) -> str:
    """The single family whose patterns match model_id(model); JudgeIndependenceError if none or several."""
    fams = families if families is not None else load_families()
    mid = model_id(model)
    hits = [f for f, pats in fams.items() if any(re.search(p, mid, flags=re.I) for p in pats)]
    if len(hits) != 1:
        what = "no family" if not hits else f"several families ({', '.join(hits)})"
        raise JudgeIndependenceError(f"model '{model}' matches {what}; add or fix it in config/model_families.yaml")
    return hits[0]


def assert_judge_independent(judge_model: str, models_under_test: Iterable[str],
                             families: Mapping[str, list[str]] | None = None) -> str:
    """Raise JudgeIndependenceError if the judge's family equals any tested model's family; else return it.

    Example: judge 'groq/openai/gpt-oss-120b' (openai) vs ['ollama/qwen3:8b', 'mistral/mistral-small-latest']
    -> 'openai'; judge 'ollama/qwen3:14b' vs ['ollama/qwen3:8b'] -> raises (qwen grading qwen).
    """
    judge_family = family_of(judge_model, families)
    clashes = [m for m in models_under_test if family_of(m, families) == judge_family]
    if clashes:
        raise JudgeIndependenceError(
            f"judge '{judge_model}' is in family '{judge_family}', the same as model(s) under test: "
            f"{', '.join(clashes)}. A judge may not grade its own family.")
    return judge_family
