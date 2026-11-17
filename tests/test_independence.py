"""Judge-family independence."""
import pytest

from evals.judge_validation.independence import (JudgeIndependenceError, assert_judge_independent, family_of,
                                                 model_id)


@pytest.mark.parametrize("model,family", [
    ("groq/openai/gpt-oss-120b", "openai"), ("openai/gpt-4o", "openai"), ("openai/o3-mini", "openai"),
    ("ollama/qwen3:8b", "qwen"), ("ollama/qwq:32b", "qwen"),
    ("mistral/mistral-small-latest", "mistral"), ("ollama/mixtral:8x7b", "mistral"),
    ("anthropic/claude-sonnet-5-5", "anthropic"), ("google/gemini-2.5-pro", "google"),
    ("ollama/gemma3:4b", "google"), ("ollama/llama3.1:8b", "meta"), ("ollama/phi4", "microsoft"),
    ("ollama/deepseek-r1:8b", "deepseek"),
])
def test_family_of(model, family):
    assert family_of(model) == family


def test_provider_prefix_is_ignored():
    assert model_id("groq/openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert family_of("ollama/qwen3:8b") == "qwen"      # 'ollama' itself matches no family


def test_current_setup_is_independent():
    assert assert_judge_independent("groq/openai/gpt-oss-120b",
                                    ["ollama/qwen3:8b", "mistral/mistral-small-latest"]) == "openai"


def test_same_family_fails_loudly():
    # the README's excluded plumbing run: qwen3:8b judged by itself
    with pytest.raises(JudgeIndependenceError, match="own family"):
        assert_judge_independent("ollama/qwen3:8b", ["ollama/qwen3:8b"])
    with pytest.raises(JudgeIndependenceError, match="mistral/mistral-small-latest"):
        assert_judge_independent("mistral/mistral-large-latest", ["ollama/qwen3:8b", "mistral/mistral-small-latest"])


def test_unknown_or_ambiguous_model_fails_loudly():
    with pytest.raises(JudgeIndependenceError, match="no family"):
        family_of("ollama/some-new-model:7b")
    with pytest.raises(JudgeIndependenceError, match="several families"):
        family_of("x/llama-qwen-merge", {"meta": [r"llama"], "qwen": [r"qwen"]})
    with pytest.raises(JudgeIndependenceError):
        assert_judge_independent("groq/openai/gpt-oss-120b", ["ollama/unknown-thing"])
