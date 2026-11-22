 # ruff: noqa: E501, E702
import asyncio

import httpx
import pytest

from evals.providers import (VERDICT_FIRST_PROMPT, VERDICT_SCHEMA, ProviderAdapter, ProviderError, ProviderExecutionPolicy,
                             credential_diagnostics, judge_call, normalize_credential, target_call)


async def ok_transport(method, url, headers, payload):
    if "anthropic" in url:
        return 200, {"request-id": "a"}, {"model": payload["model"], "content": [{"text": "answer"}], "usage": {"input_tokens": 1}}
    if "google" in url:
        return 200, {"x-request-id": "g"}, {"candidates": [{"content": {"parts": [{"text": "answer"}]}}], "usageMetadata": {"promptTokenCount": 1}}
    return 200, {"x-request-id": "o"}, {"model": payload["model"], "choices": [{"message": {"content": "VERDICT: YES"}}], "usage": {"prompt_tokens": 1}}


@pytest.mark.parametrize("provider,key", [("anthropic", "ANTHROPIC_API_KEY"), ("google", "GEMINI_API_KEY"), ("mistral", "MISTRAL_API_KEY"), ("groq", "GROQ_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY"), ("xai", "XAI_API_KEY")])
def test_all_provider_payloads_and_metadata(monkeypatch, provider, key):
    monkeypatch.setenv(key, "test")
    adapter = ProviderAdapter(provider, ok_transport)
    result = asyncio.run(adapter.request("model", [{"role": "user", "content": "x"}], structured=True))
    assert result["response"] and result["request_id"] and result["usage"]


def test_missing_credential_and_nonretryable_model(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="missing"):
        asyncio.run(ProviderAdapter("groq", ok_transport).request("x", []))
    async def unavailable(*args): return 404, {}, {"error": {"message": "model unavailable"}}
    monkeypatch.setenv("GROQ_API_KEY", "test")
    with pytest.raises(ProviderError, match="404"):
        asyncio.run(ProviderAdapter("groq", unavailable).request("x", []))


def test_credential_normalization_and_nonsecret_override_diagnostic(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  'test-key'  ")
    assert normalize_credential("  'test-key'  ") == "test-key"
    assert credential_diagnostics("groq")["needs_normalization"] is True


def test_rate_limit_retry_and_judge_verdict(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    calls = 0
    async def limited(*args):
        nonlocal calls; calls += 1
        return (429, {}, {"error": "slow"}) if calls == 1 else await ok_transport(*args)
    adapter = ProviderAdapter("groq", limited, retries=1)
    result = asyncio.run(judge_call(adapter)("c", "r", "criterion", {"model": "groq/openai/gpt-oss-120b"}))
    assert result["verdict"] is True and result["attempts"] == 2


def test_target_reuses_case_turns(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    result = asyncio.run(target_call(ProviderAdapter("deepseek", ok_transport))({"turns": [{"role": "user", "content": "u"}]}, {"model": "deepseek/deepseek-flash"}))
    assert result["response"] == "VERDICT: YES"


def test_deepseek_target_disables_default_thinking_mode(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    _, _, payload = ProviderAdapter("deepseek", ok_transport)._payload(
        "deepseek-flash", [{"role": "user", "content": "u"}], "test", False, 2048)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 2048


def test_provider_routers_select_deepseek_target_and_anthropic_judge(monkeypatch):
    from evals.providers import judge_router, target_router
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    targets = target_router({"deepseek": ProviderAdapter("deepseek", ok_transport)})
    result = asyncio.run(targets({"turns": [{"role": "user", "content": "u"}]}, {"model": "deepseek/deepseek-flash"}))
    assert result["provider_model"] == "deepseek-flash"
    judges = judge_router({"anthropic": ProviderAdapter("anthropic", ok_transport)})
    result = asyncio.run(judges("c", "r", "criterion", {"model": "anthropic/claude-haiku-4-5-20251001"}))
    assert result["request_id"] == "a"


def test_structured_and_malformed_judge_outputs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    async def structured(*args):
        return 200, {}, {"model": "x", "choices": [{"message": {"content": '{"verdict":"NO"}'}}]}
    result = asyncio.run(judge_call(ProviderAdapter("groq", structured))("c", "r", "x", {"model": "groq/openai/gpt-oss-120b"}))
    assert result["verdict"] is False
    async def malformed(*args):
        return 200, {}, {"model": "x", "choices": [{"message": {"content": "maybe"}}]}
    result = asyncio.run(judge_call(ProviderAdapter("groq", malformed))("c", "r", "x", {"model": "groq/openai/gpt-oss-120b"}))
    assert result["verdict"] is None


def test_xai_judge_uses_strict_verdict_schema(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test")
    adapter = ProviderAdapter("xai", ok_transport)
    _, _, payload = adapter._payload("grok-4.7", [{"role": "user", "content": "x"}], "test", True)
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_google_probe_payload_caps_output_and_records_model_version(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    adapter = ProviderAdapter("google", ok_transport)
    _, _, payload = adapter._payload("gemini-3.5-flash-lite", [{"role": "user", "content": "x"}], "test", False, 1)
    assert payload["generationConfig"]["maxOutputTokens"] == 1
    result = adapter._result({"modelVersion": "gemini-3.5-flash-lite-001", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "OK"}]}}]}, {}, "gemini-3.5-flash-lite", 1)
    assert result["provider_model"] == "gemini-3.5-flash-lite-001"
    assert result["finish_reason"] == "STOP"


def test_router_limits_are_sent_to_all_provider_payload_shapes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("XAI_API_KEY", "test")
    _, _, anthropic = ProviderAdapter("anthropic", ok_transport)._payload("claude", [], "test", False, 64)
    _, _, xai = ProviderAdapter("xai", ok_transport)._payload("grok", [], "test", True, 64)
    assert anthropic["max_tokens"] == 64 and xai["max_tokens"] == 64


def test_active_judges_use_native_minimal_verdict_schemas(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    _, _, anthropic = ProviderAdapter("anthropic", ok_transport)._payload("claude", [], "test", True, 128)
    _, _, google = ProviderAdapter("google", ok_transport)._payload("gemini", [], "test", True, 128)
    assert anthropic["output_config"]["format"]["schema"] == VERDICT_SCHEMA
    assert google["generationConfig"]["responseJsonSchema"] == VERDICT_SCHEMA
    assert "Return JSON only" in VERDICT_FIRST_PROMPT


def test_retry_after_and_circuit_breaker_are_offline_and_bounded(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test")
    calls = 0
    async def limited(*_args):
        nonlocal calls
        calls += 1
        return 429, {"retry-after": "0"}, {"error": "capacity"}
    policy = ProviderExecutionPolicy(max_retries=1, circuit_breaker_failures=1, circuit_breaker_cooldown_s=60)
    adapter = ProviderAdapter("xai", limited, policy=policy)
    with pytest.raises(ProviderError) as first:
        asyncio.run(adapter.request("grok", [], structured=True))
    assert calls == 1 and first.value.classification == "provider_rate_or_capacity_rejection"
    with pytest.raises(ProviderError, match="circuit breaker open"):
        asyncio.run(adapter.request("grok", [], structured=True))


def test_read_timeout_is_never_automatically_retried(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test")
    calls = 0
    async def timeout(*_args):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("offline")
    adapter = ProviderAdapter("xai", timeout, policy=ProviderExecutionPolicy(max_retries=2))
    with pytest.raises(ProviderError) as exc:
        asyncio.run(adapter.request("grok", []))
    assert calls == 1 and exc.value.classification == "uncertain_dispatched_read_timeout"
