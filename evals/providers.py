# ruff: noqa: E501, E702
"""Provider adapters for the resumable matrix; no request occurs until a caller invokes them.

The adapters keep provider payloads and billing metadata out of prompts/log output.
Tests inject a transport, so no credential or network access is needed for validation.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from dotenv import dotenv_values, load_dotenv

from evals.matrix import TargetCall, JudgeCall
from evals.run_eval import JUDGE_PROMPT, parse_verdict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

Transport = Callable[[str, str, dict[str, str], dict[str, Any]], Awaitable[tuple[int, dict[str, str], dict[str, Any]]]]


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status: int | None, message: str, *, retry_after: float | None = None,
                 classification: str | None = None, attempt_events: list[dict[str, Any]] | None = None,
                 provider_attempts: list[dict[str, Any]] | None = None):
        super().__init__(f"{provider} status={status}: {message}")
        self.provider, self.status, self.retry_after = provider, status, retry_after
        self.classification = classification or self._classify(message)
        self.attempt_events = attempt_events or []
        self.provider_attempts = provider_attempts or []

    def _classify(self, message: str) -> str:
        if self.status == 429:
            return "provider_rate_or_capacity_rejection"
        if self.status == 503:
            return "provider_temporary_unavailable"
        if self.status is None and "ReadTimeout" in message:
            return "uncertain_dispatched_read_timeout"
        if self.status is None:
            return "uncertain_transport_failure"
        if self.status and self.status >= 500:
            return "provider_server_error"
        return "nonretryable_provider_error"

    @property
    def retryable(self) -> bool:
        return self.status in {429, 503}


@dataclass(frozen=True)
class ProviderExecutionPolicy:
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 90.0
    write_timeout_s: float = 30.0
    pool_timeout_s: float = 30.0
    min_interval_s: float = 0.0
    max_retries: int = 0
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 20.0
    backoff_schedule_s: tuple[float, ...] = ()
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 60.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None, fallback_retries: int = 0) -> "ProviderExecutionPolicy":
        values = values or {}
        return cls(**{**{"max_retries": fallback_retries}, **{k: v for k, v in values.items() if k in cls.__dataclass_fields__}})


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: str
    base_url: str
    price_input_m: float | None
    price_output_m: float | None


SPECS = {
    "anthropic": ProviderSpec("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages", None, None),
    "google": ProviderSpec("google", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/models", None, None),
    "mistral": ProviderSpec("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions", None, None),
    "groq": ProviderSpec("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", None, None),
    # OpenAI-compatible Chat Completions. Model availability is deliberately not
    # inferred from a key; the offline preflight checks only configuration.
    "deepseek": ProviderSpec("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", None, None),
    "xai": ProviderSpec("xai", "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", None, None),
}

# Deliberately minimal and portable across the native schema subsets used by the
# three active judge providers.  The prompt, not a non-portable maxLength keyword,
# constrains the reason to a short phrase.
VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

VERDICT_FIRST_PROMPT = """Grade ONE criterion about an AI assistant reply. Be literal.
Return JSON only. Put the verdict first. `verdict` must be exactly YES or NO.
`reason` must be a concise explanation of at most 12 words.

[CONVERSATION]
{conversation}

[ASSISTANT'S FINAL REPLY]
{reply}

[CRITERION]
{criterion}"""


def split_model(model: str) -> tuple[str, str]:
    provider, raw = model.split("/", 1)
    return provider, raw


def credential_status(providers: set[str] | None = None) -> dict[str, bool]:
    """Names and presence only; never return credential material."""
    selected = SPECS if providers is None else {name: SPECS[name] for name in providers}
    return {name: bool(os.environ.get(spec.key_env)) for name, spec in selected.items()}


def normalize_credential(value: str) -> str:
    """Remove accidental outer whitespace and a single pair of surrounding quotes.

    Never log the return value. dotenv already handles correctly quoted .env values,
    but inherited shell variables may retain literal quotes or whitespace.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def credential_value(provider: str) -> str:
    """The normalized credential used by both access checks and provider requests."""
    return normalize_credential(os.environ.get(SPECS[provider].key_env, ""))


def credential_diagnostics(provider: str) -> dict[str, bool]:
    """Non-secret format/source diagnostics for an environment variable."""
    name = SPECS[provider].key_env
    active = os.environ.get(name, "")
    dotfile = dotenv_values(ROOT / ".env").get(name) or ""
    return {"credential_present": bool(active), "needs_normalization": bool(active) and active != normalize_credential(active),
            "shell_override_suspected": bool(active and dotfile and normalize_credential(active) != normalize_credential(dotfile))}


async def http_transport(method: str, url: str, headers: dict[str, str], payload: dict[str, Any],
                         timeout: httpx.Timeout | None = None):
    async with httpx.AsyncClient(timeout=timeout or httpx.Timeout(90.0, connect=10.0, write=30.0, pool=30.0)) as client:
        response = await client.request(method, url, headers=headers, json=payload)
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"message": response.text[:300]}}
    return response.status_code, dict(response.headers), body


class ProviderAdapter:
    def __init__(self, provider: str, transport: Transport = http_transport, retries: int = 0,
                 policy: ProviderExecutionPolicy | None = None):
        self.spec, self.transport = SPECS[provider], transport
        self.policy = policy or ProviderExecutionPolicy(max_retries=retries)
        self._last_request_at = 0.0
        self._transient_failures = 0
        self._circuit_open_until = 0.0

    def _key(self) -> str:
        key = credential_value(self.spec.name)
        if not key:
            raise ProviderError(self.spec.name, None, f"missing {self.spec.key_env}")
        return key

    async def request(self, raw_model: str, messages: list[dict[str, str]], structured: bool = False,
                      max_output_tokens: int | None = None) -> dict[str, Any]:
        key = self._key()
        url, headers, payload = self._payload(raw_model, messages, key, structured, max_output_tokens)
        if time.monotonic() < self._circuit_open_until:
            raise ProviderError(self.spec.name, 503, "circuit breaker open", classification="local_circuit_open")
        events: list[dict[str, Any]] = []
        provider_attempts: list[dict[str, Any]] = []
        last: ProviderError | None = None
        for attempt in range(self.policy.max_retries + 1):
            delay = self.policy.min_interval_s - (time.monotonic() - self._last_request_at)
            if delay > 0:
                events.append({"event": "rate_limit_wait", "seconds": delay, "attempt": attempt + 1})
                await asyncio.sleep(delay)
            try:
                self._last_request_at = time.monotonic()
                if self.transport is http_transport:
                    timeout = httpx.Timeout(self.policy.read_timeout_s, connect=self.policy.connect_timeout_s,
                                            write=self.policy.write_timeout_s, pool=self.policy.pool_timeout_s)
                    status, response_headers, body = await http_transport("POST", url, headers, payload, timeout)
                else:
                    status, response_headers, body = await self.transport("POST", url, headers, payload)
            except Exception as exc:
                provider_attempts.append({"lifecycle_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                last = ProviderError(self.spec.name, None, type(exc).__name__, attempt_events=events,
                                     provider_attempts=provider_attempts)
            else:
                if 200 <= status < 300:
                    self._transient_failures = 0
                    result = self._result(body, response_headers, raw_model, attempt + 1)
                    provider_attempts.append({"lifecycle_status": "completed", "provider_model": result.get("provider_model"),
                                              "provider_request_id": result.get("request_id"), "usage": result.get("usage"),
                                              "response": result})
                    result["attempt_events"] = events
                    result["provider_attempts"] = provider_attempts
                    return result
                message = str(body.get("error", body))[:500]
                retry_after = response_headers.get("retry-after")
                retry_after_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                provider_attempts.append({"lifecycle_status": "failed", "error": message,
                                          "provider_request_id": response_headers.get("x-request-id") or response_headers.get("request-id")})
                last = ProviderError(self.spec.name, status, message, retry_after=retry_after_s, attempt_events=events,
                                     provider_attempts=provider_attempts)
                if not last.retryable:
                    raise last
            self._transient_failures += 1
            # A response-less transport failure may have reached the provider.
            # Never automatically retry it: preserve it for operator reconciliation.
            if not last.retryable:
                break
            if self._transient_failures >= self.policy.circuit_breaker_failures:
                self._circuit_open_until = time.monotonic() + self.policy.circuit_breaker_cooldown_s
                events.append({"event": "circuit_opened", "attempt": attempt + 1})
                break
            if last.retryable and attempt < self.policy.max_retries:
                if last.retry_after is not None:
                    delay = last.retry_after
                elif attempt < len(self.policy.backoff_schedule_s):
                    delay = self.policy.backoff_schedule_s[attempt]
                else:
                    delay = min(self.policy.backoff_cap_s, self.policy.backoff_base_s * 2**attempt)
                delay += random.uniform(0, min(1.0, delay * 0.2))
                events.append({"event": "retry_scheduled", "seconds": delay, "attempt": attempt + 1,
                               "classification": last.classification})
                await asyncio.sleep(delay)
        raise last or ProviderError(self.spec.name, None, "unknown failure")

    def _payload(self, raw: str, messages: list[dict[str, str]], key: str, structured: bool,
                 max_output_tokens: int | None = None):
        if self.spec.name == "anthropic":
            payload = {
                "model": raw, "max_tokens": max_output_tokens or 512, "temperature": 0, "messages": messages}
            if structured:
                payload["output_config"] = {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
            return self.spec.base_url, {"x-api-key": key, "anthropic-version": "2023-06-01"}, payload
        if self.spec.name == "google":
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
            generation = {"temperature": 0, "responseMimeType": "application/json" if structured else "text/plain"}
            if structured:
                generation["responseJsonSchema"] = VERDICT_SCHEMA
            if max_output_tokens is not None:
                generation["maxOutputTokens"] = max_output_tokens
            return f"{self.spec.base_url}/{raw}:generateContent?key={key}", {}, {"contents": contents, "generationConfig": generation}
        payload = {"model": raw, "messages": messages, "temperature": 0}
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        # DeepSeek thinking is enabled by default. Disable it for target generation
        # so a finite cap remains available for the visible clinical response.
        if self.spec.name == "deepseek" and not structured:
            payload["thinking"] = {"type": "disabled"}
        if structured:
            if self.spec.name == "xai":
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "rubric_verdict", "strict": True, "schema": VERDICT_SCHEMA}}
            else:
                payload["response_format"] = {"type": "json_object"}
        return self.spec.base_url, {"Authorization": f"Bearer {key}"}, payload

    def _result(self, body: dict[str, Any], headers: dict[str, str], requested: str, attempts: int) -> dict[str, Any]:
        if self.spec.name == "anthropic":
            text = "".join(x.get("text", "") for x in body.get("content", [])); usage = body.get("usage", {})
            request_id = headers.get("request-id") or body.get("id")
        elif self.spec.name == "google":
            candidate = body.get("candidates", [{}])[0]
            text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = body.get("usageMetadata", {}); request_id = headers.get("x-request-id") or body.get("responseId")
        else:
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            usage = body.get("usage", {}); request_id = headers.get("x-request-id") or body.get("id")
        finish_reason = candidate.get("finishReason") if self.spec.name == "google" else None
        provider_model = body.get("modelVersion", body.get("model", requested)) if self.spec.name == "google" else body.get("model", requested)
        return {"response": text, "text": text, "provider_model": provider_model, "usage": usage,
                "request_id": request_id, "finish_reason": finish_reason, "attempts": attempts}


def target_call(adapter: ProviderAdapter) -> TargetCall:
    async def call(case: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        _, raw = split_model(meta["model"])
        return await adapter.request(raw, case["turns"])
    return call


def target_router(adapters: dict[str, ProviderAdapter], max_output_tokens: int | None = None) -> TargetCall:
    """Route a matrix target cell to its configured provider without regenerating it.

    The matrix owns idempotency and only invokes this router for an unfinished target
    cell. This function makes no network request until the matrix invokes the returned
    callable.
    """
    async def call(case: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        provider, raw = split_model(meta["model"])
        if provider not in adapters:
            raise ProviderError(provider, None, "no configured adapter")
        result = await adapters[provider].request(raw, case["turns"], max_output_tokens=max_output_tokens)
        result["request_max_output_tokens"] = max_output_tokens
        if provider == "deepseek":
            result["generation_policy"] = {"thinking": {"type": "disabled"}}
        return result
    return call


def judge_call(adapter: ProviderAdapter, max_output_tokens: int | None = None) -> JudgeCall:
    async def call(conversation: str, response: str, criterion: str, meta: dict[str, Any]) -> dict[str, Any]:
        _, raw = split_model(meta["model"])
        prompt = meta.get("judge_prompt", JUDGE_PROMPT).format(conversation=conversation, reply=response, criterion=criterion)
        result = await adapter.request(raw, [{"role": "user", "content": prompt}], structured=True,
                                       max_output_tokens=max_output_tokens)
        result["request_max_output_tokens"] = max_output_tokens
        try:
            structured = json.loads(result["text"])
            verdict = structured.get("verdict") if isinstance(structured, dict) else None
            result["verdict"] = str(verdict).upper() == "YES" if str(verdict).upper() in {"YES", "NO"} else None
        except json.JSONDecodeError:
            result["verdict"] = parse_verdict(result["text"])
        return result
    return call


def judge_router(adapters: dict[str, ProviderAdapter], max_output_tokens: int | dict[str, int] | None = None) -> JudgeCall:
    """Route an immutable stored response to its judge provider."""
    async def call(conversation: str, response: str, criterion: str, meta: dict[str, Any]) -> dict[str, Any]:
        provider, _ = split_model(meta["model"])
        if provider not in adapters:
            raise ProviderError(provider, None, "no configured adapter")
        limit = max_output_tokens.get(provider) if isinstance(max_output_tokens, dict) else max_output_tokens
        return await judge_call(adapters[provider], max_output_tokens=limit)(conversation, response, criterion, meta)
    return call
