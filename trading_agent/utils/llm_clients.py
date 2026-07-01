from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("trading_agent.llm_clients")


class _ChatJsonClientMixin:
    provider_name: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        message = self._model().invoke([("system", system_prompt), ("user", user_prompt)])
        return _message_content(message, self.provider_name)

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[Any]) -> Any:
        structured_model = getattr(self, "_structured_model", None)
        model = structured_model() if callable(structured_model) else self._model()
        return _structured_response(
            model,
            system_prompt,
            user_prompt,
            schema,
            method=getattr(self, "structured_method", "json_schema"),
            max_tokens=getattr(self, "structured_max_tokens", 1024),
            provider=self.provider_name,
        )

    def complete_tool_plan(self, system_prompt: str, user_prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        message = self._model().bind_tools(tools).invoke([("system", system_prompt), ("user", user_prompt)])
        return {"tool_calls": _standard_tool_calls(message)}

    def invoke_tools(self, system_prompt: str, user_prompt: str, tools: list[Any]) -> tuple[list[dict[str, Any]], dict]:
        return _invoke_bound_tools(self._model(), system_prompt, user_prompt, tools)

    def metadata(self) -> dict:
        return _metadata(self.provider_name)


class OpenAiJsonClient(_ChatJsonClientMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float("OPENAI_TIMEOUT_SECONDS", 60)
        self.temperature = temperature if temperature is not None else _env_float("OPENAI_TEMPERATURE", 0)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("OPENAI_MAX_TOKENS", 1024)
        self.structured_max_tokens = _env_int("OPENAI_STRUCTURED_MAX_TOKENS", 1024)
        self.structured_method = _structured_method("OPENAI_STRUCTURED_METHOD", "json_schema")
        self.http_max_retries = _env_int("OPENAI_HTTP_MAX_RETRIES", 0)
        self.provider_name = "openai"
        self._model_cache = None

    def _model(self):
        if self._model_cache is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as error:
                raise RuntimeError("Install LangChain dependencies with `uv sync` to use OpenAI models.") from error
            self._model_cache = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                max_tokens=self.max_tokens,
                max_retries=self.http_max_retries,
            )
        return self._model_cache


class OllamaJsonClient(_ChatJsonClientMixin):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float("OLLAMA_TIMEOUT_SECONDS", 90)
        self.temperature = temperature if temperature is not None else _env_float("OLLAMA_TEMPERATURE", 0)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("OLLAMA_MAX_TOKENS", 1024)
        self.structured_max_tokens = _env_int("OLLAMA_STRUCTURED_MAX_TOKENS", 1024)
        self.structured_method = _structured_method("OLLAMA_STRUCTURED_METHOD", "json_schema")
        self.provider_name = "ollama"
        self._model_cache = None
        self._structured_model_cache = None

    def _model(self):
        if self._model_cache is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as error:
                raise RuntimeError("Install LangChain dependencies with `uv sync` to use Ollama models.") from error
            self._model_cache = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                num_predict=self.max_tokens,
            )
        return self._model_cache

    def _structured_model(self):
        if self._structured_model_cache is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as error:
                raise RuntimeError("Install LangChain dependencies with `uv sync` to use Ollama models.") from error
            self._structured_model_cache = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                num_predict=self.structured_max_tokens,
            )
        return self._structured_model_cache


class OpenRouterJsonClient(_ChatJsonClientMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float("OPENROUTER_TIMEOUT_SECONDS", 60)
        self.temperature = temperature if temperature is not None else _env_float("OPENROUTER_TEMPERATURE", 0)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("OPENROUTER_MAX_TOKENS", 1024)
        self.structured_max_tokens = _env_int("OPENROUTER_STRUCTURED_MAX_TOKENS", 1024)
        self.structured_method = _structured_method("OPENROUTER_STRUCTURED_METHOD", "json_schema")
        self.http_max_retries = _env_int("OPENROUTER_HTTP_MAX_RETRIES", 0)
        self.provider_name = "openrouter"
        self._model_cache = None

    def _model(self):
        if self._model_cache is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as error:
                raise RuntimeError("Install LangChain OpenAI dependencies with `uv sync` to use OpenRouter models.") from error
            self._model_cache = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                max_tokens=self.max_tokens,
                max_retries=self.http_max_retries,
            )
        return self._model_cache


@dataclass
class FallbackLlmClient:
    primary: object
    fallback: object
    _metadata: dict = field(default_factory=dict, init=False)

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        return self._with_fallback(
            lambda client: client.complete_json(system_prompt, user_prompt),
            "primary provider returned empty content",
        )

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[Any]) -> Any:
        return self._with_fallback(
            lambda client: client.complete_structured(system_prompt, user_prompt, schema),
            "primary provider returned empty structured response",
        )

    def complete_tool_plan(self, system_prompt: str, user_prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        return self._with_fallback(
            lambda client: client.complete_tool_plan(system_prompt, user_prompt, tools),
            "primary provider returned empty tool plan",
        )

    def invoke_tools(self, system_prompt: str, user_prompt: str, tools: list[Any]) -> tuple[list[dict[str, Any]], dict]:
        return self._with_fallback(
            lambda client: client.invoke_tools(system_prompt, user_prompt, tools),
            "primary provider returned empty tool observations",
        )

    def metadata(self) -> dict:
        return dict(self._metadata)

    def _with_fallback(self, call, empty_message: str):
        primary_started = time.perf_counter()
        try:
            value = call(self.primary)
            if not value:
                raise RuntimeError(empty_message)
            self._metadata = _metadata(_provider_name(self.primary))
            return value
        except Exception as error:
            logger.warning(
                "Primary LLM provider %s failed after %dms: %s. Falling back to %s.",
                _provider_name(self.primary),
                int((time.perf_counter() - primary_started) * 1000),
                error,
                _provider_name(self.fallback),
            )
            fallback_started = time.perf_counter()
            try:
                value = call(self.fallback)
            except Exception as fallback_error:
                logger.error(
                    "Fallback LLM provider %s failed after %dms: %s",
                    _provider_name(self.fallback),
                    int((time.perf_counter() - fallback_started) * 1000),
                    fallback_error,
                )
                raise RuntimeError(
                    f"primary provider failed: {error}; fallback provider failed: {fallback_error}"
                ) from fallback_error
            logger.info(
                "Fallback LLM provider %s completed after %dms",
                _provider_name(self.fallback),
                int((time.perf_counter() - fallback_started) * 1000),
            )
            self._metadata = _metadata(
                _provider_name(self.fallback),
                fallback_used=True,
                fallback_reason=f"{_provider_name(self.primary)} failed: {error}",
            )
            return value


def _structured_response(
    model,
    system_prompt,
    user_prompt,
    schema,
    *,
    method: str,
    max_tokens: int,
    provider: str,
):
    invoke_kwargs = {} if provider == "ollama" else {"max_tokens": max_tokens}
    return model.with_structured_output(schema, method=method).invoke(
        [("system", system_prompt), ("user", user_prompt)],
        **invoke_kwargs,
    )


def should_retry_llm_error(error: Exception) -> bool:
    text = str(error).lower()
    terminal_markers = (
        "length limit was reached",
        "finish_reason='length'",
        'finish_reason="length"',
        "context_length_exceeded",
        "maximum context length",
        "unexpected keyword argument",
        "unsupported parameter",
        "not supported",
        "does not support",
    )
    return not any(marker in text for marker in terminal_markers)


def _structured_method(env_name: str, default: str) -> str:
    method = os.getenv(env_name, default).strip().lower()
    allowed = {"json_schema", "function_calling", "json_mode"}
    return method if method in allowed else default


def _invoke_bound_tools(model, system_prompt: str, user_prompt: str, tools: list[Any]) -> tuple[list[dict[str, Any]], dict]:
    tool_by_name = {tool.name: tool for tool in tools}
    message = model.bind_tools(tools).invoke([("system", system_prompt), ("user", user_prompt)])
    observations: list[dict[str, Any]] = []
    for call in _standard_tool_calls(message)[:3]:
        tool_name = call["name"]
        tool_obj = tool_by_name.get(tool_name)
        if tool_obj is None:
            observations.append({"tool": tool_name or "unknown", "error": "tool not allowed"})
            continue
        observations.append({"tool": tool_name, "observation": tool_obj.invoke(call.get("args") or {})})
    return observations, {"tool_calls": _standard_tool_calls(message)}


def _standard_tool_calls(message) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            normalized.append({"name": call.get("name", ""), "args": call.get("args") or {}})
            continue
        normalized.append({"name": getattr(call, "name", ""), "args": getattr(call, "args", {}) or {}})
    return normalized


def _message_content(message, provider_name: str) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    if not content:
        raise RuntimeError(f"{provider_name} returned empty content")
    return str(content)


def _metadata(provider: str, fallback_used: bool = False, fallback_reason: str | None = None) -> dict:
    return {
        "llm_provider": provider,
        "llm_fallback_used": fallback_used,
        "llm_fallback_provider": provider if fallback_used else None,
        "llm_fallback_reason": fallback_reason,
    }


def _provider_name(client: object) -> str:
    return str(getattr(client, "provider_name", client.__class__.__name__.lower()))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))
