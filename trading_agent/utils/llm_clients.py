from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class _ChatJsonClientMixin:
    provider_name: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        message = self._model().invoke([("system", system_prompt), ("user", user_prompt)])
        return _message_content(message, self.provider_name)

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[Any]) -> Any:
        return _structured_response(self._model(), system_prompt, user_prompt, schema)

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
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float("OPENAI_TIMEOUT_SECONDS", 60)
        self.temperature = temperature if temperature is not None else _env_float("OPENAI_TEMPERATURE", 0)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("OPENAI_MAX_TOKENS", 1024)
        self.provider_name = "openai"
        self._model_cache = None

    def _model(self):
        if self._model_cache is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as error:
                raise RuntimeError("Install LangChain dependencies with `uv sync` to use OpenAI models.") from error
            self._model_cache = ChatOpenAI(model=self.model, temperature=self.temperature, timeout=self.timeout_seconds, max_tokens=self.max_tokens)
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
        self.provider_name = "ollama"
        self._model_cache = None

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
    

class OpenRouterJsonClient(_ChatJsonClientMixin):
    def __init__(
        self,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float("OPENROUTER_TIMEOUT_SECONDS", 60)
        self.temperature = temperature if temperature is not None else _env_float("OPENROUTER_TEMPERATURE", 0)
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("OPENROUTER_MAX_TOKENS", 1024)
        self.provider_name = "openrouter"
        self._model_cache = None

    def _model(self):
        if self._model_cache is None:
            try:
                from langchain_openrouter import ChatOpenRouter
            except ImportError as error:
                raise RuntimeError("Install LangChain dependencies with `uv sync` to use OpenRouter models.") from error
            self._model_cache = ChatOpenRouter(
                model=self.model, 
                base_url=self.base_url,
                temperature=self.temperature, 
                timeout=self.timeout_seconds, 
                max_tokens=self.max_tokens
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
        try:
            value = call(self.primary)
            if not value:
                raise RuntimeError(empty_message)
            self._metadata = _metadata(_provider_name(self.primary))
            return value
        except Exception as error:
            value = call(self.fallback)
            self._metadata = _metadata(
                _provider_name(self.fallback),
                fallback_used=True,
                fallback_reason=f"{_provider_name(self.primary)} failed: {error}",
            )
            return value


def _structured_response(model, system_prompt, user_prompt, schema):
    return model.with_structured_output(schema).invoke([("system", system_prompt), ("user", user_prompt)])


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
