from __future__ import annotations

from typing import Any, Protocol


class MarketDataProvider(Protocol):
    def get_price(self, ticker: str) -> dict[str, Any]: ...

    def get_closes(self, ticker: str, limit: int = 60) -> list[float]: ...


class NewsProvider(Protocol):
    def search_news(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]: ...

    def search_market_news(
        self,
        ticker: str,
        strategy: str = "everything",
        limit: int = 5,
        sort_by: str = "publishedAt",
        search_in: str | None = None,
    ) -> list[dict[str, Any]]: ...


class LlmClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> str: ...

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: type[Any]) -> Any: ...

    def complete_tool_plan(self, system_prompt: str, user_prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]: ...

    def invoke_tools(self, system_prompt: str, user_prompt: str, tools: list[Any]) -> tuple[list[dict[str, Any]], dict]: ...

    def metadata(self) -> dict[str, Any]: ...


class BrokerClient(Protocol):
    def get_portfolio(self) -> dict[str, Any]: ...

    def place_order(self, ticker: str, action: str, quantity: int) -> dict[str, Any]: ...

    def get_order(self, order_id: str) -> dict[str, Any]: ...
