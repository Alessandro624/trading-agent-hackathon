from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trading_agent.core import JournalEntry, MarketSnapshot, NewsProvider, validate_news_query


def market_context_tool_schemas() -> list[dict[str, Any]]:
    """Schemas for deterministic tools available to the ReAct Analyst."""
    return [
        _tool_schema("inspect_snapshot", "Inspect validated market snapshot fields and confidence labels."),
        _tool_schema(
            "explain_technical_signals",
            "Explain RSI, MACD, SMA and technical confidence from validated indicators.",
        ),
        _tool_schema("summarize_news_urls", "List available news titles, sources and URLs from the Scout snapshot."),
        _tool_schema("read_news_url", "Fetch a news URL and return a bounded article excerpt.", {"url": {"type": "string"}}),
        _tool_schema(
            "search_market_news",
            "Search NewsAPI with a validated market-news strategy.",
            {
                "ticker": {"type": "string"},
                "strategy": {"type": "string", "enum": ["everything", "top_headlines"]},
                "sort_by": {"type": "string", "enum": ["publishedAt", "relevancy", "popularity"]},
                "search_in": {"type": "string"},
            },
        ),
    ]


def run_market_context_tools(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    tool_calls: list[dict[str, Any]],
    news_provider: NewsProvider | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    tools = {
        "inspect_snapshot": lambda: snapshot_payload(snapshot),
        "explain_technical_signals": lambda: {
            "summary": snapshot.technical_indicators.summary(),
            "confidence": snapshot.technical_indicators.confidence,
            "raw": asdict(snapshot.technical_indicators),
        },
        "summarize_news_urls": lambda: {
            "items": [
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                }
                for item in snapshot.news[:3]
            ]
        },
        "read_news_url": lambda call=None: read_news_url_content(str((call or {}).get("args", {}).get("url", ""))),
    }
    if news_provider is not None:
        tools["search_market_news"] = lambda call=None: safe_search_market_news(
            news_provider,
            ticker=str((call or {}).get("args", {}).get("ticker", snapshot.ticker)),
            strategy=str((call or {}).get("args", {}).get("strategy", "everything")),
            sort_by=str((call or {}).get("args", {}).get("sort_by", "publishedAt")),
            search_in=(call or {}).get("args", {}).get("search_in", "title,description"),
        )
    observations: list[dict[str, Any]] = []
    blocked: list[str] = []
    for call in tool_calls:
        name = str(call.get("name", ""))
        if name not in tools:
            blocked.append(name or "unknown")
            observations.append({"tool": name or "unknown", "error": "tool not allowed"})
            continue
        if name in {"read_news_url", "search_market_news"}:
            observations.append({"tool": name, "observation": tools[name](call)})
        else:
            observations.append({"tool": name, "observation": tools[name]()})
    if not observations:
        observations.append({"tool": "inspect_snapshot", "observation": snapshot_payload(snapshot)})
    return observations, blocked


def langchain_market_context_tools(snapshot: MarketSnapshot, recent_entries: list[JournalEntry]) -> list[Any]:
    try:
        from langchain_core.tools import tool
    except ImportError as error:
        raise RuntimeError("Install LangChain dependencies with `uv sync` to use LangChain tools.") from error

    @tool
    def inspect_snapshot() -> dict:
        """Inspect validated market snapshot fields and confidence labels."""
        return snapshot_payload(snapshot)

    @tool
    def explain_technical_signals() -> dict:
        """Explain RSI, MACD, SMA and technical confidence from validated indicators."""
        return {
            "summary": snapshot.technical_indicators.summary(),
            "confidence": snapshot.technical_indicators.confidence,
            "raw": asdict(snapshot.technical_indicators),
        }

    @tool
    def summarize_news_urls() -> dict:
        """List available news titles, sources and URLs from the Scout snapshot."""
        return {
            "items": [
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                }
                for item in snapshot.news[:3]
            ]
        }

    @tool
    def read_news_url(url: str) -> dict:
        """Fetch a news URL and return a bounded article excerpt."""
        return read_news_url_content(url)

    return [inspect_snapshot, explain_technical_signals, summarize_news_urls, read_news_url]


def langchain_news_search_tool(news_provider: NewsProvider) -> Any:
    try:
        from langchain_core.tools import tool
    except ImportError as error:
        raise RuntimeError("Install LangChain dependencies with `uv sync` to use LangChain tools.") from error

    @tool
    def search_market_news(
        ticker: str,
        strategy: str = "everything",
        sort_by: str = "publishedAt",
        search_in: str = "title,description",
    ) -> dict:
        """Search NewsAPI with bounded strategies for ticker-aware market news."""
        return safe_search_market_news(
            news_provider,
            ticker=ticker,
            strategy=strategy,
            sort_by=sort_by,
            search_in=search_in or None,
        )

    return search_market_news


def safe_search_market_news(
    news_provider: NewsProvider,
    ticker: str,
    strategy: str = "everything",
    sort_by: str = "publishedAt",
    search_in: str | None = "title,description",
) -> dict[str, Any]:
    try:
        strategy, sort_by, search_in = validate_news_query(strategy, sort_by, search_in)
        rows = news_provider.search_market_news(
            ticker=ticker,
            strategy=strategy,
            limit=5,
            sort_by=sort_by,
            search_in=search_in or None,
        )
        return {
            "status": "ok",
            "ticker": ticker.upper(),
            "strategy": strategy,
            "sort_by": sort_by,
            "search_in": search_in,
            "items": rows,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "ticker": ticker.upper(),
            "strategy": strategy,
            "sort_by": sort_by,
            "search_in": search_in,
            "items": [],
            "error": str(error),
        }


def read_news_url_content(url: str, timeout_seconds: float = 6.0, max_chars: int = 5000) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        return {"url": url, "status": "failed", "title": None, "body_excerpt": "", "error": "unsupported URL scheme"}
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "TradingAgentDemo/1.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        body = " ".join(soup.get_text(" ").split())
        return {"url": url, "status": "ok", "title": title, "body_excerpt": body[:max_chars], "error": None}
    except Exception as error:
        return {"url": url, "status": "failed", "title": None, "body_excerpt": "", "error": str(error)}


def snapshot_payload(snapshot: MarketSnapshot) -> dict[str, Any]:
    return {
        "ticker": snapshot.ticker,
        "price": snapshot.price,
        "price_confidence": snapshot.price_confidence,
        "news_confidence": snapshot.news_confidence,
        "technical_indicators": asdict(snapshot.technical_indicators),
        "failures": snapshot.failures,
        "guardrails": snapshot.guardrails_triggered,
    }


def _tool_schema(name: str, description: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "additionalProperties": False},
        },
    }
