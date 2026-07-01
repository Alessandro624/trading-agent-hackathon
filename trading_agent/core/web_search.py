from __future__ import annotations

import logging
from typing import Any

from trading_agent.core.entity_discovery import discover_candidates

logger = logging.getLogger("trading_agent.web_search")


def fetch_web_search_context(
    query: str,
    llm_client: Any | None = None,
    *,
    news_provider: Any | None = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    if not query:
        return []
    candidates, evidence = discover_candidates(
        query,
        news_provider=news_provider,
        tavily_api_key=None,
        alpaca_validator=None,
        max_discovered_tickers=max_results,
    )
    return [ev.to_dict() for ev in evidence[:max_results]]


def extract_ticker_candidates(text: str) -> list[str]:
    from trading_agent.core.entity_discovery import _TICKER_PATTERN, _TICKER_STOPWORDS

    if not text:
        return []
    seen: set[str] = set()
    candidates: list[str] = []
    for match in _TICKER_PATTERN.finditer(text):
        ticker = match.group(1)
        if ticker in _TICKER_STOPWORDS:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        candidates.append(ticker)
        if len(candidates) >= 5:
            break
    return candidates
