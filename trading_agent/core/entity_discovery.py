from __future__ import annotations

import logging
import os
import re
from typing import Any

from trading_agent.core.evidence import Evidence, dedupe_evidence, make_evidence

logger = logging.getLogger("trading_agent.entity_discovery")


_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")

_TICKER_STOPWORDS = {
    "THE",
    "AND",
    "FOR",
    "WITH",
    "NOT",
    "BUT",
    "YET",
    "NOR",
    "VIA",
    "ITS",
    "HIS",
    "HER",
    "OUR",
    "YOU",
    "ARE",
    "WAS",
    "WERE",
    "HAS",
    "HAD",
    "HAVE",
    "BEEN",
    "FROM",
    "INTO",
    "OVER",
    "UNDER",
    "THIS",
    "THAT",
    "THESE",
    "THOSE",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHICH",
    "WHILE",
    "ABOUT",
    "AFTER",
    "BEFORE",
    "BECAUSE",
    "BUY",
    "SELL",
    "HOLD",
    "USA",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CEO",
    "CFO",
    "API",
    "DNA",
    "RNA",
    "ETF",
    "GDP",
    "CPI",
    "IPO",
    "SEC",
    "FED",
    "NYC",
    "LAX",
    "WAR",
    "OIL",
    "GAS",
    "GOLD",
    "BOND",
    "BONDS",
    "STOCK",
    "STOCKS",
    "SPAC",
    "M&A",
    "AI",
    "IT",
    "IS",
    "OF",
    "ON",
    "TO",
    "IN",
    "AT",
    "BY",
    "OR",
    "AS",
    "AN",
    "BE",
    "DO",
    "IF",
    "NO",
    "SO",
    "UP",
    "HE",
    "WE",
    "ME",
    "MY",
    "I",
    "US",
    "URL",
    "HN",
    "FOMO",
}


class DiscoveredCandidate(dict):
    pass


def discover_candidates(
    query: str,
    *,
    news_provider: Any | None = None,
    tavily_api_key: str | None = None,
    alpaca_validator: Any | None = None,
    ticker_provider: Any | None = None,
    max_discovered_tickers: int = 3,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    if not query:
        return [], []

    newsapi_evidence = _newsapi_discovery(query, news_provider)

    initial_candidates = _extract_candidates_from_evidence(query, newsapi_evidence)
    if alpaca_validator is not None:
        initial_candidates = _validate_candidates_through_alpaca(initial_candidates, alpaca_validator)
    has_usable_news_candidate = any(candidate.get("alpaca_tradable") is not False for candidate in initial_candidates)
    tavily_evidence: list[Evidence] = []
    if not newsapi_evidence or not has_usable_news_candidate:
        tavily_evidence = _tavily_discovery(query, tavily_api_key or os.getenv("TAVILY_API_KEY"))

    all_evidence = dedupe_evidence(newsapi_evidence + tavily_evidence)
    if not all_evidence:
        logger.info("entity_discovery.no_evidence query=%s", query[:80])
        return [], []

    candidates = _extract_candidates_from_evidence(query, all_evidence)
    if not candidates:
        logger.info("entity_discovery.no_candidates query=%s evidence=%d", query[:80], len(all_evidence))
        return [], all_evidence

    if alpaca_validator is not None:
        candidates = _validate_candidates_through_alpaca(candidates, alpaca_validator)

    if alpaca_validator is not None:
        tradable = [c for c in candidates if c.get("alpaca_tradable") is True]
        if not tradable:
            all_text = " ".join(f"{ev.title} {ev.excerpt}" for ev in all_evidence)
            extra_tickers = [m.group(1) for m in _TICKER_PATTERN.finditer(all_text) if m.group(1) not in _TICKER_STOPWORDS and m.group(1) not in {c["ticker"] for c in candidates}]
            if extra_tickers:
                extra_candidates = [
                    {"ticker": t, "relationship": "mentioned_in_evidence", "confidence": 0.45, "classification": "unknown", "alpaca_tradable": None, "alpaca_validation_error": None, "evidence": []}
                    for t in dict.fromkeys(extra_tickers)
                ]
                extra_validated = _validate_candidates_through_alpaca(extra_candidates, alpaca_validator)
                newly_tradable = [c for c in extra_validated if c.get("alpaca_tradable") is True]
                if newly_tradable:
                    logger.info(
                        "entity_discovery.alpaca_rescue tickers=%s query=%s",
                        ",".join(c["ticker"] for c in newly_tradable),
                        query[:80],
                    )
                    candidates = candidates + newly_tradable

    candidates.sort(key=lambda c: -c.get("confidence", 0.0))
    capped = candidates[: max(1, max_discovered_tickers)]
    return capped, all_evidence


def _newsapi_discovery(query: str, news_provider: Any | None) -> list[Evidence]:
    if news_provider is None:
        return []
    evidence: list[Evidence] = []
    search = getattr(news_provider, "search_market_news", None)
    if callable(search):
        try:
            result = search(query, strategy="everything", sort_by="publishedAt") or []
            if isinstance(result, dict):
                items = result.get("items") or result.get("articles") or []
            elif isinstance(result, list):
                items = result
            else:
                items = []
        except Exception as error:
            logger.info("entity_discovery.newsapi.search_market_news.fail reason=%s query=%s", error, query[:80])
            items = []
    else:
        items = []
    if not items:
        search_basic = getattr(news_provider, "search_news", None)
        if callable(search_basic):
            try:
                items = search_basic(query) or []
            except Exception as error:
                logger.info("entity_discovery.newsapi.search_news.fail reason=%s query=%s", error, query[:80])
                items = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("summary") or item.get("description") or item.get("content") or "").strip()
        published_at = item.get("published_at") or item.get("publishedAt") or item.get("date")
        if not (title or snippet):
            continue
        evidence.append(
            make_evidence(
                title=title,
                url=url,
                published_at=str(published_at) if published_at else None,
                provider="newsapi",
                query=query,
                excerpt=snippet,
                confidence=0.7,
            )
        )
    return evidence


def _tavily_discovery(query: str, api_key: str | None) -> list[Evidence]:
    if not api_key:
        return []
    try:
        import requests
    except ImportError:
        logger.warning("entity_discovery.tavily.unavailable reason=requests_not_installed")
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "include_answer": False},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        logger.info("entity_discovery.tavily.fail reason=%s query=%s", error, query[:80])
        return []
    evidence: list[Evidence] = []
    for item in (data.get("results") or [])[:5]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        if not (title or snippet):
            continue
        evidence.append(
            make_evidence(
                title=title,
                url=url,
                published_at=None,
                provider="tavily",
                query=query,
                excerpt=snippet,
                confidence=0.6,
            )
        )
    return evidence


def _extract_candidates_from_evidence(query: str, evidence: list[Evidence]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for ev in evidence:
        text = f"{ev.title} {ev.excerpt}"
        for match in _TICKER_PATTERN.finditer(text):
            ticker = match.group(1).upper()
            if ticker in _TICKER_STOPWORDS:
                continue
            if ticker in candidates:
                existing = candidates[ticker]
                existing["evidence"].append(ev.to_dict())
                existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
                continue
            candidates[ticker] = {
                "ticker": ticker,
                "relationship": "mentioned_in_evidence",
                "confidence": 0.5,
                "classification": "unknown",
                "alpaca_tradable": None,
                "alpaca_validation_error": None,
                "evidence": [ev.to_dict()],
            }
    return list(candidates.values())


def _validate_candidates_through_alpaca(candidates: list[dict[str, Any]], alpaca_validator: Any) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            is_tradable = bool(alpaca_validator(ticker))
            candidate = {**candidate, "alpaca_tradable": is_tradable, "alpaca_validation_error": None}
        except Exception as error:
            candidate = {
                **candidate,
                "alpaca_tradable": False,
                "alpaca_validation_error": str(error)[:200],
            }
        validated.append(candidate)
    return validated
