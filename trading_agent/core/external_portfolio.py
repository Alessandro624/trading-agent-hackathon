from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("trading_agent.external_portfolio")


_TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b")


_MATCH_PROMPT = """You are a financial research assistant. The user wants to match
an external public portfolio (Warren Buffett / Berkshire Hathaway, Congress,
famous investors, etc.). Given the recent news articles below, extract the most
frequently mentioned BUY candidates (long positions only).

Return ONLY JSON: {"tickers": ["AAPL", "MSFT", ...], "rationale": "<one sentence>"}

Rules:
- Only include tickers explicitly mentioned as recent BUY additions or current
  holdings in the articles. Do not invent tickers.
- Limit to 8 tickers, most frequently mentioned first.
- Skip tickers mentioned only as a sell or a reduction.
- If the articles do not mention any specific tickers, return an empty list.
"""


def fetch_external_portfolio(
    query: str,
    news_provider: Any | None,
    llm_client: Any | None = None,
    *,
    max_articles: int = 8,
) -> dict[str, Any]:
    articles = _fetch_articles(query, news_provider, max_articles=max_articles)
    if not articles:
        return {"tickers": [], "rationale": f"No recent articles found for query: {query!r}.", "articles": []}
    if llm_client is None:
        tickers = _heuristic_extract(articles)
        return {
            "tickers": tickers,
            "rationale": f"Heuristic extraction from {len(articles)} article(s).",
            "articles": articles,
        }
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        tickers = _heuristic_extract(articles)
        return {
            "tickers": tickers,
            "rationale": f"Heuristic extraction (LLM complete_json unavailable) from {len(articles)} article(s).",
            "articles": articles,
        }
    try:
        payload = {
            "query": query,
            "articles": [{"title": a.get("title"), "snippet": (a.get("snippet") or a.get("summary") or "")[:500]} for a in articles],
        }
        raw = complete_json(_MATCH_PROMPT, json.dumps(payload, default=str))
        text = str(raw or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return {"tickers": [], "rationale": "LLM returned invalid JSON.", "articles": articles}
            parsed = json.loads(text[start : end + 1])
        tickers = [str(t).upper().strip() for t in (parsed.get("tickers") or []) if isinstance(t, str) and t.strip() and 1 <= len(t) <= 6]
        rationale = str(parsed.get("rationale") or f"LLM extracted {len(tickers)} tickers from {len(articles)} articles.")
        return {"tickers": tickers[:8], "rationale": rationale, "articles": articles}
    except Exception as error:
        logger.info("external_portfolio.llm.fail reason=%s query=%s", error, query[:80])
        tickers = _heuristic_extract(articles)
        return {
            "tickers": tickers,
            "rationale": f"LLM extraction failed ({error}); heuristic fallback from {len(articles)} article(s).",
            "articles": articles,
        }


def _fetch_articles(query: str, news_provider: Any | None, *, max_articles: int) -> list[dict[str, Any]]:
    if news_provider is None:
        return []
    search = getattr(news_provider, "search_market_news", None)
    if callable(search):
        try:
            result = search(query, strategy="everything", sort_by="publishedAt")
            items = (result or {}).get("items") or []
        except Exception as error:
            logger.info("external_portfolio.news.search_market_news.fail reason=%s", error)
            items = []
    else:
        items = []
    if not items:
        search_basic = getattr(news_provider, "search_news", None)
        if callable(search_basic):
            try:
                items = search_basic(query) or []
            except Exception as error:
                logger.info("external_portfolio.news.search_news.fail reason=%s", error)
                items = []
    articles: list[dict[str, Any]] = []
    for item in items[:max_articles]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("summary") or item.get("description") or item.get("content") or "").strip()
        if not (title or snippet):
            continue
        articles.append({"title": title, "url": url, "snippet": snippet[:500]})
    return articles


_STOPWORDS = {
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
}


def _heuristic_extract(articles: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for article in articles:
        text = f"{article.get('title', '')} {article.get('snippet', '')}"
        for match in _TICKER_PATTERN.finditer(text):
            token = match.group(0).upper()
            if token in _STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    frequent = [(t, c) for t, c in counts.items() if c >= 2]
    frequent.sort(key=lambda item: -item[1])
    return [t for t, _ in frequent[:8]]
