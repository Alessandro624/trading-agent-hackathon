from __future__ import annotations

import json
from typing import Any

from trading_agent.core import JournalEntry, LlmClient, MarketSnapshot, NewsOpinion, NewsOpinionOutput, NewsProvider, RetryPolicy
from trading_agent.tools import langchain_market_context_tools, langchain_news_search_tool, run_market_context_tools, snapshot_payload
from trading_agent.utils import get_logger

logger = get_logger("news_analyst")

SYSTEM_PROMPT = """You are the News Analyst in a multi-agent trading system.
Use validated tool observations and snapshot news only. Produce a structured NewsOpinion.
Do not invent articles, sources, URLs, or full article bodies. NewsAPI content is a preview unless read_news_url succeeded.
relevance and confidence must be decimal scores from 0.0 to 1.0, not counts, percentages, rankings, or article totals.
evidence must contain concrete points from validated observations. risks must contain uncertainty or news-specific risks.
Focus on ticker relevance, sentiment, concrete evidence, and news-specific risks.
"""


def news_opinion(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    llm_client: LlmClient,
    news_provider: NewsProvider,
    retry_policy: RetryPolicy | None = None,
) -> NewsOpinion:
    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    observations = _news_observations(snapshot, recent_entries, llm_client, news_provider)
    base_payload = {"snapshot": snapshot_payload(snapshot), "tool_observations": observations}
    errors: list[str] = []

    def call_and_parse() -> NewsOpinionOutput:
        payload = dict(base_payload)
        if errors:
            payload["validation_feedback"] = errors[-2:]
        structured = getattr(llm_client, "complete_structured", None)
        if callable(structured):
            return structured(SYSTEM_PROMPT, json.dumps(payload, default=str), NewsOpinionOutput)
        raw = llm_client.complete_json(SYSTEM_PROMPT, json.dumps(payload, default=str))
        from trading_agent.core import parse_news_opinion_output

        return parse_news_opinion_output(raw)

    def on_failure(attempt: int, error: Exception) -> None:
        errors.append(f"news opinion retry {attempt}: {error}")
        logger.warning("news.validation.retry ticker=%s attempt=%s error=%s", snapshot.ticker, attempt, error)

    try:
        parsed = retry_policy.run(call_and_parse, on_failure=on_failure)
    except Exception:
        logger.error("news.validation.failed ticker=%s", snapshot.ticker)
        return NewsOpinion(
            sentiment="unknown",
            relevance=0.0,
            confidence=0.1,
            evidence=["News analysis failed structured validation"],
            risks=errors or ["News analysis unavailable"],
            sources=[],
            summary="News analysis unavailable; decision should treat news confidence as low.",
        )

    opinion = NewsOpinion(
        sentiment=parsed.sentiment,
        relevance=parsed.relevance,
        confidence=parsed.confidence,
        evidence=parsed.evidence,
        risks=parsed.risks,
        sources=parsed.sources,
        summary=parsed.summary,
    )
    logger.info("news.opinion ticker=%s sentiment=%s confidence=%.2f", snapshot.ticker, opinion.sentiment, opinion.confidence)
    return opinion


def _news_observations(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    llm_client: LlmClient,
    news_provider: NewsProvider,
) -> list[dict[str, Any]]:
    invoke_tools = getattr(llm_client, "invoke_tools", None)
    if callable(invoke_tools):
        try:
            tools = langchain_market_context_tools(snapshot, recent_entries)
            tools.append(langchain_news_search_tool(news_provider))
            payload = json.dumps(
                {
                    "ticker": snapshot.ticker,
                    "available_tools": ["summarize_news_urls", "read_news_url", "search_market_news"],
                    "allowed_search_market_news_strategy": ["everything", "top_headlines"],
                    "allowed_sort_by": ["publishedAt", "relevancy", "popularity"],
                    "news_count": len(snapshot.news),
                }
            )
            observations, _metadata = invoke_tools(SYSTEM_PROMPT, payload, tools)
            if observations:
                return observations
        except Exception as error:
            logger.warning("news.tools.failed ticker=%s error=%s", snapshot.ticker, error)

    observations, _blocked = run_market_context_tools(snapshot, recent_entries, [{"name": "summarize_news_urls", "args": {}}])
    return observations
