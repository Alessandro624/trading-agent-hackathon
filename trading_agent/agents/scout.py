from __future__ import annotations

from trading_agent.core import (
    MarketDataProvider,
    MarketSnapshot,
    NewsProvider,
    RetryPolicy,
    news_confidence,
    price_confidence,
    utc_now_iso,
)
from trading_agent.technicals import calculate_indicators
from trading_agent.utils import get_logger

logger = get_logger("scout")


def scout_snapshot(
    ticker: str,
    market_data: MarketDataProvider,
    news_provider: NewsProvider,
    retry_policy: RetryPolicy | None = None,
) -> MarketSnapshot:
    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    symbol = ticker.upper()
    failures: list[str] = []
    guardrails: list[str] = []
    retry_count = 0
    logger.info("scout.start ticker=%s", symbol)

    def track_failure(attempt: int, error: Exception) -> None:
        nonlocal retry_count
        retry_count = attempt
        failures.append(f"price attempt {attempt}: {error}")

    price: float | None = None
    timestamp = utc_now_iso()
    try:
        payload = retry_policy.run(lambda: market_data.get_price(ticker), on_failure=track_failure)
        price = float(payload["price"])
        timestamp = payload.get("timestamp") or timestamp
        logger.info("scout.price.ok ticker=%s price=%.2f", symbol, price)
        if price <= 0:
            guardrails.append("guardrail:anomalous_price_detected")
            failures.append(f"anomalous price received: {price}")
            logger.warning("scout.price.anomalous ticker=%s price=%.2f", symbol, price)
            price = None
    except Exception as error:
        failures.append(f"price unavailable after retry: {error}")
        logger.warning("scout.price.fail ticker=%s error=%s", symbol, error)

    news_failures: list[str] = []
    try:
        news = news_provider.search_news(ticker)
        logger.info("scout.news.ok ticker=%s count=%s", symbol, len(news))
    except Exception as error:
        news = []
        news_failures.append(f"news unavailable: {error}")
        failures.extend(news_failures)
        logger.warning("scout.news.fail ticker=%s error=%s", symbol, error)

    try:
        closes = market_data.get_closes(ticker, limit=60)
        technicals = calculate_indicators(closes)
        if technicals.confidence == "none":
            logger.warning(
                "scout.technicals.insufficient ticker=%s closes=%s required=26 confidence=%s",
                symbol,
                len(closes),
                technicals.confidence,
            )
        else:
            logger.info(
                "scout.technicals.ok ticker=%s closes=%s confidence=%s",
                symbol,
                len(closes),
                technicals.confidence,
            )
    except Exception as error:
        technicals = calculate_indicators([])
        failures.append(f"technical indicators unavailable: {error}")
        logger.warning("scout.technicals.fail ticker=%s error=%s", symbol, error)

    data_sources = []
    if price is not None:
        data_sources.append("market_data")
    if news:
        data_sources.append("news")
    if technicals.confidence != "none":
        data_sources.append("technicals")

    return MarketSnapshot(
        ticker=ticker,
        timestamp=timestamp,
        price=price,
        price_confidence=price_confidence(price, failures),
        news=news,
        news_confidence=news_confidence(news, news_failures),
        data_sources=data_sources or ["none"],
        technical_indicators=technicals,
        failures=failures,
        guardrails_triggered=guardrails,
        retry_count=retry_count,
    )
