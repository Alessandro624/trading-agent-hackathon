from __future__ import annotations

from trading_agent.core import MarketSnapshot, TechnicalOpinion
from trading_agent.utils import get_logger

logger = get_logger("technical_analyst")


def technical_opinion(snapshot: MarketSnapshot) -> TechnicalOpinion:
    """Turn validated indicators into a compact technical view for the decision agent."""
    indicators = snapshot.technical_indicators
    summary = indicators.summary()
    evidence = [summary]
    risks: list[str] = []

    bullish = 0
    bearish = 0
    if indicators.macd is not None and indicators.macd_signal is not None:
        bullish += indicators.macd > indicators.macd_signal
        bearish += indicators.macd <= indicators.macd_signal
    if indicators.sma_20 is not None and indicators.sma_50 is not None:
        bullish += indicators.sma_20 > indicators.sma_50
        bearish += indicators.sma_20 <= indicators.sma_50
    if indicators.rsi_14 is not None:
        if indicators.rsi_14 >= 70:
            risks.append("RSI is overbought")
        elif indicators.rsi_14 <= 30:
            evidence.append("RSI is oversold")

    trend = "neutral"
    if bullish > bearish:
        trend = "bullish"
    elif bearish > bullish:
        trend = "bearish"

    confidence = {"high": 0.85, "medium": 0.6, "low": 0.35, "none": 0.1}[indicators.confidence]
    strength = min(1.0, max(bullish, bearish) / 2) if bullish or bearish else 0.0
    if indicators.notes:
        risks.extend(indicators.notes)

    opinion = TechnicalOpinion(
        trend=trend,
        strength=strength,
        confidence=confidence,
        evidence=evidence,
        risks=risks or ["No major technical risk identified"],
    )
    logger.info("technical.opinion ticker=%s trend=%s confidence=%.2f", snapshot.ticker, opinion.trend, opinion.confidence)
    return opinion
