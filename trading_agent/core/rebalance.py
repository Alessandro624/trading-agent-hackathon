from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("trading_agent.rebalance")

SECTOR_UNIVERSE = {
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "automotive": ["TSLA", "F", "GM"],
    "manufacturing": ["CAT", "DE", "GE", "HON"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "finance": ["JPM", "BAC", "GS"],
    "consumer": ["KO", "PG", "WMT"],
}

TICKER_SECTOR: dict[str, str] = {ticker: sector for sector, tickers in SECTOR_UNIVERSE.items() for ticker in tickers}


@dataclass(frozen=True)
class RebalancePlan:
    current_sectors: dict[str, list[str]]
    missing_sectors: list[str]
    suggested_buys: list[str]
    rationale: str


def build_rebalance_plan(
    portfolio: dict[str, Any] | None,
    llm_client: Any | None = None,
) -> RebalancePlan:
    from trading_agent.core.portfolio import positions as portfolio_positions

    positions = portfolio_positions(portfolio)
    held_tickers = [s for s, p in positions.items() if p.get("qty", 0) > 0]

    current_sectors: dict[str, list[str]] = {}
    for ticker in held_tickers:
        sector = _sector_for(ticker, llm_client)
        current_sectors.setdefault(sector, []).append(ticker)

    missing = [s for s in SECTOR_UNIVERSE if s not in current_sectors]
    suggested: list[str] = []
    for sector in missing[:3]:
        candidates = [t for t in SECTOR_UNIVERSE[sector] if t not in held_tickers]
        if candidates:
            suggested.append(candidates[0])

    rationale = f"Current sectors: {list(current_sectors.keys()) or ['none']}. " f"Missing: {missing[:3]}. " f"Suggested buys: {suggested}."
    logger.info("rebalance.plan current=%s missing=%s suggested=%s", list(current_sectors.keys()), missing[:3], suggested)
    return RebalancePlan(current_sectors, missing[:3], suggested, rationale)


def _sector_for(ticker: str, llm_client: Any | None) -> str:
    known = TICKER_SECTOR.get(ticker.upper())
    if known:
        return known
    if llm_client is None:
        return "other"
    try:
        prompt = 'Return ONLY JSON: {"sector": "<one word sector>"}. ' f"What sector is {ticker}? " "Use: technology, manufacturing, energy, healthcare, finance, consumer, automotive, other."
        raw = llm_client.complete_json(prompt, ticker)
        parsed = json.loads(raw)
        return str(parsed.get("sector", "other")).lower()
    except Exception:
        return "other"
