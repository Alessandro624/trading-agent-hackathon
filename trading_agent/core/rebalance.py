from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
class AllocationTarget:
    sector: str
    target_fraction: float


@dataclass(frozen=True)
class RebalancePlan:
    current_sectors: dict[str, list[str]]
    missing_sectors: list[str]
    suggested_buys: list[str]
    rationale: str
    allocation_targets: list[AllocationTarget] = field(default_factory=list)
    current_allocation: dict[str, float] = field(default_factory=dict)
    allocation_gaps: dict[str, float] = field(default_factory=dict)


def build_rebalance_plan(
    portfolio: dict[str, Any] | None,
    llm_client: Any | None = None,
    *,
    allocation_targets: list[AllocationTarget] | None = None,
    sector_classifier: Any | None = None,
) -> RebalancePlan:
    from trading_agent.core.portfolio import positions as portfolio_positions
    from trading_agent.core.sector_classifier import SectorClassifier

    classifier = sector_classifier or SectorClassifier()
    positions = portfolio_positions(portfolio)
    held_tickers = [s for s, p in positions.items() if p.get("qty", 0) > 0]

    current_sectors: dict[str, list[str]] = {}
    current_allocation: dict[str, float] = {}
    sector_values: dict[str, float] = {}
    total_value = 0.0
    for ticker in held_tickers:
        sector = _sector_for(ticker, llm_client, classifier)
        current_sectors.setdefault(sector, []).append(ticker)
        try:
            mv = float((positions.get(ticker) or {}).get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        sector_values[sector] = sector_values.get(sector, 0.0) + mv
        total_value += mv
    if total_value > 0:
        current_allocation = {s: round(v / total_value, 4) for s, v in sector_values.items()}

    if allocation_targets:
        targets = [(t.sector.lower(), float(t.target_fraction)) for t in allocation_targets if t.target_fraction > 0]
        gaps: dict[str, float] = {}
        for sector, target_fraction in targets:
            current_fraction = current_allocation.get(sector, 0.0)
            gap = target_fraction - current_fraction
            if gap > 0.01:
                gaps[sector] = round(gap, 4)
        suggested: list[str] = []
        for sector, gap in sorted(gaps.items(), key=lambda item: -item[1]):
            candidates = classifier.suggest_tickers_for_sector(sector, llm_client, limit=3)
            for ticker in candidates:
                if ticker not in held_tickers and ticker not in suggested:
                    suggested.append(ticker)
                    break
            if len(suggested) >= 3:
                break
        rationale = f"Current allocation: {current_allocation}. " f"Targets: {dict(targets)}. " f"Gaps: {gaps}. " f"Suggested buys: {suggested}."
        logger.info(
            "rebalance.plan.targets current=%s targets=%s gaps=%s suggested=%s",
            current_allocation,
            dict(targets),
            gaps,
            suggested,
        )
        return RebalancePlan(
            current_sectors=current_sectors,
            missing_sectors=[s for s, _ in targets if s not in current_sectors],
            suggested_buys=suggested,
            rationale=rationale,
            allocation_targets=[AllocationTarget(sector=s, target_fraction=f) for s, f in targets],
            current_allocation=current_allocation,
            allocation_gaps=gaps,
        )

    missing = [s for s in SECTOR_UNIVERSE if s not in current_sectors]
    suggested = []
    for sector in missing[:3]:
        candidates = [t for t in SECTOR_UNIVERSE[sector] if t not in held_tickers]
        if candidates:
            suggested.append(candidates[0])
        else:
            for ticker in classifier.suggest_tickers_for_sector(sector, llm_client, limit=3):
                if ticker not in held_tickers and ticker not in suggested:
                    suggested.append(ticker)
                    break

    rationale = f"Current sectors: {list(current_sectors.keys()) or ['none']}. " f"Missing: {missing[:3]}. " f"Suggested buys: {suggested}."
    logger.info("rebalance.plan current=%s missing=%s suggested=%s", list(current_sectors.keys()), missing[:3], suggested)
    return RebalancePlan(
        current_sectors=current_sectors,
        missing_sectors=missing[:3],
        suggested_buys=suggested,
        rationale=rationale,
        current_allocation=current_allocation,
    )


def _sector_for(ticker: str, llm_client: Any | None, classifier: Any | None = None) -> str:
    from trading_agent.core.sector_classifier import SectorClassifier

    cls = classifier or SectorClassifier()
    return cls.classify(ticker, llm_client)
