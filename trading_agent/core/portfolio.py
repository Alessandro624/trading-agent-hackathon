from __future__ import annotations

from typing import Any


def cash(portfolio: dict[str, Any] | None) -> float | None:
    if not portfolio:
        return None
    return float_or_none(portfolio.get("cash"))


def portfolio_value(portfolio: dict[str, Any] | None) -> float | None:
    if not portfolio:
        return None
    return float_or_none(portfolio.get("portfolio_value"))


def normalize_positions(positions: Any) -> dict[str, dict[str, float]]:
    if isinstance(positions, dict):
        return {str(symbol).upper(): position_payload(payload) for symbol, payload in positions.items()}
    if isinstance(positions, list):
        normalized: dict[str, dict[str, float]] = {}
        for item in positions:
            if isinstance(item, dict) and item.get("symbol"):
                normalized[str(item["symbol"]).upper()] = position_payload(item)
        return normalized
    return {}


def positions(portfolio: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not portfolio:
        return {}
    return normalize_positions(portfolio.get("positions"))


def position_for(portfolio: dict[str, Any] | None, ticker: str) -> dict[str, float]:
    return positions(portfolio).get(ticker.upper(), {})


def position_payload(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {"qty": float_or_zero(payload), "market_value": 0.0, "avg_entry_price": 0.0}
    return {
        "qty": float_or_zero(payload.get("qty")),
        "market_value": float_or_zero(payload.get("market_value")),
        "avg_entry_price": float_or_zero(payload.get("avg_entry_price")),
    }


def float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def float_or_zero(value: Any) -> float:
    parsed = float_or_none(value)
    return parsed if parsed is not None else 0.0
