from __future__ import annotations

import os
from dataclasses import dataclass
from math import floor
from typing import Any

from trading_agent.core.models import MarketSnapshot


@dataclass(frozen=True)
class RiskPolicy:
    max_quantity_absolute: int = 10
    max_notional_per_order: float = 1000.0
    cash_risk_fraction: float = 0.10
    portfolio_risk_fraction: float = 0.02

    @classmethod
    def from_env(cls) -> "RiskPolicy":
        return cls(
            max_quantity_absolute=int(os.getenv("TRADING_MAX_QUANTITY_ABSOLUTE", "10")),
            max_notional_per_order=float(os.getenv("TRADING_MAX_NOTIONAL_PER_ORDER", "1000")),
            cash_risk_fraction=float(os.getenv("TRADING_CASH_RISK_FRACTION", "0.10")),
            portfolio_risk_fraction=float(os.getenv("TRADING_PORTFOLIO_RISK_FRACTION", "0.02")),
        )

    def max_quantity(
        self,
        *,
        cash: float | None,
        estimated_price: float | None,
        portfolio_value: float | None = None,
    ) -> int:
        limits = [self.max_quantity_absolute]
        if estimated_price is not None and estimated_price > 0:
            limits.append(max(0, floor(self.max_notional_per_order / estimated_price)))
            if cash is not None and cash > 0:
                limits.append(max(0, floor((cash * self.cash_risk_fraction) / estimated_price)))
            if portfolio_value is not None and portfolio_value > 0:
                limits.append(max(0, floor((portfolio_value * self.portfolio_risk_fraction) / estimated_price)))
        return max(0, min(limits))

    def explain(
        self,
        *,
        cash: float | None,
        estimated_price: float | None,
        portfolio_value: float | None = None,
    ) -> str:
        computed = self.max_quantity(cash=cash, estimated_price=estimated_price, portfolio_value=portfolio_value)
        cash_text = f"{cash:.2f}" if cash is not None else "unknown"
        price_text = f"{estimated_price:.2f}" if estimated_price is not None else "unknown"
        portfolio_text = f"{portfolio_value:.2f}" if portfolio_value is not None else "unknown"
        return (
            f"risk policy max_quantity={computed} based on "
            f"absolute_cap={self.max_quantity_absolute}, "
            f"max_notional_per_order={self.max_notional_per_order:.2f}, "
            f"cash_risk_fraction={self.cash_risk_fraction:.2f}, "
            f"portfolio_risk_fraction={self.portfolio_risk_fraction:.2f}, "
            f"cash={cash_text}, estimated_price={price_text}"
            f", portfolio_value={portfolio_text}"
        )


def build_position_sizing_context(
    snapshot: MarketSnapshot,
    portfolio: dict[str, Any] | None,
    risk_policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    """Expose deterministic risk limits to the Analyst prompt."""
    risk_policy = risk_policy or RiskPolicy.from_env()
    cash = _cash(portfolio)
    portfolio_value = _portfolio_value(portfolio)
    max_quantity = risk_policy.max_quantity(
        cash=cash,
        estimated_price=snapshot.price,
        portfolio_value=portfolio_value,
    )
    positions = _positions(portfolio)
    current_position = positions.get(snapshot.ticker, {})
    return {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "positions": positions,
        "current_position": current_position,
        "estimated_price": snapshot.price,
        "max_quantity": max_quantity,
        "valid_quantity_rule": (
            f"If action is BUY or SELL, quantity must be an integer from 1 to {max_quantity}. "
            "Prefer quantity 1 for conservative BUY/SELL unless the rationale justifies a higher value. "
            "If max_quantity is 0, choose HOLD with quantity 0. If action is HOLD, quantity must be 0."
        ),
        "portfolio_context": _portfolio_context(snapshot.ticker, positions),
        "risk_explanation": risk_policy.explain(
            cash=cash,
            estimated_price=snapshot.price,
            portfolio_value=portfolio_value,
        ),
    }


def _cash(portfolio: dict[str, Any] | None) -> float | None:
    if not portfolio:
        return None
    value = portfolio.get("cash")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _portfolio_value(portfolio: dict[str, Any] | None) -> float | None:
    if not portfolio:
        return None
    value = portfolio.get("portfolio_value")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positions(portfolio: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not portfolio:
        return {}
    raw = portfolio.get("positions")
    if isinstance(raw, dict):
        return {str(symbol).upper(): _position_payload(payload) for symbol, payload in raw.items()}
    if isinstance(raw, list):
        positions: dict[str, dict[str, float]] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("symbol"):
                positions[str(item["symbol"]).upper()] = _position_payload(item)
        return positions
    return {}


def _position_payload(payload) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {"qty": _float_or_zero(payload), "market_value": 0.0}
    return {
        "qty": _float_or_zero(payload.get("qty")),
        "market_value": _float_or_zero(payload.get("market_value")),
    }


def _float_or_zero(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _portfolio_context(ticker: str, positions: dict[str, dict[str, float]]) -> str:
    current = positions.get(ticker.upper())
    if not positions:
        return "No open positions."
    if not current or current.get("qty", 0.0) == 0:
        return f"No current position in {ticker.upper()}; other open positions: {', '.join(sorted(positions))}."
    return f"Current position in {ticker.upper()}: qty={current.get('qty', 0.0):.2f}, " f"market_value={current.get('market_value', 0.0):.2f}. Consider this before adding exposure."
