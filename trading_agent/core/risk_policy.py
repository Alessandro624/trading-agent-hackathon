from __future__ import annotations

import os
from dataclasses import dataclass
from math import floor
from typing import Any

from trading_agent.core.actions import is_trade_action
from trading_agent.core.human_risk import HumanRiskProfile
from trading_agent.core.models import MarketSnapshot
from trading_agent.core.portfolio import cash as portfolio_cash
from trading_agent.core.portfolio import portfolio_value as portfolio_total_value
from trading_agent.core.portfolio import positions as portfolio_positions


@dataclass(frozen=True)
class RiskPolicy:
    max_quantity_absolute: int = 10
    max_notional_per_order: float = 1000.0
    cash_risk_fraction: float = 0.10
    portfolio_risk_fraction: float = 0.02
    stop_loss_fraction: float = 0.03
    take_profit_fraction: float = 0.02

    def adjusted_for_human_profile(self, profile: HumanRiskProfile | dict[str, Any] | None) -> "RiskPolicy":
        risk_profile = _coerce_human_risk_profile(profile)
        if risk_profile is None or risk_profile.risk_preference == "neutral":
            return self
        if risk_profile.risk_preference == "risk_off":
            scale = 0.50
        else:
            aggressiveness = max(risk_profile.buy_aggressiveness, risk_profile.sell_aggressiveness)
            scale = min(2.0, 1.0 + aggressiveness)
        return RiskPolicy(
            max_quantity_absolute=max(1, floor(self.max_quantity_absolute * scale)),
            max_notional_per_order=max(1.0, self.max_notional_per_order * scale),
            cash_risk_fraction=max(0.01, min(1.0, self.cash_risk_fraction * scale)),
            portfolio_risk_fraction=max(0.005, min(1.0, self.portfolio_risk_fraction * scale)),
            stop_loss_fraction=self.stop_loss_fraction,
            take_profit_fraction=self.take_profit_fraction,
        )

    @classmethod
    def from_env(cls) -> "RiskPolicy":
        return cls(
            max_quantity_absolute=int(os.getenv("TRADING_MAX_QUANTITY_ABSOLUTE", "10")),
            max_notional_per_order=float(os.getenv("TRADING_MAX_NOTIONAL_PER_ORDER", "1000")),
            cash_risk_fraction=float(os.getenv("TRADING_CASH_RISK_FRACTION", "0.10")),
            portfolio_risk_fraction=float(os.getenv("TRADING_PORTFOLIO_RISK_FRACTION", "0.02")),
            stop_loss_fraction=float(os.getenv("TRADING_STOP_LOSS_FRACTION", "0.03")),
            take_profit_fraction=float(os.getenv("TRADING_TAKE_PROFIT_FRACTION", "0.02")),
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

    def max_buy_quantity(
        self,
        *,
        cash: float | None,
        estimated_price: float | None,
        portfolio_value: float | None,
        current_market_value: float,
    ) -> int:
        base_quantity = self.max_quantity(cash=cash, estimated_price=estimated_price, portfolio_value=portfolio_value)
        if estimated_price is None or estimated_price <= 0 or portfolio_value is None or portfolio_value <= 0:
            return base_quantity
        max_position_notional = portfolio_value * self.portfolio_risk_fraction
        remaining_position_notional = max(0.0, max_position_notional - current_market_value)
        return min(base_quantity, max(0, floor(remaining_position_notional / estimated_price)))

    def max_sell_quantity(self, *, current_quantity: float) -> int:
        return max(0, min(self.max_quantity_absolute, floor(max(0.0, current_quantity))))

    def max_quantity_for_action(
        self,
        *,
        action: str,
        ticker: str,
        portfolio: dict[str, Any] | None,
        estimated_price: float | None,
    ) -> int:
        if not is_trade_action(action):
            return 0
        positions = portfolio_positions(portfolio)
        current_position = positions.get(ticker.upper(), {})
        if action == "SELL":
            return self.max_sell_quantity(current_quantity=current_position.get("qty", 0.0))
        return self.max_buy_quantity(
            cash=portfolio_cash(portfolio),
            estimated_price=estimated_price,
            portfolio_value=portfolio_total_value(portfolio),
            current_market_value=current_position.get("market_value", 0.0),
        )

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
            f"stop_loss_fraction={self.stop_loss_fraction:.2f}, "
            f"take_profit_fraction={self.take_profit_fraction:.2f}, "
            f"cash={cash_text}, estimated_price={price_text}"
            f", portfolio_value={portfolio_text}"
        )


def build_position_sizing_context(
    snapshot: MarketSnapshot,
    portfolio: dict[str, Any] | None,
    risk_policy: RiskPolicy | None = None,
    human_risk_profile: HumanRiskProfile | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose deterministic risk limits to the Analyst prompt."""
    risk_policy = risk_policy or RiskPolicy.from_env()
    risk_profile = _coerce_human_risk_profile(human_risk_profile)
    effective_policy = risk_policy.adjusted_for_human_profile(risk_profile)
    cash = portfolio_cash(portfolio)
    portfolio_value = portfolio_total_value(portfolio)
    positions = portfolio_positions(portfolio)
    current_position = positions.get(snapshot.ticker, {})
    current_qty = current_position.get("qty", 0.0)
    current_market_value = current_position.get("market_value", 0.0)
    max_buy_quantity = effective_policy.max_buy_quantity(
        cash=cash,
        estimated_price=snapshot.price,
        portfolio_value=portfolio_value,
        current_market_value=current_market_value,
    )
    max_sell_quantity = effective_policy.max_sell_quantity(current_quantity=current_qty)
    stop_loss_triggered = _stop_loss_triggered(snapshot.price, current_position, effective_policy.stop_loss_fraction)
    take_profit_triggered = _take_profit_triggered(snapshot.price, current_position, effective_policy.take_profit_fraction)
    risk_flags: list[str] = []
    if stop_loss_triggered:
        risk_flags.append("stop_loss_triggered")
    if take_profit_triggered:
        risk_flags.append("take_profit_triggered")
    return {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "positions": positions,
        "current_position": current_position,
        "estimated_price": snapshot.price,
        "max_quantity": max(max_buy_quantity, max_sell_quantity),
        "max_buy_quantity": max_buy_quantity,
        "max_sell_quantity": max_sell_quantity,
        "take_profit_triggered": take_profit_triggered,
        "stop_loss_triggered": stop_loss_triggered,
        "risk_flags": risk_flags,
        "human_risk_profile": risk_profile.to_dict() if risk_profile else HumanRiskProfile().to_dict(),
        "valid_quantity_rule": (
            f"BUY quantity must be from 1 to {max_buy_quantity}. "
            f"SELL quantity must be from 1 to {max_sell_quantity}. "
            "Prefer quantity 1 for conservative BUY/SELL unless the rationale justifies a higher value. "
            "If the selected action limit is 0, choose HOLD or WAIT with quantity 0. "
            "HOLD/WAIT quantity must always be 0."
        ),
        "portfolio_context": _portfolio_context(snapshot.ticker, positions),
        "risk_explanation": effective_policy.explain(
            cash=cash,
            estimated_price=snapshot.price,
            portfolio_value=portfolio_value,
        )
        + _human_risk_explanation(risk_profile)
        + _position_risk_explanation(snapshot.price, current_position, portfolio_value, effective_policy),
    }


def _portfolio_context(ticker: str, positions: dict[str, dict[str, float]]) -> str:
    current = positions.get(ticker.upper())
    if not positions:
        return "No open positions."
    if not current or current.get("qty", 0.0) == 0:
        return f"No current position in {ticker.upper()}; other open positions: {', '.join(sorted(positions))}."
    avg_entry = current.get("avg_entry_price", 0.0)
    avg_text = f", avg_entry_price={avg_entry:.2f}" if avg_entry > 0 else ""
    return f"Current position in {ticker.upper()}: qty={current.get('qty', 0.0):.2f}, " f"market_value={current.get('market_value', 0.0):.2f}{avg_text}. Consider this before adding exposure."


def _stop_loss_triggered(
    current_price: float | None,
    current_position: dict[str, float],
    stop_loss_fraction: float,
) -> bool:
    avg_entry_price = current_position.get("avg_entry_price", 0.0)
    qty = current_position.get("qty", 0.0)
    if current_price is None or current_price <= 0 or avg_entry_price <= 0 or qty <= 0:
        return False
    return current_price <= avg_entry_price * (1 - stop_loss_fraction)


def _position_risk_explanation(
    current_price: float | None,
    current_position: dict[str, float],
    portfolio_value: float | None,
    risk_policy: RiskPolicy,
) -> str:
    current_market_value = current_position.get("market_value", 0.0)
    avg_entry_price = current_position.get("avg_entry_price", 0.0)
    max_position_notional = None if portfolio_value is None else portfolio_value * risk_policy.portfolio_risk_fraction
    remaining_position_notional = None if max_position_notional is None else max(0.0, max_position_notional - current_market_value)
    parts = [
        f", current_position_market_value={current_market_value:.2f}",
        f", remaining_position_notional={remaining_position_notional:.2f}" if remaining_position_notional is not None else ", remaining_position_notional=unknown",
    ]
    if avg_entry_price > 0:
        parts.append(f", avg_entry_price={avg_entry_price:.2f}")
    if _take_profit_triggered(current_price, current_position, risk_policy.take_profit_fraction):
        parts.append(f", take_profit_triggered=True threshold={risk_policy.take_profit_fraction:.2f}")
    if _stop_loss_triggered(current_price, current_position, risk_policy.stop_loss_fraction):
        parts.append(f", stop_loss_triggered=True threshold={risk_policy.stop_loss_fraction:.2f}")
    return "".join(parts)


def _take_profit_triggered(
    current_price: float | None,
    current_position: dict[str, float],
    take_profit_fraction: float,
) -> bool:
    avg_entry_price = current_position.get("avg_entry_price", 0.0)
    qty = current_position.get("qty", 0.0)
    if current_price is None or current_price <= 0 or avg_entry_price <= 0 or qty <= 0:
        return False
    return current_price >= avg_entry_price * (1 + take_profit_fraction)


def _coerce_human_risk_profile(profile: HumanRiskProfile | dict[str, Any] | None) -> HumanRiskProfile | None:
    if profile is None:
        return None
    if isinstance(profile, HumanRiskProfile):
        return profile
    if isinstance(profile, dict):
        try:
            return HumanRiskProfile(
                risk_preference=str(profile.get("risk_preference") or "neutral"),
                buy_aggressiveness=float(profile.get("buy_aggressiveness") or 0.0),
                sell_aggressiveness=float(profile.get("sell_aggressiveness") or 0.0),
                rationale=str(profile.get("rationale") or "Human risk profile supplied as dict."),
            )
        except (TypeError, ValueError):
            return None
    return None


def _human_risk_explanation(profile: HumanRiskProfile | None) -> str:
    if profile is None or profile.risk_preference == "neutral":
        return ", human_risk_profile=neutral"
    return (
        f", human_risk_profile={profile.risk_preference}"
        f" buy_aggressiveness={profile.buy_aggressiveness:.2f}"
        f" sell_aggressiveness={profile.sell_aggressiveness:.2f}"
        f" rationale={profile.rationale}"
    )
