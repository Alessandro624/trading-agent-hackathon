from __future__ import annotations

from trading_agent.core import MarketSnapshot, RiskAssessment, RiskPolicy, build_position_sizing_context, force_hold_reasons
from trading_agent.core.human_risk import resolve_human_risk_profile
from trading_agent.utils import get_logger

logger = get_logger("risk_manager")


def assess_risk(
    snapshot: MarketSnapshot,
    portfolio: dict | None,
    risk_policy: RiskPolicy | None = None,
    human_input: list[str] | None = None,
    llm_client=None,
    human_risk_profile=None,
) -> RiskAssessment:
    """Apply deterministic tradeability and sizing constraints before the LLM decides."""
    base_policy = risk_policy or RiskPolicy.from_env()
    human_risk_profile = human_risk_profile or resolve_human_risk_profile(human_input or [], llm_client)
    sizing = build_position_sizing_context(snapshot, portfolio, base_policy, human_risk_profile=human_risk_profile)
    portfolio_context = sizing["portfolio_context"]
    reasons = [sizing["risk_explanation"], portfolio_context, *sizing["risk_flags"]]
    hold_reasons = force_hold_reasons(snapshot.price_confidence, snapshot.guardrails_triggered)
    if hold_reasons:
        reason = "; ".join(hold_reasons)
        logger.warning("risk.blocked ticker=%s reason=%s", snapshot.ticker, reason)
        return RiskAssessment(
            can_trade=False,
            max_quantity=0,
            reasons=[*reasons, *hold_reasons],
            blocked_reason=reason,
            portfolio_context=portfolio_context,
            max_buy_quantity=int(sizing["max_buy_quantity"]),
            max_sell_quantity=int(sizing["max_sell_quantity"]),
            stop_loss_triggered=bool(sizing["stop_loss_triggered"]),
            take_profit_triggered=bool(sizing["take_profit_triggered"]),
            risk_flags=list(sizing["risk_flags"]),
            human_risk_profile=dict(sizing["human_risk_profile"]),
        )

    max_quantity = int(sizing["max_quantity"])
    if max_quantity <= 0:
        logger.warning("risk.blocked ticker=%s reason=max_quantity_zero", snapshot.ticker)
        return RiskAssessment(
            can_trade=False,
            max_quantity=0,
            reasons=[*reasons, "risk policy allows no quantity"],
            blocked_reason="risk policy allows no quantity",
            portfolio_context=portfolio_context,
            max_buy_quantity=0,
            max_sell_quantity=0,
            stop_loss_triggered=bool(sizing["stop_loss_triggered"]),
            take_profit_triggered=bool(sizing["take_profit_triggered"]),
            risk_flags=list(sizing["risk_flags"]),
            human_risk_profile=dict(sizing["human_risk_profile"]),
        )

    assessment = RiskAssessment(
        can_trade=True,
        max_quantity=max_quantity,
        reasons=reasons,
        blocked_reason=None,
        portfolio_context=portfolio_context,
        max_buy_quantity=int(sizing["max_buy_quantity"]),
        max_sell_quantity=int(sizing["max_sell_quantity"]),
        stop_loss_triggered=bool(sizing["stop_loss_triggered"]),
        take_profit_triggered=bool(sizing["take_profit_triggered"]),
        risk_flags=list(sizing["risk_flags"]),
        human_risk_profile=dict(sizing["human_risk_profile"]),
    )
    logger.info(
        "risk.ok ticker=%s max_buy=%s max_sell=%s stop_loss=%s take_profit=%s",
        snapshot.ticker,
        assessment.max_buy_quantity,
        assessment.max_sell_quantity,
        assessment.stop_loss_triggered,
        assessment.take_profit_triggered,
    )
    return assessment
