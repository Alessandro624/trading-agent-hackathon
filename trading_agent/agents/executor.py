from __future__ import annotations

import os
from time import sleep

from trading_agent.core import BrokerClient, ExecutionResult, RetryPolicy, RiskPolicy, TradingDecision
from trading_agent.core.actions import is_passive_action
from trading_agent.core.portfolio import cash as portfolio_cash
from trading_agent.core.portfolio import float_or_none
from trading_agent.utils import get_logger, safe_portfolio_snapshot

logger = get_logger("executor")


def execute_decision(
    decision: TradingDecision,
    broker: BrokerClient,
    retry_policy: RetryPolicy | None = None,
    risk_policy: RiskPolicy | None = None,
    estimated_price: float | None = None,
    fill_poll_attempts: int | None = None,
    fill_poll_seconds: float | None = None,
) -> ExecutionResult:
    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    risk_policy = risk_policy or RiskPolicy.from_env()
    fill_poll_attempts = fill_poll_attempts if fill_poll_attempts is not None else int(os.getenv("BROKER_FILL_POLL_ATTEMPTS", "5"))
    fill_poll_seconds = fill_poll_seconds if fill_poll_seconds is not None else float(os.getenv("BROKER_FILL_POLL_SECONDS", "2"))
    logger.info("executor.start ticker=%s action=%s quantity=%s", decision.ticker, decision.action, decision.quantity)
    if is_passive_action(decision.action):
        portfolio = safe_portfolio_snapshot(broker)
        status = "waiting" if decision.action == "WAIT" else "skipped"
        reason = "wait" if decision.action == "WAIT" else "hold"
        logger.info("executor.result ticker=%s status=%s reason=%s", decision.ticker, status, reason)
        return ExecutionResult(
            decision.ticker,
            decision.action,
            status,
            None,
            f"{decision.action} decision, no order sent.",
            portfolio,
            portfolio_before=portfolio,
            requested_quantity=decision.quantity,
            current_price_at_order=estimated_price,
        )

    try:
        portfolio_before = broker.get_portfolio()
        cash = portfolio_cash(portfolio_before) or 0.0
        portfolio_value = float_or_none(portfolio_before.get("portfolio_value"))
        allowed_quantity = risk_policy.max_quantity_for_action(
            action=decision.action,
            ticker=decision.ticker,
            portfolio=portfolio_before,
            estimated_price=estimated_price,
        )
        risk_explanation = risk_policy.explain(
            cash=cash,
            estimated_price=estimated_price,
            portfolio_value=portfolio_value,
        )
        if decision.quantity <= 0:
            logger.warning(
                "executor.result ticker=%s status=blocked reason=invalid_quantity requested=%s allowed=%s",
                decision.ticker,
                decision.quantity,
                allowed_quantity,
            )
            return _blocked_execution_result(
                decision,
                f"Invalid trade quantity: requested={decision.quantity}; BUY/SELL requires quantity > 0. {risk_explanation}.",
                portfolio_before=portfolio_before,
                allowed_quantity=allowed_quantity,
                risk_explanation=risk_explanation,
                estimated_price=estimated_price,
            )
        if decision.quantity > allowed_quantity:
            logger.warning(
                "executor.risk_block ticker=%s requested=%s allowed=%s",
                decision.ticker,
                decision.quantity,
                allowed_quantity,
            )
            return _blocked_execution_result(
                decision,
                f"Quantity outside risk limits: requested={decision.quantity}, allowed={allowed_quantity}. {risk_explanation}.",
                portfolio_before=portfolio_before,
                allowed_quantity=allowed_quantity,
                risk_explanation=risk_explanation,
                estimated_price=estimated_price,
            )
        if cash <= 0 and decision.action == "BUY":
            logger.warning("executor.risk_block ticker=%s reason=insufficient_cash", decision.ticker)
            return _blocked_execution_result(
                decision,
                "Insufficient cash.",
                portfolio_before=portfolio_before,
                allowed_quantity=allowed_quantity,
                risk_explanation=risk_explanation,
                estimated_price=estimated_price,
            )
    except Exception as error:
        logger.error("executor.precheck.fail ticker=%s error=%s", decision.ticker, error)
        return ExecutionResult(
            decision.ticker,
            decision.action,
            "blocked",
            None,
            f"Portfolio pre-check failed: {error}",
            {},
            requested_quantity=decision.quantity,
            current_price_at_order=estimated_price,
        )

    attempts = 0

    def place():
        nonlocal attempts
        attempts += 1
        return broker.place_order(decision.ticker, decision.action, decision.quantity)

    try:
        logger.info("executor.order.start ticker=%s action=%s quantity=%s", decision.ticker, decision.action, decision.quantity)
        order = retry_policy.run(place)
        order = _wait_for_order_update(broker, order, fill_poll_attempts, fill_poll_seconds)
        order_status = _status_text(order.get("status", "unknown"))
        filled_avg_price = float_or_none(order.get("filled_avg_price"))
        portfolio_after = safe_portfolio_snapshot(broker)
        if "portfolio_error" in portfolio_after:
            logger.warning("executor.portfolio_after.fail ticker=%s error=%s", decision.ticker, portfolio_after["portfolio_error"])
        execution_status = "filled" if order_status == "filled" else "submitted"
        logger.info(
            "executor.result ticker=%s status=%s broker_status=%s order_id=%s",
            decision.ticker,
            execution_status,
            order_status,
            order.get("id"),
        )
        return ExecutionResult(
            decision.ticker,
            decision.action,
            execution_status,
            str(order.get("id")),
            f"Order submitted to broker with status {order_status}. Cash changes only after broker fill.",
            portfolio_after,
            retry_count=max(0, attempts - 1),
            portfolio_before=portfolio_before,
            requested_quantity=decision.quantity,
            allowed_quantity=allowed_quantity,
            risk_explanation=risk_explanation,
            current_price_at_order=estimated_price,
            filled_avg_price=filled_avg_price,
        )
    except Exception as order_error:
        portfolio_after = safe_portfolio_snapshot(broker)
        if "portfolio_error" in portfolio_after:
            logger.warning("executor.portfolio_after.fail ticker=%s error=%s", decision.ticker, portfolio_after["portfolio_error"])
        logger.error("executor.result ticker=%s status=rejected error=%s", decision.ticker, order_error)
        return ExecutionResult(
            decision.ticker,
            decision.action,
            "rejected",
            None,
            f"Order failed after retry: {order_error}",
            portfolio_after,
            retry_count=max(0, attempts - 1),
            portfolio_before=portfolio_before,
            requested_quantity=decision.quantity,
            allowed_quantity=allowed_quantity,
            risk_explanation=risk_explanation,
            current_price_at_order=estimated_price,
        )


def _wait_for_order_update(
    broker: BrokerClient,
    order: dict,
    attempts: int,
    seconds: float,
) -> dict:
    order_id = order.get("id")
    get_order = getattr(broker, "get_order", None)
    if not order_id or not callable(get_order):
        return order
    latest = dict(order)
    for attempt in range(max(1, attempts)):
        if attempt > 0 and seconds > 0:
            sleep(seconds)
        try:
            latest = get_order(str(order_id))
        except Exception as error:
            logger.warning("executor.order.poll.fail order_id=%s error=%s", order_id, error)
            return latest or order
        if _status_text(latest.get("status", "")) == "filled":
            return latest
    return latest


def _blocked_execution_result(
    decision: TradingDecision,
    message: str,
    *,
    portfolio_before: dict,
    allowed_quantity: int,
    risk_explanation: str,
    estimated_price: float | None,
) -> ExecutionResult:
    return ExecutionResult(
        decision.ticker,
        decision.action,
        "blocked",
        None,
        message,
        portfolio_before,
        portfolio_before=portfolio_before,
        requested_quantity=decision.quantity,
        allowed_quantity=allowed_quantity,
        risk_explanation=risk_explanation,
        current_price_at_order=estimated_price,
    )


def _status_text(value) -> str:
    text = str(getattr(value, "value", value)).lower()
    return text.rsplit(".", 1)[-1]
