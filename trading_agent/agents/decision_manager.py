from __future__ import annotations

import json
from dataclasses import asdict

from trading_agent.core import (
    AnalystDecisionOutput,
    LlmClient,
    MarketSnapshot,
    NewsOpinion,
    RetryPolicy,
    RiskAssessment,
    TechnicalOpinion,
    TradingDecision,
    parse_analyst_output,
    parse_human_intent,
    safe_rationale_details,
)
from trading_agent.journal import compact_recent_entries
from trading_agent.utils import get_logger, llm_metadata

logger = get_logger("decision_manager")

SYSTEM_PROMPT = """
You are the Decision Manager in a multi-agent trading system. 
INPUTS: Scout snapshot, NewsOpinion, TechnicalOpinion, RiskAssessment, journal memory, and optional HumanIntent.
OUTPUT: Produce a SINGLE structured TradingDecision derived from the inputs below. 
DO NOT invent, infer, or hallucinate any price, news item, indicator value, portfolio state, or agent opinion.

HARD CONSTRAINTS (DO NOT OVERRIDE):
1. If risk_assessment.can_trade is false THEN HOLD, quantity 0.
2. BUY quantity must not exceed risk_assessment.max_buy_quantity.
3. SELL quantity must not exceed risk_assessment.max_sell_quantity.
4. BUY and SELL require quantity > 0. HOLD and WAIT require quantity = 0.
5. If any required input is missing or flagged stale, default to WAIT and state which input is unavailable.

PRIORITIES:
- When Signals Conflict, ALWAYS prefer RiskAssessment, THEN TechnicalOpinion OR NewsOpinion, THEN Journal Memory. 
- HumanIntent carries the same weight as TechnicalOpinion UNLESS it violates a hard constraint; explain why to use or exclude the HumanIntent.
  If HumanIntent contains a conditional_sell:
    1. Determine whether the current ticker has material EXPOSURE to human_intent.impact_topic using validated evidence only.
    2. If EXPOSURE is CONFIRMED and impact is plausible THEN SELL and EXPLAIN.
    3. If EXPOSURE is UNCONFIRMED or speculative THEN HOLD and EXPLAIN.

ACTION DEFINITIONS:
  BUY  - Positive cross-agent consensus, within risk limits, no blocker.
  SELL - Risk signal triggered, negative consensus, or justified HumanIntent.
  HOLD - conflicting signals with no clear edge.
  WAIT - Temporary blocker only: a data gap, a pending news event, or a specific condition to re-evaluate next cycle. Always name the blocker and the expected resolution.

SELL GUIDELINES:
Actively evaluate SELL whenever holding a position. Do not default to HOLD when bearish signals are present.

SELL is the expected whenever:
- take_profit_triggered is true THEN SELL UNLESS two or more agents provide strong contrary evidence.
- stop_loss_triggered is true THEN SELL UNLESS two or more agents provide strong contrary evidence.
- Holding AND technical_opinion.trend = "bearish" with strength > 0.5 THEN POSSIBLY SELL.
- Holding AND RSI_14 > 65 THEN POSSIBLY SELL (lock in gains).
- Holding AND news_opinion.sentiment = "negative" with relevance > 0.6 THEN POSSIBLY SELL.
- Holding AND price > SMA20 by more than 3% THEN POSSIBLY SELL. Decide the portion to sell, quantity must be > 0 and < risk_assessment.max_sell_quantity.

WHEN Multiple signals are present, SELL weight increases significantly and a HOLD requires strong justification.
IF HOLD position is maintained over SELL, THEN state which signal(s) were present and teh contrary evidence that overrides them.

JOURNAL MEMORY:
Use recent journal entries to detect recency patterns (e.g., repeated WAIT cycles, prior stop-loss events). 
Flag if current decision repeats a recent pattern without new justification.
"""



def decide_from_opinions(
    snapshot: MarketSnapshot,
    recent_entries: list,
    technical: TechnicalOpinion,
    news: NewsOpinion,
    risk: RiskAssessment,
    llm_client: LlmClient,
    retry_policy: RetryPolicy | None = None,
    human_input: list[str] | None = None,
    human_intent: dict | None = None,
) -> TradingDecision:
    metadata = llm_metadata(llm_client)
    if not risk.can_trade:
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=0.25,
            rationale=f"Risk Manager blocked trading: {risk.blocked_reason}",
            used_data_sources=[*snapshot.data_sources, "technical_opinion", "news_opinion", "risk_assessment"],
            guardrails_triggered=["guardrail:risk_manager_hold"],
            llm_metadata=metadata,
            rationale_details=_multi_agent_details(technical, news, risk, "Risk Manager blocked trading."),
        )

    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    parsed_human_intent = parse_human_intent(human_input or [])
    human_intent_payload = human_intent or parsed_human_intent.to_dict()
    payload = json.dumps(
        {
            "snapshot": {
                "ticker": snapshot.ticker,
                "price": snapshot.price,
                "price_confidence": snapshot.price_confidence,
                "news_confidence": snapshot.news_confidence,
                "technical_confidence": snapshot.technical_indicators.confidence,
                "failures": snapshot.failures,
                "guardrails": snapshot.guardrails_triggered,
            },
            "technical_opinion": asdict(technical),
            "news_opinion": asdict(news),
            "risk_assessment": asdict(risk),
            "recent_journal": compact_recent_entries(recent_entries, ticker=snapshot.ticker),
            "human_input": human_input or [],
            "human_intent": human_intent_payload,
        },
        default=str,
    )
    errors: list[str] = []

    def call_and_parse() -> AnalystDecisionOutput:
        nonlocal metadata
        structured = getattr(llm_client, "complete_structured", None)
        if callable(structured):
            parsed = structured(SYSTEM_PROMPT, payload, AnalystDecisionOutput)
        else:
            raw = llm_client.complete_json(SYSTEM_PROMPT, payload)
            parsed = parse_analyst_output(raw)
        metadata = llm_metadata(llm_client)
        return parsed

    def on_failure(attempt: int, error: Exception) -> None:
        errors.append(f"decision manager retry {attempt}: {error}")
        logger.warning("decision.validation.retry ticker=%s attempt=%s error=%s", snapshot.ticker, attempt, error)

    try:
        parsed = retry_policy.run(call_and_parse, on_failure=on_failure)
    except Exception:
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=0.2,
            rationale="Decision Manager output failed structured validation after retry; safe HOLD.",
            used_data_sources=snapshot.data_sources,
            guardrails_triggered=["guardrail:invalid_decision_manager_output", *errors],
            llm_metadata=metadata,
            rationale_details=safe_rationale_details(snapshot, "Decision Manager validation failed.", errors),
        )

    deterministic_human_trade = _deterministic_human_trade_decision(snapshot, parsed, risk, parsed_human_intent)
    if deterministic_human_trade is not None:
        return deterministic_human_trade

    action_limit = _action_limit(parsed.action, risk)
    if parsed.action in {"BUY", "SELL"} and parsed.quantity > action_limit:
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=min(parsed.confidence, 0.35),
            rationale=f"Decision Manager requested {parsed.action} quantity {parsed.quantity}, above risk max {action_limit}; safe HOLD.",
            used_data_sources=parsed.used_data_sources or snapshot.data_sources,
            guardrails_triggered=["guardrail:decision_quantity_above_action_risk_limit"],
            llm_metadata=metadata,
            rationale_details=_multi_agent_details(technical, news, risk, "Requested quantity exceeded deterministic action-specific risk limit."),
        )

    details = _multi_agent_details(technical, news, risk, parsed.rationale_details.get("summary", parsed.rationale))
    details.update(parsed.rationale_details or {})
    return TradingDecision(
        ticker=snapshot.ticker,
        action=parsed.action,
        quantity=parsed.quantity,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        used_data_sources=[*(parsed.used_data_sources or snapshot.data_sources), "multi_agent_opinions"],
        llm_metadata=metadata,
        rationale_details=details,
    )


def _multi_agent_details(
    technical: TechnicalOpinion,
    news: NewsOpinion,
    risk: RiskAssessment,
    summary: str,
) -> dict:
    return {
        "summary": summary,
        "evidence": [*technical.evidence[:2], *news.evidence[:2]],
        "risks": [*technical.risks[:2], *news.risks[:2], *risk.reasons[:2]],
        "data_quality": f"technical {technical.confidence:.2f}, news {news.confidence:.2f}, risk max_quantity {risk.max_quantity}",
        "multi_agent": {
            "technical": asdict(technical),
            "news": asdict(news),
            "risk": asdict(risk),
        },
    }


def _action_limit(action: str, risk: RiskAssessment) -> int:
    if action == "BUY":
        return risk.max_buy_quantity
    if action == "SELL":
        return risk.max_sell_quantity
    return 0


def _deterministic_human_trade_decision(
    snapshot: MarketSnapshot,
    parsed: AnalystDecisionOutput,
    risk: RiskAssessment,
    human_intent,
) -> TradingDecision | None:
    action: str | None = None
    quantity = 0
    reasons: list[str] = []
    guardrails: list[str] = []

    if human_intent.requested_action == "BUY" and _human_intent_applies_to_ticker(human_intent, snapshot.ticker):
        action = "BUY"
        quantity = min(1, risk.max_buy_quantity)
        reasons.append("Human input requested BUY")
        guardrails.append("guardrail:human_buy_intent")

    if human_intent.requested_action == "SELL" and _human_intent_applies_to_ticker(human_intent, snapshot.ticker):
        action = "SELL"
        quantity = risk.max_sell_quantity
        reasons.append("Human input requested SELL")
        guardrails.append("guardrail:human_sell_intent")

    if action is None and "position_sweep" in human_intent.intents and risk.max_sell_quantity > 0:
        action = "SELL"
        quantity = risk.max_sell_quantity
        reasons.append(_human_position_sweep_reason(snapshot.ticker, human_intent))
        guardrails.append("guardrail:human_position_sweep_sell")

    if risk.stop_loss_triggered:
        action = action or "SELL"
        quantity = quantity or risk.max_sell_quantity
        reasons.append("stop-loss risk flag is active")
        guardrails.append("guardrail:stop_loss_sell")
    if risk.take_profit_triggered:
        action = action or "SELL"
        quantity = quantity or risk.max_sell_quantity
        reasons.append("take-profit risk flag is active")
        guardrails.append("guardrail:take_profit_sell")

    if action is None or quantity <= 0 or not reasons:
        return None
    return TradingDecision(
        ticker=snapshot.ticker,
        action=action,
        quantity=quantity,
        confidence=max(parsed.confidence, 0.65),
        rationale=(
            "; ".join(reasons) + f". Overriding draft {parsed.action} to {action} within risk limits " f"(max_buy_quantity={risk.max_buy_quantity}, max_sell_quantity={risk.max_sell_quantity})."
        ),
        used_data_sources=[*(parsed.used_data_sources or snapshot.data_sources), "risk_assessment", "human_intent"],
        guardrails_triggered=guardrails,
        rationale_details={
            **(parsed.rationale_details or {}),
            "summary": "; ".join(reasons),
            "human_intent": human_intent.to_dict(),
            "human_impact_topic": human_intent.impact_topic,
            "human_selected_ticker": snapshot.ticker,
            "risk_flags": risk.risk_flags,
        },
    )


def _human_intent_applies_to_ticker(human_intent, ticker: str) -> bool:
    return not human_intent.tickers or ticker.upper() in human_intent.tickers


def _human_position_sweep_reason(ticker: str, human_intent) -> str:
    topic = human_intent.impact_topic or "the user's broad risk topic"
    return f"Human input requested a broad SELL sweep for open positions potentially impacted by {topic}. " f"Selected {ticker.upper()} because it is an open position in the human-instruction sweep"