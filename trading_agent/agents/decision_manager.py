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
    safe_rationale_details,
)
from trading_agent.journal import compact_recent_entries
from trading_agent.utils import get_logger, llm_metadata

logger = get_logger("decision_manager")

SYSTEM_PROMPT = """You are the Decision Manager in a multi-agent trading system.
Produce one structured TradingDecision from the Scout snapshot, NewsOpinion, TechnicalOpinion, RiskAssessment, and recent journal memory.
Never exceed risk_assessment.max_quantity. If risk_assessment.can_trade is false, choose HOLD with quantity 0.
Quantity rule is strict: BUY/SELL require quantity > 0. HOLD requires quantity 0.
Use any current position information from the risk assessment/portfolio context before adding exposure or deciding to SELL.
Do not invent prices, news, indicators, portfolio state, or agent opinions.
Write a dashboard-ready rationale with the final decision, strongest cross-agent evidence, key risks, and data-quality limits.
"""


def decide_from_opinions(
    snapshot: MarketSnapshot,
    recent_entries: list,
    technical: TechnicalOpinion,
    news: NewsOpinion,
    risk: RiskAssessment,
    llm_client: LlmClient,
    retry_policy: RetryPolicy | None = None,
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
            "recent_journal": compact_recent_entries(recent_entries),
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

    if parsed.quantity > risk.max_quantity:
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=min(parsed.confidence, 0.35),
            rationale=f"Decision Manager requested quantity {parsed.quantity}, above risk max {risk.max_quantity}; safe HOLD.",
            used_data_sources=parsed.used_data_sources or snapshot.data_sources,
            guardrails_triggered=["guardrail:decision_quantity_above_risk_limit"],
            llm_metadata=metadata,
            rationale_details=_multi_agent_details(technical, news, risk, "Requested quantity exceeded deterministic risk limit."),
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
