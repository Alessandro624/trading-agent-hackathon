from __future__ import annotations

import json

from trading_agent.core import (
    LlmClient,
    MarketSnapshot,
    ReflectionOutput,
    RetryPolicy,
    TradingDecision,
    parse_reflection_output,
    snapshot_grounded_hold_rationale,
)
from trading_agent.utils import get_logger, llm_metadata

logger = get_logger("reflection")

SYSTEM_PROMPT = """
You are the Reflection Agent. 
Your role is adversarial: challenge the draft TradingDecision and block it if it is incoherent, unsafe, or unjustified.
INPUT: Draft based on TradingDecision
OUTPUT: Produce ONLY a valid JSON with this structure:
{
    "verdict": "PROCEED" | "HOLD",
    "reflection": (reasoning),
    "confidence_adjustment": (float in [-0.5, +0.5])
}

CONFIDENCE ADJUSTMENT:
Positive - draft is well-supported; increase confidence.
Negative - draft has weaknesses; decrease confidence.
Neutral  - 0.0

OVERRIDE draft decision (verdict = "HOLD") if ANY of the following apply:
- Required market data is missing and the draft ignores them (decision is not "WAIT").
- can_trade = false but action is not HOLD.
- Rationale does not support the chosen action (e.g., bearish evidence cited but BUY chosen without counter-argument).
- Active SELL signal present, HOLD chosen, and no hold_override justification provided.

ALWAYS PROCEED (verdict = "PROCEED") if:
  - The draft is already HOLD or WAIT with a named blocker THEN pass through unless a hard constraint is violated.
  - Issues are infrastructure/configuration only (missing API keys, provider fallback, model routing). These are not market data failures.

IMPORTANT: 
DO the evaluation using ONLY validated market data, technical/news confidence, guardrails, failures, quantity/risk consistency, and rationale coherence.
LLM provider fallback, missing API_KEY, or use of Ollama are NOT missing market data and are NOT by themselves a reason to HOLD.
"""


def reflect_decision(
    snapshot: MarketSnapshot,
    draft: TradingDecision,
    llm_client: LlmClient,
    retry_policy: RetryPolicy | None = None,
) -> TradingDecision:
    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    metadata = llm_metadata(llm_client)
    logger.info("reflection.llm.start ticker=%s draft_action=%s", draft.ticker, draft.action)
    prompt = json.dumps(
        {
            "snapshot_confidence": {
                "price": snapshot.price_confidence,
                "news": snapshot.news_confidence,
                "technical": snapshot.technical_indicators.confidence,
                "failures": snapshot.failures,
                "guardrails": snapshot.guardrails_triggered,
            },
            "draft": _reflection_draft_payload(draft),
        },
        default=str,
    )
    errors: list[str] = []

    def call_and_parse():
        nonlocal metadata
        structured = getattr(llm_client, "complete_structured", None)
        if callable(structured):
            parsed = structured(SYSTEM_PROMPT, prompt, ReflectionOutput)
        else:
            raw = llm_client.complete_json(SYSTEM_PROMPT, prompt)
            parsed = parse_reflection_output(raw)
        metadata = llm_metadata(llm_client)
        return parsed

    def on_failure(attempt: int, error: Exception) -> None:
        errors.append(f"reflection structured output retry {attempt}: {error}")
        logger.warning("reflection.validation.retry ticker=%s attempt=%s error=%s", draft.ticker, attempt, error)

    try:
        parsed = retry_policy.run(call_and_parse, on_failure=on_failure)
    except Exception:
        logger.error("reflection.validation.failed ticker=%s retries=%s", draft.ticker, len(errors))
        return TradingDecision(
            ticker=draft.ticker,
            action="HOLD",
            quantity=0,
            confidence=min(draft.confidence, 0.25),
            rationale=snapshot_grounded_hold_rationale(
                snapshot,
                "Reflection output failed structured validation after retry",
            ),
            used_data_sources=draft.used_data_sources,
            guardrails_triggered=[*draft.guardrails_triggered, "guardrail:invalid_reflection_output", *errors],
            reflection="Reflection output invalid after retry.",
            llm_metadata=_merge_metadata(draft.llm_metadata, metadata),
            rationale_details=draft.rationale_details,
        )

    new_confidence = max(0.0, min(1.0, draft.confidence + parsed.confidence_adjustment))
    if parsed.verdict == "HOLD":
        logger.warning("reflection.hold_override ticker=%s confidence=%.2f", draft.ticker, new_confidence)
        return TradingDecision(
            ticker=draft.ticker,
            action="HOLD",
            quantity=0,
            confidence=min(new_confidence, 0.45),
            rationale=draft.rationale + " Reflection override: " + parsed.reflection,
            used_data_sources=draft.used_data_sources,
            guardrails_triggered=[*draft.guardrails_triggered, "guardrail:reflection_hold"],
            reflection=_merge_reflection(draft.reflection, parsed.reflection),
            llm_metadata=_merge_metadata(draft.llm_metadata, metadata),
            rationale_details=_hold_override_details(draft, parsed.reflection),
        )
    draft.confidence = new_confidence
    draft.reflection = _merge_reflection(draft.reflection, parsed.reflection)
    draft.llm_metadata = _merge_metadata(draft.llm_metadata, metadata)
    draft.validate()
    logger.info("reflection.llm.result ticker=%s verdict=%s confidence=%.2f", draft.ticker, parsed.verdict, draft.confidence)
    return draft


def _merge_metadata(first: dict, second: dict) -> dict:
    if second.get("llm_fallback_used"):
        return second
    if first.get("llm_fallback_used"):
        return first
    return second or first


def _merge_reflection(existing: str | None, reflection: str) -> str:
    if not existing:
        return reflection
    return f"{existing}. Reflection: {reflection}"


def _reflection_draft_payload(draft: TradingDecision) -> dict:
    return {
        "ticker": draft.ticker,
        "action": draft.action,
        "quantity": draft.quantity,
        "confidence": draft.confidence,
        "rationale": draft.rationale,
        "used_data_sources": draft.used_data_sources,
        "guardrails_triggered": draft.guardrails_triggered,
        "rationale_details": draft.rationale_details,
    }


def _hold_override_details(draft: TradingDecision, reflection: str) -> dict:
    return {
        "summary": f"Reflection changed draft {draft.action} quantity {draft.quantity} to HOLD.",
        "evidence": [
            f"draft_action={draft.action}",
            f"draft_quantity={draft.quantity}",
            f"draft_confidence={draft.confidence:.2f}",
        ],
        "risks": [reflection],
        "data_quality": (draft.rationale_details or {}).get("data_quality", "see snapshot confidence fields"),
    }


