from __future__ import annotations

import json
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from trading_agent.core.actions import Action, QUANTITY_RULE_TEXT, quantity_is_valid_for_action
from trading_agent.core.models import MarketSnapshot


class AnalystDecisionOutput(BaseModel):
    action: Action
    quantity: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=10)
    used_data_sources: list[str] = Field(default_factory=list)
    rationale_details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("quantity")
    @classmethod
    def action_quantity_is_consistent(cls, value: int, info):
        action = info.data.get("action")
        if action and not quantity_is_valid_for_action(action, value):
            raise ValueError(QUANTITY_RULE_TEXT)
        return value


class ReflectionOutput(BaseModel):
    verdict: Literal["PROCEED", "HOLD"]
    reflection: str = Field(min_length=10)
    confidence_adjustment: float = Field(ge=-0.5, le=0.5)


class NewsOpinionOutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed", "unknown"]
    relevance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=10)


def _loads_object(raw: str) -> dict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object")
    return parsed


def parse_analyst_output(raw: str) -> AnalystDecisionOutput:
    try:
        return AnalystDecisionOutput.model_validate(_loads_object(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid analyst structured output: {error}") from error


def parse_reflection_output(raw: str) -> ReflectionOutput:
    try:
        return ReflectionOutput.model_validate(_loads_object(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid reflection structured output: {error}") from error


def parse_news_opinion_output(raw: str) -> NewsOpinionOutput:
    try:
        return NewsOpinionOutput.model_validate(_loads_object(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid news opinion structured output: {error}") from error


def rationale_snapshot_mismatches(rationale: str, snapshot: MarketSnapshot) -> list[str]:
    """Find obvious contradictions between LLM rationale and validated data."""
    text = rationale.lower()
    mismatches: list[str] = []
    has_reliable_price = snapshot.price is not None and snapshot.price_confidence in {"high", "medium"}
    has_usable_news = snapshot.news_confidence in {"high", "medium"}
    has_no_guardrails = not snapshot.guardrails_triggered

    if has_reliable_price and ("missing price" in text or "price data is missing" in text):
        mismatches.append("rationale says price is missing but snapshot has reliable price data")
    if has_usable_news and ("low news" in text or "missing news" in text or "news confidence is low" in text):
        mismatches.append("rationale says news is missing/low but snapshot has usable news data")
    if has_no_guardrails and ("guardrail" in text or "anomalous" in text):
        mismatches.append("rationale says guardrails/anomaly fired but snapshot has none")

    return mismatches


def snapshot_grounded_hold_rationale(snapshot: MarketSnapshot, reason: str) -> str:
    """Build a HOLD rationale that cites only validated snapshot fields."""
    price_part = "price unavailable"
    if snapshot.price is not None:
        price_part = f"price {snapshot.price:.2f} with {snapshot.price_confidence} confidence"

    news_part = f"news confidence {snapshot.news_confidence}"
    technical_part = f"technical confidence {snapshot.technical_indicators.confidence}"
    if snapshot.technical_indicators.notes:
        technical_part += f" ({'; '.join(snapshot.technical_indicators.notes)})"

    risk_parts = [price_part, news_part, technical_part]
    if snapshot.guardrails_triggered:
        risk_parts.append("guardrails: " + "; ".join(snapshot.guardrails_triggered))
    if snapshot.failures:
        risk_parts.append("failures: " + "; ".join(snapshot.failures))

    return f"Safe HOLD: {reason}. Snapshot check: " + "; ".join(risk_parts) + "."


def safe_rationale_details(snapshot: MarketSnapshot, summary: str, risks: list[str]) -> dict:
    return {
        "summary": summary,
        "evidence": [
            f"price_confidence={snapshot.price_confidence}",
            f"news_confidence={snapshot.news_confidence}",
            f"technical_confidence={snapshot.technical_indicators.confidence}",
        ],
        "risks": risks,
        "data_quality": (
            f"price {snapshot.price_confidence}, news {snapshot.news_confidence}, "
            f"technical {snapshot.technical_indicators.confidence}"
        ),
    }

