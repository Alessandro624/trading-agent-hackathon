from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.human_intent import parse_human_intent

logger = logging.getLogger("trading_agent.human_risk")


@dataclass(frozen=True)
class HumanRiskProfile:
    risk_preference: str = "neutral"
    buy_aggressiveness: float = 0.0
    sell_aggressiveness: float = 0.0
    rationale: str = "No human risk adjustment requested."

    def __post_init__(self) -> None:
        if self.risk_preference not in {"risk_on", "risk_off", "neutral"}:
            raise ValueError("risk_preference must be risk_on, risk_off, or neutral")
        if not 0.0 <= self.buy_aggressiveness <= 1.0:
            raise ValueError("buy_aggressiveness must be between 0 and 1")
        if not 0.0 <= self.sell_aggressiveness <= 1.0:
            raise ValueError("sell_aggressiveness must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_human_risk_profile(notes: list[str], llm_client: Any | None = None) -> HumanRiskProfile:
    cleaned_notes = [clean_text(note, max_chars=1000) for note in notes if clean_text(note, max_chars=1000)]
    if not cleaned_notes:
        return HumanRiskProfile()
    if not _risk_strategy_requested(cleaned_notes):
        logger.info("human.risk.skip reason=no_explicit_risk_or_sizing_request")
        return HumanRiskProfile()

    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        try:
            payload = {
                "human_notes": cleaned_notes,
                "instructions": (
                    "Return a JSON risk profile only. Interpret whether the user wants the agent "
                    "to take more or less trading risk. Use buy_aggressiveness for buy-side sizing "
                    "requests and sell_aggressiveness for sell-side sizing requests. Do not encode "
                    "a trade instruction here; this only changes risk limits."
                ),
            }
            logger.info("human.risk.llm.start notes=%s", " | ".join(note[:80] for note in cleaned_notes))
            raw = complete_json(_RISK_PROFILE_PROMPT, json.dumps(payload, default=str))
            profile = _parse_profile_json(raw)
            logger.info(
                "human.risk.llm.profile preference=%s buy=%.2f sell=%.2f",
                profile.risk_preference,
                profile.buy_aggressiveness,
                profile.sell_aggressiveness,
            )
            return profile
        except Exception as error:
            logger.warning("human.risk.llm.fail reason=%s", error)

    return _fallback_profile(cleaned_notes)


def update_persistent_human_risk_profile(
    current_profile: HumanRiskProfile | dict[str, Any] | None,
    notes: list[str],
    llm_client: Any | None = None,
) -> HumanRiskProfile:
    current = _coerce_profile(current_profile) or HumanRiskProfile()
    cleaned_notes = [clean_text(note, max_chars=1000) for note in notes if clean_text(note, max_chars=1000)]
    if not cleaned_notes or not _risk_strategy_requested(cleaned_notes):
        return current
    return resolve_human_risk_profile(cleaned_notes, llm_client)


def _parse_profile_json(raw: str) -> HumanRiskProfile:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_human_risk_profile")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("human_risk_profile_must_be_object")
    preference = str(parsed.get("risk_preference") or "neutral").lower().strip()
    if preference not in {"risk_on", "risk_off", "neutral"}:
        preference = "neutral"
    return HumanRiskProfile(
        risk_preference=preference,
        buy_aggressiveness=_clamp_float(parsed.get("buy_aggressiveness")),
        sell_aggressiveness=_clamp_float(parsed.get("sell_aggressiveness")),
        rationale=clean_text(str(parsed.get("rationale") or "Human risk profile resolved by LLM."), max_chars=500),
    )


def _fallback_profile(notes: list[str]) -> HumanRiskProfile:
    intent = parse_human_intent(notes)
    if intent.risk_preference == "risk_on":
        return HumanRiskProfile("risk_on", 0.5, 0.5, "Human input requested more risk; keyword fallback applied.")
    if intent.risk_preference == "risk_off":
        return HumanRiskProfile("risk_off", 0.0, 0.0, "Human input requested less risk; keyword fallback applied.")
    return HumanRiskProfile()


def _coerce_profile(profile: HumanRiskProfile | dict[str, Any] | None) -> HumanRiskProfile | None:
    if profile is None:
        return None
    if isinstance(profile, HumanRiskProfile):
        return profile
    if isinstance(profile, dict):
        try:
            return HumanRiskProfile(
                risk_preference=str(profile.get("risk_preference") or "neutral"),
                buy_aggressiveness=_clamp_float(profile.get("buy_aggressiveness")),
                sell_aggressiveness=_clamp_float(profile.get("sell_aggressiveness")),
                rationale=clean_text(str(profile.get("rationale") or "Persisted human risk profile."), max_chars=500),
            )
        except ValueError:
            return None
    return None


def _risk_strategy_requested(notes: list[str]) -> bool:
    text = " ".join(notes).lower()
    if "risk" in text or "risch" in text:
        return True
    sizing_terms = (
        "aggressive",
        "aggressivo",
        "aggressiva",
        "conservative",
        "conservativo",
        "conservativa",
        "quantity",
        "quantita",
        "quantità",
        "sizing",
        "size",
        "position size",
        "buy more",
        "sell more",
        "comprare tanto",
        "acquistare tanto",
        "vendere tanto",
    )
    return any(term in text for term in sizing_terms)


def _clamp_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


_RISK_PROFILE_PROMPT = """You are a human risk preference resolver for a trading agent.
You receive human notes written during the live loop.

Return ONLY JSON:
{
  "risk_preference": "risk_on|risk_off|neutral",
  "buy_aggressiveness": 0.0,
  "sell_aggressiveness": 0.0,
  "rationale": "one sentence"
}

Rules:
- This is NOT a trade decision. Do not choose tickers or actions here.
- risk_on means the user accepts larger position sizing.
- risk_off means the user wants smaller position sizing.
- buy_aggressiveness controls BUY sizing from 0.0 conservative to 1.0 most aggressive.
- sell_aggressiveness controls SELL sizing from 0.0 conservative to 1.0 most aggressive.
- If the user only says to buy/sell a ticker without a sizing/risk preference, return neutral.
- If the user describes a market crisis, sector exposure, or sell condition without asking for
  more/less risk or larger/smaller sizing, return neutral.
"""
