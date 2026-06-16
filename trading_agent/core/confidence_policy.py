from __future__ import annotations


def price_confidence(price: float | None, failures: list[str]) -> str:
    if price is None:
        return "low" if failures else "none"
    return "medium" if failures else "high"


def news_confidence(news: list[dict], failures: list[str]) -> str:
    if failures:
        return "low"
    return "medium" if news else "none"


def force_hold_reasons(price_confidence_value: str, guardrails: list[str]) -> list[str]:
    reasons: list[str] = []
    if price_confidence_value in {"low", "none"}:
        reasons.append("price confidence is degraded")
    # Empty news can be normal; only price degradation or explicit guardrails force HOLD.
    if guardrails:
        reasons.append("guardrails triggered: " + ", ".join(guardrails))
    return reasons
