from __future__ import annotations

from dataclasses import asdict, dataclass

from trading_agent.core.data_hygiene import clean_text


@dataclass(frozen=True)
class HumanIntent:
    intents: list[str]
    tickers: list[str]
    requested_action: str | None = None
    risk_preference: str | None = None
    impact_topic: str | None = None
    summary: str = "No explicit human trading intent."

    def to_dict(self) -> dict:
        return asdict(self)


def parse_human_intent(notes: list[str]) -> HumanIntent:
    text = clean_text(" ".join(notes), max_chars=4000)
    if not text:
        return HumanIntent([], [])

    intents: list[str] = []
    requested_action: str | None = None
    tickers: list[str] = []
    upper = text.upper()
    risk_preference = _risk_preference(upper)
    impact_topic = _extract_impact_topic(text)

    has_sell = _has_word(upper, "SELL") or _has_word(upper, "VENDI") or _has_word(upper, "VENDERE")
    has_buy = _has_word(upper, "BUY") or _has_word(upper, "COMPRA") or _has_word(upper, "ACQUISTA")
    has_cancel = _has_word(upper, "CANCEL") or _has_word(upper, "ANNULLA") or _has_word(upper, "UNDO")

    if has_cancel:
        intents.append("cancel")
    if has_sell and not has_cancel:
        intents.append("sell")
        if impact_topic:
            intents.append("conditional_sell")
        if _is_position_sweep_request(upper):
            intents.append("position_sweep")
    if has_buy and not has_cancel:
        intents.append("buy")
    if _has_news_request(upper):
        intents.append("news_request")
    if risk_preference:
        intents.append(risk_preference)

    summary_parts = []
    if requested_action:
        summary_parts.append(f"requested_action={requested_action}")
    if risk_preference:
        summary_parts.append(f"risk_preference={risk_preference}")
    if impact_topic:
        summary_parts.append(f"impact_topic={impact_topic}")
    if tickers:
        summary_parts.append("tickers=" + ",".join(tickers))
    summary = "; ".join(summary_parts) if summary_parts else "Human input requires LLM interpretation or is advisory context."
    return HumanIntent(_dedupe(intents), tickers, requested_action, risk_preference, impact_topic, summary)


def split_compound_note(note: str) -> list[str]:
    cleaned = clean_text(note, max_chars=2000)
    if not cleaned:
        return []
    return [cleaned]


def _extract_impact_topic(text: str) -> str | None:
    patterns = [
        r"\b(?:related\s+or\s+impacted|related\s+or\s+affected|related|impacted|affected)\s+by\s+(?P<topic>.+)$",
        r"\b(?:because\s+of|due\s+to)\s+(?P<topic>.+)$",
        r"\b(?:a causa di|per via di|colpiti da|impattati da)\s+(?P<topic>.+)$",
    ]
    for pattern in patterns:
        match = _search(pattern, text)
        if not match:
            continue
        topic = clean_text(match.group("topic").strip(" .,:;"), max_chars=240)
        return topic.lower() if topic else None
    return None


def _is_position_sweep_request(upper_text: str) -> bool:
    return any(
        phrase in upper_text
        for phrase in (
            "ALL POSITIONS",
            "ALL OPEN POSITIONS",
            "EVERY POSITION",
            "EVERY OPEN POSITION",
            "TUTTE LE POSIZIONI",
            "TUTTE POSIZIONI",
            "LIQUIDA TUTTO",
            "LIQUIDATE EVERYTHING",
        )
    )


def _risk_preference(upper_text: str) -> str | None:
    if "RISK ON" in upper_text or "MORE RISK" in upper_text or "PIU RISCHIO" in upper_text or "PIÙ RISCHIO" in upper_text:
        return "risk_on"
    if "RISK OFF" in upper_text or "LESS RISK" in upper_text or "MENO RISCHIO" in upper_text:
        return "risk_off"
    return None


def _has_news_request(upper_text: str) -> bool:
    return any(word in upper_text for word in ("NEWS", "NOTIZIE", "READ", "LEGGI", "HEADLINE"))


def _has_word(upper_text: str, word: str) -> bool:
    padded = f" {upper_text} "
    return f" {word} " in padded


def _search(pattern: str, text: str):
    import re

    return re.search(pattern, text, flags=re.IGNORECASE)


def _dedupe(items) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
