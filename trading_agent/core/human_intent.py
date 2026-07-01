from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from trading_agent.core.data_hygiene import clean_text

_NON_USD_SYMBOL_PATTERNS = [
    re.compile(r"€\s*\d"),
    re.compile(r"£\s*\d"),
    re.compile(r"¥\s*\d"),
    re.compile(r"\b\d[\d.,]*\s*(EUR|GBP|JPY|CHF|CAD|AUD|SEK|NOK|CNY|INR|BRL)\b", re.IGNORECASE),
]
_NON_USD_SYMBOL_LABEL = {
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
    "CHF": "CHF",
    "CAD": "CAD",
    "AUD": "AUD",
    "SEK": "SEK",
    "NOK": "NOK",
    "CNY": "CNY",
    "INR": "INR",
    "BRL": "BRL",
}


def detect_non_usd_currency(note: str) -> str | None:
    if not note:
        return None
    for pattern in _NON_USD_SYMBOL_PATTERNS:
        match = pattern.search(note)
        if not match:
            continue
        token = match.group(0).strip()
        for symbol, label in _NON_USD_SYMBOL_LABEL.items():
            if symbol in token:
                return label
        return "NON_USD"
    return None


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


_LLM_SPLITTER_PROMPT = """Split this human note into atomic intent units.
Return ONLY JSON: {"notes": ["intent1", "intent2"]}
Each note must contain exactly one intent.
- Preserve the original language (English/Italian/...).
- Preserve tickers, sectors, currencies, and percentages verbatim.
- "and" / "then" / commas are the typical splitters.
- Keep qualifiers such as "associated shares" attached to the company or industry they qualify.
- Keep all sectors requested in one rebalance intent; do not emit one rebalance note per sector.
- If the note is a single intent, return a one-element list.
- Do NOT invent content; only split what the user wrote.
- If the note is purely conversational with no actionable intent, return [original].

Examples:
  "Buy AAPL, sell META" -> ["Buy AAPL", "sell META"]
  "Buy tech and energy, sell consumer" -> ["Buy tech", "Buy energy", "sell consumer"]
  "Buy SpaceX and balance with manufacturing" -> ["Buy SpaceX", "balance portfolio with manufacturing"]
  "Buy an aircraft company and associated market shares" -> ["Buy an aircraft company and associated market shares"]
  "balance with manufacturing and goods retail" -> ["balance with manufacturing and goods retail"]
  "Take profits on NVDA, then short TSLA" -> ["Take profits on NVDA", "short TSLA"]
  "What's my P&L?" -> ["What's my P&L?"]
"""


def llm_split_compound_note(note: str, llm_client: Any) -> list[str]:
    import json as _json

    if not note or not llm_client:
        return split_compound_note(note)
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        return split_compound_note(note)
    try:
        raw = complete_json(_LLM_SPLITTER_PROMPT, note)
        text = str(raw or "").strip()
        try:
            parsed = _json.loads(text)
        except _json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return split_compound_note(note)
            parsed = _json.loads(text[start : end + 1])
        notes_list = parsed.get("notes") if isinstance(parsed, dict) else None
        if not isinstance(notes_list, list) or not notes_list:
            return split_compound_note(note)
        cleaned: list[str] = []
        for item in notes_list:
            text_item = clean_text(str(item), max_chars=1000)
            if text_item:
                cleaned.append(text_item)
        return cleaned or split_compound_note(note)
    except Exception:
        return split_compound_note(note)


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
