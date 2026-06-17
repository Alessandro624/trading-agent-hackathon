from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from trading_agent.core.data_hygiene import clean_text

_SYMBOL_RE = re.compile(r"\b[A-Z]{1,5}\b")
_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
}

_SELL_TERMS = {"SELL", "VENDI", "VENDERE", "VENDO", "LIQUIDA", "LIQUIDARE", "RIDUCI", "RIDURRE"}
_BUY_TERMS = {"BUY", "COMPRA", "COMPRARE", "ACQUISTA", "ACQUISTARE"}
_RISK_ON_TERMS = {"PIU RISCHIO", "PIÙ RISCHIO", "PIU' RISCHIO", "RISCHIO ALTO", "AGGRESSIVO", "RISK ON", "MORE RISK"}
_RISK_OFF_TERMS = {"MENO RISCHIO", "CONSERVATIVO", "CONSERVATIVA", "RISK OFF", "LESS RISK", "RIDURRE RISCHIO"}
_NEWS_TERMS = {"NEWS", "NOTIZIA", "NOTIZIE", "LEGGI", "READ", "ARTICOLO", "HEADLINE"}
_AVOID_TERMS = {"EVITA", "NON COMPRARE", "NON ACQUISTARE", "AVOID"}


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
    upper = text.upper()
    intents: list[str] = []
    requested_action: str | None = None
    risk_preference: str | None = None
    impact_topic = _impact_topic(text)
    is_conditional_sell = bool(impact_topic) and _contains_any_word(upper, _SELL_TERMS)

    if is_conditional_sell:
        intents.extend(["conditional_sell", "position_sweep"])
    elif _is_open_position_sweep(upper):
        intents.extend(["sell", "position_sweep"])
    elif _contains_any_word(upper, _SELL_TERMS):
        intents.append("sell")
        requested_action = "SELL"
    if _contains_any_word(upper, _BUY_TERMS):
        intents.append("buy")
        requested_action = requested_action or "BUY"
    if any(term in upper for term in _AVOID_TERMS):
        intents.append("avoid")
    if any(term in upper for term in _NEWS_TERMS):
        intents.append("news_request")
    if any(term in upper for term in _RISK_ON_TERMS):
        intents.append("risk_on")
        risk_preference = "risk_on"
    if any(term in upper for term in _RISK_OFF_TERMS):
        intents.append("risk_off")
        risk_preference = "risk_off"

    ticker_text = _remove_impact_topic(upper, impact_topic)
    tickers = _mentioned_tickers(ticker_text)
    summary_parts = []
    if requested_action:
        summary_parts.append(f"requested_action={requested_action}")
    if risk_preference:
        summary_parts.append(f"risk_preference={risk_preference}")
    if tickers:
        summary_parts.append("tickers=" + ",".join(tickers))
    if impact_topic:
        summary_parts.append(f"impact_topic={impact_topic}")
    summary = "; ".join(summary_parts) if summary_parts else "Human input contains context but no explicit trade intent."
    return HumanIntent(_dedupe(intents), tickers, requested_action, risk_preference, impact_topic, summary)


def _mentioned_tickers(upper_text: str) -> list[str]:
    ignored = {
        "ALL",
        "AND",
        "BY",
        "CRYSIS",
        "DROP",
        "GAS",
        "HOLD",
        "NEWS",
        "OIL",
        "OPEN",
        "OR",
        "POSITIONS",
        "POSITION",
        "WAIT",
        "BUY",
        "SELL",
        *_ALIASES.keys(),
    }
    tickers = [symbol for symbol in _SYMBOL_RE.findall(upper_text) if symbol not in ignored]
    for alias, symbol in _ALIASES.items():
        if alias in upper_text:
            tickers.append(symbol)
    return _dedupe(tickers)


def _contains_any_word(upper_text: str, terms: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", upper_text) for term in terms)


def _impact_topic(text: str) -> str | None:
    match = re.search(r"\b(?:related|impacted)\s+by\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    topic = clean_text(match.group(1), max_chars=240).strip(" .")
    return topic or None


def _is_open_position_sweep(upper_text: str) -> bool:
    return _contains_any_word(upper_text, _SELL_TERMS) and (
        "ALL OPEN POSITIONS" in upper_text
        or "ALL POSITIONS" in upper_text
        or "TUTTE LE POSIZIONI" in upper_text
    )


def _remove_impact_topic(upper_text: str, impact_topic: str | None) -> str:
    if not impact_topic:
        return upper_text
    return upper_text.replace(impact_topic.upper(), "")


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
