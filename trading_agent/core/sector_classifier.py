from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("trading_agent.sector_classifier")

_SEED_TICKER_SECTOR = {
    "AAPL": "technology",
    "MSFT": "technology",
    "NVDA": "technology",
    "GOOGL": "technology",
    "META": "technology",
    "TSLA": "automotive",
    "F": "automotive",
    "GM": "automotive",
    "CAT": "manufacturing",
    "DE": "manufacturing",
    "GE": "manufacturing",
    "HON": "manufacturing",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "JNJ": "healthcare",
    "UNH": "healthcare",
    "PFE": "healthcare",
    "JPM": "finance",
    "BAC": "finance",
    "GS": "finance",
    "KO": "consumer",
    "PG": "consumer",
    "WMT": "consumer",
}

_KNOWN_SECTORS = {
    "technology",
    "automotive",
    "manufacturing",
    "energy",
    "healthcare",
    "finance",
    "consumer",
    "materials",
    "utilities",
    "real_estate",
    "communication",
    "industrials",
    "bonds",
    "cash",
    "crypto",
    "other",
}


_CLASSIFIER_PROMPT = """Return ONLY JSON: {"sector": "<one of: technology, automotive, manufacturing, energy, healthcare, finance, consumer, materials, utilities, real_estate, communication, industrials, bonds, cash, crypto, other>"}.
What economic sector is the company with ticker {ticker} primarily classified in?
Use "other" only if you genuinely don't know. Never invent sectors not in the list."""


class SectorClassifier:
    def __init__(self) -> None:
        self._cache: dict[str, str] = dict(_SEED_TICKER_SECTOR)

    def classify(self, ticker: str, llm_client: Any | None = None) -> str:
        if not ticker:
            return "other"
        symbol = ticker.upper().strip()
        cached = self._cache.get(symbol)
        if cached:
            return cached
        if llm_client is None:
            return "other"
        complete_json = getattr(llm_client, "complete_json", None)
        if not callable(complete_json):
            return "other"
        try:
            raw = complete_json(_CLASSIFIER_PROMPT, json.dumps({"ticker": symbol}))
            text = str(raw or "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start < 0 or end <= start:
                    return "other"
                parsed = json.loads(text[start : end + 1])
            sector = str(parsed.get("sector") or "other").lower().strip()
            if sector not in _KNOWN_SECTORS:
                sector = "other"
            self._cache[symbol] = sector
            logger.info("sector_classifier.classify ticker=%s sector=%s", symbol, sector)
            return sector
        except Exception as error:
            logger.info("sector_classifier.fail ticker=%s reason=%s", symbol, error)
            return "other"

    def suggest_tickers_for_sector(self, sector: str, llm_client: Any | None = None, *, limit: int = 5) -> list[str]:
        if not sector:
            return []
        if llm_client is None:
            return [t for t, s in _SEED_TICKER_SECTOR.items() if s == sector.lower()][:limit]
        complete_json = getattr(llm_client, "complete_json", None)
        if not callable(complete_json):
            return []
        try:
            prompt = (
                'Return ONLY JSON: {"tickers": ["AAPL", "MSFT", ...]}. '
                f"List up to {limit} liquid US public tickers in the '{sector}' sector. "
                "Use canonical uppercase ticker symbols. No commentary."
            )
            raw = complete_json(prompt, sector)
            text = str(raw or "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start < 0 or end <= start:
                    return []
                parsed = json.loads(text[start : end + 1])
            tickers = parsed.get("tickers") if isinstance(parsed, dict) else None
            if not isinstance(tickers, list):
                return []
            result: list[str] = []
            for ticker in tickers:
                if isinstance(ticker, str) and 1 <= len(ticker) <= 5 and ticker.isalpha():
                    result.append(ticker.upper())
                if len(result) >= limit:
                    break
            return result
        except Exception:
            return []


_DEFAULT_CLASSIFIER = SectorClassifier()


def classify_sector(ticker: str, llm_client: Any | None = None) -> str:
    return _DEFAULT_CLASSIFIER.classify(ticker, llm_client)
