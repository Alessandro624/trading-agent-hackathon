from __future__ import annotations

import re


_US_TICKER = re.compile(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?")


def normalize_ticker(value: object) -> str | None:
    symbol = str(value or "").strip().upper()
    return symbol if _US_TICKER.fullmatch(symbol) else None


def is_ticker_symbol(value: object) -> bool:
    return normalize_ticker(value) is not None
