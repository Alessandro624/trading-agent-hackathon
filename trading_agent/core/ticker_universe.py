from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.portfolio import positions as portfolio_positions
from trading_agent.core.ticker_selection import parse_watchlist


@dataclass(frozen=True)
class TickerUniverse:
    symbols: list[str]
    sources: dict[str, list[str]] = field(default_factory=dict)


def build_ticker_universe(
    *,
    configured_watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    human_input: list[str] | None,
    fallback_ticker: str | None,
    recent_news: list[dict[str, Any]] | None = None,
    provider_symbols: list[str] | None = None,
) -> TickerUniverse:
    symbols: list[str] = []
    sources: dict[str, list[str]] = {}

    for symbol, position in portfolio_positions(portfolio).items():
        if position.get("qty", 0.0) > 0:
            _add_symbol(symbols, sources, symbol, "open_position")

    for symbol in parse_watchlist(configured_watchlist):
        _add_symbol(symbols, sources, symbol, "configured_watchlist")

    for symbol in provider_symbols or []:
        _add_symbol(symbols, sources, symbol, "ticker_provider")

    if fallback_ticker:
        _add_symbol(symbols, sources, fallback_ticker, "fallback")

    return TickerUniverse(symbols, sources)


def _add_symbol(symbols: list[str], sources: dict[str, list[str]], raw_symbol: str, source: str) -> None:
    symbol = clean_text(str(raw_symbol), max_chars=16).upper()
    if not _is_symbol(symbol):
        return
    if symbol not in symbols:
        symbols.append(symbol)
    if source not in sources.setdefault(symbol, []):
        sources[symbol].append(source)


def _is_symbol(value: str) -> bool:
    return 1 <= len(value) <= 5 and value.isalpha() and value.isupper()
