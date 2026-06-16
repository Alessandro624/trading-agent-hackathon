from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.portfolio import position_for
from trading_agent.adapters.ticker_provider import TickerProvider

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


@dataclass(frozen=True)
class TickerSelection:
    ticker: str
    reason: str
    rationale: str
    mentioned_tickers: list[str]


def parse_watchlist(value: str | list[str] | tuple[str, ...]) -> list[str]:
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    symbols: list[str] = []
    for item in raw_items:
        symbol = clean_text(item, max_chars=16).upper()
        if not symbol or not re.fullmatch(r"[A-Z]{1,5}", symbol):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def select_ticker_v0(
    watchlist: list[str],
    *,
    human_input: list[str],
    portfolio: dict[str, Any] | None,
    cycle_index: int,
    fallback: str | None = None,
) -> TickerSelection:
    symbols = parse_watchlist(watchlist or ([fallback] if fallback else []))
    if not symbols:
        raise ValueError("watchlist must include at least one valid ticker")
    mentioned = _mentioned_tickers(human_input, symbols)
    if mentioned:
        ticker = mentioned[-1]
        return TickerSelection(
            ticker=ticker,
            reason="human_input",
            rationale=_selection_rationale(ticker, "Human input mentioned this ticker.", human_input, portfolio),
            mentioned_tickers=mentioned,
        )
    ticker = symbols[cycle_index % len(symbols)]
    return TickerSelection(
        ticker=ticker,
        reason="watchlist_rotation",
        rationale=_selection_rationale(ticker, "No new human ticker mention; rotating watchlist.", human_input, portfolio),
        mentioned_tickers=[],
    )


def _mentioned_tickers(human_input: list[str], watchlist: list[str]) -> list[str]:
    allowed = set(watchlist)
    mentioned: list[str] = []
    for note in human_input:
        text = clean_text(note, max_chars=2000)
        upper = text.upper()
        candidates = [symbol for symbol in _SYMBOL_RE.findall(upper) if symbol in allowed]
        for alias, symbol in _ALIASES.items():
            if alias in upper and symbol in allowed:
                candidates.append(symbol)
        for symbol in candidates:
            if symbol not in mentioned:
                mentioned.append(symbol)
    return mentioned


def _selection_rationale(
    ticker: str,
    reason: str,
    human_input: list[str],
    portfolio: dict[str, Any] | None,
) -> str:
    position = position_for(portfolio, ticker)
    position_text = "no current position"
    if position:
        position_text = f"current position qty={position.get('qty', 0)} market_value={position.get('market_value', 0)}"
    input_text = "; ".join(human_input) if human_input else "no new human input"
    return f"{reason} Selected {ticker}; {position_text}; human_input={input_text}"

def select_ticker(
    watchlist: str,
    *,
    human_input: list[str],
    portfolio: dict[str, Any] | None,
    cycle_index: int,
    fallback: str | None = None,
) -> TickerSelection:
    
    ticker_provider: TickerProvider = TickerProvider()

    tickers_list = ticker_provider.get_tickers_with_info(watchlist)

    mentioned = _mentioned_tickers(human_input, [ t.name for t in tickers_list])
    if mentioned:
        ticker = mentioned[-1]
        return TickerSelection(
            ticker=ticker,
            reason="human_input",
            rationale=_selection_rationale(ticker, "Human input mentioned this ticker.", human_input, portfolio),
            mentioned_tickers=mentioned,
        )
    
    ticker = tickers_list[0]
    return TickerSelection(
        ticker=ticker,
        reason="most_valuable_by_metrics",
        rationale=_selection_rationale(ticker, "Metrics used suggest this ticker as the most valuable.", human_input, portfolio),
        mentioned_tickers=[],
    )