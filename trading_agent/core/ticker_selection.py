from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.human_intent import parse_human_intent
from trading_agent.core.portfolio import position_for
from trading_agent.core.portfolio import positions as portfolio_positions


@dataclass(frozen=True)
class TickerSelection:
    ticker: str
    reason: str
    rationale: str
    mentioned_tickers: list[str] = field(default_factory=list)


def parse_watchlist(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    symbols: list[str] = []
    for item in raw_items:
        symbol = clean_text(item, max_chars=16).upper()
        if not _is_symbol(symbol):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def select_ticker(
    watchlist: list[str] | str | None = None,
    *,
    human_input: list[str],
    portfolio: dict[str, Any] | None,
    recent_news: list[dict[str, Any]] | None = None,
    cycle_index: int = 0,
    fallback: str | None = None,
    ticker_provider: Any | None = None,
) -> TickerSelection:
    watchlist_symbols = parse_watchlist(watchlist)
    human_intent = parse_human_intent(human_input)
    positioned_symbols = _open_position_tickers(portfolio)
    if "position_sweep" in human_intent.intents and positioned_symbols:
        candidates = [symbol for symbol in watchlist_symbols if symbol in positioned_symbols] or positioned_symbols
        ticker = _rank(candidates, ticker_provider) or candidates[cycle_index % len(candidates)]
        return TickerSelection(
            ticker=ticker,
            reason="human_position_sweep",
            rationale=_selection_rationale(ticker, "Human input requested a sweep or review of open positions.", human_input, portfolio),
            mentioned_tickers=candidates,
        )

    if positioned_symbols:
        ticker = _rank(positioned_symbols, ticker_provider) or positioned_symbols[cycle_index % len(positioned_symbols)]
        return TickerSelection(
            ticker=ticker,
            reason="open_position",
            rationale=_selection_rationale(ticker, "Open position selected for review.", human_input, portfolio),
            mentioned_tickers=positioned_symbols,
        )

    if watchlist_symbols:
        ticker = _rank(watchlist_symbols, ticker_provider) or watchlist_symbols[cycle_index % len(watchlist_symbols)]
        return TickerSelection(
            ticker=ticker,
            reason="watchlist_rotation",
            rationale=_selection_rationale(ticker, "No open position or explicit human ticker; selecting from current ticker universe.", human_input, portfolio),
            mentioned_tickers=[],
        )

    if fallback:
        fallback_symbol = fallback.upper().strip()
        if _is_symbol(fallback_symbol):
            return TickerSelection(
                ticker=fallback_symbol,
                reason="fallback",
                rationale=_selection_rationale(fallback_symbol, "No dynamic universe available; using fallback ticker.", human_input, portfolio),
                mentioned_tickers=[],
            )

    raise ValueError("select_ticker could not produce a ticker from human input, positions, news, universe, or fallback.")


def _rank(candidates: list[str], ticker_provider: Any | None) -> str | None:
    if ticker_provider is None or not candidates:
        return None
    pick_best = getattr(ticker_provider, "pick_best_by_metrics", None)
    if not callable(pick_best):
        return None
    try:
        picked = pick_best(candidates)
    except Exception:
        return None
    if isinstance(picked, str) and picked.upper() in candidates:
        return picked.upper()
    return None


def _is_symbol(value: str) -> bool:
    return 1 <= len(value) <= 5 and value.isalpha() and value.isupper()


def _open_position_tickers(portfolio: dict[str, Any] | None) -> list[str]:
    open_positions = portfolio_positions(portfolio)
    return [symbol for symbol, position in open_positions.items() if position.get("qty", 0.0) > 0]


def _selection_rationale(
    ticker: str,
    reason: str,
    human_input: list[str],
    portfolio: dict[str, Any] | None,
) -> str:
    position = position_for(portfolio, ticker)
    if position:
        position_text = f"current position qty={position.get('qty', 0)} market_value={position.get('market_value', 0)}"
    else:
        position_text = "no current position"
    input_text = "; ".join(human_input) if human_input else "no new human input"
    return f"{reason} Selected {ticker}; {position_text}; human_input={input_text}"
