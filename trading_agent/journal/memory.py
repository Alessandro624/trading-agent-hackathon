from __future__ import annotations

from typing import Any

from trading_agent.core import JournalEntry


def compact_recent_entries(
    entries: list[JournalEntry],
    limit: int = 3,
    ticker: str | None = None,
    per_ticker_limit: int | None = None,
    global_limit: int = 1,
) -> list[dict[str, Any]]:
    """Keep recent decisions small enough to inject into Analyst prompts."""
    scoped_entries = _scoped_entries(entries, limit, ticker, per_ticker_limit, global_limit)
    compact: list[dict[str, Any]] = []
    for entry, scope in scoped_entries:
        decision = entry.decision or {}
        execution = entry.execution_result or {}
        snapshot = entry.market_snapshot or {}
        technicals = snapshot.get("technical_indicators") or {}
        rationale_details = decision.get("rationale_details") or {}
        compact.append(
            {
                "timestamp": entry.timestamp,
                "ticker": entry.ticker,
                "action": entry.action,
                "quantity": decision.get("quantity"),
                "confidence": entry.confidence,
                "outcome": entry.outcome,
                "summary": entry.cycle_summary,
                "technical_confidence": technicals.get("confidence"),
                "requested_quantity": execution.get("requested_quantity"),
                "allowed_quantity": execution.get("allowed_quantity"),
                "rationale_summary": rationale_details.get("summary") or _truncate(entry.rationale, 180),
                "memory_scope": scope,
            }
        )
    return compact


def _scoped_entries(
    entries: list[JournalEntry],
    limit: int,
    ticker: str | None,
    per_ticker_limit: int | None,
    global_limit: int,
) -> list[tuple[JournalEntry, str]]:
    if ticker is None:
        return [(entry, "recent") for entry in entries[-limit:]]
    symbol = ticker.upper()
    ticker_limit = per_ticker_limit if per_ticker_limit is not None else max(1, limit - global_limit)
    ticker_entries = [entry for entry in entries if entry.ticker == symbol][-ticker_limit:]
    global_entries = [entry for entry in entries if entry.ticker != symbol][-global_limit:]
    return [(entry, "ticker") for entry in ticker_entries] + [(entry, "global") for entry in global_entries]


def _truncate(value: str, max_length: int) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
