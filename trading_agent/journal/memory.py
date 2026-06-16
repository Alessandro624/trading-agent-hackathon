from __future__ import annotations

from typing import Any

from trading_agent.core import JournalEntry


def compact_recent_entries(entries: list[JournalEntry], limit: int = 3) -> list[dict[str, Any]]:
    """Keep recent decisions small enough to inject into Analyst prompts."""
    compact: list[dict[str, Any]] = []
    for entry in entries[-limit:]:
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
            }
        )
    return compact


def _truncate(value: str, max_length: int) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
