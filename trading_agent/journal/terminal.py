from __future__ import annotations

from typing import Any

from trading_agent.core import JournalEntry


def format_cycle_log(cycle_index: int, entry: JournalEntry) -> str:
    execution = entry.execution_result or {}
    before = execution.get("portfolio_before") or {}
    after = execution.get("portfolio_after") or {}
    llm_fallback = "yes" if entry.llm_fallback_used else "no"
    guardrails = len(entry.guardrails_triggered)
    failures = len(entry.failures)

    lines = [
        f"[cycle {cycle_index}] {entry.ticker} {entry.action} outcome={entry.outcome} confidence={entry.confidence:.2f}",
        f"  llm={entry.llm_provider} fallback={llm_fallback}",
        f"  decision_path={_decision_path(entry)}",
        f"  quantity=requested:{_value(execution.get('requested_quantity'))} allowed:{_value(execution.get('allowed_quantity'))}",
        f"  current_price_at_order={_money(execution.get('current_price_at_order'))}",
        f"  filled_avg_price={_money(execution.get('filled_avg_price'))}",
        f"  portfolio_cash={_money(before.get('cash'))} -> {_money(after.get('cash'))}",
        f"  guardrails={guardrails} failures={failures}",
        f"  summary={entry.cycle_summary}",
    ]
    if execution.get("risk_explanation"):
        lines.append(f"  risk={execution.get('risk_explanation')}")
    if entry.guardrails_triggered:
        lines.append("  guardrail_details=" + " | ".join(entry.guardrails_triggered))
    if entry.failures:
        lines.append("  failure_details=" + " | ".join(entry.failures))
    return "\n".join(lines)


def print_cycle_log(cycle_index: int, entry: JournalEntry) -> None:
    print(format_cycle_log(cycle_index, entry))


def _money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:,.2f}"
    return "n/a"


def _value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _decision_path(entry: JournalEntry) -> str:
    draft = entry.draft_decision or {}
    if not draft:
        return f"final:{entry.action}"
    draft_action = draft.get("action", "n/a")
    draft_quantity = draft.get("quantity", "n/a")
    final_quantity = (entry.decision or {}).get("quantity", "n/a")
    return f"draft:{draft_action} {draft_quantity} -> final:{entry.action} {final_quantity}"
