from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.agent_reply")


def generate_reply(
    question_type: str,
    topic: str | None,
    portfolio: dict[str, Any] | None,
    journal_entries: list[dict[str, Any]],
    note: str,
    llm_client: Any | None,
) -> str:
    portfolio = portfolio or {}
    summary = _build_deterministic_summary(question_type, topic, portfolio, journal_entries, note)
    if llm_client is None:
        return summary
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        return summary
    try:
        prompt = _build_reply_prompt(question_type, topic, note)
        payload = {
            "question_type": question_type,
            "topic": topic,
            "note": note,
            "portfolio": _portfolio_payload(portfolio),
            "recent_decisions": _recent_decisions_payload(journal_entries),
        }
        raw = complete_json(prompt, json.dumps(payload, default=str))
        text = str(raw or "").strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("reply"):
                return str(parsed["reply"]).strip()
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, dict) and parsed.get("reply"):
                        return str(parsed["reply"]).strip()
                except json.JSONDecodeError:
                    pass
        return text or summary
    except Exception as error:
        logger.info("agent_reply.llm.fail reason=%s question_type=%s — using deterministic summary", error, question_type)
        return summary


def _build_deterministic_summary(
    question_type: str,
    topic: str | None,
    portfolio: dict[str, Any],
    journal_entries: list[dict[str, Any]],
    note: str,
) -> str:
    positions = portfolio.get("positions") or {}
    cash = _safe_float(portfolio.get("cash"))
    portfolio_value = _safe_float(portfolio.get("portfolio_value"))

    if question_type == "cash":
        return f"Your current cash balance is ${cash:.2f}. " f"Total portfolio value (cash + positions) is ${portfolio_value:.2f}."

    if question_type == "portfolio":
        if not positions:
            return "You currently hold no open positions. Portfolio is all cash."
        lines = [f"You currently hold {len(positions)} position(s):"]
        for ticker, data in sorted(positions.items()):
            qty = _safe_float(data.get("qty"))
            mv = _safe_float(data.get("market_value"))
            pnl = _safe_float(data.get("unrealized_pl"))
            lines.append(f"  • {ticker}: {qty} shares, market value ${mv:.2f}, unrealized P&L ${pnl:+.2f}")
        lines.append(f"Cash: ${cash:.2f}. Total portfolio value: ${portfolio_value:.2f}.")
        return "\n".join(lines)

    if question_type == "pnl":
        total_pnl = sum(_safe_float((data or {}).get("unrealized_pl")) for data in positions.values())
        return f"Unrealized P&L across all open positions: ${total_pnl:+.2f}. " f"Portfolio value: ${portfolio_value:.2f} (cash ${cash:.2f})."

    if question_type == "decision_history":
        if not journal_entries:
            return "No recent decisions have been recorded in this run yet."
        lines = [f"Last {min(len(journal_entries), 5)} decision(s):"]
        for entry in journal_entries[-5:]:
            timestamp = entry.get("timestamp", "-")
            ticker = entry.get("ticker", "-")
            action = entry.get("action", "-")
            outcome = entry.get("outcome", "-")
            qty = ((entry.get("decision") or {}).get("quantity")) or 0
            lines.append(f"  • {timestamp} | {action} {qty} {ticker} -> {outcome}")
        return "\n".join(lines)

    if question_type == "market_opinion":
        if not topic:
            return "I don't have a market opinion on that topic. Please specify a ticker or sector."
        return f"I don't have a prepared market opinion on '{topic}'. " "Run a normal cycle on the relevant ticker to get a technical + news analysis."

    return f"I received your question: '{note}'. I'll get back to you with a detailed answer."


def _build_reply_prompt(question_type: str, topic: str | None, note: str) -> str:
    topic_text = f" Topic hint: '{topic}'." if topic else ""
    return (
        "You are the trading agent's reply generator. Answer the user's question "
        "concisely using only the provided portfolio and recent decisions. "
        'Return ONLY JSON: {"reply": "<your answer, max 4 sentences>"}. '
        f"Question type: {question_type}.{topic_text} Original note: {note!r}."
    )


def _portfolio_payload(portfolio: dict[str, Any]) -> dict[str, Any]:
    positions = portfolio.get("positions") or {}
    return {
        "cash": _safe_float(portfolio.get("cash")),
        "portfolio_value": _safe_float(portfolio.get("portfolio_value")),
        "positions": {
            ticker: {
                "qty": _safe_float(data.get("qty")),
                "market_value": _safe_float(data.get("market_value")),
                "avg_entry_price": _safe_float(data.get("avg_entry_price")),
                "unrealized_pl": _safe_float(data.get("unrealized_pl")),
            }
            for ticker, data in positions.items()
        },
    }


def _recent_decisions_payload(journal_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in journal_entries[-10:]:
        decision = entry.get("decision") or {}
        payload.append(
            {
                "timestamp": entry.get("timestamp"),
                "ticker": entry.get("ticker"),
                "action": entry.get("action"),
                "quantity": decision.get("quantity"),
                "outcome": entry.get("outcome"),
                "rationale": (entry.get("rationale") or "")[:200],
            }
        )
    return payload


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def append_agent_reply(replies_path: Path, reply: str, note: str, timestamp: str | None = None, question_type: str | None = None) -> None:
    replies_path.parent.mkdir(parents=True, exist_ok=True)
    when = timestamp or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    qtype = question_type or "general"
    block = f"\n## {when} — [{qtype}]\n" f"**User note:** {note}\n\n" f"**Agent reply:**\n\n{reply}\n"
    if not replies_path.exists():
        replies_path.write_text("# Agent Replies\n", encoding="utf-8")
    with replies_path.open("a", encoding="utf-8") as handle:
        handle.write(block)

    jsonl_path = replies_path.with_suffix(".jsonl")
    entry = {
        "timestamp": when,
        "question_type": qtype,
        "note": note,
        "reply": reply,
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_agent_replies(replies_path: Path) -> list[dict[str, Any]]:
    jsonl_path = replies_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        entries: list[dict[str, Any]] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    if not replies_path.exists():
        return []
    text = replies_path.read_text(encoding="utf-8")
    entries = []
    current: dict[str, Any] = {}
    body_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                current["reply"] = "\n".join(body_lines).strip()
                entries.append(current)
            header = line[3:].strip()
            ts, _, qtype_part = header.partition(" — ")
            qtype = qtype_part.strip("[]").strip() if qtype_part else "general"
            current = {"timestamp": ts.strip(), "question_type": qtype, "note": "", "reply": ""}
            body_lines = []
        elif line.startswith("**User note:**"):
            current["note"] = line[len("**User note:**") :].strip()
        elif line.startswith("**Agent reply:**"):
            continue
        else:
            body_lines.append(line)
    if current:
        current["reply"] = "\n".join(body_lines).strip()
        entries.append(current)
    return entries


def search_journal(
    journal_path: Path,
    *,
    ticker: str | None = None,
    instruction_id: str | None = None,
    entry_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not journal_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ticker and str(row.get("ticker") or "").upper() != ticker.upper():
            continue
        if instruction_id and row.get("instruction_id") != instruction_id:
            continue
        if entry_type and row.get("entry_type", "cycle") != entry_type:
            continue
        ts = row.get("timestamp") or ""
        if since and ts < since:
            continue
        if until and ts > until:
            continue
        rows.append(row)
    return rows[-limit:]
