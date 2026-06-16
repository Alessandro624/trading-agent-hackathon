from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trading_agent.core import (
    ExecutionResult,
    JournalEntry,
    MarketSnapshot,
    TradingDecision,
    build_cycle_summary,
    utc_now_iso,
)


class JournalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        snapshot: MarketSnapshot,
        decision: TradingDecision,
        execution: ExecutionResult | None,
        draft_decision: TradingDecision | None = None,
        cycle_started_at: str | None = None,
        human_input: list[str] | None = None,
    ) -> JournalEntry:
        summary = build_cycle_summary(snapshot, decision, execution)
        llm_metadata = decision.llm_metadata or {}
        entry = JournalEntry(
            timestamp=cycle_started_at or utc_now_iso(),
            ticker=snapshot.ticker,
            action=decision.action,
            rationale=decision.rationale,
            data_source=", ".join(snapshot.data_sources),
            confidence=decision.confidence,
            outcome=execution.status if execution else "not_executed",
            cycle_summary=summary,
            market_snapshot=asdict(snapshot),
            decision=asdict(decision),
            execution_result=asdict(execution) if execution else None,
            draft_decision=asdict(draft_decision) if draft_decision else None,
            llm_provider=llm_metadata.get("llm_provider", "none"),
            llm_fallback_used=bool(llm_metadata.get("llm_fallback_used", False)),
            llm_fallback_provider=llm_metadata.get("llm_fallback_provider"),
            llm_fallback_reason=llm_metadata.get("llm_fallback_reason"),
            guardrails_triggered=[*snapshot.guardrails_triggered, *decision.guardrails_triggered],
            failures=snapshot.failures,
            human_input_used=human_input or [],
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def append_stage(
        self,
        *,
        ticker: str,
        stage: str,
        status: str,
        message: str,
        cycle_started_at: str | None = None,
        snapshot: MarketSnapshot | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "entry_type": "stage",
            "timestamp": cycle_started_at or utc_now_iso(),
            "ticker": ticker.upper(),
            "stage": stage,
            "status": status,
            "message": message,
            "action": "STAGE",
            "confidence": 0.0,
            "outcome": status,
            "rationale": message,
            "cycle_summary": message,
            "data_source": ", ".join(snapshot.data_sources) if snapshot else "",
            "market_snapshot": asdict(snapshot) if snapshot else {},
            "decision": {},
            "execution_result": None,
            "draft_decision": None,
            "guardrails_triggered": list(snapshot.guardrails_triggered) if snapshot else [],
            "failures": list(snapshot.failures) if snapshot else [],
            "details": details or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def read_all(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        rows: list[JournalEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("entry_type", "cycle") != "cycle":
                continue
            rows.append(JournalEntry(**payload))
        return rows
