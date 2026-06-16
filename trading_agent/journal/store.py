from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

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
    ) -> JournalEntry:
        summary = build_cycle_summary(snapshot, decision, execution)
        llm_metadata = decision.llm_metadata or {}
        entry = JournalEntry(
            timestamp=utc_now_iso(),
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
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def read_all(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        rows: list[JournalEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(JournalEntry(**json.loads(line)))
        return rows
