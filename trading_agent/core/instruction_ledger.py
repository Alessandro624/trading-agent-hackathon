from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.instruction_ledger")


def make_request_id(cursor_before: int, cursor_after: int, *, run_dir_name: str | None = None) -> str:
    raw = f"{run_dir_name or 'run'}:{cursor_before}:{cursor_after}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"req-{digest}"


def make_idempotency_key(request_id: str, subnote_index: int, intent_type: str, ticker: str | None) -> str:
    raw = f"{request_id}:{subnote_index}:{intent_type or ''}:{ticker or ''}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"idem-{digest}"


@dataclass(frozen=True)
class LedgerEntry:
    idempotency_key: str
    request_id: str
    subnote_index: int
    intent_type: str
    target_ticker: str | None
    instruction_id: str
    created_at: str
    status: str = "queued"
    retry_count: int = 0
    updated_at: str = ""
    completed_at: str | None = None
    outcome_summary: str | None = None
    workflow_id: str | None = None
    workflow_type: str | None = None
    originating_confirmation_id: str | None = None
    operation_index: int = 1
    operation_total: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        created = str(data.get("created_at") or now)
        return cls(
            idempotency_key=str(data.get("idempotency_key") or ""),
            request_id=str(data.get("request_id") or ""),
            subnote_index=int(data.get("subnote_index") or 0),
            intent_type=str(data.get("intent_type") or ""),
            target_ticker=(str(data.get("target_ticker") or "").upper() or None) if data.get("target_ticker") else None,
            instruction_id=str(data.get("instruction_id") or ""),
            created_at=created,
            status=str(data.get("status") or "queued"),
            retry_count=int(data.get("retry_count") or 0),
            updated_at=str(data.get("updated_at") or created),
            completed_at=str(data.get("completed_at") or "") or None,
            outcome_summary=str(data.get("outcome_summary") or "") or None,
            workflow_id=str(data.get("workflow_id") or "") or None,
            workflow_type=str(data.get("workflow_type") or "") or None,
            originating_confirmation_id=str(data.get("originating_confirmation_id") or "") or None,
            operation_index=int(data.get("operation_index") or 1),
            operation_total=int(data.get("operation_total") or 1),
        )


class InstructionLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [LedgerEntry.from_dict(item) for item in items if isinstance(item, dict)]

    def save(self, entries: list[LedgerEntry]) -> None:
        payload = {"entries": [e.to_dict() for e in entries]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def has_key(self, idempotency_key: str) -> bool:
        return any(e.idempotency_key == idempotency_key for e in self.load())

    def get_entry(self, idempotency_key: str) -> LedgerEntry | None:
        for entry in self.load():
            if entry.idempotency_key == idempotency_key:
                return entry
        return None

    def record_queued(
        self,
        *,
        idempotency_key: str,
        request_id: str,
        subnote_index: int,
        intent_type: str,
        target_ticker: str | None,
        instruction_id: str,
        workflow_id: str | None = None,
        workflow_type: str | None = None,
        originating_confirmation_id: str | None = None,
        operation_index: int = 1,
        operation_total: int = 1,
    ) -> LedgerEntry:
        entries = self.load()
        for entry in entries:
            if entry.idempotency_key == idempotency_key:
                logger.info(
                    "ledger.idempotent_skip key=%s status=%s instruction_id=%s",
                    idempotency_key,
                    entry.status,
                    entry.instruction_id,
                )
                return entry
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        new_entry = LedgerEntry(
            idempotency_key=idempotency_key,
            request_id=request_id,
            subnote_index=subnote_index,
            intent_type=intent_type,
            target_ticker=target_ticker,
            instruction_id=instruction_id,
            created_at=now,
            status="queued",
            retry_count=0,
            updated_at=now,
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            originating_confirmation_id=originating_confirmation_id,
            operation_index=operation_index,
            operation_total=operation_total,
        )
        entries.append(new_entry)
        self.save(entries)
        logger.info(
            "ledger.queued key=%s intent=%s ticker=%s instruction_id=%s",
            idempotency_key,
            intent_type,
            target_ticker,
            instruction_id,
        )
        return new_entry

    def update_status(
        self,
        instruction_id: str,
        status: str,
        *,
        outcome_summary: str | None = None,
        increment_retry: bool = False,
    ) -> bool:
        entries = self.load()
        updated = False
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for index, entry in enumerate(entries):
            if entry.instruction_id == instruction_id:
                entries[index] = replace(
                    entry,
                    status=status,
                    retry_count=entry.retry_count + (1 if increment_retry else 0),
                    updated_at=now,
                    completed_at=now if status in {"completed", "cancelled", "abandoned"} else entry.completed_at,
                    outcome_summary=outcome_summary or entry.outcome_summary,
                )
                updated = True
                break
        if updated:
            self.save(entries)
        return updated

    def cancel_by_request_id(self, request_id: str) -> int:
        entries = self.load()
        count = 0
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for index, entry in enumerate(entries):
            if entry.request_id == request_id and entry.status == "queued":
                entries[index] = replace(
                    entry,
                    status="cancelled",
                    retry_count=entry.retry_count,
                    updated_at=now,
                    completed_at=now,
                    outcome_summary="cancelled by user request",
                )
                count += 1
        if count:
            self.save(entries)
        return count

    def cancel_by_instruction_ids(self, instruction_ids: list[str] | tuple[str, ...]) -> int:
        targets = {str(value) for value in instruction_ids if value}
        if not targets:
            return 0
        entries = self.load()
        count = 0
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for index, entry in enumerate(entries):
            if entry.instruction_id not in targets or entry.status != "queued":
                continue
            entries[index] = replace(
                entry,
                status="cancelled",
                retry_count=entry.retry_count,
                updated_at=now,
                completed_at=now,
                outcome_summary="cancelled by user request",
            )
            count += 1
        if count:
            self.save(entries)
        return count


def make_confirmation_id() -> str:
    return f"CF-{uuid.uuid4().hex[:4].upper()}"


_CONFIRM_PATTERN = re.compile(r"\bCF-[0-9A-Fa-f]{4}\b")


def parse_confirmation_reference(text: str) -> str | None:
    if not text:
        return None
    match = _CONFIRM_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).upper()
