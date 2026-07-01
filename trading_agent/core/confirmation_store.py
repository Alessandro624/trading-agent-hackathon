from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_agent.core.evidence import Evidence
from trading_agent.core.instruction_ledger import make_confirmation_id

logger = logging.getLogger("trading_agent.confirmation_store")


def _operation_fingerprint(operations: list[dict[str, Any]]) -> str:
    normalized = []
    for operation in operations:
        normalized.append({str(key): value for key, value in sorted(operation.items()) if value is not None})
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


DEFAULT_EXPIRY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class PendingConfirmation:
    confirmation_id: str
    originating_request_id: str
    proposal_text: str
    proposed_operations: tuple[dict[str, Any], ...]
    rationale: str
    evidence: tuple[dict[str, Any], ...]
    created_at: str
    expires_at: str
    status: str = "pending"
    resolved_at: str | None = None
    resolved_by_request_id: str | None = None
    modification_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_id": self.confirmation_id,
            "originating_request_id": self.originating_request_id,
            "proposal_text": self.proposal_text,
            "proposed_operations": [dict(o) for o in self.proposed_operations],
            "rationale": self.rationale,
            "evidence": [dict(e) for e in self.evidence],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "resolved_by_request_id": self.resolved_by_request_id,
            "modification_text": self.modification_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingConfirmation":
        now = datetime.now(timezone.utc)
        return cls(
            confirmation_id=str(data.get("confirmation_id") or make_confirmation_id()),
            originating_request_id=str(data.get("originating_request_id") or ""),
            proposal_text=str(data.get("proposal_text") or ""),
            proposed_operations=tuple(data.get("proposed_operations") or []),
            rationale=str(data.get("rationale") or ""),
            evidence=tuple(data.get("evidence") or []),
            created_at=str(data.get("created_at") or now.isoformat(timespec="seconds").replace("+00:00", "Z")),
            expires_at=str(data.get("expires_at") or (now + timedelta(seconds=DEFAULT_EXPIRY_SECONDS)).isoformat(timespec="seconds").replace("+00:00", "Z")),
            status=str(data.get("status") or "pending"),
            resolved_at=str(data.get("resolved_at") or "") or None,
            resolved_by_request_id=str(data.get("resolved_by_request_id") or "") or None,
            modification_text=str(data.get("modification_text") or "") or None,
        )

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if expires.tzinfo is not None:
            now_cmp = now.replace(tzinfo=expires.tzinfo) if now.tzinfo is None else now.astimezone(expires.tzinfo)
        else:
            now_cmp = now.replace(tzinfo=None) if now.tzinfo else now
        return now_cmp >= expires


class ConfirmationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[PendingConfirmation]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("confirmations") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [PendingConfirmation.from_dict(item) for item in items if isinstance(item, dict)]

    def save(self, confirmations: list[PendingConfirmation]) -> None:
        payload = {"confirmations": [c.to_dict() for c in confirmations]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(
        self,
        *,
        originating_request_id: str,
        proposal_text: str,
        proposed_operations: list[dict[str, Any]],
        rationale: str,
        evidence: list[Evidence] | list[dict[str, Any]],
        confirmation_id: str | None = None,
        expiry_seconds: int = DEFAULT_EXPIRY_SECONDS,
    ) -> PendingConfirmation:
        operation_fingerprint = _operation_fingerprint(proposed_operations)
        for existing in self.load():
            if existing.status == "pending" and existing.originating_request_id == originating_request_id and _operation_fingerprint(list(existing.proposed_operations)) == operation_fingerprint:
                logger.info(
                    "confirmation.reuse id=%s request=%s operations=%d",
                    existing.confirmation_id,
                    originating_request_id,
                    len(proposed_operations),
                )
                return existing
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat(timespec="seconds").replace("+00:00", "Z")
        expires_iso = (now + timedelta(seconds=expiry_seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")
        # Normalize evidence to dicts.
        evidence_dicts = []
        for item in evidence:
            if isinstance(item, Evidence):
                evidence_dicts.append(item.to_dict())
            elif isinstance(item, dict):
                evidence_dicts.append(item)
        confirmation = PendingConfirmation(
            confirmation_id=confirmation_id or make_confirmation_id(),
            originating_request_id=originating_request_id,
            proposal_text=proposal_text,
            proposed_operations=tuple(proposed_operations),
            rationale=rationale,
            evidence=tuple(evidence_dicts),
            created_at=now_iso,
            expires_at=expires_iso,
            status="pending",
        )
        confirmations = self.load()
        confirmations.append(confirmation)
        self.save(confirmations)
        logger.info(
            "confirmation.add id=%s request=%s operations=%d expires=%s",
            confirmation.confirmation_id,
            originating_request_id,
            len(proposed_operations),
            expires_iso,
        )
        return confirmation

    def get(self, confirmation_id: str) -> PendingConfirmation | None:
        for c in self.load():
            if c.confirmation_id.upper() == confirmation_id.upper():
                return c
        return None

    def get_pending(self) -> list[PendingConfirmation]:
        result: list[PendingConfirmation] = []
        for c in self.load():
            if c.status != "pending":
                continue
            if c.is_expired():
                continue
            result.append(c)
        return result

    def mark_status(
        self,
        confirmation_id: str,
        status: str,
        *,
        resolved_by_request_id: str | None = None,
        modification_text: str | None = None,
    ) -> PendingConfirmation | None:
        if status not in {"confirmed", "rejected", "modified", "expired"}:
            raise ValueError(f"invalid confirmation status: {status}")
        confirmations = self.load()
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        updated: PendingConfirmation | None = None
        for index, c in enumerate(confirmations):
            if c.confirmation_id.upper() == confirmation_id.upper():
                confirmations[index] = PendingConfirmation(
                    confirmation_id=c.confirmation_id,
                    originating_request_id=c.originating_request_id,
                    proposal_text=c.proposal_text,
                    proposed_operations=c.proposed_operations,
                    rationale=c.rationale,
                    evidence=c.evidence,
                    created_at=c.created_at,
                    expires_at=c.expires_at,
                    status=status,
                    resolved_at=now_iso,
                    resolved_by_request_id=resolved_by_request_id,
                    modification_text=modification_text,
                )
                updated = confirmations[index]
                break
        if updated is not None:
            self.save(confirmations)
            logger.info(
                "confirmation.mark id=%s status=%s modification=%s",
                confirmation_id,
                status,
                "yes" if modification_text else "no",
            )
        return updated

    def expire_overdue(self) -> int:
        confirmations = self.load()
        count = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        for index, c in enumerate(confirmations):
            if c.status == "pending" and c.is_expired():
                confirmations[index] = PendingConfirmation(
                    confirmation_id=c.confirmation_id,
                    originating_request_id=c.originating_request_id,
                    proposal_text=c.proposal_text,
                    proposed_operations=c.proposed_operations,
                    rationale=c.rationale,
                    evidence=c.evidence,
                    created_at=c.created_at,
                    expires_at=c.expires_at,
                    status="expired",
                    resolved_at=now_iso,
                )
                count += 1
        if count:
            self.save(confirmations)
        return count
