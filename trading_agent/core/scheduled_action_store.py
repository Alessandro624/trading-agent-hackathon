from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.scheduled_action_store")


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class ScheduledAction:
    action_id: str
    note: str
    wrapped_intent_type: str
    target_ticker: str | None
    trigger_type: str
    trigger_value: str
    created_at: str
    status: str
    instruction_id: str | None = None
    rationale: str | None = None
    triggered_at: str | None = None
    requested_notional_usd: float | None = None
    requested_quantity: int | None = None
    partial_fraction: float | None = None
    override_constraints: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledAction":
        return cls(
            action_id=str(data.get("action_id") or uuid.uuid4().hex[:8]),
            note=str(data.get("note") or ""),
            wrapped_intent_type=str(data.get("wrapped_intent_type") or "advisory").lower(),
            target_ticker=(str(data.get("target_ticker") or "").upper() or None) if data.get("target_ticker") else None,
            trigger_type=str(data.get("trigger_type") or "").lower(),
            trigger_value=str(data.get("trigger_value") or ""),
            created_at=str(data.get("created_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z"),
            status=str(data.get("status") or "pending").lower(),
            instruction_id=str(data.get("instruction_id") or "") or None,
            rationale=str(data.get("rationale") or "") or None,
            triggered_at=str(data.get("triggered_at") or "") or None,
            requested_notional_usd=_safe_float(data.get("requested_notional_usd")),
            requested_quantity=_safe_int(data.get("requested_quantity")),
            partial_fraction=_safe_float(data.get("partial_fraction")),
            override_constraints=bool(data.get("override_constraints") or False),
        )


class ScheduledActionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ScheduledAction]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [ScheduledAction.from_dict(item) for item in items if isinstance(item, dict)]

    def save(self, actions: list[ScheduledAction]) -> None:
        payload = {"actions": [a.to_dict() for a in actions]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, action: ScheduledAction) -> ScheduledAction:
        actions = self.load()
        actions.append(action)
        self.save(actions)
        logger.info(
            "scheduled_action.add id=%s trigger=%s/%s wrapped=%s ticker=%s",
            action.action_id,
            action.trigger_type,
            action.trigger_value,
            action.wrapped_intent_type,
            action.target_ticker,
        )
        return action

    def get_due(self, now: datetime | None = None, *, market_open: bool = False, market_close: bool = False) -> list[ScheduledAction]:
        now = now or datetime.utcnow()
        actions = self.load()
        due: list[ScheduledAction] = []
        for action in actions:
            if action.status != "pending":
                continue
            if _is_due(action, now, market_open=market_open, market_close=market_close):
                due.append(action)
        return due

    def mark_triggered(self, action_id: str, *, triggered_at: str | None = None) -> bool:
        actions = self.load()
        updated = False
        for index, action in enumerate(actions):
            if action.action_id == action_id:
                actions[index] = ScheduledAction(
                    action_id=action.action_id,
                    note=action.note,
                    wrapped_intent_type=action.wrapped_intent_type,
                    target_ticker=action.target_ticker,
                    trigger_type=action.trigger_type,
                    trigger_value=action.trigger_value,
                    created_at=action.created_at,
                    status="triggered",
                    instruction_id=action.instruction_id,
                    rationale=action.rationale,
                    triggered_at=triggered_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    requested_notional_usd=action.requested_notional_usd,
                    requested_quantity=action.requested_quantity,
                    partial_fraction=action.partial_fraction,
                    override_constraints=action.override_constraints,
                )
                updated = True
                break
        if updated:
            self.save(actions)
            logger.info("scheduled_action.triggered id=%s", action_id)
        return updated

    def cancel(self, action_id: str) -> bool:
        actions = self.load()
        updated = False
        for index, action in enumerate(actions):
            if action.action_id == action_id and action.status == "pending":
                actions[index] = ScheduledAction(
                    action_id=action.action_id,
                    note=action.note,
                    wrapped_intent_type=action.wrapped_intent_type,
                    target_ticker=action.target_ticker,
                    trigger_type=action.trigger_type,
                    trigger_value=action.trigger_value,
                    created_at=action.created_at,
                    status="cancelled",
                    instruction_id=action.instruction_id,
                    rationale=action.rationale,
                    triggered_at=action.triggered_at,
                    requested_notional_usd=action.requested_notional_usd,
                    requested_quantity=action.requested_quantity,
                    partial_fraction=action.partial_fraction,
                    override_constraints=action.override_constraints,
                )
                updated = True
                break
        if updated:
            self.save(actions)
            logger.info("scheduled_action.cancel id=%s", action_id)
        return updated


def _is_due(action: ScheduledAction, now: datetime, *, market_open: bool, market_close: bool) -> bool:
    if action.trigger_type == "datetime":
        try:
            trigger_at = datetime.fromisoformat(action.trigger_value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if trigger_at.tzinfo is not None:
            now_cmp = now.astimezone(trigger_at.tzinfo) if now.tzinfo else now.replace(tzinfo=trigger_at.tzinfo)
        else:
            now_cmp = now.replace(tzinfo=None) if now.tzinfo else now
        return now_cmp >= trigger_at
    if action.trigger_type == "market_open":
        return market_open
    if action.trigger_type == "market_close":
        return market_close
    if action.trigger_type == "day_of_week":
        value = action.trigger_value.lower().strip()
        if value == "tomorrow":
            try:
                created = datetime.fromisoformat(action.created_at.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                return False
            target_date = (created + timedelta(days=1)).date()
            return now.date() >= target_date
        target_weekday = _WEEKDAYS.get(value)
        if target_weekday is None:
            return False
        return now.weekday() == target_weekday
    return False


def make_action_id() -> str:
    return uuid.uuid4().hex[:8]


def make_scheduled_action(
    *,
    note: str,
    wrapped_intent_type: str,
    trigger_type: str,
    trigger_value: str,
    target_ticker: str | None = None,
    instruction_id: str | None = None,
    rationale: str | None = None,
    requested_notional_usd: float | None = None,
    requested_quantity: int | None = None,
    partial_fraction: float | None = None,
    override_constraints: bool = False,
    action_id: str | None = None,
) -> ScheduledAction:
    return ScheduledAction(
        action_id=action_id or make_action_id(),
        note=note,
        wrapped_intent_type=wrapped_intent_type.lower(),
        target_ticker=target_ticker.upper() if target_ticker else None,
        trigger_type=trigger_type.lower(),
        trigger_value=trigger_value,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        status="pending",
        instruction_id=instruction_id,
        rationale=rationale,
        requested_notional_usd=requested_notional_usd,
        requested_quantity=requested_quantity,
        partial_fraction=partial_fraction,
        override_constraints=override_constraints,
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _safe_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
