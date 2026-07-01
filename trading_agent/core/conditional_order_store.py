from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.conditional_order_store")


@dataclass(frozen=True)
class ConditionalOrder:
    order_id: str
    ticker: str
    side: str
    trigger_type: str
    trigger_price: float | None
    trigger_fraction: float | None
    created_at: str
    status: str
    instruction_id: str | None = None
    rationale: str | None = None
    triggered_at: str | None = None
    triggered_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionalOrder":
        return cls(
            order_id=str(data.get("order_id") or uuid.uuid4().hex[:8]),
            ticker=str(data.get("ticker") or "").upper(),
            side=str(data.get("side") or "sell").lower(),
            trigger_type=str(data.get("trigger_type") or "").lower(),
            trigger_price=_safe_float(data.get("trigger_price")),
            trigger_fraction=_safe_float(data.get("trigger_fraction")),
            created_at=str(data.get("created_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z"),
            status=str(data.get("status") or "pending").lower(),
            instruction_id=str(data.get("instruction_id") or "") or None,
            rationale=str(data.get("rationale") or "") or None,
            triggered_at=str(data.get("triggered_at") or "") or None,
            triggered_price=_safe_float(data.get("triggered_price")),
        )


class ConditionalOrderStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ConditionalOrder]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("orders") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [ConditionalOrder.from_dict(item) for item in items if isinstance(item, dict)]

    def save(self, orders: list[ConditionalOrder]) -> None:
        payload = {"orders": [o.to_dict() for o in orders]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, order: ConditionalOrder) -> ConditionalOrder:
        orders = self.load()
        orders = [o for o in orders if not (o.ticker == order.ticker and o.trigger_type == order.trigger_type and o.status == "pending")]
        orders.append(order)
        self.save(orders)
        logger.info(
            "conditional_order.add id=%s ticker=%s trigger=%s price=%s fraction=%s",
            order.order_id,
            order.ticker,
            order.trigger_type,
            order.trigger_price,
            order.trigger_fraction,
        )
        return order

    def get_pending(self, ticker: str | None = None) -> list[ConditionalOrder]:
        orders = self.load()
        pending = [o for o in orders if o.status == "pending"]
        if ticker is None:
            return pending
        upper = ticker.upper()
        return [o for o in pending if o.ticker == upper]

    def mark_triggered(
        self,
        order_id: str,
        triggered_price: float,
        *,
        triggered_at: str | None = None,
    ) -> bool:
        orders = self.load()
        updated = False
        for index, order in enumerate(orders):
            if order.order_id == order_id:
                orders[index] = ConditionalOrder(
                    order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    trigger_type=order.trigger_type,
                    trigger_price=order.trigger_price,
                    trigger_fraction=order.trigger_fraction,
                    created_at=order.created_at,
                    status="triggered",
                    instruction_id=order.instruction_id,
                    rationale=order.rationale,
                    triggered_at=triggered_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    triggered_price=triggered_price,
                )
                updated = True
                break
        if updated:
            self.save(orders)
            logger.info(
                "conditional_order.triggered id=%s price=%s",
                order_id,
                triggered_price,
            )
        return updated

    def cancel(self, order_id: str) -> bool:
        orders = self.load()
        updated = False
        for index, order in enumerate(orders):
            if order.order_id == order_id and order.status == "pending":
                orders[index] = ConditionalOrder(
                    order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    trigger_type=order.trigger_type,
                    trigger_price=order.trigger_price,
                    trigger_fraction=order.trigger_fraction,
                    created_at=order.created_at,
                    status="cancelled",
                    instruction_id=order.instruction_id,
                    rationale=order.rationale,
                    triggered_at=order.triggered_at,
                    triggered_price=order.triggered_price,
                )
                updated = True
                break
        if updated:
            self.save(orders)
            logger.info("conditional_order.cancel id=%s", order_id)
        return updated


def make_order_id() -> str:
    return uuid.uuid4().hex[:8]


def make_conditional_order(
    *,
    ticker: str,
    trigger_type: str,
    trigger_price: float | None = None,
    trigger_fraction: float | None = None,
    side: str = "sell",
    instruction_id: str | None = None,
    rationale: str | None = None,
    order_id: str | None = None,
) -> ConditionalOrder:
    return ConditionalOrder(
        order_id=order_id or make_order_id(),
        ticker=ticker.upper(),
        side=side.lower(),
        trigger_type=trigger_type.lower(),
        trigger_price=trigger_price,
        trigger_fraction=trigger_fraction,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        status="pending",
        instruction_id=instruction_id,
        rationale=rationale,
    )


def check_trigger(order: ConditionalOrder, current_price: float | None, avg_entry_price: float | None = None) -> bool:
    if current_price is None or current_price <= 0:
        return False
    if order.trigger_type in {"price_above", "take_profit"}:
        threshold = order.trigger_price
        if threshold is None and order.trigger_type == "take_profit" and avg_entry_price:
            threshold = avg_entry_price * 1.20
        if threshold is None:
            return False
        return current_price >= threshold
    if order.trigger_type in {"price_below", "stop_loss"}:
        threshold = order.trigger_price
        if threshold is None and order.trigger_type == "stop_loss" and avg_entry_price:
            threshold = avg_entry_price * 0.90
        if threshold is None:
            return False
        return current_price <= threshold
    return False


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
