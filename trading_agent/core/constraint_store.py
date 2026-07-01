from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.constraint_store")


@dataclass(frozen=True)
class PortfolioConstraint:
    constraint_id: str
    type: str
    value: str
    rationale: str
    created_at: str
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioConstraint":
        return cls(
            constraint_id=str(data.get("constraint_id") or uuid.uuid4().hex[:8]),
            type=str(data.get("type") or "exclude_ticker"),
            value=str(data.get("value") or ""),
            rationale=str(data.get("rationale") or ""),
            created_at=str(data.get("created_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z"),
            active=bool(data.get("active", True)),
        )


class ConstraintStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[PortfolioConstraint]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = data.get("constraints") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [PortfolioConstraint.from_dict(item) for item in items if isinstance(item, dict)]

    def save(self, constraints: list[PortfolioConstraint]) -> None:
        payload = {"constraints": [c.to_dict() for c in constraints]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, constraint: PortfolioConstraint) -> PortfolioConstraint:
        constraints = self.load()
        constraints = [c for c in constraints if not (c.type == constraint.type and c.value.lower() == constraint.value.lower())]
        constraints.append(constraint)
        self.save(constraints)
        logger.info("constraint.add id=%s type=%s value=%s", constraint.constraint_id, constraint.type, constraint.value)
        return constraint

    def deactivate(self, constraint_id: str) -> bool:
        constraints = self.load()
        updated = False
        for index, constraint in enumerate(constraints):
            if constraint.constraint_id == constraint_id:
                constraints[index] = PortfolioConstraint(
                    constraint_id=constraint.constraint_id,
                    type=constraint.type,
                    value=constraint.value,
                    rationale=constraint.rationale,
                    created_at=constraint.created_at,
                    active=False,
                )
                updated = True
                break
        if updated:
            self.save(constraints)
            logger.info("constraint.deactivate id=%s", constraint_id)
        return updated

    def deactivate_by_value(self, type: str, value: str) -> int:
        constraints = self.load()
        count = 0
        for index, constraint in enumerate(constraints):
            if constraint.type == type and constraint.value.lower() == value.lower() and constraint.active:
                constraints[index] = PortfolioConstraint(
                    constraint_id=constraint.constraint_id,
                    type=constraint.type,
                    value=constraint.value,
                    rationale=constraint.rationale,
                    created_at=constraint.created_at,
                    active=False,
                )
                count += 1
        if count:
            self.save(constraints)
            logger.info("constraint.deactivate_by_value type=%s value=%s count=%d", type, value, count)
        return count

    def active(self) -> list[PortfolioConstraint]:
        return [c for c in self.load() if c.active]

    def applies_to(
        self,
        ticker: str,
        sector: str | None = None,
        price: float | None = None,
    ) -> list[PortfolioConstraint]:
        violations: list[PortfolioConstraint] = []
        upper_ticker = (ticker or "").upper()
        for constraint in self.active():
            if constraint.type == "exclude_ticker" and constraint.value.upper() == upper_ticker:
                violations.append(constraint)
                continue
            if constraint.type == "exclude_sector" and sector and constraint.value.lower() == sector.lower():
                violations.append(constraint)
                continue
            if constraint.type == "max_price":
                try:
                    threshold = float(constraint.value)
                    if price is not None and price > threshold:
                        violations.append(constraint)
                except ValueError:
                    continue
                continue
            if constraint.type == "esg_filter":
                continue
        return violations


def make_constraint_id() -> str:
    return uuid.uuid4().hex[:8]


def make_constraint(
    *,
    type: str,
    value: str,
    rationale: str,
    constraint_id: str | None = None,
) -> PortfolioConstraint:
    return PortfolioConstraint(
        constraint_id=constraint_id or make_constraint_id(),
        type=type,
        value=value,
        rationale=rationale,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        active=True,
    )
