from __future__ import annotations

from typing import Literal

Action = Literal["BUY", "SELL", "HOLD", "WAIT"]

TRADE_ACTIONS = {"BUY", "SELL"}
PASSIVE_ACTIONS = {"HOLD", "WAIT"}
ALL_ACTIONS = {*TRADE_ACTIONS, *PASSIVE_ACTIONS}

ACTION_SCHEMA_TEXT = "BUY|SELL|HOLD|WAIT"
QUANTITY_RULE_TEXT = "BUY/SELL require quantity > 0. HOLD/WAIT require quantity 0."


def is_trade_action(action: str) -> bool:
    return action in TRADE_ACTIONS


def is_passive_action(action: str) -> bool:
    return action in PASSIVE_ACTIONS


def validate_action(action: str, *, label: str = "action") -> None:
    if action not in ALL_ACTIONS:
        raise ValueError(f"{label} must be BUY, SELL, HOLD, or WAIT")


def quantity_is_valid_for_action(action: str, quantity: int) -> bool:
    if is_passive_action(action):
        return quantity == 0
    if is_trade_action(action):
        return quantity > 0
    return False


def normalize_quantity_for_action(action: str, quantity: int) -> int:
    if is_passive_action(action):
        return 0
    return quantity
