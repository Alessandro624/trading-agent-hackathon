from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

FailureType = Literal[
    "tool_transient",
    "risk_block",
    "market_closed",
    "broker_permanent",
    "broker_unknown",
]


_DEFAULT_MAX_RETRIES: dict[str, int] = {
    "tool_transient": 2,
    "risk_block": 0,
    "market_closed": 1,
    "broker_permanent": 0,
    "broker_unknown": 0,
}


_TRANSIENT_PATTERNS = [
    r"\brate\s*limit\b",
    r"\b429\b",
    r"\btimeout\b",
    r"\btimed?\s*out\b",
    r"\btemporary\b",
    r"\btransient\b",
    r"\bnetwork\b",
    r"\bconnection\s*(reset|refused|closed)\b",
    r"\b503\b",
    r"\b502\b",
    r"\bgateway\b",
    r"\bservice\s*unavailable\b",
]
_MARKET_CLOSED_PATTERNS = [
    r"\bmarket\s*(is\s*)?closed\b",
    r"\bmarket_closed\b",
    r"\boutside\s*market\s*hours\b",
    r"\bnot\s*open\b",
    r"\btrading\s*halted\b",
]
_BROKER_PERMANENT_PATTERNS = [
    r"\binsufficient\s*(buying\s*power|cash|funds|equity)\b",
    r"\bnot\s*enough\s*cash\b",
    r"\bpattern\s*day\s*trader\b",
    r"\bnot\s*tradable\b",
    r"\bnot\s*found\b",
    r"\bunknown\s*symbol\b",
    r"\bfractional\s*not\s*supported\b",
    r"\bprice\s*out\s*of\s*band\b",
]


@dataclass(frozen=True)
class ExecutionFailure:
    instruction_id: str | None
    failure_type: FailureType
    error_message: str
    retryable: bool
    suggested_retry_delay_seconds: float | None
    rationale: str


def classify_failure(
    *,
    instruction_id: str | None,
    outcome: str,
    error_message: str,
    execution_status: str | None = None,
    risk_can_trade: bool | None = None,
    market_is_open: bool | None = None,
) -> ExecutionFailure:
    message = str(error_message or "")
    outcome_lower = (outcome or "").lower()
    status_lower = (execution_status or "").lower()

    if risk_can_trade is False or outcome_lower == "blocked" or status_lower == "blocked":
        return ExecutionFailure(
            instruction_id=instruction_id,
            failure_type="risk_block",
            error_message=message,
            retryable=False,
            suggested_retry_delay_seconds=None,
            rationale="Risk manager blocked the trade (can_trade=False or quantity above limit). Instruction is infeasible at current portfolio state; not retried.",
        )

    if market_is_open is False or _matches_any(message, _MARKET_CLOSED_PATTERNS):
        return ExecutionFailure(
            instruction_id=instruction_id,
            failure_type="market_closed",
            error_message=message,
            retryable=True,
            suggested_retry_delay_seconds=None,
            rationale="Market is closed. Instruction will be retried at the next market open.",
        )

    if _matches_any(message, _TRANSIENT_PATTERNS):
        return ExecutionFailure(
            instruction_id=instruction_id,
            failure_type="tool_transient",
            error_message=message,
            retryable=True,
            suggested_retry_delay_seconds=None,
            rationale="Transient tool error (rate limit / timeout / network). Instruction will be retried.",
        )

    if _matches_any(message, _BROKER_PERMANENT_PATTERNS):
        return ExecutionFailure(
            instruction_id=instruction_id,
            failure_type="broker_permanent",
            error_message=message,
            retryable=False,
            suggested_retry_delay_seconds=None,
            rationale="Permanent broker rejection (insufficient funds / not tradable / unknown symbol). Instruction abandoned; user must correct manually.",
        )

    return ExecutionFailure(
        instruction_id=instruction_id,
        failure_type="broker_unknown",
        error_message=message,
        retryable=False,
        suggested_retry_delay_seconds=None,
        rationale="Unrecognised failure. Instruction abandoned to avoid retrying an unknown error.",
    )


def should_retry(failure: ExecutionFailure, retry_count: int) -> bool:
    if not failure.retryable:
        return False
    max_retries = max_retries_for_failure_type(failure.failure_type)
    return retry_count < max_retries


def max_retries_for_failure_type(failure_type: str) -> int:
    return _DEFAULT_MAX_RETRIES.get(failure_type, 0)


def _matches_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)
