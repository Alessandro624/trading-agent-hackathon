from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.core.evidence import Evidence


@dataclass(frozen=True)
class OutcomeRationale:
    user_summary: str
    next_step: str
    evidence: tuple[Evidence, ...] = ()
    technical_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_summary": self.user_summary,
            "next_step": self.next_step,
            "evidence": [e.to_dict() for e in self.evidence],
            "technical_details": dict(self.technical_details),
        }


@dataclass(frozen=True)
class ResolverOutcome:
    outcome_type: str
    request_id: str
    subnote_index: int
    rationale: OutcomeRationale
    idempotency_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_type": self.outcome_type,
            "request_id": self.request_id,
            "subnote_index": self.subnote_index,
            "idempotency_key": self.idempotency_key,
            "rationale": self.rationale.to_dict(),
        }


@dataclass(frozen=True)
class ExecutableInstruction(ResolverOutcome):
    intent_type: str
    target_ticker: str | None = None
    target_tickers: tuple[str, ...] = ()
    topic: str | None = None
    confidence: float = 0.0
    requested_notional_usd: float | None = None
    requested_quantity: int | None = None
    partial_fraction: float | None = None
    override_constraints: bool = False
    cancel_target_ids: tuple[str, ...] = ()
    cancel_scope: str | None = None
    cancel_reversal_action: str | None = None
    excluded_tickers: tuple[dict[str, Any], ...] = ()
    discovered_candidates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "intent_type": self.intent_type,
                "target_ticker": self.target_ticker,
                "target_tickers": list(self.target_tickers),
                "topic": self.topic,
                "confidence": self.confidence,
                "requested_notional_usd": self.requested_notional_usd,
                "requested_quantity": self.requested_quantity,
                "partial_fraction": self.partial_fraction,
                "override_constraints": self.override_constraints,
                "cancel_target_ids": list(self.cancel_target_ids),
                "cancel_scope": self.cancel_scope,
                "cancel_reversal_action": self.cancel_reversal_action,
                "excluded_tickers": [dict(t) for t in self.excluded_tickers],
                "discovered_candidates": [dict(c) for c in self.discovered_candidates],
            }
        )
        return base


@dataclass(frozen=True)
class PendingConfirmation(ResolverOutcome):
    proposal_text: str
    proposed_operations: tuple[dict[str, Any], ...]
    confirmation_id: str
    expires_at: str
    originating_request_id: str

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "proposal_text": self.proposal_text,
                "proposed_operations": [dict(o) for o in self.proposed_operations],
                "confirmation_id": self.confirmation_id,
                "expires_at": self.expires_at,
                "originating_request_id": self.originating_request_id,
            }
        )
        return base


@dataclass(frozen=True)
class ConfirmedOperations(ResolverOutcome):
    confirmation_id: str
    operations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "confirmation_id": self.confirmation_id,
                "operations": [dict(operation) for operation in self.operations],
            }
        )
        return base


@dataclass(frozen=True)
class InformationRequest(ResolverOutcome):
    question_type: str
    reply: str | None = None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"question_type": self.question_type, "reply": self.reply})
        return base


@dataclass(frozen=True)
class PersistentStateUpdate(ResolverOutcome):
    action: str
    constraint_type: str
    value: str
    constraint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "action": self.action,
                "constraint_type": self.constraint_type,
                "value": self.value,
                "constraint_id": self.constraint_id,
            }
        )
        return base


@dataclass(frozen=True)
class ScheduledActionOutcome(ResolverOutcome):
    wrapped_intent_type: str
    target_ticker: str | None
    trigger_type: str
    trigger_value: str
    requested_notional_usd: float | None = None
    requested_quantity: int | None = None
    partial_fraction: float | None = None
    override_constraints: bool = False

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "wrapped_intent_type": self.wrapped_intent_type,
                "target_ticker": self.target_ticker,
                "trigger_type": self.trigger_type,
                "trigger_value": self.trigger_value,
                "requested_notional_usd": self.requested_notional_usd,
                "requested_quantity": self.requested_quantity,
                "partial_fraction": self.partial_fraction,
                "override_constraints": self.override_constraints,
            }
        )
        return base


@dataclass(frozen=True)
class ConditionalOrderOutcome(ResolverOutcome):
    target_ticker: str
    trigger_type: str
    trigger_price: float | None
    trigger_fraction: float | None
    is_immediate: bool = False

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "target_ticker": self.target_ticker,
                "trigger_type": self.trigger_type,
                "trigger_price": self.trigger_price,
                "trigger_fraction": self.trigger_fraction,
                "is_immediate": self.is_immediate,
            }
        )
        return base


@dataclass(frozen=True)
class AdvisoryReply(ResolverOutcome):
    advisory_type: str
    reply: str

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"advisory_type": self.advisory_type, "reply": self.reply})
        return base


def outcome_to_human_instructions(outcome: ResolverOutcome) -> list:
    from trading_agent.core.human_instruction import HumanInstruction

    if isinstance(outcome, ConfirmedOperations):
        instructions = []
        from trading_agent.core.instruction_ledger import make_idempotency_key

        for index, operation in enumerate(outcome.operations, start=1):
            ticker = str(operation.get("ticker") or "").upper() or None
            intent_type = str(operation.get("intent_type") or "forced_buy")
            instructions.append(
                HumanInstruction(
                    note=outcome.rationale.user_summary,
                    target_ticker=ticker,
                    reason="human_confirmed_proposal",
                    sequence_index=index,
                    sequence_total=len(outcome.operations),
                    resolver_confidence=0.9,
                    resolver_rationale=outcome.rationale.user_summary,
                    resolver_intent_type=intent_type,
                    requested_notional_usd=operation.get("requested_notional_usd"),
                    requested_quantity=operation.get("requested_quantity"),
                    partial_fraction=operation.get("partial_fraction"),
                    override_constraints=bool(operation.get("override_constraints", False)),
                    request_id=outcome.request_id,
                    subnote_index=outcome.subnote_index,
                    idempotency_key=make_idempotency_key(outcome.request_id, outcome.subnote_index, intent_type, ticker),
                    evidence=tuple(e.to_dict() for e in outcome.rationale.evidence),
                    user_summary=outcome.rationale.user_summary,
                    next_step=outcome.rationale.next_step,
                    technical_details=dict(outcome.rationale.technical_details),
                    originating_confirmation_id=outcome.confirmation_id,
                    workflow_id=f"workflow-{outcome.confirmation_id}",
                    workflow_type=str(operation.get("workflow_type") or "confirmed_operations"),
                    resolver_topic=str(operation.get("topic") or operation.get("workflow_type") or "confirmed_operations"),
                )
            )
        return instructions

    if not isinstance(outcome, ExecutableInstruction):
        return []

    common_kwargs = dict(
        note=outcome.rationale.user_summary,
        resolver_confidence=outcome.confidence,
        resolver_rationale=outcome.rationale.user_summary,
        resolver_topic=outcome.topic,
        requested_notional_usd=outcome.requested_notional_usd,
        requested_quantity=outcome.requested_quantity,
        partial_fraction=outcome.partial_fraction,
        override_constraints=outcome.override_constraints,
        request_id=outcome.request_id,
        idempotency_key=outcome.idempotency_key,
        subnote_index=outcome.subnote_index,
        evidence=outcome.rationale.evidence and tuple(e.to_dict() for e in outcome.rationale.evidence) or (),
        user_summary=outcome.rationale.user_summary,
        next_step=outcome.rationale.next_step,
        technical_details=dict(outcome.rationale.technical_details) if outcome.rationale.technical_details else None,
        discovered_candidates=outcome.discovered_candidates,
    )

    if outcome.intent_type == "cancel":
        instructions = [
            HumanInstruction(
                **common_kwargs,
                target_ticker=None,
                reason="human_cancel",
                resolver_intent_type="cancel",
                cancel_target_ids=outcome.cancel_target_ids,
                cancel_scope=outcome.cancel_scope,
                cancel_reversal_action=outcome.cancel_reversal_action,
            )
        ]
        return instructions

    targets = outcome.target_tickers or ((outcome.target_ticker,) if outcome.target_ticker else ())
    if not targets:
        return []
    reason = "human_forced_action"
    if outcome.intent_type in {"broad_sell", "conditional_sell", "position_sweep"}:
        reason = "human_llm_position_sweep"
    if outcome.intent_type == "news_request":
        reason = "human_news_request"
    instructions: list[HumanInstruction] = []
    for index, ticker in enumerate(targets, start=1):
        if index == 1:
            key = outcome.idempotency_key
        else:
            from trading_agent.core.instruction_ledger import make_idempotency_key

            key = make_idempotency_key(outcome.request_id, outcome.subnote_index, outcome.intent_type, ticker)
        instructions.append(
            HumanInstruction(
                **{**common_kwargs, "idempotency_key": key},
                target_ticker=ticker,
                reason=reason,
                sequence_index=index,
                sequence_total=len(targets),
                resolver_intent_type=outcome.intent_type,
                resolver_excluded_tickers=outcome.excluded_tickers,
            )
        )
    return instructions
