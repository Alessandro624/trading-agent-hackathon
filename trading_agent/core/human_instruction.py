from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.human_intent import (
    detect_non_usd_currency,
    llm_split_compound_note,
    parse_human_intent,
    split_compound_note,
)
from trading_agent.core.portfolio import positions as portfolio_positions

logger = logging.getLogger("trading_agent.human_instruction")

_RESOLVER_CACHE: dict[str, list[dict[str, Any]]] = {}


def clear_resolver_cache() -> None:
    _RESOLVER_CACHE.clear()


def _resolver_cache_key(
    note: str,
    buyable_universe: list[str],
    sellable_targets: list[str],
    pending_payload: list[dict[str, Any]],
    recent_executed_payload: list[dict[str, Any]],
) -> str:
    payload = {
        "note": note,
        "buyable": buyable_universe,
        "sellable": sellable_targets,
        "pending_ids": sorted(f"{item.get('instruction_id')}:{item.get('intent_type')}:{item.get('target_ticker')}" for item in pending_payload),
        "recent_ids": sorted(f"{item.get('instruction_id')}:{item.get('outcome')}" for item in recent_executed_payload),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


_CACHE_PRESERVED_FIELDS = (
    "note",
    "target_ticker",
    "reason",
    "sequence_index",
    "sequence_total",
    "resolver_confidence",
    "resolver_rationale",
    "resolver_topic",
    "resolver_intent_type",
    "reversal_of",
    "cancel_target_ids",
    "cancel_scope",
    "cancel_reversal_action",
    "resolver_excluded_tickers",
    "requested_notional_usd",
    "requested_quantity",
    "partial_fraction",
    "question_type",
    "override_constraints",
    "constraint_update",
    "conditional_order",
    "scheduled_trigger",
    "allocation_targets",
    "web_search_context",
    "workflow_id",
    "workflow_type",
)


def _cache_serialize_instruction(instruction: HumanInstruction) -> dict[str, Any]:
    return {field: getattr(instruction, field) for field in _CACHE_PRESERVED_FIELDS}


def _reissue_from_cached(cached: dict[str, Any]) -> HumanInstruction:
    kwargs = {field: cached.get(field) for field in _CACHE_PRESERVED_FIELDS}
    cleaned = {key: value for key, value in kwargs.items() if value is not None or key in {"note", "target_ticker", "reason"}}
    return HumanInstruction(**cleaned)


def _generate_instruction_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class HumanInstruction:
    note: str
    target_ticker: str | None
    reason: str
    sequence_index: int = 1
    sequence_total: int = 1
    resolver_confidence: float | None = None
    resolver_rationale: str | None = None
    resolver_topic: str | None = None
    resolver_intent_type: str | None = None
    instruction_id: str = field(default_factory=_generate_instruction_id)
    cancelled_by: str | None = None
    reversal_of: str | None = None
    retry_count: int = 0
    max_retries: int | None = None
    cancel_target_ids: tuple[str, ...] = ()
    cancel_scope: str | None = None
    cancel_reversal_action: str | None = None
    resolver_excluded_tickers: tuple[dict[str, Any], ...] = ()
    requested_notional_usd: float | None = None
    requested_quantity: int | None = None
    partial_fraction: float | None = None
    question_type: str | None = None
    override_constraints: bool = False
    constraint_update: dict[str, Any] | None = None
    conditional_order: dict[str, Any] | None = None
    scheduled_trigger: dict[str, Any] | None = None
    allocation_targets: tuple[dict[str, Any], ...] = ()
    web_search_context: tuple[dict[str, Any], ...] = ()
    request_id: str | None = None
    idempotency_key: str | None = None
    subnote_index: int = 0
    evidence: tuple[dict[str, Any], ...] = ()
    user_summary: str | None = None
    next_step: str | None = None
    technical_details: dict[str, Any] | None = None
    discovered_candidates: tuple[dict[str, Any], ...] = ()
    originating_confirmation_id: str | None = None
    workflow_id: str | None = None
    workflow_type: str | None = None


def plan_human_instructions(
    notes: list[str],
    *,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None = None,
    pending_instructions: list[HumanInstruction] | None = None,
    recent_executed_instructions: list[dict[str, Any]] | None = None,
    ticker_validator: Callable[[str], bool] | None = None,
) -> list[HumanInstruction]:
    instructions: list[HumanInstruction] = []
    pending = pending_instructions or []
    recent_executed = recent_executed_instructions or []
    for note in notes:
        cleaned = clean_text(note, max_chars=2000)
        if not cleaned:
            continue
        if llm_client is not None:
            sub_notes = llm_split_compound_note(cleaned, llm_client)
        else:
            sub_notes = split_compound_note(cleaned)
        if not sub_notes:
            continue
        for sub_note in sub_notes:
            instructions.extend(
                _instructions_for_note(
                    sub_note,
                    watchlist,
                    portfolio,
                    fallback_ticker,
                    llm_client,
                    pending,
                    recent_executed,
                    ticker_validator,
                )
            )
    return instructions


def _instructions_for_note(
    note: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None,
    pending_instructions: list[HumanInstruction] | None = None,
    recent_executed_instructions: list[dict[str, Any]] | None = None,
    ticker_validator: Callable[[str], bool] | None = None,
) -> list[HumanInstruction]:
    non_usd_currency = detect_non_usd_currency(note)
    if non_usd_currency:
        logger.info(
            "human.instruction.non_usd_rejected currency=%s note=%s",
            non_usd_currency,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_context",
                resolver_intent_type="advisory",
                resolver_rationale=(f"Currency {non_usd_currency} is not supported. " "Please specify amounts in USD (e.g. 'invest $50,000')."),
                resolver_topic="non_usd_currency",
            )
        ]

    short_match = _detect_short_selling_intent(note)
    if short_match:
        logger.info(
            "human.instruction.short_rejected cue=%s note=%s",
            short_match,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_context",
                resolver_intent_type="advisory",
                resolver_rationale=("Short selling is not supported. Only long positions " "(buy/sell) are available in this simulation."),
                resolver_topic="short_selling_not_supported",
            )
        ]

    if _is_vague_delegative_note(note):
        logger.info("human.instruction.vague_delegative note=%s", _short_note(note))
        return [
            HumanInstruction(
                note,
                None,
                "human_context",
                resolver_intent_type="advisory",
                resolver_confidence=0.2,
                resolver_rationale=("Your message was too vague for me to act on. " "Please specify a ticker, sector, or action. " "Example: 'Buy tech stocks' or 'Sell my AAPL position'."),
                resolver_topic="vague_delegation",
            )
        ]

    intent = parse_human_intent([note])

    if _requires_llm_resolution(note, intent):
        logger.info("human.instruction.route_to_llm reason=rich_intent_cues note=%s", _short_note(note))
        resolved = _llm_resolve_note(
            note,
            watchlist,
            portfolio,
            llm_client,
            pending_instructions=pending_instructions or [],
            recent_executed_instructions=recent_executed_instructions or [],
            ticker_validator=ticker_validator,
        )
        if resolved:
            return resolved

    if intent.requested_action in {"BUY", "SELL"} and intent.tickers:
        targets = intent.tickers
        intent_type = "forced_buy" if intent.requested_action == "BUY" else "forced_sell"
        logger.info(
            "human.instruction.deterministic intent_type=%s targets=%s note=%s",
            intent_type,
            ",".join(targets),
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                target,
                "human_forced_action",
                resolver_intent_type=intent_type,
            )
            for target in targets
        ]

    if "news_request" in intent.intents and intent.tickers:
        logger.info(
            "human.instruction.news_request targets=%s note=%s",
            ",".join(intent.tickers),
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                target,
                "human_news_request",
                resolver_intent_type="news_request",
                resolver_rationale=f"User asked to read news about {target}.",
            )
            for target in intent.tickers
        ]

    if "position_sweep" in intent.intents and not intent.impact_topic:
        targets = _open_position_targets(watchlist, portfolio, fallback_ticker=fallback_ticker)
        if targets:
            total = len(targets)
            logger.info(
                "human.instruction.position_sweep targets=%s note=%s",
                ",".join(targets),
                _short_note(note),
            )
            return [
                HumanInstruction(
                    note,
                    target,
                    "human_position_sweep",
                    index,
                    total,
                    resolver_intent_type="position_sweep",
                )
                for index, target in enumerate(targets, start=1)
            ]

    if intent.tickers and "cancel" not in intent.intents:
        logger.info(
            "human.instruction.advisory_ticker target=%s note=%s",
            intent.tickers[-1],
            _short_note(note),
        )
        return [HumanInstruction(note, intent.tickers[-1], "human_input", resolver_intent_type="advisory")]

    resolved = _llm_resolve_note(
        note,
        watchlist,
        portfolio,
        llm_client,
        pending_instructions=pending_instructions or [],
        recent_executed_instructions=recent_executed_instructions or [],
        ticker_validator=ticker_validator,
    )
    if resolved:
        return resolved

    if intent.impact_topic or "conditional_sell" in intent.intents:
        logger.info(
            "human.instruction.semantic_required topic=%s reason=no_valid_llm_resolution note=%s",
            intent.impact_topic,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_context_requires_resolver",
                resolver_intent_type="advisory",
                resolver_topic=intent.impact_topic,
            )
        ]

    logger.info(
        "human.instruction.unresolved reason=no_resolver_output note=%s",
        _short_note(note),
    )
    return [HumanInstruction(note, None, "human_context", resolver_intent_type="advisory")]


def _llm_resolve_note(
    note: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    llm_client: Any | None,
    *,
    pending_instructions: list[HumanInstruction] | None = None,
    recent_executed_instructions: list[dict[str, Any]] | None = None,
    ticker_validator: Callable[[str], bool] | None = None,
) -> list[HumanInstruction]:
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        logger.info("human.resolver.skip reason=no_llm_client note=%s", _short_note(note))
        return []

    sellable_targets = _open_position_targets(watchlist, portfolio)
    buyable_universe = _watchlist_symbols(watchlist)
    pending_payload = _serialize_pending_instructions(pending_instructions or [])
    recent_executed_payload = _serialize_recent_executed_instructions(recent_executed_instructions or [])

    cache_key = _resolver_cache_key(
        note,
        buyable_universe,
        sellable_targets,
        pending_payload,
        recent_executed_payload,
    )
    if cache_key in _RESOLVER_CACHE:
        logger.info("human.resolver.cache.hit note=%s", _short_note(note))
        cached = _RESOLVER_CACHE[cache_key]
        return [_reissue_from_cached(item) for item in cached]

    result = _llm_resolve_note_impl(
        note,
        watchlist,
        portfolio,
        llm_client,
        pending_instructions=pending_instructions,
        recent_executed_instructions=recent_executed_instructions,
        ticker_validator=ticker_validator,
    )

    if result and cache_key:
        _RESOLVER_CACHE[cache_key] = [_cache_serialize_instruction(item) for item in result]
        logger.info(
            "human.resolver.cache.store key=%s items=%d note=%s",
            cache_key[:8],
            len(result),
            _short_note(note),
        )
    return result


def _llm_resolve_note_impl(
    note: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    llm_client: Any | None,
    *,
    pending_instructions: list[HumanInstruction] | None = None,
    recent_executed_instructions: list[dict[str, Any]] | None = None,
    ticker_validator: Callable[[str], bool] | None = None,
) -> list[HumanInstruction]:
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        logger.info("human.resolver.skip reason=no_llm_client note=%s", _short_note(note))
        return []

    sellable_targets = _open_position_targets(watchlist, portfolio)
    buyable_universe = _watchlist_symbols(watchlist)
    portfolio_payload = _enrich_positions_with_llm(portfolio_positions(portfolio), llm_client)
    pending_payload = _serialize_pending_instructions(pending_instructions or [])
    recent_executed_payload = _serialize_recent_executed_instructions(recent_executed_instructions or [])

    web_search_context = _maybe_fetch_web_search_context(note, buyable_universe, sellable_targets, llm_client)

    try:
        payload = {
            "human_note": note,
            "watchlist": buyable_universe,
            "sellable_open_position_tickers": sellable_targets,
            "portfolio_positions": portfolio_payload,
            "pending_instructions": pending_payload,
            "recent_executed_instructions": recent_executed_payload,
            "web_search_context": web_search_context,
            "instructions": (
                "Interpret the human note into the JSON schema defined by the system prompt. "
                "Every sell-side ticker MUST come from sellable_open_position_tickers. "
                "Buy-side tickers should come from watchlist unless the user explicitly names a ticker "
                "outside that initial universe; those external buy candidates will be validated by the agent. "
                "Do not invent unnamed tickers. If no open position is materially exposed to the topic, "
                "return intent_type='advisory' with an empty target_tickers list and confidence below 0.6. "
                "For cancel intents, target_instruction_ids MUST come from pending_instructions or "
                "recent_executed_instructions. Never invent instruction IDs. "
                "If `web_search_context` is provided, use it to identify the public ticker the user "
                "likely means (e.g. SpaceX -> SPCX if search results mention SPCX). "
                "Prefer tickers that appear as `ticker_candidates` in the search results."
            ),
        }
        logger.info(
            "human.resolver.llm.start sellable=%s watchlist=%s pending=%d recent_executed=%d note=%s",
            ",".join(sellable_targets),
            ",".join(buyable_universe),
            len(pending_payload),
            len(recent_executed_payload),
            _short_note(note),
        )
        raw = complete_json(_RESOLVER_PROMPT, json.dumps(payload, default=str))
        resolved = _parse_resolver_json(raw)
        logger.info(
            "human.resolver.llm.raw intent_type=%s targets=%s cancel_targets=%s confidence=%s topic=%s",
            resolved.get("intent_type"),
            ",".join(_target_ticker_codes(resolved.get("target_tickers"))),
            ",".join(_as_str_list(resolved.get("target_instruction_ids"))),
            resolved.get("confidence"),
            clean_text(str(resolved.get("topic") or ""), max_chars=120),
        )
    except Exception as error:
        raw_preview = _raw_preview(locals().get("raw", ""))
        logger.warning("human.resolver.llm.fail reason=%s raw_preview=%s note=%s", error, raw_preview, _short_note(note))
        return []

    intent_type = str(resolved.get("intent_type", "")).lower().strip()
    confidence = _confidence(resolved.get("confidence"))
    topic = clean_text(str(resolved.get("topic") or ""), max_chars=240) or None
    rationale_global = clean_text(str(resolved.get("rationale") or ""), max_chars=500) or None

    if intent_type == "cancel":
        target_ids = _as_str_list(resolved.get("target_instruction_ids"))
        if not target_ids:
            logger.info("human.resolver.llm.cancel.reject reason=no_target_ids note=%s", _short_note(note))
            return [
                HumanInstruction(
                    note,
                    None,
                    "human_context",
                    resolver_intent_type="advisory",
                    resolver_confidence=confidence,
                    resolver_rationale=rationale_global or "Cancel instruction had no target_instruction_ids.",
                    resolver_topic=topic,
                )
            ]
        cancel_scope = str(resolved.get("cancel_scope") or "all").lower().strip()
        if cancel_scope not in {"queued", "executed", "all"}:
            cancel_scope = "all"
        raw_reversal = str(resolved.get("reversal_action") or "AUTO").upper().strip()
        if raw_reversal not in {"BUY", "SELL", "NONE", "AUTO"}:
            raw_reversal = "AUTO"
        target_description = clean_text(str(resolved.get("target_description") or ""), max_chars=240) or None
        logger.info(
            "human.resolver.llm.cancel.accept scope=%s targets=%s reversal=%s confidence=%.2f",
            cancel_scope,
            ",".join(target_ids),
            raw_reversal,
            confidence,
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_cancel",
                resolver_intent_type="cancel",
                resolver_confidence=confidence,
                resolver_rationale=rationale_global,
                resolver_topic=topic,
                cancel_target_ids=tuple(target_ids),
                cancel_scope=cancel_scope,
                cancel_reversal_action=raw_reversal,
            )
        ]

    if intent_type in {"forced_buy", "forced_sell"}:
        return _forced_action_instructions(
            note,
            intent_type,
            resolved,
            buyable_universe,
            sellable_targets,
            ticker_validator,
            confidence,
            topic,
            rationale_global,
        )

    if intent_type in {"broad_sell", "conditional_sell", "position_sweep"}:
        if confidence < 0.6:
            logger.info("human.resolver.llm.reject reason=confidence_below_threshold confidence=%.2f", confidence)
            return []
        if not sellable_targets:
            logger.info("human.resolver.llm.reject reason=no_sellable_positions intent_type=%s", intent_type)
            return []
        targets = _validate_targets(resolved.get("target_tickers"), sellable_targets)
        excluded_tickers = _validate_targets(resolved.get("excluded_tickers"), sellable_targets)
        if not targets:
            validation_error = "no_valid_sellable_targets: " f"requested={','.join(_target_ticker_codes(resolved.get('target_tickers')))} " f"sellable={','.join(sellable_targets)}"
            logger.info(
                "human.resolver.llm.reject reason=no_valid_sellable_targets requested=%s sellable=%s",
                ",".join(_target_ticker_codes(resolved.get("target_tickers"))),
                ",".join(sellable_targets),
            )
            repaired = _repair_invalid_resolution(
                complete_json,
                payload,
                resolved,
                [validation_error],
                note,
            )
            if repaired:
                repaired_intent_type = str(repaired.get("intent_type", "")).lower().strip()
                repaired_confidence = _confidence(repaired.get("confidence"))
                repaired_topic = clean_text(str(repaired.get("topic") or ""), max_chars=240) or None
                repaired_rationale = clean_text(str(repaired.get("rationale") or ""), max_chars=500) or None
                if repaired_intent_type in {"forced_buy", "forced_sell"}:
                    return _forced_action_instructions(
                        note,
                        repaired_intent_type,
                        repaired,
                        buyable_universe,
                        sellable_targets,
                        ticker_validator,
                        repaired_confidence,
                        repaired_topic,
                        repaired_rationale,
                    )
                if repaired_intent_type in {"broad_sell", "conditional_sell", "position_sweep"} and repaired_confidence >= 0.6:
                    repaired_targets = _validate_targets(repaired.get("target_tickers"), sellable_targets)
                    if repaired_targets:
                        repaired_excluded_tickers = _validate_targets(repaired.get("excluded_tickers"), sellable_targets)
                        return _position_sweep_instructions(
                            note,
                            repaired_intent_type,
                            repaired_targets,
                            repaired_confidence,
                            repaired_rationale,
                            repaired_topic,
                            repaired_excluded_tickers,
                        )
            return []
        return _position_sweep_instructions(
            note,
            intent_type,
            targets,
            confidence,
            rationale_global,
            topic,
            excluded_tickers,
        )

    if intent_type == "rebalance_request":
        if confidence < 0.7:
            logger.info(
                "human.resolver.llm.rebalance.reject reason=confidence_below_threshold confidence=%.2f",
                confidence,
            )
            return []
        try:
            from trading_agent.core.rebalance import build_rebalance_plan

            plan = build_rebalance_plan(portfolio, llm_client)
        except Exception as error:
            logger.warning("human.resolver.llm.rebalance.fail reason=%s", error)
            return []
        if not plan.suggested_buys:
            logger.info(
                "human.resolver.llm.rebalance.skip reason=portfolio_already_diversified note=%s",
                _short_note(note),
            )
            return [
                HumanInstruction(
                    note,
                    None,
                    "human_context",
                    resolver_intent_type="advisory",
                    resolver_confidence=confidence,
                    resolver_rationale="Portfolio is already diversified across sectors. No rebalance needed.",
                    resolver_topic=topic,
                )
            ]
        total = len(plan.suggested_buys)
        logger.info(
            "human.resolver.llm.rebalance.accept suggested=%s rationale=%s",
            ",".join(plan.suggested_buys),
            plan.rationale,
        )
        return [
            HumanInstruction(
                note,
                ticker,
                "human_rebalance",
                index,
                total,
                confidence,
                plan.rationale,
                topic or "portfolio_rebalance",
                "forced_buy",
            )
            for index, ticker in enumerate(plan.suggested_buys, start=1)
        ]

    if intent_type == "information_request":
        question_type = str(resolved.get("question_type") or "general").lower().strip()
        if question_type not in {"pnl", "portfolio", "decision_history", "market_opinion", "cash"}:
            question_type = "general"
        logger.info(
            "human.resolver.llm.information_request question_type=%s confidence=%.2f note=%s",
            question_type,
            confidence,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_information_request",
                resolver_intent_type="information_request",
                resolver_confidence=confidence,
                resolver_rationale=rationale_global or f"User asked an information question ({question_type}).",
                resolver_topic=topic or question_type,
                question_type=question_type,
            )
        ]

    if intent_type == "constraint_update":
        constraint_update = resolved.get("constraint_update")
        if not isinstance(constraint_update, dict):
            logger.info("human.resolver.llm.constraint_update.reject reason=no_payload note=%s", _short_note(note))
            return [
                HumanInstruction(
                    note,
                    None,
                    "human_context",
                    resolver_intent_type="advisory",
                    resolver_confidence=confidence,
                    resolver_rationale="Constraint update missing required payload.",
                    resolver_topic=topic,
                )
            ]
        action = str(constraint_update.get("action") or "add").lower().strip()
        if action not in {"add", "deactivate", "override"}:
            action = "add"
        logger.info(
            "human.resolver.llm.constraint_update action=%s confidence=%.2f note=%s",
            action,
            confidence,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_constraint_update",
                resolver_intent_type="constraint_update",
                resolver_confidence=confidence,
                resolver_rationale=rationale_global,
                resolver_topic=topic,
                constraint_update={
                    "action": action,
                    "type": str(constraint_update.get("type") or "").lower().strip(),
                    "value": str(constraint_update.get("value") or "").strip(),
                    "rationale": clean_text(str(constraint_update.get("rationale") or ""), max_chars=500) or None,
                    "constraint_id": str(constraint_update.get("constraint_id") or "").strip() or None,
                },
            )
        ]

    if intent_type == "conditional_order":
        conditional_order = resolved.get("conditional_order")
        if not isinstance(conditional_order, dict):
            logger.info("human.resolver.llm.conditional_order.reject reason=no_payload note=%s", _short_note(note))
            return []
        trigger_type = str(conditional_order.get("trigger_type") or "").lower().strip()
        if trigger_type not in {"take_profit", "stop_loss", "price_above", "price_below"}:
            logger.info("human.resolver.llm.conditional_order.reject reason=invalid_trigger_type note=%s", _short_note(note))
            return []
        targets = _validate_targets(resolved.get("target_tickers"), sellable_targets)
        if not targets:
            logger.info("human.resolver.llm.conditional_order.reject reason=no_targets note=%s", _short_note(note))
            return []
        trigger_price = _safe_float(conditional_order.get("trigger_price"))
        trigger_fraction = _safe_fraction(conditional_order.get("trigger_fraction"))
        logger.info(
            "human.resolver.llm.conditional_order trigger=%s targets=%s price=%s fraction=%s note=%s",
            trigger_type,
            ",".join(t["ticker"] for t in targets),
            trigger_price,
            trigger_fraction,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                target["ticker"],
                "human_conditional_order",
                resolver_intent_type="conditional_order",
                resolver_confidence=confidence,
                resolver_rationale=clean_text(target.get("rationale") or rationale_global or "", max_chars=500) or None,
                resolver_topic=topic,
                conditional_order={
                    "trigger_type": trigger_type,
                    "trigger_price": trigger_price,
                    "trigger_fraction": trigger_fraction,
                },
            )
            for target in targets
        ]

    if intent_type == "scheduled_action":
        scheduled_trigger = resolved.get("scheduled_trigger")
        if not isinstance(scheduled_trigger, dict):
            logger.info("human.resolver.llm.scheduled_action.reject reason=no_payload note=%s", _short_note(note))
            return []
        trigger_type = str(scheduled_trigger.get("trigger_type") or "").lower().strip()
        if trigger_type not in {"datetime", "market_open", "market_close", "day_of_week"}:
            logger.info("human.resolver.llm.scheduled_action.reject reason=invalid_trigger_type note=%s", _short_note(note))
            return []
        trigger_value = str(scheduled_trigger.get("trigger_value") or "").strip()
        if not trigger_value:
            logger.info("human.resolver.llm.scheduled_action.reject reason=no_trigger_value note=%s", _short_note(note))
            return []
        targets = _validate_targets(resolved.get("target_tickers"), sellable_targets) or _validate_forced_buy_targets(resolved.get("target_tickers"), buyable_universe, ticker_validator)
        wrapped_intent_type = str(scheduled_trigger.get("wrapped_intent_type") or "advisory").lower().strip()
        logger.info(
            "human.resolver.llm.scheduled_action trigger=%s/%s wrapped=%s targets=%s note=%s",
            trigger_type,
            trigger_value,
            wrapped_intent_type,
            ",".join(t["ticker"] for t in targets),
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                target["ticker"] if targets else None,
                "human_scheduled_action",
                resolver_intent_type="scheduled_action",
                resolver_confidence=confidence,
                resolver_rationale=rationale_global,
                resolver_topic=topic,
                scheduled_trigger={
                    "trigger_type": trigger_type,
                    "trigger_value": trigger_value,
                    "wrapped_intent_type": wrapped_intent_type,
                },
            )
            for target in (targets or [{"ticker": None}])
        ]

    if intent_type == "match_external":
        logger.info(
            "human.resolver.llm.match_external topic=%s confidence=%.2f note=%s",
            topic,
            confidence,
            _short_note(note),
        )
        return [
            HumanInstruction(
                note,
                None,
                "human_match_external",
                resolver_intent_type="match_external",
                resolver_confidence=confidence,
                resolver_rationale=rationale_global or "User wants to match an external portfolio.",
                resolver_topic=topic or "external_portfolio",
            )
        ]

    logger.info(
        "human.resolver.llm.advisory intent_type=%s confidence=%.2f topic=%s note=%s",
        intent_type,
        confidence,
        topic,
        _short_note(note),
    )
    return [
        HumanInstruction(
            note,
            None,
            "human_context",
            resolver_intent_type="advisory",
            resolver_confidence=confidence,
            resolver_rationale=rationale_global,
            resolver_topic=topic,
        )
    ]


def _enrich_positions_with_llm(positions: dict, llm_client: Any) -> dict:
    if not positions or not llm_client:
        return positions
    tickers = list(positions.keys())
    prompt = 'Return ONLY JSON. For each ticker provide a one-sentence business description focusing on energy consumption, supply chain, and macro exposure. Schema: {"TICKER": "description", ...}'
    try:
        raw = llm_client.complete_json(prompt, ", ".join(tickers))
        descriptions = json.loads(raw)
    except Exception as e:
        return positions

    return {symbol: {**data, "business": descriptions.get(symbol, "")} for symbol, data in positions.items()}


def _serialize_pending_instructions(instructions: list[HumanInstruction]) -> list[dict[str, Any]]:
    cancellable_types = {"forced_buy", "forced_sell", "broad_sell", "conditional_sell", "position_sweep"}
    serialized: list[dict[str, Any]] = []
    for instr in instructions:
        if instr.resolver_intent_type not in cancellable_types:
            continue
        serialized.append(
            {
                "instruction_id": instr.instruction_id,
                "intent_type": instr.resolver_intent_type,
                "target_ticker": instr.target_ticker,
                "note": clean_text(instr.note, max_chars=200),
                "retry_count": instr.retry_count,
                "state": "queued",
            }
        )
    return serialized


def _serialize_recent_executed_instructions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for entry in entries:
        instruction_id = entry.get("instruction_id")
        if not instruction_id:
            continue
        serialized.append(
            {
                "instruction_id": str(instruction_id),
                "intent_type": str(entry.get("intent_type") or entry.get("resolver_intent_type") or "unknown"),
                "target_ticker": str(entry.get("ticker") or entry.get("target_ticker") or ""),
                "executed_action": str(entry.get("action") or ""),
                "executed_quantity": entry.get("quantity") or 0,
                "outcome": str(entry.get("outcome") or ""),
                "failure_type": entry.get("failure_type"),
                "note": clean_text(str(entry.get("note") or entry.get("human_input_used") or ""), max_chars=200),
                "state": "executed",
            }
        )
    return serialized


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, (str, int)) and str(item).strip()]


def parse_cancel_resolver_output(raw: str) -> dict[str, Any] | None:
    try:
        parsed = _parse_resolver_json(raw)
    except Exception:
        return None
    if str(parsed.get("intent_type", "")).lower() != "cancel":
        return None
    return {
        "cancel_scope": str(parsed.get("cancel_scope") or "all").lower().strip(),
        "target_instruction_ids": _as_str_list(parsed.get("target_instruction_ids")),
        "target_description": clean_text(str(parsed.get("target_description") or ""), max_chars=240) or None,
        "reversal_action": str(parsed.get("reversal_action") or "NONE").upper().strip(),
        "confidence": _confidence(parsed.get("confidence")),
        "rationale": clean_text(str(parsed.get("rationale") or ""), max_chars=500) or None,
        "topic": clean_text(str(parsed.get("topic") or ""), max_chars=240) or None,
    }


def _validate_targets(raw_targets: Any, allowed_universe: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_targets, list):
        return []
    allowed = {symbol.upper() for symbol in allowed_universe}
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_targets:
        if isinstance(item, str):
            ticker = item.upper().strip()
            entry = {"ticker": ticker, "exposure": None, "rationale": None}
        elif isinstance(item, dict):
            ticker = str(item.get("ticker") or item.get("symbol") or "").upper().strip()
            entry = {
                "ticker": ticker,
                "exposure": clean_text(str(item.get("exposure") or ""), max_chars=40) or None,
                "rationale": clean_text(str(item.get("rationale") or ""), max_chars=500) or None,
            }
        else:
            continue
        if not ticker or not re.fullmatch(r"[A-Z]{1,5}", ticker):
            continue
        if allowed and ticker not in allowed:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        normalised.append(entry)
    return normalised


def _forced_action_instructions(
    note: str,
    intent_type: str,
    resolved: dict[str, Any],
    buyable_universe: list[str],
    sellable_targets: list[str],
    ticker_validator: Callable[[str], bool] | None,
    confidence: float,
    topic: str | None,
    rationale_global: str | None,
) -> list[HumanInstruction]:
    raw_targets = resolved.get("target_tickers")
    targets = _validate_forced_buy_targets(raw_targets, buyable_universe, ticker_validator) if intent_type == "forced_buy" else _validate_targets(raw_targets, sellable_targets)
    if not targets or confidence < 0.6:
        logger.info(
            "human.resolver.llm.reject reason=forced_action_missing_ticker_or_confidence intent_type=%s confidence=%.2f",
            intent_type,
            confidence,
        )
        if intent_type == "forced_buy" and confidence >= 0.6:
            requested = ", ".join(_target_ticker_codes(raw_targets)) or "requested ticker"
            return [
                HumanInstruction(
                    note,
                    None,
                    "human_context_requires_resolver",
                    resolver_intent_type="advisory",
                    resolver_confidence=confidence,
                    resolver_rationale=(f"{requested} could not be validated as a tradable buy candidate outside the initial buy universe."),
                    resolver_topic=topic,
                )
            ]
        return []
    action = "BUY" if intent_type == "forced_buy" else "SELL"
    requested_notional_usd = _safe_float(resolved.get("requested_notional_usd"))
    if requested_notional_usd is not None and requested_notional_usd <= 0:
        requested_notional_usd = None
    partial_fraction = _safe_fraction(resolved.get("partial_fraction"))
    if partial_fraction is not None and intent_type != "forced_sell":
        partial_fraction = None
    override_constraints = bool(resolved.get("override_constraints") or False)
    logger.info(
        "human.resolver.llm.forced action=%s targets=%s confidence=%.2f topic=%s notional=%s partial=%s override=%s",
        action,
        ",".join(t["ticker"] for t in targets),
        confidence,
        topic,
        requested_notional_usd,
        partial_fraction,
        override_constraints,
    )
    return [
        HumanInstruction(
            note,
            target["ticker"],
            "human_forced_action",
            resolver_intent_type=intent_type,
            resolver_confidence=confidence,
            resolver_rationale=clean_text(target.get("rationale") or rationale_global or "", max_chars=500) or None,
            resolver_topic=topic,
            requested_notional_usd=requested_notional_usd,
            partial_fraction=partial_fraction,
            override_constraints=override_constraints,
        )
        for target in targets
    ]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _safe_fraction(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed <= 0 or parsed > 1:
        return None
    return parsed


def _position_sweep_instructions(
    note: str,
    intent_type: str,
    targets: list[dict[str, Any]],
    confidence: float,
    rationale_global: str | None,
    topic: str | None,
    excluded_tickers: list[dict[str, Any]],
) -> list[HumanInstruction]:
    total = len(targets)
    logger.info(
        "human.resolver.llm.accept intent_type=%s targets=%s confidence=%.2f topic=%s",
        intent_type,
        ",".join(t["ticker"] for t in targets),
        confidence,
        topic,
    )
    return [
        HumanInstruction(
            note,
            target["ticker"],
            "human_llm_position_sweep",
            index,
            total,
            confidence,
            clean_text(target.get("rationale") or rationale_global or "", max_chars=500) or None,
            topic,
            intent_type,
            resolver_excluded_tickers=tuple(excluded_tickers),
        )
        for index, target in enumerate(targets, start=1)
    ]


def _repair_invalid_resolution(
    complete_json: Callable[[str, str], str],
    base_payload: dict[str, Any],
    previous_response: dict[str, Any],
    validation_errors: list[str],
    note: str,
) -> dict[str, Any] | None:
    repair_payload = {
        **base_payload,
        "previous_invalid_response": previous_response,
        "validation_errors": validation_errors,
        "instructions": (
            "Repair the previous JSON response. It failed validation and MUST be corrected. "
            "Use the same schema. Respect validation_errors exactly. "
            "If the human note is a buy request, do not return a sell-side intent. "
            "For broad thematic SELL notes, re-evaluate direct and indirect exposure for each open position. "
            "For oil and gas crisis topics, data-center energy sensitivity can be material for META, "
            "while AAPL should be excluded when exposure is too weak or generic. "
            "Return target_tickers only for positions with concrete per-ticker exposure rationales. "
            "Return ONLY corrected JSON."
        ),
    }
    try:
        logger.info(
            "human.resolver.llm.repair.start errors=%s note=%s",
            " | ".join(validation_errors),
            _short_note(note),
        )
        raw = complete_json(_RESOLVER_PROMPT, json.dumps(repair_payload, default=str))
        repaired = _parse_resolver_json(raw)
        logger.info(
            "human.resolver.llm.repair.raw intent_type=%s targets=%s confidence=%s topic=%s",
            repaired.get("intent_type"),
            ",".join(_target_ticker_codes(repaired.get("target_tickers"))),
            repaired.get("confidence"),
            clean_text(str(repaired.get("topic") or ""), max_chars=120),
        )
        return repaired
    except Exception as error:
        raw_preview = _raw_preview(locals().get("raw", ""))
        logger.warning("human.resolver.llm.repair.fail reason=%s raw_preview=%s note=%s", error, raw_preview, _short_note(note))
        return None


def _validate_forced_buy_targets(
    raw_targets: Any,
    buyable_universe: list[str],
    ticker_validator: Callable[[str], bool] | None,
) -> list[dict[str, Any]]:
    targets = _validate_targets(raw_targets, buyable_universe)
    seen = {target["ticker"] for target in targets}
    allowed = {symbol.upper() for symbol in buyable_universe}
    if not callable(ticker_validator) or not isinstance(raw_targets, list):
        return targets

    for item in raw_targets:
        if isinstance(item, str):
            ticker = item.upper().strip()
            rationale = None
        elif isinstance(item, dict):
            ticker = str(item.get("ticker") or item.get("symbol") or "").upper().strip()
            rationale = clean_text(str(item.get("rationale") or ""), max_chars=500) or None
        else:
            continue
        if not ticker or not re.fullmatch(r"[A-Z]{1,5}", ticker):
            continue
        if ticker in allowed or ticker in seen:
            continue
        if not ticker_validator(ticker):
            logger.info("human.resolver.ticker_validation.reject ticker=%s", ticker)
            continue
        logger.info("human.resolver.ticker_validation.accept ticker=%s", ticker)
        detail = "validated outside initial buy universe"
        targets.append(
            {
                "ticker": ticker,
                "exposure": None,
                "rationale": f"{rationale} ({detail})." if rationale else detail,
            }
        )
        seen.add(ticker)
    return targets


def _target_ticker_codes(raw_targets: Any) -> list[str]:
    if not isinstance(raw_targets, list):
        return []
    codes: list[str] = []
    for item in raw_targets:
        if isinstance(item, str):
            codes.append(item)
        elif isinstance(item, dict):
            code = item.get("ticker") or item.get("symbol")
            if isinstance(code, str):
                codes.append(code)
    return codes


def _open_position_targets(
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    *,
    fallback_ticker: str | None = None,
) -> list[str]:
    open_positions = portfolio_positions(portfolio)
    watchlist_symbols = _watchlist_symbols(watchlist)
    if watchlist_symbols:
        targets = [symbol for symbol in watchlist_symbols if open_positions.get(symbol, {}).get("qty", 0.0) > 0]
    else:
        targets = [symbol for symbol, position in open_positions.items() if position.get("qty", 0.0) > 0]
    return _deprioritize_fallback_ticker(targets, fallback_ticker)


def _watchlist_symbols(watchlist: list[str] | str | None) -> list[str]:
    if watchlist is None:
        return []
    raw_items = watchlist.split(",") if isinstance(watchlist, str) else list(watchlist)
    symbols: list[str] = []
    for item in raw_items:
        symbol = clean_text(str(item), max_chars=16).upper()
        if re.fullmatch(r"[A-Z]{1,5}", symbol) and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _deprioritize_fallback_ticker(targets: list[str], fallback_ticker: str | None) -> list[str]:
    fallback = (fallback_ticker or "").upper()
    if not fallback or len(targets) <= 1 or fallback not in targets:
        return targets
    return [target for target in targets if target != fallback] + [fallback]


def _short_note(note: str) -> str:
    return clean_text(note, max_chars=120)


_LLM_RESOLUTION_CUES = (
    "invest $",
    "invest $",
    "spend $",
    "buy $",
    "sell $",
    "half",
    "half of",
    "half my",
    "trim",
    "a quarter",
    "25%",
    "50%",
    "75%",
    "10%",
    "20%",
    "30%",
    "40%",
    "60%",
    "70%",
    "80%",
    "90%",
    "take profits",
    "take profit",
    "realize gains",
    "what's my",
    "what is my",
    "show me",
    "show my",
    "how much",
    "why did you",
    "why did the",
    "p&l",
    "pnl",
    "balance",
    "explain",
    "don't buy",
    "do not buy",
    "never buy",
    "no more",
    "stop buying",
    "exclude",
    "remove that",
    "ignore that rule",
    "esg",
    "fossil",
    "tobacco",
    "stop loss",
    "stop-loss",
    "if it hits",
    "if it reaches",
    "when it hits",
    "when it reaches",
    "at $",
    "tomorrow",
    "at open",
    "at close",
    "market open",
    "market close",
    "on monday",
    "on tuesday",
    "on wednesday",
    "on thursday",
    "on friday",
    "next week",
    "next month",
    "make me",
    "target allocation",
    "rebalance to",
    "60%",
    "40%",
    "30%",
    "tech and bonds",
    "match buffett",
    "match warren",
    "congress is buying",
    "congress buying",
    "match congress",
    "match ackman",
    "match wood",
    "match cathy",
    "match cathie",
    "match berkshire",
    "invest like",
    "follow buffett",
    "follow congress",
)


def _requires_llm_resolution(note: str, intent: Any) -> bool:
    if not note:
        return False
    text_lower = note.lower()
    for cue in _LLM_RESOLUTION_CUES:
        if cue in text_lower:
            return True
    if note.rstrip().endswith("?"):
        return True
    return False


_BRAND_KEYWORDS = (
    "spacex",
    "tesla",
    "apple",
    "boeing",
    "rocket lab",
    "rocketlab",
    "openai",
    "chatgpt",
    "stripe",
    "discord",
    "reddit",
    "instagram",
    "whatsapp",
    "youtube",
    "netflix",
    "disney",
    "nvidia",
    "amazon",
    "google",
    "microsoft",
    "meta ",
    "facebook",
    "bytedance",
    "tiktok",
    "epic games",
    "valve",
    "klarna",
    "plaid",
    "figma",
    "databricks",
)


def _maybe_fetch_web_search_context(
    note: str,
    buyable_universe: list[str],
    sellable_targets: list[str],
    llm_client: Any | None,
) -> list[dict[str, Any]]:
    if not note or llm_client is None:
        return []
    note_lower = note.lower()
    should_search = any(brand in note_lower for brand in _BRAND_KEYWORDS)
    if not should_search:
        ticker_tokens = re.findall(r"\b[A-Z]{2,5}\b", note)
        known = {symbol.upper() for symbol in buyable_universe} | {symbol.upper() for symbol in sellable_targets}
        external_tickers = [
            token for token in ticker_tokens if token.upper() not in known and token.upper() not in {"BUY", "SELL", "USA", "USD", "EUR", "GBP", "JPY", "ETC", "AND", "NOT", "FOR", "ALL", "NEW", "FAQ"}
        ]
        if not external_tickers:
            return []
        should_search = True
    try:
        from trading_agent.core.web_search import fetch_web_search_context

        results = fetch_web_search_context(note, llm_client, max_results=5)
    except Exception as error:
        logger.info("human.resolver.web_search.fail reason=%s note=%s", error, _short_note(note))
        return []
    if not results:
        return []
    logger.info(
        "human.resolver.web_search.ok results=%d note=%s",
        len(results),
        _short_note(note),
    )
    return results


_VAGUE_PHRASES = (
    "do something good",
    "do something",
    "surprise me",
    "help me make money",
    "make me money",
    "i don't know",
    "i dont know",
    "you decide",
    "do whatever",
    "do whatever you want",
    "anything you want",
    "your call",
    "your choice",
    "non lo so",
    "decidi tu",
    "quello che vuoi",
    "fai tu",
    "sorprendimi",
    "fai qualcosa di buono",
)


def _is_vague_delegative_note(note: str) -> bool:
    if not note:
        return False
    text = note.lower().strip()
    if len(text) > 200:
        return False
    if re.search(r"\b[A-Z]{1,5}\b", note):
        return False
    for phrase in _VAGUE_PHRASES:
        if phrase in text:
            return True
    return False


_SHORT_SELLING_CUES = (
    "short ",
    "shorting",
    "short-selling",
    "short sell",
    "bet against",
    "betting against",
    "put options on",
    "put option on",
    "buy puts on",
    "buy a put on",
    "shorta",
    "vendi allo scoperto",
    "allo scoperto",
)


def _detect_short_selling_intent(note: str) -> str | None:
    if not note:
        return None
    text_lower = note.lower()
    for cue in _SHORT_SELLING_CUES:
        if cue in text_lower:
            return cue.strip()
    return None


def _parse_resolver_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("resolver_response_must_be_object")
    return parsed


def _raw_preview(raw: Any) -> str:
    text = clean_text(str(raw or ""), max_chars=160)
    return text if text else "<empty>"


_RESOLVER_PROMPT = """
You are the Human Instruction Resolver for an autonomous trading agent.
INPUT: one human note, current open positions, optional watchlist.
OUTPUT: Return ONLY valid JSON. No prose, no markdown fences.

LOCAL BRAND ALIAS HINTS (apply only when no current relational claim is involved):
These stable product aliases are convenience hints, not evidence for ownership,
acquisitions, private-company exposure, or other time-sensitive relationships.
`web_search_context` always overrides these hints for relational requests.
  Facebook/WhatsApp/Instagram/Threads       -> META
  Google/YouTube/Android/Chrome/Gmail       -> GOOGL
  Amazon/AWS/Alexa/Prime/Kindle             -> AMZN
  Microsoft/Windows/Azure/LinkedIn/GitHub   -> MSFT
  Apple/iPhone/iPad/Mac/iOS/iCloud          -> AAPL
  Tesla/Model 3/Model Y                     -> TSLA
  Nvidia/GeForce/CUDA                       -> NVDA
  OpenAI/ChatGPT                            -> no direct public ticker; any proxy requires confirmation
  Netflix → NFLX | Disney → DIS
If two brands resolve to the same ticker, emit only one entry.

INTENT TYPES:
| Intent               | Trigger                              | target_tickers source                        |
|----------------------|--------------------------------------|----------------------------------------------|
| broad_sell           | Reduce exposure to a theme/sector    | sellable_open_position_tickers only           |
| conditional_sell     | "Sell X if Y happens"                | sellable_open_position_tickers only           |
| position_sweep       | Explicit "sell everything"           | all sellable positions (only if unambiguous)  |
| forced_buy           | Explicit BUY a named ticker          | watchlist; named outside ticker if explicit   |
| forced_sell          | Explicit SELL a named ticker         | sellable_open_position_tickers only           |
| news_request         | "Read/check news for company X"      | open positions or watchlist                   |
| cancel               | Undo a prior instruction             | IDs from pending/executed instructions        |
| rebalance_request    | "Rebalance/diversify portfolio"      | always empty                                  |
| information_request  | Question about portfolio, P&L, etc.  | always empty                                  |
| constraint_update    | Persistent rule ("don't buy fossil") | always empty                                  |
| conditional_order    | "Take profits"/"stop loss on X"      | sellable_open_position_tickers (or watchlist) |
| scheduled_action     | "Tomorrow"/"at open"/"on Friday"     | wrapped_intent target_tickers                 |
| match_external       | "Match Buffett"/"Congress is buying" | filled by agent from external_portfolio       |
| advisory             | Opinion, context, no trade action    | always empty                                  |

INTENT RULES:
- PRIORITY RULE: If the human note explicitly requests to buy or sell a specific brand, company, or ticker (e.g., "sell meta stocks", "vendi apple"), you MUST classify it as `forced_sell` or `forced_buy` targeting ONLY that specific ticker. NEVER generalize an explicit single-company request into a `broad_sell` or `position_sweep`.
- topic mandatory for: broad_sell, conditional_sell, position_sweep.
- position_sweep: if note is ambiguous, downgrade to broad_sell + topic.
- forced_buy: do not invent unnamed tickers. If the note mentions a company,
  industry, or event outside the watchlist, use only validated candidates and
  cited `web_search_context`; select direct exposure before proxies.
- forced_sell: if named ticker not in sellable_open_position_tickers →
  add to excluded_tickers with rationale "not held".
- advisory: confidence < 0.6, target_tickers empty.
- rebalance_request: confidence ≥ 0.7 (you are confident the user wants
  rebalancing, not a specific sell); target_tickers empty; the agent
  computes rebalance targets itself.
- For rebalance requests, preserve all explicitly requested sectors in
  `allocation_targets`. If percentages are omitted, assign equal fractions.
- Notes in Italian are parsed identically
  ("vendi le tech" = broad_sell, topic "tech").

NEW INTENT DETAILS:

information_request:
- User asks a question about portfolio, P&L, past decisions, or cash.
- Examples: "What's my P&L?", "Show me my portfolio", "Why did you buy X?",
  "How much cash do I have?".
- target_tickers MUST be empty. Set `question_type` to one of:
  "pnl" | "portfolio" | "decision_history" | "market_opinion" | "cash".
- The agent will produce a textual reply (no trade action).

constraint_update:
- User sets, removes, or overrides a persistent rule.
- Examples:
  "Don't buy fossil fuels" -> constraint_update with action="add",
    type="exclude_sector", value="fossil_fuels".
  "Never buy tobacco" -> action="add", type="exclude_sector", value="tobacco".
  "Don't buy XOM" -> action="add", type="exclude_ticker", value="XOM".
  "Remove that constraint" -> action="deactivate" (use constraint_id from
    context if available, else target_description explains which one).
  "Ignore that rule for now" -> action="override" (sets override_constraints=true
    on the wrapped intent; only valid when combined with another trade intent).
- target_tickers MUST be empty for add/deactivate. For override, the wrapped
  intent's target_tickers apply and override_constraints=true.

conditional_order:
- "Take profits on NVDA", "Stop loss on TSLA", "Sell AAPL if it hits $200".
- trigger_type ∈ {"take_profit", "stop_loss", "price_above", "price_below"}.
- trigger_price: float or null. null means use a default percentage
  (e.g. +20% for take_profit, -10% for stop_loss).
- trigger_fraction: optional float (0..1). For "sell half if it hits X" the
  fraction is 0.5. null means sell the entire position.
- target_tickers must reference sellable_open_position_tickers (for sells) or
  any buyable ticker (for trailing buy orders).

scheduled_action:
- "Buy AAPL tomorrow", "Sell META at market open", "Don't trade on Friday".
- trigger_type ∈ {"datetime", "market_open", "market_close", "day_of_week"}.
- trigger_value: ISO datetime string, or "open"/"close", or "monday".."sunday".
- wrapped_intent_type: the underlying intent (forced_buy, forced_sell, ...).
- The wrapped intent's target_tickers carry through to the eventual instruction.

match_external (Feature 10):
- "Match Warren Buffett's portfolio", "Buy what Congress is buying",
  "Invest like Cathy Wood".
- target_tickers MUST be empty. Set `topic` to the entity being matched
  ("warren_buffett", "congress", "cathy_wood", ...).
- The agent will fetch the external portfolio via NewsAPI and produce
  forced_buy instructions for the top tickers.

PARTIAL SELL ("sell half my AAPL", "trim 25% of NVDA", "take profits on TSLA"):
- Use `forced_sell` with `partial_fraction` set:
  "half" / "trim" without explicit % → partial_fraction = 0.5
  "trim 25%" → partial_fraction = 0.25
  "take profits" (no fraction specified) → partial_fraction = 0.5
- The agent computes requested_quantity = max(1, int(held_qty * partial_fraction)).
- If the user wants a conditional trigger instead ("take profits when +20%"),
  use `conditional_order` with trigger_type="take_profit".

USD NOTIONAL ("invest $50,000 on SpaceX", "buy €10k of AAPL" — € rejected pre-resolver):
- Use `forced_buy` with `requested_notional_usd` = the dollar amount (50000.0).
- The agent computes qty = floor(requested_notional_usd / current_price),
  capped by max_quantity from risk policy.
- Non-USD currencies (€, £, ¥, EUR, GBP, JPY, ...) are rejected BEFORE the
  resolver runs; you will never see them in `human_note`.

CANCEL RESOLUTION:
Match user's reference to pending_instructions and/or
recent_executed_instructions by: ticker name, action type, recency cue
("that one", "quello di prima"), or topic keyword.
If multiple match and user did not disambiguate → return ALL matching IDs.
If confidence in match < 0.6 → return advisory, topic "ambiguous_cancel".
Never invent instruction IDs.

cancel_scope     : "queued" | "executed" | "all"
reversal_action  : "BUY" or "SELL" (opposite of original) if outcome = "filled"
                   "NONE" if outcome = "failed" | "rejected" | "blocked"

TOPIC MATCHING & ANTI-SWEEP RULE:
For broad_sell / conditional_sell / position_sweep: only include positions
with genuine, material exposure to the topic (sector, industry, geography,
supply chain, or recent news).
  "tech selloff"       → NVDA, AAPL, MSFT, GOOGL, META — NOT KO, XOM
  "energy crash"       → XOM, CVX, COP — NOT AAPL, MSFT
  "semiconductor"      → NVDA, AMD, INTC, TSM — NOT JNJ, KO
  "oil and gas crisis" → only if direct/indirect exposure can be justified
                         (e.g. data-center energy costs for META); exclude
                         if exposure is too weak or generic.
If you cannot justify a ticker in one sentence → exclude it.
A smaller, well-justified list is always better than a broad sweep.

PER-TICKER RATIONALE (mandatory for all sell-side intents):
Every target_tickers entry must include a one-sentence rationale.
No rationale → ticker excluded. No exceptions.

CONFIDENCE:
confidence is your interpretation confidence in [0.0, 1.0].
If unsure whether the user meant to trade → return advisory, confidence < 0.6.

OUTPUT SCHEMA:
Cancel-only fields (target_instruction_ids, cancel_scope, reversal_action,
target_description) must be omitted for all non-cancel intents.

{
  "intent_type": "<see intent table>",
  "topic": "<impact theme or null>",
  "topic_scope": "sector|macro_event|company_specific|geopolitical|other|null",
  "target_tickers": [
    { 
      "ticker": "AAPL", 
      "exposure": "high|medium|low", 
      "classification": "direct|proxy|unknown",
      "rationale": (one-sentece explanation) 
    }
  ],
  "excluded_tickers": [
    { 
      "ticker": "KO", 
      "rationale": (one-sentece explanation) 
    }
  ],
  "target_instruction_ids": ["id1", "id2"],
  "cancel_scope": "queued|executed|all",
  "reversal_action": "BUY|SELL|NONE",
  "target_description": "<which instruction(s) the user meant>",
  "confidence": 0.0,
  "rationale": "<global interpretation summary>",
  "requires_validation": true,
  "requested_notional_usd": 50000.0,
  "requested_quantity": 1,
  "partial_fraction": 0.5,
  "question_type": "pnl|portfolio|decision_history|market_opinion|cash",
  "constraint_update": {
    "action": "add|deactivate|override",
    "type": "exclude_ticker|exclude_sector|max_price|esg_filter",
    "value": "fossil_fuels",
    "rationale": "User requested ESG exclusion.",
    "constraint_id": "optional-for-deactivate"
  },
  "conditional_order": {
    "trigger_type": "take_profit|stop_loss|price_above|price_below",
    "trigger_price": 200.0,
    "trigger_fraction": 0.5
  },
  "scheduled_trigger": {
    "trigger_type": "datetime|market_open|market_close|day_of_week",
    "trigger_value": "2026-06-20T14:30:00Z|open|close|friday",
    "wrapped_intent_type": "forced_buy|forced_sell|..."
  },
  "allocation_targets": [
    {"sector": "technology", "fraction": 0.6},
    {"sector": "bonds", "fraction": 0.4}
  ]
}
"""
