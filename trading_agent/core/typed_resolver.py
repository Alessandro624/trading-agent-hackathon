from __future__ import annotations

import json
import logging
from typing import Any, Callable

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.evidence import Evidence, make_inference
from trading_agent.core.human_intent import (
    detect_non_usd_currency,
    llm_split_compound_note,
    split_compound_note,
)
from trading_agent.core.instruction_ledger import (
    make_idempotency_key,
    make_request_id,
    parse_confirmation_reference,
)
from trading_agent.core.typed_outcomes import (
    AdvisoryReply,
    ConfirmedOperations,
    ConditionalOrderOutcome,
    ExecutableInstruction,
    InformationRequest,
    OutcomeRationale,
    PendingConfirmation,
    PersistentStateUpdate,
    ResolverOutcome,
    ScheduledActionOutcome,
)

logger = logging.getLogger("trading_agent.typed_resolver")


def resolve_human_input(
    notes: list[str],
    *,
    request_id: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None = None,
    pending_instructions: list | None = None,
    recent_executed_instructions: list[dict[str, Any]] | None = None,
    ticker_validator: Callable[[str], bool] | None = None,
    news_provider: Any | None = None,
    alpaca_validator: Callable[[str], bool] | None = None,
    ticker_provider: Any | None = None,
    max_discovered_tickers: int = 3,
    confirmation_store: Any | None = None,
) -> list[ResolverOutcome]:
    outcomes: list[ResolverOutcome] = []
    for note in notes:
        cleaned = clean_text(note, max_chars=2000)
        if not cleaned:
            continue
        if llm_client is not None:
            sub_notes = llm_split_compound_note(cleaned, llm_client)
        else:
            sub_notes = split_compound_note(cleaned)
        sub_notes = _coalesce_related_subnotes(sub_notes)
        seen: set[str] = set()
        unique_sub_notes: list[str] = []
        for sub in sub_notes:
            key = sub.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique_sub_notes.append(sub)
        for subnote_index, sub_note in enumerate(unique_sub_notes):
            outcome = _resolve_atomic_note(
                sub_note,
                request_id=request_id,
                subnote_index=subnote_index,
                watchlist=watchlist,
                portfolio=portfolio,
                fallback_ticker=fallback_ticker,
                llm_client=llm_client,
                pending_instructions=pending_instructions or [],
                recent_executed_instructions=recent_executed_instructions or [],
                ticker_validator=ticker_validator,
                news_provider=news_provider,
                alpaca_validator=alpaca_validator,
                ticker_provider=ticker_provider,
                max_discovered_tickers=max_discovered_tickers,
                confirmation_store=confirmation_store,
            )
            if outcome is not None:
                outcomes.append(outcome)
    return outcomes


def _coalesce_related_subnotes(sub_notes: list[str]) -> list[str]:
    result: list[str] = []
    for sub_note in sub_notes:
        text = clean_text(sub_note, max_chars=1000)
        if not text:
            continue
        lower = text.lower()
        related_continuation = lower.startswith(("associated ", "related ", "and associated ", "and related ", "buy associated ", "buy related "))
        is_rebalance = any(cue in lower for cue in ("balance", "rebalance", "diversif"))
        previous_rebalance = bool(result) and any(cue in result[-1].lower() for cue in ("balance", "rebalance", "diversif"))
        if result and (related_continuation or (is_rebalance and previous_rebalance)):
            result[-1] = f"{result[-1]}; {text}"
        else:
            result.append(text)
    return result


def _resolve_atomic_note(
    note: str,
    *,
    request_id: str,
    subnote_index: int,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None,
    pending_instructions: list,
    recent_executed_instructions: list[dict[str, Any]],
    ticker_validator: Callable[[str], bool] | None,
    news_provider: Any | None,
    alpaca_validator: Callable[[str], bool] | None,
    ticker_provider: Any | None,
    max_discovered_tickers: int,
    confirmation_store: Any | None,
) -> ResolverOutcome | None:
    non_usd = detect_non_usd_currency(note)
    if non_usd:
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Currency {non_usd} is not supported. Please specify amounts in USD.",
                next_step="Re-enter the request using a USD amount (e.g. 'invest $50,000').",
                evidence=(),
                technical_details={"advisory_type": "non_usd", "currency": non_usd},
            ),
            advisory_type="non_usd",
            reply=f"Currency {non_usd} is not supported. Please specify amounts in USD.",
        )

    if _detect_short_selling_intent(note):
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary="Short selling is not supported in this simulation.",
                next_step="Use a long position (buy/sell) instead, or rephrase to a bearish view without shorting.",
                evidence=(),
                technical_details={"advisory_type": "short_selling"},
            ),
            advisory_type="short_selling",
            reply="Short selling is not supported. Only long positions (buy/sell) are available in this simulation.",
        )

    if _is_vague_delegative_note(note):
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary="Your message was too vague for me to act on.",
                next_step="Specify a ticker, sector, or action. Example: 'Buy tech stocks' or 'Sell my AAPL position'.",
                evidence=(),
                technical_details={"advisory_type": "vague", "confidence": 0.2},
            ),
            advisory_type="vague",
            reply=("Your message was too vague for me to act on. " "Please specify a ticker, sector, or action. " "Example: 'Buy tech stocks' or 'Sell my AAPL position'."),
        )

    confirm_ref = parse_confirmation_reference(note)
    if confirm_ref and confirmation_store is not None:
        confirmation_outcome = _handle_confirmation_reference(
            note,
            confirm_ref,
            confirmation_store=confirmation_store,
            request_id=request_id,
            subnote_index=subnote_index,
        )
        modification_text = confirmation_outcome.rationale.technical_details.get("modification_text")
        if modification_text and isinstance(confirmation_outcome, AdvisoryReply):
            return _llm_resolve_atomic_note(
                modification_text,
                request_id=request_id,
                subnote_index=subnote_index,
                watchlist=watchlist,
                portfolio=portfolio,
                fallback_ticker=fallback_ticker,
                llm_client=llm_client,
                pending_instructions=pending_instructions,
                recent_executed_instructions=recent_executed_instructions,
                ticker_validator=ticker_validator,
                news_provider=news_provider,
                alpaca_validator=alpaca_validator,
                ticker_provider=ticker_provider,
                max_discovered_tickers=max_discovered_tickers,
            )
        return confirmation_outcome

    scheduled_cancel_id = _scheduled_action_cancel_id(note)
    if scheduled_cancel_id:
        return ExecutableInstruction(
            outcome_type="ExecutableInstruction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "cancel", scheduled_cancel_id),
            rationale=OutcomeRationale(
                user_summary=f"Cancellation requested for scheduled action {scheduled_cancel_id}.",
                next_step="The matching scheduled action will be cancelled if it is still pending.",
                technical_details={"scheduled_action_id": scheduled_cancel_id, "cancel_kind": "scheduled_action"},
            ),
            intent_type="cancel",
            confidence=1.0,
            cancel_target_ids=(scheduled_cancel_id,),
            cancel_scope="queued",
            cancel_reversal_action="NONE",
        )

    rebalance_targets = [instruction for instruction in pending_instructions if getattr(instruction, "workflow_type", None) == "portfolio_rebalance"]
    if rebalance_targets and _is_stop_rebalance_request(note):
        target_ids = tuple(str(getattr(instruction, "instruction_id", "")) for instruction in rebalance_targets if getattr(instruction, "instruction_id", None))
        return ExecutableInstruction(
            outcome_type="ExecutableInstruction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "cancel", None),
            rationale=OutcomeRationale(
                user_summary=f"Stopping the active portfolio rebalance; {len(target_ids)} queued operation(s) will be cancelled.",
                next_step="Completed rebalance trades remain unchanged.",
                technical_details={"workflow_ids": sorted({getattr(item, "workflow_id", None) for item in rebalance_targets if getattr(item, "workflow_id", None)})},
            ),
            intent_type="cancel",
            confidence=1.0,
            cancel_target_ids=target_ids,
            cancel_scope="queued",
            cancel_reversal_action="NONE",
        )

    return _llm_resolve_atomic_note(
        note,
        request_id=request_id,
        subnote_index=subnote_index,
        watchlist=watchlist,
        portfolio=portfolio,
        fallback_ticker=fallback_ticker,
        llm_client=llm_client,
        pending_instructions=pending_instructions,
        recent_executed_instructions=recent_executed_instructions,
        ticker_validator=ticker_validator,
        news_provider=news_provider,
        alpaca_validator=alpaca_validator,
        ticker_provider=ticker_provider,
        max_discovered_tickers=max_discovered_tickers,
    )


def _is_stop_rebalance_request(note: str) -> bool:
    text = (note or "").lower()
    stop_cues = ("stop", "cancel", "halt", "ferma", "annulla", "interrompi")
    rebalance_cues = ("balanc", "rebalance", "rebalancing", "diversif")
    return any(cue in text for cue in stop_cues) and any(cue in text for cue in rebalance_cues)


def _scheduled_action_cancel_id(note: str) -> str | None:
    import re

    text = note or ""
    lower = text.lower()
    if "scheduled" not in lower and "schedule" not in lower:
        return None
    if not any(cue in lower for cue in ("cancel", "stop", "annulla", "ferma")):
        return None
    match = re.search(r"\b([0-9a-fA-F]{8})\b", text)
    return match.group(1) if match else None


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
    import re

    if not note:
        return False
    text = note.lower().strip()
    if len(text) > 200:
        return False
    if re.search(r"\b[A-Z]{2,5}\b", note):
        return False
    for phrase in _VAGUE_PHRASES:
        if phrase in text:
            return True
    return False


def _handle_confirmation_reference(
    note: str,
    confirmation_id: str,
    *,
    confirmation_store: Any,
    request_id: str,
    subnote_index: int,
) -> ResolverOutcome:
    confirmation = confirmation_store.get(confirmation_id)
    if confirmation is None:
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Confirmation {confirmation_id} not found.",
                next_step="Check the Confirmations tab for pending confirmations.",
                evidence=(),
                technical_details={"advisory_type": "confirmation_not_found", "confirmation_id": confirmation_id},
            ),
            advisory_type="unsupported",
            reply=f"Confirmation {confirmation_id} not found.",
        )
    if confirmation.is_expired():
        confirmation_store.mark_status(confirmation_id, "expired", resolved_by_request_id=request_id)
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Confirmation {confirmation_id} has expired.",
                next_step="Re-enter the original request to generate a fresh proposal with current prices and constraints.",
                evidence=(),
                technical_details={"advisory_type": "confirmation_expired", "confirmation_id": confirmation_id},
            ),
            advisory_type="unsupported",
            reply=f"Confirmation {confirmation_id} has expired. Please re-enter the original request to regenerate a proposal.",
        )
    note_lower = note.lower()
    if note_lower.startswith("reject") or " reject " in note_lower:
        confirmation_store.mark_status(confirmation_id, "rejected", resolved_by_request_id=request_id)
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Confirmation {confirmation_id} rejected.",
                next_step="No trade will be placed. Enter a new request to start over.",
                evidence=(),
                technical_details={"advisory_type": "confirmation_rejected", "confirmation_id": confirmation_id},
            ),
            advisory_type="unsupported",
            reply=f"Confirmation {confirmation_id} rejected. No trade will be placed.",
        )
    modification_text = None
    if " but " in note_lower:
        idx = note_lower.find(" but ")
        modification_text = note[idx + 5 :].strip()
    if modification_text:
        proposed = list(confirmation.proposed_operations)
        cap_usd = _extract_usd_cap(modification_text)
        if cap_usd is not None and proposed:
            modified_operations = _apply_total_notional_cap(proposed, cap_usd)
            confirmation_store.mark_status(confirmation_id, "confirmed", resolved_by_request_id=request_id, modification_text=modification_text)
            return ConfirmedOperations(
                outcome_type="ConfirmedOperations",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "confirmed_operations", confirmation_id),
                rationale=OutcomeRationale(
                    user_summary=f"Confirmation {confirmation_id} accepted with a ${cap_usd:,.0f} total cap.",
                    next_step="The capped operations will be queued for execution.",
                    evidence=tuple(Evidence.from_dict(e) for e in (confirmation.evidence or [])),
                    technical_details={
                        "originating_confirmation_id": confirmation_id,
                        "modification_text": modification_text,
                        "total_notional_cap_usd": cap_usd,
                        "proposed_operations": modified_operations,
                    },
                ),
                confirmation_id=confirmation_id,
                operations=tuple(modified_operations),
            )
        confirmation_store.mark_status(confirmation_id, "modified", resolved_by_request_id=request_id, modification_text=modification_text)
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Confirmation {confirmation_id} modified. Re-resolving with: {modification_text!r}.",
                next_step="The agent will re-resolve your modification and produce a fresh proposal or instruction next cycle.",
                evidence=(),
                technical_details={
                    "advisory_type": "confirmation_modified",
                    "confirmation_id": confirmation_id,
                    "modification_text": modification_text,
                },
            ),
            advisory_type="unsupported",
            reply=f"Confirmation {confirmation_id} modified with: {modification_text!r}. Re-resolving in the next cycle.",
        )
    confirmation_store.mark_status(confirmation_id, "confirmed", resolved_by_request_id=request_id)
    proposed = list(confirmation.proposed_operations)
    if not proposed:
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"Confirmation {confirmation_id} had no proposed operations.",
                next_step="Enter a new request.",
                evidence=(),
                technical_details={"advisory_type": "confirmation_empty", "confirmation_id": confirmation_id},
            ),
            advisory_type="unsupported",
            reply=f"Confirmation {confirmation_id} had no proposed operations.",
        )
    return ConfirmedOperations(
        outcome_type="ConfirmedOperations",
        request_id=request_id,
        subnote_index=subnote_index,
        idempotency_key=make_idempotency_key(request_id, subnote_index, "confirmed_operations", confirmation_id),
        rationale=OutcomeRationale(
            user_summary=f"Confirmation {confirmation_id} accepted for {len(proposed)} operation(s).",
            next_step="The confirmed operations will be queued for execution.",
            evidence=tuple(Evidence.from_dict(e) for e in (confirmation.evidence or [])),
            technical_details={
                "originating_confirmation_id": confirmation_id,
                "proposed_operations": proposed,
            },
        ),
        confirmation_id=confirmation_id,
        operations=tuple(proposed),
    )


def _extract_usd_cap(text: str) -> float | None:
    import re

    lower = (text or "").lower()
    if not any(cue in lower for cue in ("cap", "limit", "maximum", "max", "limita", "massimo")):
        return None
    match = re.search(r"(?:\$|usd\s*)\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*([kKmM]))?", text)
    if not match:
        match = re.search(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|usd|dollars?)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value if value > 0 else None


def _apply_total_notional_cap(operations: list[dict[str, Any]], cap_usd: float) -> list[dict[str, Any]]:
    if not operations:
        return []
    buy_indexes = [
        index
        for index, operation in enumerate(operations)
        if str(operation.get("intent_type") or operation.get("side") or operation.get("action") or "").lower().replace("forced_", "") == "buy"
    ]
    target_indexes = buy_indexes or list(range(len(operations)))
    existing = []
    for index in target_indexes:
        existing.append(_coerce_float(operations[index].get("requested_notional_usd") or operations[index].get("notional"), 0.0, None, None) or 0.0)
    total_existing = sum(existing)
    capped = [dict(operation) for operation in operations]
    if total_existing > 0:
        for index, existing_notional in zip(target_indexes, existing):
            capped[index]["requested_notional_usd"] = round(min(existing_notional, cap_usd * (existing_notional / total_existing)), 2)
    else:
        per_operation = round(cap_usd / len(target_indexes), 2)
        for index in target_indexes:
            capped[index]["requested_notional_usd"] = per_operation
    return capped


def _llm_resolve_atomic_note(
    note: str,
    *,
    request_id: str,
    subnote_index: int,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None,
    pending_instructions: list,
    recent_executed_instructions: list[dict[str, Any]],
    ticker_validator: Callable[[str], bool] | None,
    news_provider: Any | None,
    alpaca_validator: Callable[[str], bool] | None,
    ticker_provider: Any | None,
    max_discovered_tickers: int,
) -> ResolverOutcome:
    from trading_agent.core.entity_discovery import discover_candidates
    from trading_agent.core.evidence import make_evidence

    complete_json = getattr(llm_client, "complete_json", None) if llm_client else None

    candidates: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    if _needs_external_discovery(note, watchlist, portfolio):
        discovery_query = _market_discovery_query(note)
        candidates, evidence = discover_candidates(
            discovery_query,
            news_provider=news_provider,
            tavily_api_key=None,
            alpaca_validator=alpaca_validator,
            ticker_provider=ticker_provider,
            max_discovered_tickers=max_discovered_tickers,
        )

    if not callable(complete_json):
        return _deterministic_outcome(
            note,
            request_id=request_id,
            subnote_index=subnote_index,
            watchlist=watchlist,
            portfolio=portfolio,
            fallback_ticker=fallback_ticker,
            candidates=candidates,
            evidence=evidence,
        )

    try:
        from trading_agent.core.human_instruction import _RESOLVER_PROMPT, _parse_resolver_json
        from trading_agent.core.portfolio import positions as portfolio_positions

        sellable_targets = _open_position_targets(watchlist, portfolio)
        buyable_universe = _watchlist_symbols(watchlist)
        payload = {
            "human_note": note,
            "watchlist": buyable_universe,
            "sellable_open_position_tickers": sellable_targets,
            "portfolio_positions": portfolio_positions(portfolio) or {},
            "pending_instructions": [
                {
                    "instruction_id": getattr(i, "instruction_id", ""),
                    "intent_type": getattr(i, "resolver_intent_type", ""),
                    "target_ticker": getattr(i, "target_ticker", None),
                    "note": (getattr(i, "note", "") or "")[:200],
                }
                for i in pending_instructions
            ],
            "recent_executed_instructions": recent_executed_instructions,
            "discovered_candidates": candidates,
            "web_search_context": [e.to_dict() for e in evidence],
            "instructions": (
                "Interpret the human note into the JSON schema defined by the system prompt. "
                "Use `discovered_candidates` and `web_search_context` to identify tickers when the note refers to an entity not in `watchlist`. "
                "For every externally discovered target add classification='direct' only when the cited evidence identifies it as the requested listed company; "
                "use classification='proxy' for indirect exposure and classification='unknown' when the relationship is uncertain. "
                "The LLM is NEVER allowed to invent web search results — only interpret the provided evidence. "
                "If the evidence is insufficient or conflicting, return intent_type='advisory' with a clarification rationale."
            ),
        }
        raw = complete_json(_RESOLVER_PROMPT, json.dumps(payload, default=str))
        resolved = raw if isinstance(raw, dict) else _parse_resolver_json(raw)
    except Exception as error:
        logger.warning("typed_resolver.llm.fail reason=%s note=%s", error, note[:80])
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary=f"I couldn't interpret your request due to a resolver error.",
                next_step="Please rephrase or try again next cycle.",
                evidence=tuple(evidence),
                technical_details={"advisory_type": "resolver_error", "error": str(error)[:200]},
            ),
            advisory_type="unsupported",
            reply=f"I couldn't interpret your request. Please rephrase.",
        )

    return _build_outcome_from_resolver_json(
        resolved,
        note=note,
        request_id=request_id,
        subnote_index=subnote_index,
        candidates=candidates,
        evidence=evidence,
        buyable_universe=buyable_universe if "buyable_universe" in dir() else [],
        sellable_targets=sellable_targets if "sellable_targets" in dir() else [],
        ticker_validator=ticker_validator,
        portfolio=portfolio,
        llm_client=llm_client,
        ticker_provider=ticker_provider,
    )


def _market_discovery_query(note: str) -> str:
    cleaned = clean_text(note, max_chars=500)
    return f"{cleaned} public companies stock tickers direct exposure listed shares tradable securities"


def _build_outcome_from_resolver_json(
    resolved: dict[str, Any],
    *,
    note: str,
    request_id: str,
    subnote_index: int,
    candidates: list[dict[str, Any]],
    evidence: list[Evidence],
    buyable_universe: list[str],
    sellable_targets: list[str],
    ticker_validator: Callable[[str], bool] | None,
    portfolio: dict[str, Any] | None = None,
    llm_client: Any | None = None,
    ticker_provider: Any | None = None,
) -> ResolverOutcome:
    intent_type = str(resolved.get("intent_type", "")).lower().strip()
    confidence = _coerce_float(resolved.get("confidence"), 0.0, 1.0, 0.0)
    topic = (resolved.get("topic") or None) and str(resolved.get("topic"))
    rationale_global = str(resolved.get("rationale") or "")[:500] or None
    target_ticker_codes = _target_ticker_codes(resolved.get("target_tickers"))

    user_summary = rationale_global or f"Resolved intent: {intent_type}."
    next_step = _next_step_for_intent(intent_type)
    technical_details = {
        "confidence": confidence,
        "intent_type": intent_type,
        "topic": topic,
        "discovered_candidates": candidates,
        "resolver_raw": {k: v for k, v in resolved.items() if k != "rationale"},
    }
    rationale = OutcomeRationale(
        user_summary=user_summary,
        next_step=next_step,
        evidence=tuple(evidence),
        technical_details=technical_details,
    )

    if intent_type == "advisory":
        tradable_candidates = [c for c in candidates if c.get("alpaca_tradable") is True]
        if not tradable_candidates and ticker_provider is not None:
            search_fn = getattr(ticker_provider, "search_assets_by_name", None)
            if callable(search_fn):
                search_query = resolved.get("topic") or note
                try:
                    name_matches = search_fn(search_query, max_results=5)
                    if name_matches:
                        logger.info(
                            "typed_resolver.name_search_rescue matches=%s note=%s",
                            ",".join(m["ticker"] for m in name_matches),
                            note[:80],
                        )
                        tradable_candidates = [
                            {"ticker": m["ticker"], "alpaca_tradable": True, "confidence": 0.6, "relationship": "name_match", "classification": "proxy", "name": m.get("name", "")}
                            for m in name_matches
                        ]
                except Exception as err:
                    logger.info("typed_resolver.name_search_rescue.fail reason=%s", str(err)[:120])

        if tradable_candidates:
            best = tradable_candidates[0]
            best_ticker = best["ticker"]
            best_name = best.get("name", best_ticker)
            logger.info(
                "typed_resolver.advisory_override ticker=%s reason=alpaca_validated_candidate_found note=%s",
                best_ticker,
                note[:80],
            )
            return PendingConfirmation(
                outcome_type="PendingConfirmation",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "pending_confirmation", best_ticker),
                rationale=OutcomeRationale(
                    user_summary=(
                        f"The entity in your request may not be directly listed, but I found "
                        f"{best_ticker} ({best_name}) as a related tradable instrument on Alpaca. "
                        f"Confirm if you'd like to proceed with {best_ticker}."
                    ),
                    next_step=f"Reply 'confirm CF-XXXX' to proceed with {best_ticker} or 'reject CF-XXXX' to cancel.",
                    evidence=tuple(evidence),
                    technical_details={
                        "advisory_type": "promoted_from_advisory",
                        "best_candidate": best,
                        "all_candidates": candidates,
                        "llm_rationale": rationale_global,
                    },
                ),
                proposal_text=f"Buy {best_ticker} ({best_name}) as a proxy for: {note}",
                proposed_operations=(
                    {
                        "intent_type": "forced_buy",
                        "ticker": best_ticker,
                        "requested_notional_usd": _coerce_float(resolved.get("requested_notional_usd"), 0.0, None, None),
                    },
                ),
                confirmation_id="",
                expires_at="",
                originating_request_id=request_id,
            )
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=rationale,
            advisory_type="unsupported",
            reply=user_summary,
        )

    if intent_type == "information_request":
        question_type = str(resolved.get("question_type") or "general").lower().strip()
        return InformationRequest(
            outcome_type="InformationRequest",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "information_request", None),
            rationale=rationale,
            question_type=question_type,
        )

    if intent_type == "constraint_update":
        cu = resolved.get("constraint_update") or {}
        action = str(cu.get("action") or "add").lower().strip()
        return PersistentStateUpdate(
            outcome_type="PersistentStateUpdate",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "constraint_update", None),
            rationale=rationale,
            action=action,
            constraint_type=str(cu.get("type") or "exclude_ticker"),
            value=str(cu.get("value") or ""),
            constraint_id=str(cu.get("constraint_id") or "") or None,
        )

    if intent_type == "rebalance_request":
        from trading_agent.core.rebalance import AllocationTarget, build_rebalance_plan

        allocation_targets = []
        sector_aliases = {"retail": "consumer", "goods retail": "consumer", "consumer goods": "consumer"}
        for item in resolved.get("allocation_targets") or []:
            if not isinstance(item, dict):
                continue
            sector = str(item.get("sector") or "").strip().lower()
            sector = sector_aliases.get(sector, sector)
            fraction = _coerce_float(item.get("fraction"), 0.0, 1.0, None)
            if sector and fraction:
                allocation_targets.append(AllocationTarget(sector=sector, target_fraction=fraction))
        plan = build_rebalance_plan(portfolio, llm_client, allocation_targets=allocation_targets or None)
        operations = []
        for ticker in plan.suggested_buys:
            if callable(ticker_validator) and not ticker_validator(ticker):
                continue
            operations.append(
                {
                    "intent_type": "forced_buy",
                    "ticker": ticker,
                    "workflow_type": "portfolio_rebalance",
                    "topic": "portfolio_rebalance",
                }
            )
        if not operations:
            return AdvisoryReply(
                outcome_type="AdvisoryReply",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
                rationale=OutcomeRationale(
                    user_summary="I reviewed the portfolio but found no validated rebalance trades to propose.",
                    next_step="You can specify sectors or target allocations if you want a narrower rebalance.",
                    evidence=tuple(evidence),
                    technical_details={"advisory_type": "rebalance_no_operations", "plan": plan.rationale},
                ),
                advisory_type="rebalance_no_operations",
                reply="No validated rebalance trades were found.",
            )
        return PendingConfirmation(
            outcome_type="PendingConfirmation",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "pending_rebalance", None),
            rationale=OutcomeRationale(
                user_summary=f"I prepared a rebalance proposal with {len(operations)} purchase(s).",
                next_step="Review the proposal and confirm or reject it.",
                evidence=tuple(evidence),
                technical_details={"rebalance_plan": plan.rationale, "operations": operations},
            ),
            proposal_text="Rebalance proposal: " + ", ".join(f"buy {op['ticker']}" for op in operations),
            proposed_operations=tuple(operations),
            confirmation_id="",
            expires_at="",
            originating_request_id=request_id,
        )

    if intent_type == "conditional_order":
        co = resolved.get("conditional_order") or {}
        trigger_type = str(co.get("trigger_type") or "").lower().strip()
        if not trigger_type:
            return AdvisoryReply(
                outcome_type="AdvisoryReply",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
                rationale=OutcomeRationale(
                    user_summary="Conditional order requested but no trigger type was provided.",
                    next_step="Specify a trigger: 'take profits at $200', 'stop loss at $150'.",
                    evidence=tuple(evidence),
                    technical_details={"advisory_type": "conditional_order_missing_trigger"},
                ),
                advisory_type="unsupported",
                reply="Conditional order missing trigger_type.",
            )
        if not target_ticker_codes:
            return AdvisoryReply(
                outcome_type="AdvisoryReply",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
                rationale=OutcomeRationale(
                    user_summary="Conditional order requested but no ticker was identified.",
                    next_step="Specify a ticker: 'take profits on NVDA at $200'.",
                    evidence=tuple(evidence),
                    technical_details={"advisory_type": "conditional_order_missing_ticker"},
                ),
                advisory_type="unsupported",
                reply="Conditional order missing target ticker.",
            )
        is_immediate = bool(co.get("is_immediate") or False)
        if is_immediate:
            return ExecutableInstruction(
                outcome_type="ExecutableInstruction",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "forced_sell", target_ticker_codes[0]),
                rationale=rationale,
                intent_type="forced_sell",
                target_ticker=target_ticker_codes[0],
                topic=topic,
                confidence=confidence,
                partial_fraction=_coerce_float(co.get("trigger_fraction"), 0.0, 1.0, None),
                discovered_candidates=tuple(candidates),
            )
        return ConditionalOrderOutcome(
            outcome_type="ConditionalOrder",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "conditional_order", target_ticker_codes[0]),
            rationale=rationale,
            target_ticker=target_ticker_codes[0],
            trigger_type=trigger_type,
            trigger_price=_coerce_float(co.get("trigger_price"), None, None, None),
            trigger_fraction=_coerce_float(co.get("trigger_fraction"), 0.0, 1.0, None),
        )

    if intent_type == "scheduled_action":
        st = resolved.get("scheduled_trigger") or {}
        trigger_type = str(st.get("trigger_type") or "").lower().strip()
        trigger_value = str(st.get("trigger_value") or "").strip()
        wrapped_intent_type = str(st.get("wrapped_intent_type") or "advisory").lower().strip()
        if not trigger_type or not trigger_value:
            return AdvisoryReply(
                outcome_type="AdvisoryReply",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
                rationale=OutcomeRationale(
                    user_summary="Scheduled action requested but trigger is incomplete.",
                    next_step="Specify when to fire: 'tomorrow', 'at open', 'on Friday'.",
                    evidence=tuple(evidence),
                    technical_details={"advisory_type": "scheduled_action_missing_trigger"},
                ),
                advisory_type="unsupported",
                reply="Scheduled action missing trigger_type or trigger_value.",
            )
        return ScheduledActionOutcome(
            outcome_type="ScheduledAction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "scheduled_action", target_ticker_codes[0] if target_ticker_codes else None),
            rationale=rationale,
            wrapped_intent_type=wrapped_intent_type,
            target_ticker=target_ticker_codes[0] if target_ticker_codes else None,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            requested_notional_usd=_coerce_float(resolved.get("requested_notional_usd"), 0.0, None, None),
            requested_quantity=int(resolved["requested_quantity"]) if str(resolved.get("requested_quantity") or "").isdigit() else None,
            partial_fraction=_coerce_float(resolved.get("partial_fraction"), 0.0, 1.0, None),
            override_constraints=bool(resolved.get("override_constraints") or False),
        )

    if intent_type == "match_external":
        return AdvisoryReply(
            outcome_type="AdvisoryReply",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
            rationale=OutcomeRationale(
                user_summary="External portfolio matching is currently unsupported.",
                next_step="This feature is pending acceptance tests. Please specify tickers directly.",
                evidence=tuple(evidence),
                technical_details={"advisory_type": "external_portfolio_disabled"},
            ),
            advisory_type="external_portfolio_disabled",
            reply=("External portfolio matching is currently unsupported. " "Please specify tickers directly (e.g. 'Buy AAPL')."),
        )

    if intent_type in {"forced_buy", "forced_sell", "broad_sell", "conditional_sell", "position_sweep", "news_request"}:
        if not target_ticker_codes:
            return AdvisoryReply(
                outcome_type="AdvisoryReply",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
                rationale=OutcomeRationale(
                    user_summary=f"{intent_type} requested but no ticker was identified.",
                    next_step="Specify a ticker or rephrase.",
                    evidence=tuple(evidence),
                    technical_details={"advisory_type": "no_ticker"},
                ),
                advisory_type="unsupported",
                reply=f"{intent_type} requested but no ticker was identified.",
            )
        first_ticker = target_ticker_codes[0]
        requested_quantity = _coerce_float(resolved.get("requested_quantity"), 0.0, None, None)
        if intent_type == "forced_buy" and requested_quantity is not None and not requested_quantity.is_integer():
            rounded_quantity = max(1, int(requested_quantity))
            return PendingConfirmation(
                outcome_type="PendingConfirmation",
                request_id=request_id,
                subnote_index=subnote_index,
                idempotency_key=make_idempotency_key(request_id, subnote_index, "fractional_rounding", first_ticker),
                rationale=OutcomeRationale(
                    user_summary=(f"Fractional shares are not supported. I can round {requested_quantity:g} shares of " f"{first_ticker} to {rounded_quantity} whole share(s)."),
                    next_step="Confirm to use the proposed whole-share quantity, or reject to cancel.",
                    evidence=tuple(evidence),
                    technical_details={"requested_quantity": requested_quantity, "rounded_quantity": rounded_quantity},
                ),
                proposal_text=f"Buy {rounded_quantity} whole share(s) of {first_ticker} instead of {requested_quantity:g} fractional shares.",
                proposed_operations=({"intent_type": "forced_buy", "ticker": first_ticker, "requested_quantity": rounded_quantity},),
                confirmation_id="",
                expires_at="",
                originating_request_id=request_id,
            )
        candidate = next((c for c in candidates if c.get("ticker") == first_ticker), None)
        if candidate is not None:
            raw_target = next(
                (item for item in (resolved.get("target_tickers") or []) if isinstance(item, dict) and str(item.get("ticker") or item.get("symbol") or "").upper().strip() == first_ticker),
                {},
            )
            classification = str(raw_target.get("classification") or candidate.get("classification") or "unknown").lower()
            is_tradable = candidate.get("alpaca_tradable")
            if is_tradable is False or (callable(ticker_validator) and not ticker_validator(first_ticker)):
                return AdvisoryReply(
                    outcome_type="AdvisoryReply",
                    request_id=request_id,
                    subnote_index=subnote_index,
                    idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", first_ticker),
                    rationale=OutcomeRationale(
                        user_summary=f"I found {first_ticker}, but it is not available for trading through the connected broker.",
                        next_step="Choose another listed company or ask for alternatives.",
                        evidence=tuple(evidence),
                        technical_details={"advisory_type": "ticker_not_tradable", "candidate": candidate},
                    ),
                    advisory_type="ticker_not_tradable",
                    reply=f"{first_ticker} is not available for trading through the connected broker.",
                )
            if classification != "direct":
                # Tranche 1 — proxy purchases require confirmation.
                return PendingConfirmation(
                    outcome_type="PendingConfirmation",
                    request_id=request_id,
                    subnote_index=subnote_index,
                    idempotency_key=make_idempotency_key(request_id, subnote_index, "pending_confirmation", first_ticker),
                    rationale=OutcomeRationale(
                        user_summary=(f"{first_ticker} is an indirect or uncertain match for the requested entity. " "I need your confirmation before trading it."),
                        next_step=f"Reply 'confirm CF-XXXX' to proceed or 'reject CF-XXXX' to cancel.",
                        evidence=tuple(evidence),
                        technical_details={
                            "classification": classification,
                            "candidate": candidate,
                        },
                    ),
                    proposal_text=f"Buy {first_ticker} as a proxy for: {note}",
                    proposed_operations=(
                        {
                            "intent_type": intent_type,
                            "ticker": first_ticker,
                            "requested_notional_usd": _coerce_float(resolved.get("requested_notional_usd"), 0.0, None, None),
                            "partial_fraction": _coerce_float(resolved.get("partial_fraction"), 0.0, 1.0, None),
                        },
                    ),
                    confirmation_id="",
                    expires_at="",
                    originating_request_id=request_id,
                )
        return ExecutableInstruction(
            outcome_type="ExecutableInstruction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, intent_type, first_ticker),
            rationale=rationale,
            intent_type=intent_type,
            target_ticker=first_ticker,
            target_tickers=tuple(target_ticker_codes),
            topic=topic,
            confidence=confidence,
            requested_notional_usd=_coerce_float(resolved.get("requested_notional_usd"), 0.0, None, None),
            requested_quantity=int(requested_quantity) if requested_quantity is not None else None,
            partial_fraction=_coerce_float(resolved.get("partial_fraction"), 0.0, 1.0, None),
            override_constraints=bool(resolved.get("override_constraints") or False),
            excluded_tickers=tuple(_validate_targets(resolved.get("excluded_tickers"), sellable_targets)),
            discovered_candidates=tuple(candidates),
        )

    return AdvisoryReply(
        outcome_type="AdvisoryReply",
        request_id=request_id,
        subnote_index=subnote_index,
        idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
        rationale=rationale,
        advisory_type="unsupported",
        reply=user_summary,
    )


def _needs_external_discovery(note: str, watchlist: list[str] | str | None, portfolio: dict[str, Any] | None) -> bool:
    import re

    note_lower = (note or "").lower()
    if any(phrase in note_lower for phrase in ("company that acquired", "parent of", "owner of", "spinoff of")):
        return True
    from trading_agent.core.portfolio import positions as portfolio_positions

    buyable = _watchlist_symbols(watchlist)
    held = list((portfolio_positions(portfolio) or {}).keys())
    known = {s.upper() for s in buyable + held}
    ticker_tokens = re.findall(r"\b[A-Z]{2,5}\b", note or "")
    external = [t for t in ticker_tokens if t.upper() not in known and t.upper() not in {"BUY", "SELL", "USA", "USD"}]
    if external:
        return True
    trade_request = any(cue in note_lower for cue in ("buy", "invest", "acquista", "compra"))
    explicit_known = any(symbol.lower() in note_lower for symbol in known)
    return trade_request and not explicit_known


def _open_position_targets(watchlist: list[str] | str | None, portfolio: dict[str, Any] | None) -> list[str]:
    from trading_agent.core.portfolio import positions as portfolio_positions

    open_positions = portfolio_positions(portfolio) or {}
    watchlist_symbols = _watchlist_symbols(watchlist)
    if watchlist_symbols:
        return [s for s in watchlist_symbols if (open_positions.get(s) or {}).get("qty", 0.0) > 0]
    return [s for s, p in open_positions.items() if p.get("qty", 0.0) > 0]


def _watchlist_symbols(watchlist: list[str] | str | None) -> list[str]:
    from trading_agent.core.data_hygiene import clean_text
    from trading_agent.core.ticker_symbols import normalize_ticker

    if watchlist is None:
        return []
    raw_items = watchlist.split(",") if isinstance(watchlist, str) else list(watchlist)
    symbols: list[str] = []
    for item in raw_items:
        symbol = normalize_ticker(clean_text(str(item), max_chars=16))
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _target_ticker_codes(raw_targets: Any) -> list[str]:
    if not isinstance(raw_targets, list):
        return []
    codes: list[str] = []
    for item in raw_targets:
        if isinstance(item, str):
            codes.append(item.upper().strip())
        elif isinstance(item, dict):
            code = item.get("ticker") or item.get("symbol")
            if isinstance(code, str):
                codes.append(code.upper().strip())
    return [c for c in codes if c]


def _validate_targets(raw_targets: Any, allowed_universe: list[str]) -> list[dict[str, Any]]:
    from trading_agent.core.ticker_symbols import normalize_ticker

    if not isinstance(raw_targets, list):
        return []
    allowed = {symbol.upper() for symbol in allowed_universe}
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_targets:
        if isinstance(item, str):
            ticker = normalize_ticker(item) or ""
            entry = {"ticker": ticker, "exposure": None, "rationale": None}
        elif isinstance(item, dict):
            ticker = normalize_ticker(item.get("ticker") or item.get("symbol")) or ""
            entry = {
                "ticker": ticker,
                "exposure": str(item.get("exposure") or "")[:40] or None,
                "rationale": str(item.get("rationale") or "")[:500] or None,
            }
        else:
            continue
        if not ticker or ticker in seen:
            continue
        if allowed and ticker not in allowed:
            continue
        seen.add(ticker)
        normalised.append(entry)
    return normalised


def _coerce_float(value: Any, low: float | None, high: float | None, default: float | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    if low is not None and result < low:
        return default
    if high is not None and result > high:
        return default
    return result


def _next_step_for_intent(intent_type: str) -> str:
    return {
        "forced_buy": "Will execute the buy in the next cycle, subject to risk and constraint checks.",
        "forced_sell": "Will execute the sell in the next cycle, subject to risk and constraint checks.",
        "broad_sell": "Will execute the sells across affected positions in the next cycles.",
        "conditional_sell": "Will execute the conditional sell when the condition is met.",
        "position_sweep": "Will liquidate the matching positions across the next cycles.",
        "news_request": "Will fetch and summarize news for the requested ticker.",
        "information_request": "Will generate a textual reply grounded in portfolio and journal state.",
        "constraint_update": "Will update the persistent constraint store.",
        "conditional_order": "Will register the conditional order for re-evaluation every cycle.",
        "scheduled_action": "Will register the scheduled action; it fires when its trigger is met.",
        "advisory": "No trade will be placed. See the rationale for what to do next.",
    }.get(intent_type, "Will process in the next cycle.")


def _deterministic_outcome(
    note: str,
    *,
    request_id: str,
    subnote_index: int,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    candidates: list[dict[str, Any]],
    evidence: list[Evidence],
) -> ResolverOutcome:
    import re

    buyable = _watchlist_symbols(watchlist)
    held = _open_position_targets(watchlist, portfolio)
    ticker_tokens = re.findall(r"\b[A-Z]{2,5}\b", note)
    explicit = [t for t in ticker_tokens if t.upper() in {s.upper() for s in buyable + held} and t.upper() not in {"BUY", "SELL", "USA", "USD"}]
    note_lower = note.lower()
    if "buy" in note_lower and explicit:
        return ExecutableInstruction(
            outcome_type="ExecutableInstruction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "forced_buy", explicit[0]),
            rationale=OutcomeRationale(
                user_summary=f"Buy {explicit[0]} (deterministic fallback, no LLM).",
                next_step="Will execute the buy in the next cycle.",
                evidence=tuple(evidence),
                technical_details={"confidence": 0.6, "resolver_source": "deterministic"},
            ),
            intent_type="forced_buy",
            target_ticker=explicit[0],
            confidence=0.6,
            discovered_candidates=tuple(candidates),
        )
    if ("sell" in note_lower or "vendi" in note_lower) and explicit:
        return ExecutableInstruction(
            outcome_type="ExecutableInstruction",
            request_id=request_id,
            subnote_index=subnote_index,
            idempotency_key=make_idempotency_key(request_id, subnote_index, "forced_sell", explicit[0]),
            rationale=OutcomeRationale(
                user_summary=f"Sell {explicit[0]} (deterministic fallback, no LLM).",
                next_step="Will execute the sell in the next cycle.",
                evidence=tuple(evidence),
                technical_details={"confidence": 0.6, "resolver_source": "deterministic"},
            ),
            intent_type="forced_sell",
            target_ticker=explicit[0],
            confidence=0.6,
            discovered_candidates=tuple(candidates),
        )
    return AdvisoryReply(
        outcome_type="AdvisoryReply",
        request_id=request_id,
        subnote_index=subnote_index,
        idempotency_key=make_idempotency_key(request_id, subnote_index, "advisory", None),
        rationale=OutcomeRationale(
            user_summary="I couldn't interpret your request without an LLM. Please rephrase or enable an LLM provider.",
            next_step="Specify an explicit ticker and action (e.g. 'Buy AAPL').",
            evidence=tuple(evidence),
            technical_details={"advisory_type": "no_llm", "resolver_source": "deterministic"},
        ),
        advisory_type="unsupported",
        reply="I couldn't interpret your request. Please specify an explicit ticker and action.",
    )
