from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> None:
        return None


from trading_agent.adapters import AlpacaBrokerClient, AlpacaMarketDataProvider, NewsApiProvider
from trading_agent.core import (
    HumanRiskProfile,
    HumanInputStore,
    HumanInstruction,
    classify_failure,
    max_retries_for_failure_type,
    parse_watchlist,
    plan_human_instructions,
    select_ticker,
    should_retry,
    build_ticker_universe,
)
from trading_agent.core.human_risk import update_persistent_human_risk_profile
from trading_agent.core.execution_policy import ExecutionFailure
from trading_agent.journal import JournalStore, RunContext, create_run_context, print_cycle_log, render_dashboard
from trading_agent.pipeline import build_graph, run_cycle
from trading_agent.utils import FallbackLlmClient, OllamaJsonClient, OpenAiJsonClient, OpenRouterJsonClient, configure_logging, safe_portfolio_snapshot

logger = logging.getLogger("trading_agent.run_agent")

RECENT_EXECUTED_WINDOW = 10
PROVIDER_UNIVERSE_LIMIT = 25


def _build_adapters():
    import os

    primary_provider: str = os.getenv("PRIMARY_PROVIDER", "openai")
    llm_client: FallbackLlmClient

    if primary_provider == "openrouter":
        llm_client = FallbackLlmClient(primary=OpenRouterJsonClient(), fallback=OllamaJsonClient())
    else:
        llm_client = FallbackLlmClient(primary=OpenAiJsonClient(), fallback=OllamaJsonClient())

    market_data = AlpacaMarketDataProvider()
    news_provider = NewsApiProvider()
    broker = AlpacaBrokerClient()

    try:
        from trading_agent.adapters.ticker_provider import TickerProvider

        ticker_provider: Any | None = TickerProvider()
    except Exception as error:
        logger.warning("ticker_provider.unavailable reason=%s", error)
        ticker_provider = None

    return market_data, news_provider, llm_client, broker, ticker_provider


def _extract_recent_news(recent_entries: list, limit: int = 10) -> list[dict[str, Any]]:
    news: list[dict[str, Any]] = []
    for entry in reversed(recent_entries):
        snapshot = getattr(entry, "snapshot", None) or {}
        articles = getattr(snapshot, "news", None)
        if articles is None and isinstance(snapshot, dict):
            articles = snapshot.get("news") or []
        if not articles:
            continue
        for article in articles:
            if isinstance(article, dict):
                news.append(article)
        if len(news) >= limit:
            break
    return news[:limit]


def _extract_recent_executed_for_resolver(recent_entries: list, limit: int = RECENT_EXECUTED_WINDOW) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for entry in reversed(recent_entries[-limit:]):
        if hasattr(entry, "to_dict"):
            payload = entry.to_dict()
        elif isinstance(entry, dict):
            payload = entry
        else:
            continue
        instruction_id = payload.get("instruction_id")
        if not instruction_id:
            continue
        decision = payload.get("decision") or {}
        execution = payload.get("execution_result") or {}
        serialized.append(
            {
                "instruction_id": str(instruction_id),
                "intent_type": str(decision.get("resolver_intent_type") or "unknown"),
                "ticker": str(payload.get("ticker") or ""),
                "action": str(payload.get("action") or ""),
                "quantity": int(decision.get("quantity") or 0),
                "outcome": str(payload.get("outcome") or ""),
                "failure_type": payload.get("failure_type"),
                "note": " ".join(payload.get("human_input_used") or [])[:200],
                "reversal_of": payload.get("reversal_of"),
            }
        )
    return serialized


def _human_intent_override_for_instruction(instruction: HumanInstruction | None) -> dict[str, Any] | None:
    if instruction is None or not instruction.target_ticker:
        return None
    intent_type = instruction.resolver_intent_type
    if intent_type is None:
        return None

    ticker = instruction.target_ticker.upper()
    topic = instruction.resolver_topic

    if intent_type in {"forced_buy", "forced_sell"}:
        action = "BUY" if intent_type == "forced_buy" else "SELL"
        return {
            "intents": ["buy" if action == "BUY" else "sell"],
            "tickers": [ticker],
            "requested_action": action,
            "risk_preference": None,
            "impact_topic": topic,
            "summary": (f"LLM-resolved forced {action} for {ticker}" + (f" due to topic '{topic}'" if topic else "") + f" (resolver_confidence={instruction.resolver_confidence})"),
        }

    if intent_type in {"broad_sell", "conditional_sell", "position_sweep"}:
        return {
            "intents": ["sell", "position_sweep"],
            "tickers": [ticker],
            "requested_action": "SELL",
            "risk_preference": None,
            "impact_topic": topic,
            "summary": (f"LLM-resolved {intent_type} for {ticker}" + (f" due to topic '{topic}'" if topic else "") + f" (resolver_confidence={instruction.resolver_confidence})"),
        }

    return None


def _handle_cancellation(
    cancel_instruction: HumanInstruction,
    instruction_queue: list[HumanInstruction],
    recent_entries: list,
    journal: JournalStore,
) -> list[HumanInstruction]:
    target_ids = list(cancel_instruction.cancel_target_ids)
    scope = cancel_instruction.cancel_scope or "all"
    if scope not in {"queued", "executed", "all"}:
        scope = "all"
    reversal_action_override = (cancel_instruction.cancel_reversal_action or "NONE").upper()
    cancel_id = cancel_instruction.instruction_id
    reversals: list[HumanInstruction] = []

    for target_id in target_ids:
        handled = False

        if scope in {"queued", "all"}:
            removed = _remove_from_queue(instruction_queue, target_id)
            if removed is not None:
                handled = True
                was_in_retry = removed.retry_count > 0
                logger.info(
                    "cancel.queued.removed target=%s was_in_retry=%s retry_count=%s cancel_id=%s",
                    target_id,
                    was_in_retry,
                    removed.retry_count,
                    cancel_id,
                )
                journal.append_stage(
                    ticker=removed.target_ticker or "UNKNOWN",
                    stage="instruction_cancelled",
                    status="completed",
                    message=(
                        f"Instruction {target_id} cancelled by {cancel_id} (was queued, "
                        f"was_in_retry={was_in_retry}, retry_count={removed.retry_count}). "
                        f"Original intent: {removed.resolver_intent_type} on {removed.target_ticker}."
                    ),
                    details={
                        "cancelled_instruction_id": target_id,
                        "cancel_instruction_id": cancel_id,
                        "scope": "queued",
                        "was_in_retry": was_in_retry,
                        "original_retry_count": removed.retry_count,
                        "original_intent_type": removed.resolver_intent_type,
                        "original_target_ticker": removed.target_ticker,
                        "original_note": removed.note[:200],
                    },
                )

        if scope in {"executed", "all"}:
            original_entry = _find_entry_by_instruction_id(recent_entries, target_id)
            if original_entry is not None:
                handled = True
                outcome = str(original_entry.get("outcome") or "").lower()
                original_ticker = str(original_entry.get("ticker") or "UNKNOWN")
                original_action = str(original_entry.get("action") or "").upper()
                original_failure_type = original_entry.get("failure_type")

                if outcome == "filled":
                    reversal_action = _resolve_reversal_action(original_action, reversal_action_override, cancel_instruction)
                    if reversal_action == "NONE":
                        logger.info(
                            "cancel.executed.no_reversal target=%s outcome=filled reason=user_requested_none cancel_id=%s",
                            target_id,
                            cancel_id,
                        )
                        journal.append_stage(
                            ticker=original_ticker,
                            stage="instruction_cancelled",
                            status="completed",
                            message=(f"Instruction {target_id} cancelled by {cancel_id} (was executed, filled). " f"No reversal requested. Original: {original_action} {original_ticker}."),
                            details={
                                "cancelled_instruction_id": target_id,
                                "cancel_instruction_id": cancel_id,
                                "scope": "executed",
                                "original_outcome": outcome,
                                "reversal_action": "NONE",
                            },
                        )
                    else:
                        reversal_intent_type = "forced_buy" if reversal_action == "BUY" else "forced_sell"
                        reversal = HumanInstruction(
                            note=f"Reversal of {target_id}: cancel_id={cancel_id} requested {reversal_action} {original_ticker}",
                            target_ticker=original_ticker,
                            reason="human_reversal",
                            resolver_intent_type=reversal_intent_type,
                            resolver_confidence=cancel_instruction.resolver_confidence,
                            resolver_rationale=f"Auto-generated reversal: user cancelled instruction {target_id} which had executed.",
                            reversal_of=target_id,
                        )
                        reversals.append(reversal)
                        journal.add_marker(target_id, "cancelled", cancel_id)
                        logger.info(
                            "cancel.executed.reversed target=%s outcome=filled reversal_action=%s reversal_id=%s cancel_id=%s",
                            target_id,
                            reversal_action,
                            reversal.instruction_id,
                            cancel_id,
                        )
                        journal.append_stage(
                            ticker=original_ticker,
                            stage="instruction_reversed",
                            status="completed",
                            message=(f"Instruction {target_id} cancelled by {cancel_id} (was executed, filled). " f"Reversal {reversal.instruction_id} enqueued: {reversal_action} {original_ticker}."),
                            details={
                                "cancelled_instruction_id": target_id,
                                "cancel_instruction_id": cancel_id,
                                "scope": "executed",
                                "original_outcome": outcome,
                                "reversal_action": reversal_action,
                                "reversal_instruction_id": reversal.instruction_id,
                                "original_action": original_action,
                                "original_ticker": original_ticker,
                            },
                        )
                elif outcome in {"failed", "rejected", "blocked"}:
                    logger.info(
                        "cancel.executed.no_effect target=%s outcome=%s failure_type=%s cancel_id=%s",
                        target_id,
                        outcome,
                        original_failure_type,
                        cancel_id,
                    )
                    journal.append_stage(
                        ticker=original_ticker,
                        stage="instruction_cancelled",
                        status="completed",
                        message=(
                            f"Instruction {target_id} cancelled by {cancel_id} but had no effect: "
                            f"original outcome was '{outcome}' (failure_type={original_failure_type}). "
                            f"No reversal generated because no trade was executed."
                        ),
                        details={
                            "cancelled_instruction_id": target_id,
                            "cancel_instruction_id": cancel_id,
                            "scope": "executed",
                            "original_outcome": outcome,
                            "original_failure_type": original_failure_type,
                            "reversal_action": "NONE",
                            "reason": "original_action_never_executed",
                        },
                    )
                else:
                    logger.warning(
                        "cancel.executed.pending target=%s outcome=%s cancel_id=%s",
                        target_id,
                        outcome,
                        cancel_id,
                    )
                    journal.append_stage(
                        ticker=original_ticker,
                        stage="instruction_cancelled",
                        status="completed",
                        message=(f"Instruction {target_id} cancelled by {cancel_id} but original outcome was " f"'{outcome}' (not filled). No reversal generated; original may still be pending."),
                        details={
                            "cancelled_instruction_id": target_id,
                            "cancel_instruction_id": cancel_id,
                            "scope": "executed",
                            "original_outcome": outcome,
                            "reversal_action": "NONE",
                            "reason": "original_action_pending_or_skipped",
                        },
                    )

        if not handled:
            logger.warning(
                "cancel.target_not_found target=%s cancel_id=%s scope=%s",
                target_id,
                cancel_id,
                scope,
            )
            journal.append_stage(
                ticker="UNKNOWN",
                stage="instruction_cancelled",
                status="completed",
                message=(f"Cancel {cancel_id} targeted instruction {target_id} but it was not found " f"in the queue or in the last {RECENT_EXECUTED_WINDOW} executed entries."),
                details={
                    "cancelled_instruction_id": target_id,
                    "cancel_instruction_id": cancel_id,
                    "scope": scope,
                    "reason": "target_not_found",
                },
            )

    return reversals


def _remove_from_queue(queue: list[HumanInstruction], target_id: str) -> HumanInstruction | None:
    for index, instr in enumerate(queue):
        if instr.instruction_id == target_id:
            return queue.pop(index)
    return None


def _find_entry_by_instruction_id(recent_entries: list, target_id: str) -> dict[str, Any] | None:
    for entry in reversed(recent_entries[-RECENT_EXECUTED_WINDOW:]):
        if hasattr(entry, "to_dict"):
            payload = entry.to_dict()
        elif isinstance(entry, dict):
            payload = entry
        else:
            continue
        if payload.get("instruction_id") == target_id:
            return payload
    return None


def _resolve_reversal_action(
    original_action: str,
    override: str,
    cancel_instruction: HumanInstruction,
) -> str:
    if override in {"BUY", "SELL"}:
        return override
    if override == "NONE":
        return "NONE"
    if original_action == "BUY":
        return "SELL"
    if original_action == "SELL":
        return "BUY"
    return "NONE"


def _handle_execution_failure(
    executed_instruction: HumanInstruction | None,
    journal_entry: Any,
    instruction_queue: list[HumanInstruction],
    broker: Any,
    journal: JournalStore,
) -> None:
    if executed_instruction is None or journal_entry is None:
        return

    if hasattr(journal_entry, "to_dict"):
        payload = journal_entry.to_dict()
    elif isinstance(journal_entry, dict):
        payload = journal_entry
    else:
        return

    outcome = str(payload.get("outcome") or "").lower()
    if outcome not in {"failed", "rejected", "blocked"}:
        return

    failure_type = payload.get("failure_type") or "broker_unknown"
    execution = payload.get("execution_result") or {}
    error_message = str(execution.get("message") or payload.get("rationale") or "")
    ticker = str(payload.get("ticker") or executed_instruction.target_ticker or "UNKNOWN")

    market_is_open = _get_market_is_open(broker)
    failure = classify_failure(
        instruction_id=executed_instruction.instruction_id,
        outcome=outcome,
        error_message=error_message,
        execution_status=outcome,
        risk_can_trade=None,
        market_is_open=market_is_open,
    )
    if failure_type and failure_type != "broker_unknown":
        failure = ExecutionFailure(
            instruction_id=failure.instruction_id,
            failure_type=failure_type,
            error_message=failure.error_message,
            retryable=_is_retryable_type(failure_type),
            suggested_retry_delay_seconds=failure.suggested_retry_delay_seconds,
            rationale=failure.rationale,
        )

    retry_count = executed_instruction.retry_count
    max_retries = executed_instruction.max_retries
    if max_retries is None:
        max_retries = max_retries_for_failure_type(failure.failure_type)

    if not should_retry(failure, retry_count):
        logger.info(
            "failure.abandon instruction_id=%s ticker=%s failure_type=%s retry_count=%s max=%s reason=%s",
            executed_instruction.instruction_id,
            ticker,
            failure.failure_type,
            retry_count,
            max_retries,
            failure.rationale,
        )
        journal.append_stage(
            ticker=ticker,
            stage="instruction_abandoned",
            status="completed",
            message=(f"Instruction {executed_instruction.instruction_id} abandoned: {failure.rationale} " f"(failure_type={failure.failure_type}, retries={retry_count}/{max_retries})."),
            details={
                "instruction_id": executed_instruction.instruction_id,
                "failure_type": failure.failure_type,
                "error_message": error_message[:500],
                "retry_count": retry_count,
                "max_retries": max_retries,
                "outcome": outcome,
                "rationale": failure.rationale,
            },
        )
        return

    retried = replace(
        executed_instruction,
        retry_count=retry_count + 1,
    )
    instruction_queue.insert(0, retried)
    logger.info(
        "failure.retry_scheduled instruction_id=%s ticker=%s failure_type=%s retry=%s/%s",
        executed_instruction.instruction_id,
        ticker,
        failure.failure_type,
        retry_count + 1,
        max_retries,
    )
    journal.append_stage(
        ticker=ticker,
        stage="instruction_retry_scheduled",
        status="completed",
        message=(f"Instruction {executed_instruction.instruction_id} re-queued for retry " f"({retry_count + 1}/{max_retries}) after {failure.failure_type}: {failure.rationale}"),
        details={
            "instruction_id": executed_instruction.instruction_id,
            "retried_instruction_id": retried.instruction_id,
            "failure_type": failure.failure_type,
            "error_message": error_message[:500],
            "retry_count": retry_count + 1,
            "max_retries": max_retries,
            "outcome": outcome,
            "rationale": failure.rationale,
        },
    )


def _is_retryable_type(failure_type: str) -> bool:
    return max_retries_for_failure_type(failure_type) > 0


def _get_market_is_open(broker: Any) -> bool | None:
    get_market_clock = getattr(broker, "get_market_clock", None)
    if not callable(get_market_clock):
        return None
    try:
        clock = get_market_clock()
        return clock.get("is_open")
    except Exception as error:
        logger.warning("market.clock.check.fail error=%s", error)
        return None


def _provider_symbols(ticker_provider: Any | None, configured_watchlist: list[str]) -> list[str]:
    if ticker_provider is None or configured_watchlist:
        return []
    get_tickers_with_info = getattr(ticker_provider, "get_tickers_with_info", None)
    if not callable(get_tickers_with_info):
        return []
    try:
        info_items = get_tickers_with_info()
    except Exception as error:
        logger.warning("ticker_provider.universe.fail reason=%s", error)
        return []
    symbols: list[str] = []
    for item in info_items or []:
        if isinstance(item, str):
            raw_symbol = item
        elif isinstance(item, dict):
            raw_symbol = item.get("symbol") or item.get("ticker") or item.get("name")
        else:
            raw_symbol = getattr(item, "symbol", None) or getattr(item, "ticker", None) or getattr(item, "name", None)
        symbol = str(raw_symbol or "").upper().strip()
        if _is_ticker_symbol(symbol) and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= PROVIDER_UNIVERSE_LIMIT:
            break
    return symbols


def _provider_ticker_validator(ticker_provider: Any | None):
    if ticker_provider is None:
        return None
    get_tickers_with_info = getattr(ticker_provider, "get_tickers_with_info", None)
    if not callable(get_tickers_with_info):
        return None

    def validate(symbol: str) -> bool:
        ticker = str(symbol or "").upper().strip()
        if not _is_ticker_symbol(ticker):
            return False
        try:
            info_items = get_tickers_with_info(ticker)
        except Exception as error:
            logger.warning("ticker_provider.validate.fail ticker=%s reason=%s", ticker, error)
            return False
        for item in info_items or []:
            if isinstance(item, str):
                raw_symbol = item
            elif isinstance(item, dict):
                raw_symbol = item.get("symbol") or item.get("ticker") or item.get("name")
            else:
                raw_symbol = getattr(item, "symbol", None) or getattr(item, "ticker", None) or getattr(item, "name", None)
            if str(raw_symbol or "").upper().strip() == ticker:
                return True
        return False

    return validate


def _is_ticker_symbol(value: str) -> bool:
    return 1 <= len(value) <= 5 and value.isalpha() and value.isupper()


def _human_event_context(ticker_universe: Any) -> dict[str, Any]:
    sources = getattr(ticker_universe, "sources", {}) or {}
    sellable = [symbol for symbol in ticker_universe.symbols if "open_position" in sources.get(symbol, [])]
    return {
        "buyable_universe": list(ticker_universe.symbols),
        "sellable_open_positions": sellable,
    }


def _planned_human_targets_by_note(instructions: list[HumanInstruction]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for instruction in instructions:
        if not instruction.target_ticker:
            continue
        grouped.setdefault(instruction.note, []).append(
            {
                "ticker": instruction.target_ticker,
                "sequence_index": instruction.sequence_index,
                "sequence_total": instruction.sequence_total,
                "intent_type": instruction.resolver_intent_type,
                "rationale": instruction.resolver_rationale,
            }
        )
    return grouped


def _latest_human_risk_profile(recent_entries: list[Any]) -> HumanRiskProfile:
    for entry in reversed(recent_entries):
        payload = entry.to_dict() if hasattr(entry, "to_dict") else entry
        if not isinstance(payload, dict):
            continue
        risk_assessment = payload.get("risk_assessment") or {}
        if not isinstance(risk_assessment, dict):
            continue
        profile = risk_assessment.get("human_risk_profile")
        if isinstance(profile, dict):
            try:
                return HumanRiskProfile(
                    risk_preference=str(profile.get("risk_preference") or "neutral"),
                    buy_aggressiveness=float(profile.get("buy_aggressiveness") or 0.0),
                    sell_aggressiveness=float(profile.get("sell_aggressiveness") or 0.0),
                    rationale=str(profile.get("rationale") or "Restored human risk profile from journal."),
                )
            except ValueError:
                continue
    return HumanRiskProfile()


def run_agent(
    ticker: str,
    cycles: int,
    interval_seconds: float,
    base_dir: Path,
    mode: str = "single_agent",
    resume_from: Path | None = None,
    memory_limit: int = 3,
    human_input: Path | None = None,
    watchlist: str | list[str] | None = None,
) -> RunContext:
    configure_logging()
    context = create_run_context(base_dir, "multi", resume_from=resume_from)
    journal = JournalStore(context.journal_path)
    human_input_path = human_input or context.human_input_path
    human_input_store = HumanInputStore(human_input_path, _human_input_cursor_path(human_input, context.human_input_cursor_path))
    market_data, news_provider, llm_client, broker, ticker_provider = _build_adapters()
    recent_entries = journal.read_all()[-memory_limit:] if context.resumed_from else []
    current_human_risk_profile = _latest_human_risk_profile(recent_entries)
    configured_watchlist_symbols = parse_watchlist(watchlist)
    agent = build_graph(
        mode=mode,
        market_data=market_data,
        news_provider=news_provider,
        llm_client=llm_client,
        broker=broker,
        journal=journal,
    )

    instruction_queue: list[HumanInstruction] = []
    for cycle_index in range(cycles):
        portfolio_snapshot = safe_portfolio_snapshot(broker)
        recent_news = _extract_recent_news(recent_entries)
        new_notes: list[str] = []
        note_ids: dict[str, str] = {}
        if not instruction_queue:
            new_notes = human_input_store.read_next_note().notes
            if new_notes:
                logger.info("human.input.received notes=%s", " | ".join(n[:80] for n in new_notes))
                previous_profile = current_human_risk_profile
                current_human_risk_profile = update_persistent_human_risk_profile(
                    current_human_risk_profile,
                    new_notes,
                    llm_client,
                )
                if current_human_risk_profile != previous_profile:
                    logger.info(
                        "human.risk.persisted preference=%s buy=%.2f sell=%.2f",
                        current_human_risk_profile.risk_preference,
                        current_human_risk_profile.buy_aggressiveness,
                        current_human_risk_profile.sell_aggressiveness,
                    )
        ticker_universe = build_ticker_universe(
            configured_watchlist=configured_watchlist_symbols,
            portfolio=portfolio_snapshot,
            human_input=new_notes,
            fallback_ticker=ticker,
            recent_news=recent_news,
            provider_symbols=_provider_symbols(ticker_provider, configured_watchlist_symbols),
        )
        watchlist_symbols = ticker_universe.symbols
        human_event_context = _human_event_context(ticker_universe)
        logger.info(
            "ticker.universe symbols=%s sources=%s",
            ",".join(watchlist_symbols) or "-",
            ticker_universe.sources,
        )
        if not instruction_queue:
            for note_index, note in enumerate(new_notes, start=1):
                note_id = f"cycle-{cycle_index + 1}-note-{note_index}"
                note_ids[note] = note_id
                journal.append_human_event(
                    note_id=note_id,
                    status="received",
                    note=note,
                    details={
                        "cycle": cycle_index + 1,
                        "human_risk_profile": current_human_risk_profile.to_dict(),
                        **human_event_context,
                    },
                )
            new_instructions = plan_human_instructions(
                new_notes,
                watchlist=watchlist_symbols,
                portfolio=portfolio_snapshot,
                fallback_ticker=ticker,
                llm_client=llm_client,
                pending_instructions=instruction_queue,
                recent_executed_instructions=_extract_recent_executed_for_resolver(recent_entries),
                ticker_validator=_provider_ticker_validator(ticker_provider),
            )
            planned_by_note = _planned_human_targets_by_note(new_instructions)
            for instr in new_instructions:
                journal.append_human_event(
                    note_id=note_ids.get(instr.note, f"instruction-{instr.instruction_id}"),
                    status="resolved",
                    note=instr.note,
                    instruction_id=instr.instruction_id,
                    ticker=instr.target_ticker,
                    details={
                        "intent_type": instr.resolver_intent_type,
                        "reason": instr.reason,
                        "confidence": instr.resolver_confidence,
                        "topic": instr.resolver_topic,
                        "rationale": instr.resolver_rationale,
                        "sequence_index": instr.sequence_index,
                        "sequence_total": instr.sequence_total,
                        "planned_targets": planned_by_note.get(instr.note, []),
                        "excluded_tickers": list(instr.resolver_excluded_tickers),
                        "human_risk_profile": current_human_risk_profile.to_dict(),
                        **human_event_context,
                    },
                )
            for instr in new_instructions:
                if instr.resolver_intent_type == "cancel":
                    logger.info(
                        "cancel.received cancel_id=%s targets=%s scope=%s reversal=%s",
                        instr.instruction_id,
                        ",".join(instr.cancel_target_ids),
                        instr.cancel_scope,
                        instr.cancel_reversal_action,
                    )
                    reversals = _handle_cancellation(
                        instr,
                        instruction_queue,
                        recent_entries,
                        journal,
                    )
                    instruction_queue.extend(reversals)
                else:
                    instruction_queue.append(instr)

        active_instruction = instruction_queue.pop(0) if instruction_queue else None
        human_notes = [active_instruction.note] if active_instruction else []

        if active_instruction and active_instruction.target_ticker:
            selected_ticker = active_instruction.target_ticker
            logger.info(
                "ticker.select source=human_instruction ticker=%s reason=%s intent=%s confidence=%s topic=%s retry=%s",
                selected_ticker,
                active_instruction.reason,
                active_instruction.resolver_intent_type,
                active_instruction.resolver_confidence,
                active_instruction.resolver_topic,
                active_instruction.retry_count,
            )
            journal.append_human_event(
                note_id=note_ids.get(active_instruction.note, f"instruction-{active_instruction.instruction_id}"),
                status="active",
                note=active_instruction.note,
                instruction_id=active_instruction.instruction_id,
                ticker=selected_ticker,
                details={
                    "intent_type": active_instruction.resolver_intent_type,
                    "reason": active_instruction.reason,
                    "topic": active_instruction.resolver_topic,
                    "rationale": active_instruction.resolver_rationale,
                    "retry_count": active_instruction.retry_count,
                    **human_event_context,
                },
            )
            journal.append_stage(
                ticker=selected_ticker,
                stage="ticker_selector",
                status="completed",
                message=(
                    f"Human instruction {active_instruction.instruction_id} selected {selected_ticker} "
                    f"(intent={active_instruction.resolver_intent_type}, "
                    f"reason={active_instruction.reason}, "
                    f"confidence={active_instruction.resolver_confidence}, "
                    f"retry={active_instruction.retry_count})."
                ),
                details={
                    "instruction_id": active_instruction.instruction_id,
                    "reason": active_instruction.reason,
                    "watchlist": watchlist_symbols,
                    "mentioned_tickers": [selected_ticker],
                    "human_input": human_notes,
                    "human_instruction_index": active_instruction.sequence_index,
                    "human_instruction_total": active_instruction.sequence_total,
                    "human_instruction_status": "active",
                    "human_resolver_source": _human_resolver_source(active_instruction),
                    "human_resolver_intent_type": active_instruction.resolver_intent_type,
                    "human_resolver_confidence": active_instruction.resolver_confidence,
                    "human_resolver_rationale": active_instruction.resolver_rationale,
                    "human_resolver_topic": active_instruction.resolver_topic,
                    "retry_count": active_instruction.retry_count,
                    "reversal_of": active_instruction.reversal_of,
                },
            )
        else:
            selection = select_ticker(
                human_input=human_notes,
                portfolio=portfolio_snapshot,
                recent_news=recent_news,
                watchlist=watchlist_symbols,
                cycle_index=cycle_index,
                fallback=ticker,
                ticker_provider=ticker_provider,
            )
            selected_ticker = selection.ticker
            logger.info(
                "ticker.select source=hybrid ticker=%s reason=%s mentioned=%s",
                selected_ticker,
                selection.reason,
                ",".join(selection.mentioned_tickers) or "-",
            )
            journal.append_stage(
                ticker=selected_ticker,
                stage="ticker_selector",
                status="completed",
                message=selection.rationale,
                details={
                    "reason": selection.reason,
                    "watchlist": watchlist_symbols,
                    "mentioned_tickers": selection.mentioned_tickers,
                    "human_input": human_notes,
                    "recent_news_count": len(recent_news),
                },
            )

        human_intent_override = _human_intent_override_for_instruction(active_instruction)
        if human_intent_override is not None:
            logger.info(
                "human.intent.override ticker=%s action=%s topic=%s",
                selected_ticker,
                human_intent_override.get("requested_action"),
                human_intent_override.get("impact_topic"),
            )

        state = run_cycle(
            agent=agent,
            ticker=selected_ticker,
            recent_entries=recent_entries,
            journal=journal,
            human_input=human_notes,
            human_intent=human_intent_override,
            human_risk_profile=current_human_risk_profile.to_dict(),
            instruction_id=active_instruction.instruction_id if active_instruction else None,
            retry_count=active_instruction.retry_count if active_instruction else 0,
            reversal_of=active_instruction.reversal_of if active_instruction else None,
        )
        journal_entry = state.get("journal_entry")
        if journal_entry:
            recent_entries.append(journal_entry)
            print_cycle_log(cycle_index + 1, journal_entry)
            if active_instruction is not None:
                payload = journal_entry.to_dict() if hasattr(journal_entry, "to_dict") else journal_entry
                risk_payload = payload.get("risk_assessment") or {}
                journal.append_human_event(
                    note_id=note_ids.get(active_instruction.note, f"instruction-{active_instruction.instruction_id}"),
                    status=str(payload.get("outcome") or "completed"),
                    note=active_instruction.note,
                    instruction_id=active_instruction.instruction_id,
                    ticker=str(payload.get("ticker") or active_instruction.target_ticker or "HUMAN"),
                    details={
                        "action": payload.get("action"),
                        "quantity": (payload.get("decision") or {}).get("quantity"),
                        "failure_type": payload.get("failure_type"),
                        "reversal_of": payload.get("reversal_of"),
                        "human_risk_profile": risk_payload.get("human_risk_profile"),
                        "risk_limits": {
                            "max_buy_quantity": risk_payload.get("max_buy_quantity"),
                            "max_sell_quantity": risk_payload.get("max_sell_quantity"),
                        },
                    },
                )
                _handle_execution_failure(
                    active_instruction,
                    journal_entry,
                    instruction_queue,
                    broker,
                    journal,
                )
        render_dashboard(context.journal_path, context.dashboard_path)
        if cycle_index < cycles - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)
    return context


def _human_input_cursor_path(human_input: Path | None, default_cursor: Path) -> Path:
    if human_input is None:
        return default_cursor
    return human_input.with_suffix(".cursor")


def _human_resolver_source(instruction: HumanInstruction) -> str:
    if instruction.reason == "human_llm_position_sweep":
        return "llm"
    if instruction.reason == "human_position_sweep":
        return "deterministic_fallback"
    if instruction.reason == "human_reversal":
        return "reversal"
    if instruction.reason == "human_cancel":
        return "cancel"
    return "deterministic"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL", help="Fallback ticker used when nothing else applies.")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--mode", choices=["single_agent", "multi_agent"], default="single_agent")
    parser.add_argument("--base-dir", type=Path, default=Path("journal_runs"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--memory-limit", type=int, default=3)
    parser.add_argument("--human-input", type=Path, default=None)
    parser.add_argument(
        "--watchlist",
        default=None,
        help=(
            "Optional soft constraint. When provided, news-mentioned tickers "
            "and the default universe are filtered against this list. Open "
            "positions and explicit human mentions are never filtered."
        ),
    )
    args = parser.parse_args()
    context = run_agent(
        args.ticker,
        args.cycles,
        args.interval,
        args.base_dir,
        args.mode,
        args.resume_from,
        args.memory_limit,
        args.human_input,
        args.watchlist,
    )
    if context.resumed_from:
        print(f"resumed_from: {context.resumed_from}")
    print(f"journal: {context.journal_path}")
    print(f"dashboard: {context.dashboard_path}")
    print(f"human_input: {args.human_input or context.human_input_path}")


if __name__ == "__main__":
    main()
