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
from trading_agent.core.autonomous_selection import AutonomousSelectionCache, choose_cycle_ticker, discover_dynamic_candidates
from trading_agent.core.execution_policy import ExecutionFailure
from trading_agent.journal import (
    DashboardProjectionWatcher,
    JournalStore,
    RunContext,
    create_run_context,
    print_cycle_log,
    render_dashboard,
    write_dashboard_projection,
)
from trading_agent.pipeline import build_graph, run_cycle
from trading_agent.utils import FallbackLlmClient, OllamaJsonClient, OpenAiJsonClient, OpenRouterJsonClient, configure_logging, safe_portfolio_snapshot

logger = logging.getLogger("trading_agent.run_agent")

RECENT_EXECUTED_WINDOW = 10


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
            "requested_notional_usd": instruction.requested_notional_usd,
            "requested_quantity": instruction.requested_quantity,
            "partial_fraction": instruction.partial_fraction,
        }

    if intent_type in {"broad_sell", "conditional_sell", "position_sweep"}:
        return {
            "intents": ["sell", "position_sweep"],
            "tickers": [ticker],
            "requested_action": "SELL",
            "risk_preference": None,
            "impact_topic": topic,
            "summary": (f"LLM-resolved {intent_type} for {ticker}" + (f" due to topic '{topic}'" if topic else "") + f" (resolver_confidence={instruction.resolver_confidence})"),
            "requested_notional_usd": instruction.requested_notional_usd,
            "partial_fraction": instruction.partial_fraction,
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
    return []


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
    from trading_agent.core.ticker_symbols import is_ticker_symbol

    return is_ticker_symbol(value)


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
    max_discovered_tickers: int = 3,
    autonomous_news_shortlist_size: int = 3,
    autonomous_market_cache_ttl: float = 30.0,
    autonomous_news_cache_ttl: float = 600.0,
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

    render_dashboard(context.journal_path, context.dashboard_path)
    dashboard_watcher = DashboardProjectionWatcher(
        context.journal_path,
        context.run_dir / "dashboard_data.js",
    )
    dashboard_watcher.start()

    from trading_agent.core.constraint_store import ConstraintStore
    from trading_agent.core.conditional_order_store import ConditionalOrderStore, check_trigger as _check_conditional_trigger
    from trading_agent.core.scheduled_action_store import ScheduledActionStore
    from trading_agent.core.agent_reply import generate_reply as _generate_agent_reply, append_agent_reply as _append_agent_reply
    from trading_agent.core.sector_classifier import SectorClassifier

    from trading_agent.core.typed_resolver import resolve_human_input as _resolve_human_input_typed
    from trading_agent.core.typed_outcomes import (
        AdvisoryReply,
        ConfirmedOperations,
        ConditionalOrderOutcome,
        ExecutableInstruction,
        InformationRequest,
        outcome_to_human_instructions,
        PendingConfirmation,
        PersistentStateUpdate,
        ScheduledActionOutcome,
    )
    from trading_agent.core.instruction_ledger import (
        InstructionLedger,
        make_request_id,
    )
    from trading_agent.core.confirmation_store import ConfirmationStore

    constraint_store = ConstraintStore(context.run_dir / "constraints.json")
    conditional_order_store = ConditionalOrderStore(context.run_dir / "conditional_orders.json")
    scheduled_action_store = ScheduledActionStore(context.run_dir / "scheduled_actions.json")
    confirmation_store = ConfirmationStore(context.run_dir / "pending_confirmations.json")
    instruction_ledger = InstructionLedger(context.run_dir / "instruction_ledger.json")
    agent_replies_path = context.run_dir / "agent_replies.md"
    sector_classifier = SectorClassifier()
    autonomous_selection_cache = AutonomousSelectionCache(
        market_ttl_seconds=autonomous_market_cache_ttl,
        news_ttl_seconds=autonomous_news_cache_ttl,
    )

    instruction_queue: list[HumanInstruction] = []
    previous_market_is_open = _get_market_is_open(broker)
    for cycle_index in range(cycles):
        portfolio_snapshot = safe_portfolio_snapshot(broker)
        recent_news = _extract_recent_news(recent_entries)
        new_notes: list[str] = []

        input_batch = human_input_store.read_next_note()
        new_notes = input_batch.notes
        request_id = make_request_id(
            input_batch.cursor_before,
            input_batch.cursor_after,
            run_dir_name=context.run_dir.name,
        )
        if new_notes:
            logger.info(
                "human.input.received request_id=%s notes=%s",
                request_id,
                " | ".join(n[:80] for n in new_notes),
            )
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
        dynamic_symbols = (
            discover_dynamic_candidates(
                news_provider,
                ticker_provider,
                cache=autonomous_selection_cache,
                max_candidates=6,
            )
            if not configured_watchlist_symbols
            else []
        )
        ticker_universe = build_ticker_universe(
            configured_watchlist=configured_watchlist_symbols,
            portfolio=portfolio_snapshot,
            human_input=new_notes,
            fallback_ticker=ticker,
            recent_news=recent_news,
            provider_symbols=dynamic_symbols,
        )
        watchlist_symbols = ticker_universe.symbols[:10]
        ticker_universe = replace(ticker_universe, symbols=watchlist_symbols)
        human_event_context = _human_event_context(ticker_universe)
        logger.info(
            "ticker.universe symbols=%s sources=%s",
            ",".join(watchlist_symbols) or "-",
            ticker_universe.sources,
        )

        try:
            market_is_open = _get_market_is_open(broker)
            opened_now = market_is_open is True and previous_market_is_open is not True
            closed_now = market_is_open is False and previous_market_is_open is True
            due_scheduled = scheduled_action_store.get_due(market_open=opened_now, market_close=closed_now)
            previous_market_is_open = market_is_open
        except Exception as error:
            logger.warning("scheduled_action.get_due.fail reason=%s", error)
            due_scheduled = []
        for scheduled in due_scheduled:
            wrapped = HumanInstruction(
                note=scheduled.note,
                target_ticker=scheduled.target_ticker,
                reason="human_scheduled_action",
                resolver_intent_type=scheduled.wrapped_intent_type or "advisory",
                resolver_rationale=scheduled.rationale or "Scheduled action fired.",
                requested_notional_usd=scheduled.requested_notional_usd,
                requested_quantity=getattr(scheduled, "requested_quantity", None),
                partial_fraction=scheduled.partial_fraction,
                override_constraints=scheduled.override_constraints,
            )
            instruction_queue.append(wrapped)
            scheduled_action_store.mark_triggered(scheduled.action_id)
            journal.append_stage(
                ticker=scheduled.target_ticker or "HUMAN",
                stage="scheduled_action_fired",
                status="completed",
                message=f"Scheduled action {scheduled.action_id} fired (trigger={scheduled.trigger_type}/{scheduled.trigger_value}). Wrapped as {wrapped.resolver_intent_type} on {wrapped.target_ticker or 'n/a'}.",
                details={
                    "scheduled_action_id": scheduled.action_id,
                    "trigger_type": scheduled.trigger_type,
                    "trigger_value": scheduled.trigger_value,
                    "wrapped_intent_type": scheduled.wrapped_intent_type,
                    "target_ticker": scheduled.target_ticker,
                    "instruction_id": wrapped.instruction_id,
                },
            )

        try:
            expired_count = confirmation_store.expire_overdue()
            if expired_count:
                logger.info("confirmation.expired count=%d", expired_count)
        except Exception as error:
            logger.warning("confirmation.expire_overdue.fail reason=%s", error)

        if new_notes:
            for note_index, note in enumerate(new_notes, start=1):
                note_id = request_id
                journal.append_human_event(
                    note_id=note_id,
                    status="received",
                    note=note,
                    details={
                        "cycle": cycle_index + 1,
                        "request_id": request_id,
                        "human_risk_profile": current_human_risk_profile.to_dict(),
                        **human_event_context,
                    },
                )
            outcomes = _resolve_human_input_typed(
                new_notes,
                request_id=request_id,
                watchlist=watchlist_symbols,
                portfolio=portfolio_snapshot,
                fallback_ticker=ticker,
                llm_client=llm_client,
                pending_instructions=instruction_queue,
                recent_executed_instructions=_extract_recent_executed_for_resolver(recent_entries),
                ticker_validator=_provider_ticker_validator(ticker_provider),
                news_provider=news_provider,
                alpaca_validator=_provider_ticker_validator(ticker_provider),
                ticker_provider=ticker_provider,
                max_discovered_tickers=max(1, int(max_discovered_tickers)),
                confirmation_store=confirmation_store,
            )
            for outcome in outcomes:
                journal.append_human_event(
                    note_id=outcome.request_id,
                    status="resolved",
                    note=outcome.rationale.user_summary,
                    instruction_id=None,
                    ticker=getattr(outcome, "target_ticker", None),
                    details={
                        "outcome_type": outcome.outcome_type,
                        "request_id": outcome.request_id,
                        "subnote_index": outcome.subnote_index,
                        "idempotency_key": outcome.idempotency_key,
                        "user_summary": outcome.rationale.user_summary,
                        "next_step": outcome.rationale.next_step,
                        "evidence": [e for e in outcome.rationale.to_dict()["evidence"]],
                        "technical_details": outcome.rationale.technical_details,
                        **human_event_context,
                    },
                )

                if isinstance(outcome, ExecutableInstruction) and outcome.intent_type == "cancel":
                    cancel_instructions = outcome_to_human_instructions(outcome)
                    for cancel_instr in cancel_instructions:
                        logger.info(
                            "cancel.received cancel_id=%s targets=%s scope=%s reversal=%s",
                            cancel_instr.instruction_id,
                            ",".join(cancel_instr.cancel_target_ids),
                            cancel_instr.cancel_scope,
                            cancel_instr.cancel_reversal_action,
                        )
                        for target_id in cancel_instr.cancel_target_ids:
                            try:
                                if scheduled_action_store.cancel(target_id):
                                    journal.append_stage(
                                        ticker="HUMAN",
                                        stage="scheduled_action_cancelled",
                                        status="completed",
                                        message=f"Scheduled action {target_id} cancelled by user request.",
                                        details={
                                            "scheduled_action_id": target_id,
                                            "cancel_instruction_id": cancel_instr.instruction_id,
                                        },
                                    )
                            except Exception as error:
                                logger.warning("scheduled_action.cancel.fail id=%s reason=%s", target_id, error)
                        reversals = _handle_cancellation(
                            cancel_instr,
                            instruction_queue,
                            recent_entries,
                            journal,
                        )
                        try:
                            instruction_ledger.cancel_by_instruction_ids(cancel_instr.cancel_target_ids)
                        except Exception:
                            pass
                        instruction_queue.extend(reversals)
                    continue

                if isinstance(outcome, PendingConfirmation):
                    if not outcome.confirmation_id:
                        confirmation = confirmation_store.add(
                            originating_request_id=outcome.originating_request_id or outcome.request_id,
                            proposal_text=outcome.proposal_text,
                            proposed_operations=list(outcome.proposed_operations),
                            rationale=outcome.rationale.user_summary,
                            evidence=list(outcome.rationale.evidence),
                        )
                    else:
                        confirmation = outcome
                    journal.append_stage(
                        ticker="HUMAN",
                        stage="pending_confirmation",
                        status="completed",
                        message=f"Pending confirmation {confirmation.confirmation_id}: {outcome.proposal_text}",
                        details={
                            "confirmation_id": confirmation.confirmation_id,
                            "proposal_text": outcome.proposal_text,
                            "proposed_operations": list(outcome.proposed_operations),
                            "expires_at": confirmation.expires_at,
                            "evidence": [e.to_dict() if hasattr(e, "to_dict") else e for e in outcome.rationale.evidence],
                        },
                    )
                    continue

                if isinstance(outcome, InformationRequest):
                    from trading_agent.core.agent_reply import search_journal

                    reply_entries = (
                        search_journal(context.journal_path, limit=50) if outcome.question_type == "decision_history" else [e.to_dict() if hasattr(e, "to_dict") else e for e in recent_entries]
                    )
                    reply = _generate_agent_reply(
                        question_type=outcome.question_type,
                        topic=None,
                        portfolio=portfolio_snapshot,
                        journal_entries=reply_entries,
                        note=outcome.rationale.user_summary,
                        llm_client=llm_client,
                    )
                    _append_agent_reply(
                        agent_replies_path,
                        reply,
                        outcome.rationale.user_summary,
                        question_type=outcome.question_type,
                    )
                    journal.append_stage(
                        ticker="HUMAN",
                        stage="agent_reply",
                        status="completed",
                        message=f"Agent replied to {outcome.question_type}: {reply[:200]}",
                        details={
                            "request_id": outcome.request_id,
                            "question_type": outcome.question_type,
                            "reply": reply,
                            "evidence": [e.to_dict() if hasattr(e, "to_dict") else e for e in outcome.rationale.evidence],
                        },
                    )
                    continue

                if isinstance(outcome, PersistentStateUpdate):
                    pseudo_instr = HumanInstruction(
                        note=outcome.rationale.user_summary,
                        target_ticker=None,
                        reason="human_constraint_update",
                        resolver_intent_type="constraint_update",
                        resolver_rationale=outcome.rationale.user_summary,
                        constraint_update={
                            "action": outcome.action,
                            "type": outcome.constraint_type,
                            "value": outcome.value,
                            "rationale": outcome.rationale.user_summary,
                            "constraint_id": outcome.constraint_id,
                        },
                        request_id=outcome.request_id,
                        idempotency_key=outcome.idempotency_key,
                        user_summary=outcome.rationale.user_summary,
                        next_step=outcome.rationale.next_step,
                        evidence=tuple(e.to_dict() for e in outcome.rationale.evidence),
                    )
                    _apply_constraint_update(pseudo_instr, constraint_store, journal)
                    continue

                if isinstance(outcome, ConditionalOrderOutcome):
                    pseudo_instr = HumanInstruction(
                        note=outcome.rationale.user_summary,
                        target_ticker=outcome.target_ticker,
                        reason="human_conditional_order",
                        resolver_intent_type="conditional_order",
                        resolver_rationale=outcome.rationale.user_summary,
                        conditional_order={
                            "trigger_type": outcome.trigger_type,
                            "trigger_price": outcome.trigger_price,
                            "trigger_fraction": outcome.trigger_fraction,
                        },
                        request_id=outcome.request_id,
                        idempotency_key=outcome.idempotency_key,
                        user_summary=outcome.rationale.user_summary,
                        next_step=outcome.rationale.next_step,
                        evidence=tuple(e.to_dict() for e in outcome.rationale.evidence),
                    )
                    _apply_conditional_order(pseudo_instr, conditional_order_store, journal)
                    continue

                if isinstance(outcome, ScheduledActionOutcome):
                    pseudo_instr = HumanInstruction(
                        note=outcome.rationale.user_summary,
                        target_ticker=outcome.target_ticker,
                        reason="human_scheduled_action",
                        resolver_intent_type="scheduled_action",
                        resolver_rationale=outcome.rationale.user_summary,
                        requested_notional_usd=outcome.requested_notional_usd,
                        requested_quantity=outcome.requested_quantity,
                        partial_fraction=outcome.partial_fraction,
                        override_constraints=outcome.override_constraints,
                        scheduled_trigger={
                            "trigger_type": outcome.trigger_type,
                            "trigger_value": outcome.trigger_value,
                            "wrapped_intent_type": outcome.wrapped_intent_type,
                        },
                        request_id=outcome.request_id,
                        idempotency_key=outcome.idempotency_key,
                        user_summary=outcome.rationale.user_summary,
                        next_step=outcome.rationale.next_step,
                        evidence=tuple(e.to_dict() for e in outcome.rationale.evidence),
                    )
                    _apply_scheduled_action(pseudo_instr, scheduled_action_store, journal)
                    continue

                if isinstance(outcome, AdvisoryReply):
                    _append_agent_reply(
                        agent_replies_path,
                        outcome.reply,
                        outcome.rationale.user_summary,
                        question_type="advisory",
                    )
                    journal.append_stage(
                        ticker="HUMAN",
                        stage="advisory_reply",
                        status="completed",
                        message=f"Advisory ({outcome.advisory_type}): {outcome.reply[:200]}",
                        details={
                            "request_id": outcome.request_id,
                            "advisory_type": outcome.advisory_type,
                            "reply": outcome.reply,
                            "evidence": [e.to_dict() if hasattr(e, "to_dict") else e for e in outcome.rationale.evidence],
                        },
                    )
                    continue

                if isinstance(outcome, (ExecutableInstruction, ConfirmedOperations)):
                    instructions = outcome_to_human_instructions(outcome)
                    for instr in instructions:
                        if instr.idempotency_key and instruction_ledger.has_key(instr.idempotency_key):
                            logger.info(
                                "ledger.idempotent_skip key=%s intent=%s ticker=%s",
                                instr.idempotency_key,
                                instr.resolver_intent_type,
                                instr.target_ticker,
                            )
                            journal.append_stage(
                                ticker=instr.target_ticker or "HUMAN",
                                stage="idempotent_skip",
                                status="completed",
                                message=f"Instruction with key {instr.idempotency_key} already queued or completed; skipping.",
                                details={
                                    "idempotency_key": instr.idempotency_key,
                                    "request_id": instr.request_id,
                                    "intent_type": instr.resolver_intent_type,
                                    "target_ticker": instr.target_ticker,
                                },
                            )
                            continue
                        if instr.idempotency_key:
                            instruction_ledger.record_queued(
                                idempotency_key=instr.idempotency_key,
                                request_id=instr.request_id or "",
                                subnote_index=instr.subnote_index,
                                intent_type=instr.resolver_intent_type or "",
                                target_ticker=instr.target_ticker,
                                instruction_id=instr.instruction_id,
                                workflow_id=instr.workflow_id,
                                workflow_type=instr.workflow_type,
                                originating_confirmation_id=instr.originating_confirmation_id,
                                operation_index=instr.sequence_index,
                                operation_total=instr.sequence_total,
                            )
                        instruction_queue.append(instr)
                    continue

            write_dashboard_projection(context.journal_path, context.run_dir / "dashboard_data.js")

        _fire_triggered_conditional_orders(
            conditional_order_store,
            instruction_queue,
            broker,
            market_data,
            journal,
        )

        active_instruction = instruction_queue.pop(0) if instruction_queue else None
        human_notes = [active_instruction.note] if active_instruction else []
        if active_instruction:
            try:
                instruction_ledger.update_status(active_instruction.instruction_id, "active")
            except Exception as error:
                logger.warning("ledger.mark_active.fail reason=%s", error)

        if active_instruction and active_instruction.target_ticker:
            sector = sector_classifier.classify(active_instruction.target_ticker, llm_client)
            try:
                current_price = market_data.get_price(active_instruction.target_ticker).get("price")
                current_price = float(current_price) if current_price else None
            except Exception:
                current_price = None
            violations = constraint_store.applies_to(
                active_instruction.target_ticker,
                sector=sector,
                price=current_price,
            )
            if violations and not active_instruction.override_constraints:
                violation = violations[0]
                logger.info(
                    "constraint.blocked ticker=%s constraint=%s/%s rationale=%s",
                    active_instruction.target_ticker,
                    violation.type,
                    violation.value,
                    violation.rationale,
                )
                journal.append_stage(
                    ticker=active_instruction.target_ticker,
                    stage="constraint_blocked",
                    status="completed",
                    message=(f"Instruction {active_instruction.instruction_id} blocked by active constraint " f"{violation.constraint_id} ({violation.type}={violation.value}): {violation.rationale}"),
                    details={
                        "instruction_id": active_instruction.instruction_id,
                        "constraint_id": violation.constraint_id,
                        "constraint_type": violation.type,
                        "constraint_value": violation.value,
                        "constraint_rationale": violation.rationale,
                        "override_attempted": False,
                    },
                )
                write_dashboard_projection(context.journal_path, context.run_dir / "dashboard_data.js")
                if cycle_index < cycles - 1 and interval_seconds > 0:
                    time.sleep(interval_seconds)
                continue
            if violations and active_instruction.override_constraints:
                journal.append_stage(
                    ticker=active_instruction.target_ticker,
                    stage="constraint_bypassed",
                    status="completed",
                    message=(f"Instruction {active_instruction.instruction_id} bypassed {len(violations)} active constraint(s) " "because override_constraints=true was set by the resolver."),
                    details={
                        "instruction_id": active_instruction.instruction_id,
                        "bypassed_constraints": [{"constraint_id": v.constraint_id, "type": v.type, "value": v.value} for v in violations],
                        "override_attempted": True,
                    },
                )

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
                note_id=active_instruction.request_id or f"instruction-{active_instruction.instruction_id}",
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
            selection = choose_cycle_ticker(
                active_ticker=None,
                tickers=watchlist_symbols,
                sources=ticker_universe.sources,
                portfolio=portfolio_snapshot,
                market_data=market_data,
                news_provider=news_provider,
                llm_client=llm_client,
                cache=autonomous_selection_cache,
                news_shortlist_size=max(1, int(autonomous_news_shortlist_size)),
            )
            selected_ticker = selection.ticker
            logger.info(
                "ticker.select source=%s ticker=%s mode=%s confidence=%.2f",
                selection.selection_source,
                selected_ticker,
                selection.selection_mode,
                selection.confidence,
            )
            journal.append_stage(
                ticker=selected_ticker,
                stage="ticker_selector",
                status="completed",
                message=selection.user_summary,
                details=selection.to_dict(),
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
                    note_id=active_instruction.request_id or f"instruction-{active_instruction.instruction_id}",
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
                try:
                    outcome_status = str(payload.get("outcome") or "completed").lower()
                    ledger_status = {
                        "filled": "completed",
                        "submitted": "completed",
                        "skipped": "completed",
                        "waiting": "queued",
                        "not_executed": "queued",
                        "blocked": "abandoned",
                        "rejected": "abandoned",
                        "failed": "queued",
                    }.get(outcome_status, "completed")
                    instruction_ledger.update_status(
                        active_instruction.instruction_id,
                        ledger_status,
                        outcome_summary=str(payload.get("cycle_summary") or "")[:200],
                        increment_retry=(outcome_status in {"failed", "rejected", "blocked"}),
                    )
                except Exception as error:
                    logger.warning("ledger.update_status.fail reason=%s", error)
                _handle_execution_failure(
                    active_instruction,
                    journal_entry,
                    instruction_queue,
                    broker,
                    journal,
                )
        write_dashboard_projection(context.journal_path, context.run_dir / "dashboard_data.js")
        if cycle_index < cycles - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)
    dashboard_watcher.stop()
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


def _apply_constraint_update(
    instruction: HumanInstruction,
    constraint_store: Any,
    journal: JournalStore,
) -> None:
    from trading_agent.core.constraint_store import make_constraint

    payload = instruction.constraint_update or {}
    action = str(payload.get("action") or "add").lower()
    if action == "add":
        constraint = make_constraint(
            type=str(payload.get("type") or "exclude_ticker"),
            value=str(payload.get("value") or ""),
            rationale=str(payload.get("rationale") or instruction.note or ""),
        )
        constraint_store.add(constraint)
        journal.append_stage(
            ticker="HUMAN",
            stage="constraint_added",
            status="completed",
            message=f"Constraint added: {constraint.type}={constraint.value} ({constraint.rationale}).",
            details={
                "instruction_id": instruction.instruction_id,
                "constraint_id": constraint.constraint_id,
                "constraint_type": constraint.type,
                "constraint_value": constraint.value,
                "constraint_rationale": constraint.rationale,
            },
        )
    elif action == "deactivate":
        constraint_id = str(payload.get("constraint_id") or "")
        type_ = str(payload.get("type") or "")
        value = str(payload.get("value") or "")
        if constraint_id:
            ok = constraint_store.deactivate(constraint_id)
        elif type_ and value:
            ok = constraint_store.deactivate_by_value(type_, value) > 0
        else:
            ok = False
        journal.append_stage(
            ticker="HUMAN",
            stage="constraint_deactivated" if ok else "constraint_deactivate_failed",
            status="completed",
            message=(
                f"Constraint deactivated (id={constraint_id}, type={type_}, value={value}). " if ok else f"Could not find a constraint to deactivate (id={constraint_id}, type={type_}, value={value})."
            ),
            details={
                "instruction_id": instruction.instruction_id,
                "constraint_id": constraint_id,
                "constraint_type": type_,
                "constraint_value": value,
                "deactivated": ok,
            },
        )
    elif action == "override":
        journal.append_stage(
            ticker="HUMAN",
            stage="constraint_override_recorded",
            status="completed",
            message="Constraint override recorded for this instruction only.",
            details={
                "instruction_id": instruction.instruction_id,
            },
        )


def _apply_conditional_order(
    instruction: HumanInstruction,
    conditional_order_store: Any,
    journal: JournalStore,
) -> None:
    from trading_agent.core.conditional_order_store import make_conditional_order

    payload = instruction.conditional_order or {}
    if not instruction.target_ticker:
        journal.append_stage(
            ticker="HUMAN",
            stage="conditional_order_rejected",
            status="completed",
            message="Conditional order rejected: no target ticker.",
            details={"instruction_id": instruction.instruction_id},
        )
        return
    order = make_conditional_order(
        ticker=instruction.target_ticker,
        trigger_type=str(payload.get("trigger_type") or ""),
        trigger_price=payload.get("trigger_price"),
        trigger_fraction=payload.get("trigger_fraction"),
        side="sell",
        instruction_id=instruction.instruction_id,
        rationale=instruction.resolver_rationale,
    )
    conditional_order_store.add(order)
    journal.append_stage(
        ticker=instruction.target_ticker,
        stage="conditional_order_added",
        status="completed",
        message=(f"Conditional order registered: {order.trigger_type} on {order.ticker} " f"price={order.trigger_price} fraction={order.trigger_fraction}."),
        details={
            "instruction_id": instruction.instruction_id,
            "order_id": order.order_id,
            "ticker": order.ticker,
            "trigger_type": order.trigger_type,
            "trigger_price": order.trigger_price,
            "trigger_fraction": order.trigger_fraction,
        },
    )


def _apply_scheduled_action(
    instruction: HumanInstruction,
    scheduled_action_store: Any,
    journal: JournalStore,
) -> None:
    from trading_agent.core.scheduled_action_store import make_scheduled_action

    payload = instruction.scheduled_trigger or {}
    action = make_scheduled_action(
        note=instruction.note,
        wrapped_intent_type=str(payload.get("wrapped_intent_type") or "advisory"),
        trigger_type=str(payload.get("trigger_type") or ""),
        trigger_value=str(payload.get("trigger_value") or ""),
        target_ticker=instruction.target_ticker,
        instruction_id=instruction.instruction_id,
        rationale=instruction.resolver_rationale,
        requested_notional_usd=instruction.requested_notional_usd,
        requested_quantity=instruction.requested_quantity,
        partial_fraction=instruction.partial_fraction,
        override_constraints=instruction.override_constraints,
    )
    scheduled_action_store.add(action)
    journal.append_stage(
        ticker=instruction.target_ticker or "HUMAN",
        stage="scheduled_action_added",
        status="completed",
        message=(f"Scheduled action registered: {action.trigger_type}/{action.trigger_value} " f"wraps {action.wrapped_intent_type} on {action.target_ticker or 'n/a'}."),
        details={
            "instruction_id": instruction.instruction_id,
            "action_id": action.action_id,
            "trigger_type": action.trigger_type,
            "trigger_value": action.trigger_value,
            "wrapped_intent_type": action.wrapped_intent_type,
            "target_ticker": action.target_ticker,
        },
    )


def _fire_triggered_conditional_orders(
    conditional_order_store: Any,
    instruction_queue: list[HumanInstruction],
    broker: Any,
    market_data: Any,
    journal: JournalStore,
) -> None:
    from trading_agent.core.conditional_order_store import check_trigger as _check_trigger

    pending = conditional_order_store.get_pending()
    if not pending:
        return
    tickers = {order.ticker for order in pending}
    prices: dict[str, float | None] = {}
    avg_entries: dict[str, float | None] = {}
    try:
        portfolio = broker.get_portfolio() or {}
        positions = portfolio.get("positions") or {}
    except Exception:
        positions = {}
    for ticker in tickers:
        try:
            price_payload = market_data.get_price(ticker)
            prices[ticker] = float(price_payload.get("price")) if price_payload else None
        except Exception:
            prices[ticker] = None
        pos = positions.get(ticker) or {}
        try:
            avg_entries[ticker] = float(pos.get("avg_entry_price")) if pos.get("avg_entry_price") else None
        except (TypeError, ValueError):
            avg_entries[ticker] = None

    for order in pending:
        price = prices.get(order.ticker)
        avg_entry = avg_entries.get(order.ticker)
        if not _check_trigger(order, current_price=price, avg_entry_price=avg_entry):
            continue
        triggered_qty = None
        try:
            held_qty = float((positions.get(order.ticker) or {}).get("qty") or 0)
            if order.trigger_fraction:
                from math import floor as _floor

                triggered_qty = max(1, int(_floor(held_qty * order.trigger_fraction)))
        except (TypeError, ValueError):
            triggered_qty = None
        sell_instruction = HumanInstruction(
            note=f"Conditional order {order.order_id} triggered ({order.trigger_type} at {price})",
            target_ticker=order.ticker,
            reason="human_conditional_order_triggered",
            resolver_intent_type="forced_sell",
            resolver_rationale=f"Conditional order {order.order_id} fired: {order.trigger_type} at price {price}.",
            partial_fraction=order.trigger_fraction if triggered_qty is None else None,
        )
        instruction_queue.append(sell_instruction)
        conditional_order_store.mark_triggered(order.order_id, triggered_price=price or 0.0)
        journal.append_stage(
            ticker=order.ticker,
            stage="conditional_order_triggered",
            status="completed",
            message=(
                f"Conditional order {order.order_id} triggered at price {price} "
                f"(type={order.trigger_type}, fraction={order.trigger_fraction}). "
                f"Queued forced_sell {sell_instruction.instruction_id}."
            ),
            details={
                "order_id": order.order_id,
                "ticker": order.ticker,
                "trigger_type": order.trigger_type,
                "trigger_price": order.trigger_price,
                "triggered_price": price,
                "trigger_fraction": order.trigger_fraction,
                "sell_instruction_id": sell_instruction.instruction_id,
            },
        )


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
        "--autonomous-shortlist-size",
        type=int,
        default=3,
        help="Number of metric-ranked candidates enriched with news before autonomous LLM selection.",
    )
    parser.add_argument(
        "--autonomous-market-cache-ttl",
        type=float,
        default=30.0,
        help="Seconds to reuse autonomous-selection prices and historical metrics.",
    )
    parser.add_argument(
        "--autonomous-news-cache-ttl",
        type=float,
        default=600.0,
        help="Seconds to reuse autonomous-selection news evidence.",
    )
    parser.add_argument(
        "--watchlist",
        default=None,
        help=("Optional autonomous candidate universe. Open positions, the fallback " "ticker, and validated explicit human targets remain eligible."),
    )
    args = parser.parse_args()
    context = run_agent(
        ticker=args.ticker,
        cycles=args.cycles,
        interval_seconds=args.interval,
        base_dir=args.base_dir,
        mode=args.mode,
        resume_from=args.resume_from,
        memory_limit=args.memory_limit,
        human_input=args.human_input,
        watchlist=args.watchlist,
        autonomous_news_shortlist_size=args.autonomous_shortlist_size,
        autonomous_market_cache_ttl=args.autonomous_market_cache_ttl,
        autonomous_news_cache_ttl=args.autonomous_news_cache_ttl,
    )
    if context.resumed_from:
        print(f"resumed_from: {context.resumed_from}")
    print(f"journal: {context.journal_path}")
    print(f"dashboard: {context.dashboard_path}")
    print(f"human_input: {args.human_input or context.human_input_path}")


if __name__ == "__main__":
    main()
