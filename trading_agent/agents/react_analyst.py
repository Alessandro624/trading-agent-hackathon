from __future__ import annotations

import json
from typing import Any

from trading_agent.core import (
    JournalEntry,
    LlmClient,
    MarketSnapshot,
    RetryPolicy,
    TradingDecision,
    build_position_sizing_context,
    parse_analyst_output,
    parse_human_intent,
    rationale_snapshot_mismatches,
    safe_rationale_details,
    snapshot_grounded_hold_rationale,
)
from trading_agent.journal import compact_recent_entries
from trading_agent.tools import (
    langchain_market_context_tools,
    langchain_news_search_tool,
    market_context_tool_schemas,
    run_market_context_tools,
    snapshot_payload,
)
from trading_agent.utils import get_logger, llm_metadata

logger = get_logger("analyst.react")


TOOL_PROMPT = """You are the ReAct Analyst tool-selection step.
Use the available tools only to inspect validated market data, news, technical context, or journal memory before the final decision.
Do not place orders and do not decide BUY/SELL/HOLD/WAIT in this step.
Prefer no extra tool when the snapshot is already sufficient.
Call search_market_news only when snapshot news is empty, stale, or too generic.
Call read_news_url for at most one important URL when you need more than NewsAPI's truncated preview.
Use at most 3 tool calls.
If the runtime requires a JSON tool plan, return exactly:
{"tool_calls":[{"name":"inspect_snapshot|explain_technical_signals|summarize_news_urls|read_news_url|search_market_news","args":{}}]}
"""

FINAL_PROMPT = """You are the ReAct Analyst final decision step.
Use the tool observations and return only valid JSON matching:
{"action":"BUY|SELL|HOLD|WAIT","quantity":0,"confidence":0.0,"rationale":"...","used_data_sources":["..."],"rationale_details":{"summary":"...","evidence":["..."],"risks":["..."],"data_quality":"..."}}
Do not invent prices, news, technical indicators, failures, guardrails, or portfolio state.
Human input is advisory but important: consider it explicitly, and if you disagree, explain why using validated data and risk limits.
Human intent is pre-parsed from the notes. If human_intent.requested_action is SELL for the current ticker and position_sizing.max_sell_quantity > 0, prefer SELL unless validated data gives a clear reason not to.
If human_intent includes conditional_sell, evaluate whether the current ticker is materially exposed to human_intent.impact_topic. SELL only if exposure/impact is justified by validated evidence; otherwise HOLD and explain why.
If human_intent.risk_preference is risk_on, you may use a higher BUY/SELL quantity within the valid risk limit. If it is risk_off, prefer lower quantity or SELL/HOLD.
If observations reveal a temporary data gap, pending market condition, or specific news/source that should be checked in a later cycle, choose WAIT.
If observations reveal a final no-trade conclusion, contradiction, or unsafe setup, choose HOLD.
NewsAPI content is a short/truncated preview; treat full article body as available only when read_news_url succeeded.
If tool failures exist, mention them as data limits, not market signals.
Quantity rule is strict: BUY/SELL require quantity > 0. HOLD/WAIT require quantity 0.
Use position_sizing.valid_quantity_rule.
If the selected action limit is 0, choose HOLD or WAIT with quantity 0.
If position_sizing.stop_loss_triggered or position_sizing.take_profit_triggered is true, treat it as a strong risk signal and explain whether to SELL or HOLD.
Consider position_sizing.current_position and portfolio_context before adding exposure or deciding to SELL.
Write the rationale for humans reading a dashboard: concise summary, concrete evidence, explicit risks, and data-quality limits.
"""


def react_analyst_decision(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    llm_client: LlmClient,
    retry_policy: RetryPolicy | None = None,
    portfolio: dict | None = None,
    news_provider=None,
    human_input: list[str] | None = None,
    human_intent: dict[str, Any] | None = None,
) -> TradingDecision:
    retry_policy = retry_policy or RetryPolicy(max_attempts=2)
    metadata = llm_metadata(llm_client)
    logger.info("analyst.llm.start ticker=%s mode=react", snapshot.ticker)

    observations, blocked_tools = _tool_observations(snapshot, recent_entries, llm_client, news_provider)
    for tool_name in blocked_tools:
        logger.warning("react.tool.blocked ticker=%s tool=%s", snapshot.ticker, tool_name)
    used_tools = [observation["tool"] for observation in observations]
    for tool_name in used_tools:
        if tool_name not in blocked_tools:
            logger.info("react.tool.call ticker=%s tool=%s", snapshot.ticker, tool_name)
    logger.info("react.tools.completed ticker=%s tools=%s", snapshot.ticker, ",".join(used_tools) or "none")

    errors: list[str] = []
    position_sizing = build_position_sizing_context(snapshot, portfolio)
    parsed_human_intent = parse_human_intent(human_input or [])
    human_intent_payload = human_intent or parsed_human_intent.to_dict()
    final_payload = json.dumps(
        {
            "snapshot": snapshot_payload(snapshot),
            "recent_journal": compact_recent_entries(recent_entries, ticker=snapshot.ticker),
            "human_input": human_input or [],
            "human_intent": human_intent_payload,
            "position_sizing": position_sizing,
            "tool_observations": observations,
        },
        default=str,
    )

    def call_and_parse():
        nonlocal metadata
        raw = llm_client.complete_json(FINAL_PROMPT, final_payload)
        metadata = llm_metadata(llm_client)
        return parse_analyst_output(raw)

    def on_failure(attempt: int, error: Exception) -> None:
        errors.append(f"react analyst structured output retry {attempt}: {error}")
        logger.warning("react.validation.retry ticker=%s attempt=%s error=%s", snapshot.ticker, attempt, error)

    try:
        parsed = retry_policy.run(call_and_parse, on_failure=on_failure)
    except Exception:
        logger.error("react.validation.failed ticker=%s retries=%s", snapshot.ticker, len(errors))
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=0.2,
            rationale=snapshot_grounded_hold_rationale(
                snapshot,
                "ReAct analyst output failed structured validation after retry",
            ),
            used_data_sources=[*snapshot.data_sources, "react_tools"],
            guardrails_triggered=["guardrail:invalid_react_analyst_output", *errors],
            reflection="ReAct tools used: " + (", ".join(used_tools) if used_tools else "none"),
            llm_metadata=metadata,
            rationale_details=_with_tool_audit(
                safe_rationale_details(snapshot, "Safe HOLD because ReAct output failed validation.", errors),
                observations,
            ),
        )

    mismatches = rationale_snapshot_mismatches(parsed.rationale, snapshot)
    if mismatches:
        logger.warning("react.rationale_mismatch ticker=%s mismatches=%s", snapshot.ticker, "|".join(mismatches))
        return TradingDecision(
            ticker=snapshot.ticker,
            action="HOLD",
            quantity=0,
            confidence=min(parsed.confidence, 0.25),
            rationale=snapshot_grounded_hold_rationale(
                snapshot,
                "ReAct rationale conflicted with validated market snapshot: " + "; ".join(mismatches),
            ),
            used_data_sources=[*(parsed.used_data_sources or snapshot.data_sources), "react_tools"],
            guardrails_triggered=["guardrail:react_rationale_snapshot_mismatch", *mismatches],
            reflection="ReAct tools used: " + (", ".join(used_tools) if used_tools else "none"),
            llm_metadata=metadata,
            rationale_details=_with_tool_audit(
                safe_rationale_details(snapshot, "Safe HOLD because ReAct rationale contradicted validated data.", mismatches),
                observations,
            ),
        )

    human_trade_decision = _human_trade_decision(snapshot, parsed, parsed_human_intent, position_sizing, metadata, observations)
    if human_trade_decision is not None:
        return human_trade_decision

    rationale_details = _with_tool_audit(parsed.rationale_details, observations)
    return TradingDecision(
        ticker=snapshot.ticker,
        action=parsed.action,
        quantity=parsed.quantity,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        used_data_sources=[*(parsed.used_data_sources or snapshot.data_sources), "react_tools"],
        reflection="ReAct tools used: " + (", ".join(used_tools) if used_tools else "none"),
        llm_metadata=metadata,
        rationale_details=rationale_details,
    )


def _tool_observations(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    llm_client: LlmClient,
    news_provider=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    invoke_tools = getattr(llm_client, "invoke_tools", None)
    if callable(invoke_tools):
        payload = _tool_payload(snapshot, recent_entries, news_provider)
        try:
            tools = langchain_market_context_tools(snapshot, recent_entries)
            if news_provider is not None:
                tools.append(langchain_news_search_tool(news_provider))
            observations, _metadata = invoke_tools(TOOL_PROMPT, payload, tools)
            if observations:
                return observations, [item["tool"] for item in observations if item.get("error")]
        except Exception as error:
            logger.warning("react.langchain_tools.failed ticker=%s error=%s", snapshot.ticker, error)

    tool_calls = _request_tool_calls(snapshot, recent_entries, llm_client, news_provider)
    return run_market_context_tools(snapshot, recent_entries, tool_calls, news_provider)


def _request_tool_calls(
    snapshot: MarketSnapshot,
    recent_entries: list[JournalEntry],
    llm_client: LlmClient,
    news_provider=None,
) -> list[dict[str, Any]]:
    payload = _tool_payload(snapshot, recent_entries, news_provider)
    try:
        planner = getattr(llm_client, "complete_tool_plan", None)
        if callable(planner):
            parsed = planner(TOOL_PROMPT, payload, market_context_tool_schemas())
        else:
            raw = llm_client.complete_json(TOOL_PROMPT, payload)
            parsed = json.loads(raw)
    except Exception as error:
        logger.warning("react.tool_plan.failed ticker=%s error=%s", snapshot.ticker, error)
        return [{"name": "inspect_snapshot", "args": {}}]
    calls = parsed.get("tool_calls", [])
    if not isinstance(calls, list):
        return [{"name": "inspect_snapshot", "args": {}}]
    return [call for call in calls[:3] if isinstance(call, dict)]


def _tool_payload(snapshot: MarketSnapshot, recent_entries: list[JournalEntry], news_provider=None) -> str:
    return json.dumps(
        {
            "snapshot": snapshot_payload(snapshot),
            "recent_journal_count": len(recent_entries),
            "available_tools": _available_tool_names(news_provider),
        },
        default=str,
    )


def _available_tool_names(news_provider=None) -> list[str]:
    return [
        "inspect_snapshot",
        "explain_technical_signals",
        "summarize_news_urls",
        "read_news_url",
        *([] if news_provider is None else ["search_market_news"]),
    ]


def _with_tool_audit(details: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(details or {})
    enriched["tool_audit"] = {
        "tools_used": [item.get("tool") for item in observations if item.get("tool")],
        "tool_failures": [item for item in observations if item.get("error") or ((item.get("observation") or {}).get("status") == "failed")],
    }
    return enriched


def _human_trade_decision(
    snapshot: MarketSnapshot,
    parsed,
    human_intent,
    position_sizing: dict[str, Any],
    metadata: dict[str, Any],
    observations: list[dict[str, Any]],
) -> TradingDecision | None:
    max_buy = int(position_sizing.get("max_buy_quantity") or 0)
    max_sell = int(position_sizing.get("max_sell_quantity") or 0)
    action: str | None = None
    quantity = 0
    summary: str | None = None
    guardrail: str | None = None

    if human_intent.requested_action == "BUY" and _human_intent_applies_to_ticker(human_intent, snapshot.ticker) and max_buy > 0:
        action = "BUY"
        quantity = 1
        summary = "Human input requested BUY and risk limits allow a conservative purchase."
        guardrail = "guardrail:human_buy_intent"
    elif human_intent.requested_action == "SELL" and _human_intent_applies_to_ticker(human_intent, snapshot.ticker) and max_sell > 0:
        action = "SELL"
        quantity = max_sell
        summary = "Human input requested SELL and a current position exists."
        guardrail = "guardrail:human_sell_intent"
    elif "position_sweep" in human_intent.intents and max_sell > 0:
        action = "SELL"
        quantity = max_sell
        summary = _human_position_sweep_summary(snapshot.ticker, human_intent)
        guardrail = "guardrail:human_position_sweep_sell"

    if action is None or quantity <= 0 or summary is None or guardrail is None:
        return None
    details = _with_tool_audit(parsed.rationale_details, observations)
    details.update(
        {
            "summary": summary,
            "human_intent": human_intent.to_dict(),
            "human_impact_topic": human_intent.impact_topic,
            "human_selected_ticker": snapshot.ticker,
            "position_sizing": position_sizing,
        }
    )
    return TradingDecision(
        ticker=snapshot.ticker,
        action=action,
        quantity=quantity,
        confidence=max(parsed.confidence, 0.65),
        rationale=(
            f"{summary} For {snapshot.ticker}, overriding draft {parsed.action} to {action} "
            f"within risk limits (max_buy_quantity={max_buy}, max_sell_quantity={max_sell})."
        ),
        used_data_sources=[*(parsed.used_data_sources or snapshot.data_sources), "human_intent", "react_tools"],
        guardrails_triggered=[guardrail],
        reflection="ReAct tools used: " + (", ".join(item["tool"] for item in observations if item.get("tool")) or "none"),
        llm_metadata=metadata,
        rationale_details=details,
    )


def _human_intent_applies_to_ticker(human_intent, ticker: str) -> bool:
    return not human_intent.tickers or ticker.upper() in human_intent.tickers


def _human_position_sweep_summary(ticker: str, human_intent) -> str:
    topic = human_intent.impact_topic or "the user's broad risk topic"
    return (
        f"Human input requested a broad SELL sweep for open positions potentially impacted by {topic}. "
        f"Selected {ticker.upper()} because it is an open position in the human-instruction sweep."
    )
