from __future__ import annotations

from typing import Any, Literal, TypedDict

from trading_agent.core import BrokerClient, LlmClient, MarketDataProvider, NewsProvider, utc_now_iso
from trading_agent.journal import JournalStore
from trading_agent.agents import (
    assess_risk,
    decide_from_opinions,
    execute_decision,
    news_opinion,
    react_analyst_decision,
    reflect_decision,
    scout_snapshot,
    technical_opinion,
)
from trading_agent.utils import get_logger, safe_portfolio_snapshot

logger = get_logger("pipeline")


class State(TypedDict, total=False):
    ticker: str
    recent_entries: list
    snapshot: Any
    technical_opinion: Any
    news_opinion: Any
    risk_assessment: Any
    draft_decision: Any
    final_decision: Any
    execution_result: Any
    journal_entry: Any
    cycle_started_at: str
    human_input: list[str]


def build_graph(
    mode: Literal["single_agent", "multi_agent"],
    market_data: MarketDataProvider,
    news_provider: NewsProvider,
    llm_client: LlmClient,
    journal: JournalStore,
    broker: BrokerClient | None = None,
):
    if broker is None:
        raise ValueError("broker is required for single_agent and multi_agent graphs")
    if mode == "multi_agent":
        return build_multi_agent_graph(market_data, news_provider, llm_client, broker, journal)
    return build_single_agent_graph(market_data, news_provider, llm_client, broker, journal)


def build_single_agent_graph(
    market_data: MarketDataProvider,
    news_provider: NewsProvider,
    llm_client: LlmClient,
    broker: BrokerClient,
    journal: JournalStore,
):
    END, StateGraph = _load_langgraph()
    graph = StateGraph(State)
    _add_scout_node(graph, market_data, news_provider, journal)

    def analyst_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "analyst",
            lambda: _single_agent_analysis(state, broker, llm_client, news_provider),
        )

    graph.add_node("analyst", analyst_node)
    _add_common_tail_nodes(graph, market_data, llm_client, broker, journal)
    graph.set_entry_point("scout")
    graph.add_edge("scout", "analyst")
    graph.add_edge("analyst", "reflection")
    graph.add_edge("reflection", "executor")
    graph.add_edge("executor", "journal")
    graph.add_edge("journal", END)
    return graph.compile()


def build_multi_agent_graph(
    market_data: MarketDataProvider,
    news_provider: NewsProvider,
    llm_client: LlmClient,
    broker: BrokerClient,
    journal: JournalStore,
):
    END, StateGraph = _load_langgraph()
    graph = StateGraph(State)
    _add_scout_node(graph, market_data, news_provider, journal)

    def technical_node(state: State) -> State:
        return _run_stage(state, journal, "technical_analyst", lambda: state.update({"technical_opinion": technical_opinion(state["snapshot"])}) or state)

    def news_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "news_analyst",
            lambda: state.update(
                {
                    "news_opinion": news_opinion(
                        state["snapshot"],
                        state.get("recent_entries", []),
                        llm_client,
                        news_provider,
                    )
                }
            )
            or state,
        )

    def risk_node(state: State) -> State:
        return _run_stage(state, journal, "risk_manager", lambda: _risk_analysis(state, broker))

    def decision_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "decision_manager",
            lambda: state.update(
                {
                    "draft_decision": decide_from_opinions(
                        state["snapshot"],
                        state.get("recent_entries", []),
                        state["technical_opinion"],
                        state["news_opinion"],
                        state["risk_assessment"],
                        llm_client,
                        human_input=state.get("human_input", []),
                    )
                }
            )
            or state,
        )

    graph.add_node("technical_analyst", technical_node)
    graph.add_node("news_analyst", news_node)
    graph.add_node("risk_manager", risk_node)
    graph.add_node("decision_manager", decision_node)
    _add_common_tail_nodes(graph, market_data, llm_client, broker, journal)
    graph.set_entry_point("scout")
    graph.add_edge("scout", "technical_analyst")
    graph.add_edge("technical_analyst", "news_analyst")
    graph.add_edge("news_analyst", "risk_manager")
    graph.add_edge("risk_manager", "decision_manager")
    graph.add_edge("decision_manager", "reflection")
    graph.add_edge("reflection", "executor")
    graph.add_edge("executor", "journal")
    graph.add_edge("journal", END)
    return graph.compile()


def _add_scout_node(
    graph,
    market_data: MarketDataProvider,
    news_provider: NewsProvider,
    journal: JournalStore,
) -> None:
    def scout_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "scout",
            lambda: state.update({"snapshot": scout_snapshot(state["ticker"], market_data, news_provider)}) or state,
        )

    graph.add_node("scout", scout_node)


def _add_common_tail_nodes(
    graph,
    market_data: MarketDataProvider,
    llm_client: LlmClient,
    broker: BrokerClient,
    journal: JournalStore,
) -> None:
    def reflection_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "reflection",
            lambda: state.update({"final_decision": reflect_decision(state["snapshot"], state["draft_decision"], llm_client)})
            or state,
        )

    def executor_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "executor",
            lambda: _execute_with_price(state, market_data, broker),
        )

    def journal_node(state: State) -> State:
        return _run_stage(
            state,
            journal,
            "journal",
            lambda: state.update(
                {
                    "journal_entry": journal.append(
                        state["snapshot"],
                        state["final_decision"],
                        state["execution_result"],
                        draft_decision=state.get("draft_decision"),
                        cycle_started_at=state.get("cycle_started_at"),
                        human_input=state.get("human_input", []),
                    )
                }
            )
            or state,
        )

    graph.add_node("reflection", reflection_node)
    graph.add_node("executor", executor_node)
    graph.add_node("journal", journal_node)


def run_cycle(
    agent,
    ticker: str,
    recent_entries: list,
    journal: JournalStore | None = None,
    cycle_started_at: str | None = None,
    human_input: list[str] | None = None,
) -> State:
    symbol = ticker.upper()
    started_at = cycle_started_at or utc_now_iso()
    human_notes = human_input or []
    logger.info("cycle.start ticker=%s started_at=%s", symbol, started_at)
    if journal is not None:
        journal.append_stage(
            ticker=symbol,
            stage="cycle",
            status="started",
            message="Cycle started.",
            cycle_started_at=started_at,
            details={"human_input": human_notes},
        )
    try:
        state = agent.invoke(
            {
                "ticker": symbol,
                "recent_entries": recent_entries,
                "cycle_started_at": started_at,
                "human_input": human_notes,
            }
        )
    except Exception as error:
        logger.error("cycle.failed ticker=%s error=%s", symbol, error)
        if journal is not None:
            journal.append_stage(
                ticker=symbol,
                stage="cycle",
                status="failed",
                message=str(error),
                cycle_started_at=started_at,
            )
        raise
    entry = state.get("journal_entry")
    if entry:
        logger.info("journal.write ticker=%s outcome=%s", symbol, entry.outcome)
        logger.info("cycle.end ticker=%s action=%s outcome=%s", symbol, entry.action, entry.outcome)
    return state


def _run_stage(state: State, journal: JournalStore, stage: str, action) -> State:
    ticker = state["ticker"]
    started_at = state.get("cycle_started_at")
    logger.info("cycle.stage.start ticker=%s stage=%s", ticker, stage)
    journal.append_stage(ticker=ticker, stage=stage, status="started", message=f"{stage} started.", cycle_started_at=started_at)
    try:
        next_state = action()
    except Exception as error:
        logger.error("cycle.stage.failed ticker=%s stage=%s error=%s", ticker, stage, error)
        journal.append_stage(ticker=ticker, stage=stage, status="failed", message=str(error), cycle_started_at=started_at)
        raise
    journal.append_stage(
        ticker=ticker,
        stage=stage,
        status="completed",
        message=f"{stage} completed.",
        cycle_started_at=started_at,
        snapshot=next_state.get("snapshot"),
    )
    logger.info("cycle.stage.completed ticker=%s stage=%s", ticker, stage)
    return next_state


def _single_agent_analysis(
    state: State,
    broker: BrokerClient,
    llm_client: LlmClient,
    news_provider: NewsProvider,
) -> State:
    portfolio = _portfolio_for_sizing(broker, state["ticker"])
    state["draft_decision"] = react_analyst_decision(
        state["snapshot"],
        state.get("recent_entries", []),
        llm_client,
        portfolio=portfolio,
        news_provider=news_provider,
        human_input=state.get("human_input", []),
    )
    return state


def _risk_analysis(state: State, broker: BrokerClient) -> State:
    portfolio = _portfolio_for_sizing(broker, state["ticker"])
    state["risk_assessment"] = assess_risk(state["snapshot"], portfolio)
    return state


def _execute_with_price(state: State, market_data: MarketDataProvider, broker: BrokerClient) -> State:
    order_price = _current_order_price(market_data, state["ticker"], state["snapshot"].price)
    state["execution_result"] = execute_decision(
        state["final_decision"],
        broker,
        estimated_price=order_price,
    )
    return state


def _portfolio_for_sizing(broker: BrokerClient, ticker: str) -> dict | None:
    portfolio = safe_portfolio_snapshot(broker)
    if "portfolio_error" in portfolio:
        logger.warning("portfolio.sizing.fail ticker=%s error=%s", ticker, portfolio["portfolio_error"])
        return None
    logger.info("portfolio.sizing.ok ticker=%s", ticker)
    return portfolio


def _current_order_price(market_data: MarketDataProvider, ticker: str, fallback_price: float | None) -> float | None:
    try:
        payload = market_data.get_price(ticker)
        price = float(payload["price"])
        if price > 0:
            logger.info("order.price.refresh.ok ticker=%s price=%.2f", ticker, price)
            return price
    except Exception as error:
        logger.warning("order.price.refresh.fail ticker=%s error=%s", ticker, error)
    return fallback_price


def _load_langgraph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as error:
        raise RuntimeError("Install langgraph to use the pipeline graph.") from error
    return END, StateGraph
