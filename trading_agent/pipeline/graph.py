from __future__ import annotations

from typing import Any, Literal, TypedDict

from trading_agent.core import BrokerClient, LlmClient, MarketDataProvider, NewsProvider
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
    _add_scout_node(graph, market_data, news_provider)

    def analyst_node(state: State) -> State:
        portfolio = _portfolio_for_sizing(broker, state["ticker"])
        state["draft_decision"] = react_analyst_decision(
            state["snapshot"],
            state.get("recent_entries", []),
            llm_client,
            portfolio=portfolio,
            news_provider=news_provider,
        )
        return state

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
    _add_scout_node(graph, market_data, news_provider)

    def technical_node(state: State) -> State:
        state["technical_opinion"] = technical_opinion(state["snapshot"])
        return state

    def news_node(state: State) -> State:
        state["news_opinion"] = news_opinion(
            state["snapshot"],
            state.get("recent_entries", []),
            llm_client,
            news_provider,
        )
        return state

    def risk_node(state: State) -> State:
        portfolio = _portfolio_for_sizing(broker, state["ticker"])
        state["risk_assessment"] = assess_risk(state["snapshot"], portfolio)
        return state

    def decision_node(state: State) -> State:
        state["draft_decision"] = decide_from_opinions(
            state["snapshot"],
            state.get("recent_entries", []),
            state["technical_opinion"],
            state["news_opinion"],
            state["risk_assessment"],
            llm_client,
        )
        return state

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


def _add_scout_node(graph, market_data: MarketDataProvider, news_provider: NewsProvider) -> None:
    def scout_node(state: State) -> State:
        state["snapshot"] = scout_snapshot(state["ticker"], market_data, news_provider)
        return state

    graph.add_node("scout", scout_node)


def _add_common_tail_nodes(
    graph,
    market_data: MarketDataProvider,
    llm_client: LlmClient,
    broker: BrokerClient,
    journal: JournalStore,
) -> None:
    def reflection_node(state: State) -> State:
        state["final_decision"] = reflect_decision(state["snapshot"], state["draft_decision"], llm_client)
        return state

    def executor_node(state: State) -> State:
        order_price = _current_order_price(market_data, state["ticker"], state["snapshot"].price)
        state["execution_result"] = execute_decision(
            state["final_decision"],
            broker,
            estimated_price=order_price,
        )
        return state

    def journal_node(state: State) -> State:
        state["journal_entry"] = journal.append(
            state["snapshot"],
            state["final_decision"],
            state["execution_result"],
            draft_decision=state.get("draft_decision"),
        )
        return state

    graph.add_node("reflection", reflection_node)
    graph.add_node("executor", executor_node)
    graph.add_node("journal", journal_node)


def run_cycle(
    agent,
    ticker: str,
    recent_entries: list,
) -> State:
    symbol = ticker.upper()
    logger.info("cycle.start ticker=%s", symbol)
    state = agent.invoke({"ticker": symbol, "recent_entries": recent_entries})
    entry = state.get("journal_entry")
    if entry:
        logger.info("journal.write ticker=%s outcome=%s", symbol, entry.outcome)
        logger.info("cycle.end ticker=%s action=%s outcome=%s", symbol, entry.action, entry.outcome)
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
