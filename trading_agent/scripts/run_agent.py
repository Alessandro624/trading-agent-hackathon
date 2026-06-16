from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from trading_agent.adapters import AlpacaBrokerClient, AlpacaMarketDataProvider, NewsApiProvider
from trading_agent.core import HumanInputStore, HumanInstruction, parse_watchlist, plan_human_instructions, select_ticker
from trading_agent.journal import JournalStore, RunContext, create_run_context, print_cycle_log, render_dashboard
from trading_agent.pipeline import build_graph, run_cycle
from trading_agent.utils import FallbackLlmClient, OllamaJsonClient, OpenAiJsonClient, configure_logging, safe_portfolio_snapshot


def _build_adapters():
    llm_client = FallbackLlmClient(primary=OpenAiJsonClient(), fallback=OllamaJsonClient())
    return AlpacaMarketDataProvider(), NewsApiProvider(), llm_client, AlpacaBrokerClient()


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
    ticker_mode: str = "fixed",
) -> RunContext:
    configure_logging()
    context = create_run_context(base_dir, ticker, resume_from=resume_from)
    journal = JournalStore(context.journal_path)
    human_input_path = human_input or context.human_input_path
    human_input_store = HumanInputStore(human_input_path, _human_input_cursor_path(human_input, context.human_input_cursor_path))
    market_data, news_provider, llm_client, broker = _build_adapters()
    recent_entries = journal.read_all()[-memory_limit:] if context.resumed_from else []
    watchlist_symbols = parse_watchlist(watchlist or ticker)
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
        if not instruction_queue:
            instruction_queue.extend(
                plan_human_instructions(
                    human_input_store.read_next_note().notes,
                    watchlist=watchlist_symbols,
                    portfolio=portfolio_snapshot,
                    fallback_ticker=ticker,
                    llm_client=llm_client,
                )
            )
        active_instruction = instruction_queue.pop(0) if instruction_queue else None
        human_notes = [active_instruction.note] if active_instruction else []
        selected_ticker = ticker
        if active_instruction and active_instruction.target_ticker:
            selected_ticker = active_instruction.target_ticker
            journal.append_stage(
                ticker=selected_ticker,
                stage="ticker_selector",
                status="completed",
                message=f"Human instruction selected {selected_ticker}.",
                details={
                    "reason": active_instruction.reason,
                    "watchlist": watchlist_symbols,
                    "mentioned_tickers": [selected_ticker],
                    "human_input": human_notes,
                    "human_instruction_index": active_instruction.sequence_index,
                    "human_instruction_total": active_instruction.sequence_total,
                    "human_instruction_status": "active",
                    "human_resolver_source": _human_resolver_source(active_instruction),
                    "human_resolver_confidence": active_instruction.resolver_confidence,
                    "human_resolver_rationale": active_instruction.resolver_rationale,
                    "human_resolver_topic": active_instruction.resolver_topic,
                },
            )
        elif ticker_mode == "auto":
            selection = select_ticker(
                watchlist_symbols,
                human_input=human_notes,
                portfolio=portfolio_snapshot,
                cycle_index=cycle_index,
                fallback=ticker,
            )
            selected_ticker = selection.ticker
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
                },
            )
        state = run_cycle(
            agent=agent,
            ticker=selected_ticker,
            recent_entries=recent_entries,
            journal=journal,
            human_input=human_notes,
        )
        if state.get("journal_entry"):
            recent_entries.append(state["journal_entry"])
            print_cycle_log(cycle_index + 1, state["journal_entry"])
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
    return "deterministic"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--mode", choices=["single_agent", "multi_agent"], default="single_agent")
    parser.add_argument("--base-dir", type=Path, default=Path("journal_runs"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--memory-limit", type=int, default=3)
    parser.add_argument("--human-input", type=Path, default=None)
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--ticker-mode", choices=["fixed", "auto"], default="fixed")
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
        args.ticker_mode,
    )
    if context.resumed_from:
        print(f"resumed_from: {context.resumed_from}")
    print(f"journal: {context.journal_path}")
    print(f"dashboard: {context.dashboard_path}")
    print(f"human_input: {args.human_input or context.human_input_path}")


if __name__ == "__main__":
    main()
