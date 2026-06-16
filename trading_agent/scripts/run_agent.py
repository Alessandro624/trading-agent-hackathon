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
from trading_agent.core import HumanInputStore, parse_watchlist, select_ticker
from trading_agent.journal import JournalStore, RunContext, create_run_context, print_cycle_log, render_dashboard
from trading_agent.pipeline import build_graph, run_cycle
from trading_agent.utils import FallbackLlmClient, OllamaJsonClient, OpenAiJsonClient, OpenRouterJsonClient, configure_logging, safe_portfolio_snapshot


def _build_adapters():
    import os

    primary_provider: str = os.getenv("PRIMARY_PROVIDER", "openai")
    llm_client: FallbackLlmClient

    if primary_provider == "openrouter":
        llm_client = FallbackLlmClient(primary=OpenRouterJsonClient(), fallback=OllamaJsonClient())
    else:
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
    human_input_store = HumanInputStore(human_input or context.human_input_path, context.human_input_cursor_path)
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

    for cycle_index in range(cycles):
        human_notes = human_input_store.read_new_notes().notes
        selected_ticker = ticker
        if ticker_mode == "auto":
            selection = select_ticker(
                watchlist_symbols,
                human_input=human_notes,
                portfolio=safe_portfolio_snapshot(broker),
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
