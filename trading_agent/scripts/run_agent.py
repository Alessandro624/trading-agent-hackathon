from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from trading_agent.adapters import AlpacaBrokerClient, AlpacaMarketDataProvider, NewsApiProvider
from trading_agent.journal import JournalStore, RunContext, create_run_context, print_cycle_log, render_dashboard
from trading_agent.pipeline import build_graph, run_cycle
from trading_agent.utils import FallbackLlmClient, OllamaJsonClient, OpenAiJsonClient, configure_logging


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
) -> RunContext:
    configure_logging()
    context = create_run_context(base_dir, ticker, resume_from=resume_from)
    journal = JournalStore(context.journal_path)
    market_data, news_provider, llm_client, broker = _build_adapters()
    recent_entries = journal.read_all()[-memory_limit:] if context.resumed_from else []
    agent = build_graph(
        mode=mode,
        market_data=market_data,
        news_provider=news_provider,
        llm_client=llm_client,
        broker=broker,
        journal=journal,
    )

    for cycle_index in range(cycles):
        state = run_cycle(
            agent=agent,
            ticker=ticker,
            recent_entries=recent_entries,
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
    args = parser.parse_args()
    context = run_agent(
        args.ticker,
        args.cycles,
        args.interval,
        args.base_dir,
        args.mode,
        args.resume_from,
        args.memory_limit,
    )
    if context.resumed_from:
        print(f"resumed_from: {context.resumed_from}")
    print(f"journal: {context.journal_path}")
    print(f"dashboard: {context.dashboard_path}")


if __name__ == "__main__":
    main()
