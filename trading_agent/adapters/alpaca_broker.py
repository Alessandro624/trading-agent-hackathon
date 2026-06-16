from __future__ import annotations

import os

from trading_agent.utils import require_env


class AlpacaBrokerClient:
    def __init__(self) -> None:
        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY"),
            paper=paper,
        )

    def get_portfolio(self) -> dict:
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "positions": [
                {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "market_value": float(position.market_value),
                }
                for position in positions
            ],
        }
