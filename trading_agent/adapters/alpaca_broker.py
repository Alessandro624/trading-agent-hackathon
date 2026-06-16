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
                    "avg_entry_price": float(getattr(position, "avg_entry_price", 0) or 0),
                }
                for position in positions
            ],
        }

    def place_order(self, ticker: str, action: str, quantity: int) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self.client.submit_order(
            MarketOrderRequest(
                symbol=ticker.upper(),
                qty=quantity,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        return {"id": str(order.id), "status": _status_value(order.status)}

    def get_order(self, order_id: str) -> dict:
        order = self.client.get_order_by_id(order_id)
        filled_avg_price = getattr(order, "filled_avg_price", None)
        return {
            "id": str(order.id),
            "status": _status_value(order.status),
            "filled_avg_price": float(filled_avg_price) if filled_avg_price is not None else None,
        }


def _status_value(status) -> str:
    return str(getattr(status, "value", status)).lower()
