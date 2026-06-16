from __future__ import annotations

from trading_agent.utils import require_env


class AlpacaMarketDataProvider:
    def __init__(self) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        self.client = StockHistoricalDataClient(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY"),
        )

    def get_price(self, ticker: str) -> dict:
        from alpaca.data.requests import StockLatestTradeRequest

        symbol = ticker.upper()
        trades = self.client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        trade = trades[symbol] if isinstance(trades, dict) else trades
        return {"ticker": symbol, "price": float(trade.price), "timestamp": trade.timestamp.isoformat()}
