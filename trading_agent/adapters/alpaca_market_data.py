from __future__ import annotations

from datetime import datetime, timedelta, timezone

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

    def get_closes(self, ticker: str, limit: int = 60) -> list[float]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        symbol = ticker.upper()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(limit * 2, 90))
        bars = self.client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=limit,
                feed=DataFeed.IEX,
            )
        )
        frame = bars.df
        if frame.empty:
            return []
        if "symbol" in frame.index.names:
            frame = frame.xs(symbol, level="symbol")
        return [float(value) for value in frame["close"].tail(limit).tolist()]
