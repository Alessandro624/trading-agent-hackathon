from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus

from trading_agent.utils.config import require_env


@dataclass
class WorldConfig:
    asset_class: AssetClass = field(default=AssetClass.US_EQUITY)
    exchages: list[AssetExchange] = field(default_factory=list)


class TickerProvider:

    def __init__(self):
        self.default_world: WorldConfig = WorldConfig()
        self.current_world: Optional[WorldConfig] = None

        self.trading_client: TradingClient = TradingClient(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY")
        )


    def _define_world(self, watchlist_params: Optional[str] = None) -> None:
        return None


    def _get_tickers(self, watchlist_params: Optional[str] = None) -> set[str]:

        if watchlist_params:
            self._define_world(watchlist_params)

        request_params: GetAssetsRequest = GetAssetsRequest(
            asset_class = self.current_world.asset_class if self.current_world else self.default_world.asset_class,
            status=AssetStatus.ACTIVE
        )

        available_assets = self.trading_client.get_all_assets(request_params)

        if not available_assets:
            return set()
        
        asset_symbols: set[str] = set()
        
        for a in available_assets:
            if a.symbol:
                asset_symbols.add(a.symbol)

        return asset_symbols
    

    def _get_stats_from_symbol(self, symbol: str) -> dict:
        return dict(name = symbol)


    def get_tickers_with_info(self, watchlist: Optional[str] = None) -> list[dict]:

        symbols = self._get_tickers(watchlist)
        info: list = []

        for s in symbols:
            info.append(self._get_stats_from_symbol(s))

        return info


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    t = TickerProvider()

    tickers = t.get_tickers_with_info()

    print(tickers[:5])

    