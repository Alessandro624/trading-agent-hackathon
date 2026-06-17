from __future__ import annotations

import re

from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from trading_agent.utils.config import require_env
from trading_agent.core.data_hygiene import clean_text


class TickerProvider:

    def __init__(self):
        self.trading_client: TradingClient = TradingClient(api_key=require_env("ALPACA_API_KEY"), secret_key=require_env("ALPACA_SECRET_KEY"))

    def _parse_watchlist(self, value: str | None) -> list[str]:
        if not value:
            return []

        raw_items = value.split(",") if isinstance(value, str) else list(value)
        symbols: list[str] = []
        for item in raw_items:
            symbol = clean_text(item, max_chars=16).upper()
            if not symbol or not re.fullmatch(r"[A-Z]{1,5}", symbol):
                continue
            if symbol not in symbols:
                symbols.append(symbol)
        return symbols

    def _get_tickers(self, watchlist_params: Optional[str] = None) -> set[str]:

        request_params: GetAssetsRequest = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)

        available_assets = self.trading_client.get_all_assets(request_params)

        if not available_assets:
            return set()

        asset_symbols: set[str] = set()

        for a in available_assets:
            if a.symbol and a.tradable:
                asset_symbols.add(a.symbol)

        watchlist_symbols: list[str] = self._parse_watchlist(watchlist_params)
        if watchlist_symbols:
            return set.intersection(asset_symbols, watchlist_symbols)

        return asset_symbols

    def _get_stats_from_symbol(self, symbol: str) -> dict:
        return dict(name=symbol)

    def get_tickers_with_info(self, watchlist: Optional[str] = None) -> list[dict]:

        symbols = self._get_tickers(watchlist)
        info: list = []

        for s in symbols:
            info.append(self._get_stats_from_symbol(s))

        return info

    def pick_best_by_metrics(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        seen: set[str] = set()
        unique: list[str] = []
        for symbol in candidates:
            symbol = symbol.upper().strip()
            if re.fullmatch(r"[A-Z]{1,5}", symbol) and symbol not in seen:
                seen.add(symbol)
                unique.append(symbol)
        if not unique:
            return None
        try:
            watchlist_param = ",".join(unique)
            info_list = self.get_tickers_with_info(watchlist_param)
        except Exception:
            return unique[0]
        if not info_list:
            return unique[0]
        first = info_list[0]
        if isinstance(first, dict):
            name = first.get("name") or first.get("symbol")
            if isinstance(name, str) and name:
                return name.upper()
        return unique[0]
