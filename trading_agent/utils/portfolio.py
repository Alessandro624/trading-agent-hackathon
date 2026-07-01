from __future__ import annotations

from trading_agent.core import BrokerClient


def safe_portfolio_snapshot(broker: BrokerClient) -> dict:
    try:
        return broker.get_portfolio()
    except Exception as error:
        return {"portfolio_error": str(error)}
