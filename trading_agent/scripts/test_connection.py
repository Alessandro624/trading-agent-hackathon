from __future__ import annotations

import argparse

from dotenv import load_dotenv

from trading_agent.adapters import AlpacaBrokerClient, AlpacaMarketDataProvider


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check Alpaca paper account and market-data connectivity.")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    broker = AlpacaBrokerClient()
    portfolio = broker.get_portfolio()

    print("Connected to Alpaca Paper Trading")
    print(f"Cash: ${portfolio.get('cash', 0):,.2f}")
    print(f"Portfolio value: ${portfolio.get('portfolio_value', 0):,.2f}")
    print(f"Open positions: {len(portfolio.get('positions') or [])}")

    try:
        price = AlpacaMarketDataProvider().get_price(args.ticker)
    except Exception as error:
        print(f"Market data check failed for {args.ticker.upper()}: {error}")
        return

    print(f"Latest {price['ticker']} price: ${price['price']:,.2f}")
    print(f"Price timestamp: {price['timestamp']}")


if __name__ == "__main__":
    main()
