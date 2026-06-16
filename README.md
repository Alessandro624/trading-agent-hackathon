# Autonomous Trading Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Alpaca Market](https://img.shields.io/badge/Trading-Alpaca_Market-yellow.svg)](https://alpaca.markets/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


{Name} is an Autonomous Trading Agent, built around the **LangChain** and **Alpaca Markets** Framework, that is able to sell, or hold decisions on stock tickers, based on real-time news.

---

## Overview 

This project implements a single LangChain agent capable of:
 
- **Buying** and **selling** stock tickers through the Alpaca Paper Trading API
- **Holding** positions when market conditions are uncertain
- **Gathering news** via NewsAPI to inform trading decisions
- Acting autonomously based on a defined strategy loop

The agent is designed for research and experimentation in a **paper trading** (simulated, risk-free) environment.


## Project Structure

```text
trading-agent-hackathon/
├── trading-agent/
│   ├── adapters/
│   │   ├── alpaca_broker.py            # Alpaca Broker, able to get Portfolio info
│   │   └── alpaca_market_data.py       # Alpaca Marker Data, able to get tickers' prices
│   ├── scripts/
│   │   └── test_connection.py          # Simple Script to test the Connection with the Alpaca Market API
│   └── utils/
│       ├── __init__.py                 # Re-exports all config symbols
│       ├── config.py                   # Configuration Info
│       └── logger.py                   # Logger Info
├── pyproject.toml                      # v0.1.0, alpaca-py>=0.30, project scripts
└── .env                                # API keys (not committed)
```

---