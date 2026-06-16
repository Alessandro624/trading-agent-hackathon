# Autonomous Trading Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Alpaca Market](https://img.shields.io/badge/Trading-Alpaca_Market-yellow.svg)](https://alpaca.markets/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


{Name} is an Autonomous Trading System, built around the **LangChain** and **Alpaca Markets** Framework, that is able to sell, or hold decisions on stock tickers, based on real-time news.

---

## Overview 

This project implements a system, based on LangChain agents, capable of:
 
- **Buying** and **selling** stock tickers through the Alpaca Paper Trading API
- **Holding** positions when market conditions are uncertain
- **Gathering news** via NewsAPI to inform trading decisions
- Acting autonomously based on a defined strategy loop

Each agent is designed for research and experimentation in a **paper trading** (simulated, risk-free) environment.


---

## Architecture

The Systems provides two different modes:
 - The Single Agent Flow (ReAct Analyst), an all-in-one Agent that execute all the workflow by itself
 - The Multi-Agent Flow, a collection a Specialised Agents built for Scouting, Risk Assessment, Decision Introspection and Trade

---


## Project Structure

```text
trading-agent-hackathon/
├── trading-agent/
│   │
│   ├── adapters/
│   │   ├── alpaca_broker.py            # Alpaca Broker, able to get Portfolio info
│   │   ├── alpaca_market_data.py       # Alpaca Marker Data, able to get tickers' prices
│   │   └── news_provider.py            # News API Helper for currents available news
│   │
│   ├── agents/
│   │   ├── decision_manager.py         # Decision Handling and Drafting of the Action to perform
│   │   ├── executor.py                 # Main Action Executor based on previous FeedBacks
│   │   ├── news_analyst.py             # Real-time News Analyst, able to retrieve/filter available news 
│   │   ├── react_analyst.py            # Single-agent Workflow Executor
│   │   ├── reflection.py               # Intrespection Agent, evaluates the decision using an Adversarial approach
│   │   ├── risk_manager.py             # Risk Assessment Expert
│   │   ├── scout.py                    # Single-agent Scout for Portfolio Information (Deterministic)
│   │   └── technical_analyst.py        # Portfolio Tracker and Expert of the Environment
│   │
│   ├── core/
│   │   ├── confidence_policy.py        # Confidence Policies on News, Prices (and others) Fidelity
│   │   ├── llm_guardrails.py           # LLM Guardrails for Output Checking
│   │   ├── models.py                   # Base Custom Models
│   │   ├── news_query.py               # Base News Template Query and Search Parameters
│   │   ├── ports.py                    # Custom Inferfaces
│   │   ├── retry_policy.py             # Retry Policy definition
│   │   └── risk_policy.py              # Risk Policy Definition
│   │
│   ├── journal/
│   │   ├── dashboard.py                # Dashboard for a single Journal
│   │   ├── memory.py                   # 
│   │   ├── run_manager.py              # Single Run Manager for Saving and Recalling
│   │   ├── store.py                    # Journal Storing and Recalling Model
│   │   └── terminal.py                 # Terminal Inferface API
│   │
│   ├── pipeline/
│   │   └── graph.py                    # Pipeline Visualization
│   │
|   ├── scripts/
│   │   ├── run_agent.py                # Base Script to run an Agent
│   │   └── test_connection.py          # Simple Script to test the Connection with the Alpaca Market API
|   |
│   ├── technicals/
│   │   └── indicators.py               # Evaluation Metrics and Procedures
│   │
│   ├── tools/
│   │   └── context_tools.py            # Tools Available for Each step of the Pipeline
│   │
│   └── utils/
│       ├── __init__.py                 # Re-exports all config symbols
│       ├── config.py                   # Configuration Info
│       ├── http.py                     # Https Utils to request external files
│       ├── llm_clients.py              # Availble Clients Config (OpenAI, Ollama)
│       ├── llm_metadata.py             # Availble Clients Metadata
│       ├── logger.py                   # Logger Info
│       └── portfolio.py                # Portfolio Availablity Checks
│
├── pyproject.toml                      # v0.1.0, alpaca-py>=0.30, project scripts
└── .env                                # API keys (not committed)
```

---