# Meet Hack-a-Ton! Our Autonomous Trading Agentic System

> An autonomous, LLM-powered trading agent, built around the **LangChain** and **Alpaca Markets** Framework, that observes the market, reasons about opportunities, and acts — without requiring human intervention between cycles.

This project was conceived and carried out as part of ***Hackathon 2026***, held at **Unical** in Rende, Italy

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Alpaca Market](https://img.shields.io/badge/Trading-Alpaca_Market-yellow.svg)](https://alpaca.markets/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Overview](#overview)
- [The Problem We Solve](#the-problem-we-solve)
- [Architecture](#architecture)
- [How It Works — Phase by Phase](#how-it-works--phase-by-phase)
  - [Phase 1 — Scout: Data Collection](#phase-1--scout-data-collection)
  - [Phase 2 — Market Analysis](#phase-2--market-analysis)
  - [Phase 3 — Risk Manager](#phase-3--risk-manager)
  - [Phase 4 — Decision Manager](#phase-4--decision-manager)
  - [Phase 5 — Reflection](#phase-5--reflection)
  - [Phase 6 — Executor](#phase-6--executor)
  - [Phase 7 — Journal](#phase-7--journal)
- [The Autonomous Loop](#the-autonomous-loop)
- [Human Input Channel](#human-input-channel)
  - [Resolver, Confirmations, and External Discovery](#resolver-confirmations-and-external-discovery)
  - [Live Dashboard](#live-dashboard)
- [Project Structure](#project-structure)

---

## Overview

Every human trader naturally does three things: **observe** the market, **reason** about what to do, and **act** accordingly. Our system replicates this loop autonomously — continuously and without manual intervention — by orchestrating a pipeline of specialized AI agents, each with a distinct responsibility.

The result is a fully traceable, auditable, and configurable trading agent that runs on real market data while keeping capital safe through paper trading.

---

## The Problem We Solve

Large Language Models (LLMs) such as GPT or Claude do not have access to real-time market data. When asked directly, they may generate plausible-looking but inaccurate values — a phenomenon known as **hallucination**.

Our architecture addresses this at the root: **the LLM never relies on information from its training data when making decisions**. Every piece of data it receives has been freshly collected through external API calls, timestamped, and tagged with a confidence level. If a data source is unavailable, the system records the failure explicitly rather than generating substitute values.

---

## Architecture

The system is organized as a sequential, single-cycle pipeline with a feedback loop.

Each component is architecturally independent and communicates through structured data objects, making the system modular, testable, and easy to extend.

---

## How It Works — Phase by Phase

### Phase 1 — Scout: Data Collection

The cycle begins with the **Scout**, whose sole responsibility is gathering real-world data through external APIs:

| Data Source | What It Collects |
|---|---|
| **Alpaca** (broker API) | Current price and trading volume |
| **Technical Indicators** | RSI-14, MACD, SMA20, SMA50 |
| **NewsAPI** | Recent news headlines for the selected ticker |

Every piece of information is tagged with:

- a **confidence level** (`"high"`, `"medium"`, `"low"`, or `"none"`)
- a **timestamp**

If a data source is unavailable, the system records the failure and lowers the confidence accordingly — it never generates substitute values.

---

### Phase 2 — Market Analysis

The collected data is then processed by two independent analytical agents:

**Technical Analyst**
Evaluates market indicators and produces a structured `TechnicalOpinion`:

- Market trend: `"bullish"` / `"bearish"` / `"neutral"`
- Signal strength
- Supporting evidence
- Identified risks

**News Analyst**
Examines recent headlines and produces a structured `NewsOpinion`:

- Sentiment: `"positive"` / `"negative"` / `"neutral"` / `"mixed"`
- Relevance score
- Confidence level

> ⚠️ Neither agent makes trading decisions. Their purpose is to provide structured, evidence-based assessments for downstream consumption.

---

### Phase 3 — Risk Manager

Before any decision is made, the **Risk Manager** evaluates portfolio constraints and trading limits.

It computes:

- Maximum **buy quantity** based on available cash and portfolio exposure
- Maximum **sell quantity** based on current holdings
- Portfolio-related risk metrics and trading constraints

The Risk Manager acts as a dedicated **safety layer**, ensuring every subsequent decision stays within the portfolio's risk profile.

---

### Phase 4 — Decision Manager

The **Decision Manager** is the primary LLM-based agent — the cognitive core of the system.

It receives the structured outputs from all previous components and produces a `TradingDecision`:

| Field | Description |
|---|---|
| `action` | `BUY`, `SELL`, `HOLD`, or `WAIT` |
| `quantity` | Number of shares |
| `confidence` | Confidence score (0–1) |
| `rationale` | Natural-language explanation of the decision |

The **rationale** is a first-class output. A well-justified decision is often more valuable than an aggressive action without a clear explanation.

The prompt follows a **grounded prompting** approach, explicitly instructing the model to rely only on the information provided in the current context — never on its training data.

---

### Phase 5 — Reflection

Before execution, a second LLM-based step performs a **critical review** of the proposed decision.

The **Reflection Agent** checks:

- Consistency between the rationale and the available evidence
- Coherence of the proposed action
- Compliance with portfolio constraints
- Potential contradictions in the reasoning

The result either **confirms** the original decision or recommends a **more conservative alternative**.

---

### Phase 6 — Executor

Once validated, the **Executor** translates the decision into a real trading order.

Orders are submitted to **Alpaca Paper Trading** — a simulated brokerage environment that uses real market data without involving real capital.

The Executor communicates via REST APIs and records the broker's response before passing control to the next stage.

---

### Phase 7 — Journal

Every cycle is persisted as a **JSONL record** containing:

```jsonc
{
  "timestamp": "...",
  "ticker": "AAPL",
  "decision": "BUY",
  "quantity": 5,
  "rationale": "...",
  "data_sources": [...],
  "execution_outcome": {...},
  "confidence_scores": {...}
}
```

The journal serves two purposes:

1. **Agent Memory** — previous cycles can be injected as context for future decisions, enabling the agent to learn from its own history.
2. **Auditability** — every action can be inspected and explained after the fact, making the system fully traceable and easier to evaluate.

---

## The Autonomous Loop

After completing the journaling phase, the system waits for a configurable interval and starts a new cycle. The process can run continuously for extended periods without manual intervention.

When no explicit watchlist is configured, the autonomous selector can also build a dynamic candidate universe from market activity and recent news. The selection step is not a simple round-robin over open positions: each candidate is scored using portfolio exposure, market data, technical indicators, news evidence, and recent selection history. If an LLM selection is unavailable or invalid, the system falls back to the strongest local score while still recording why the fallback happened.

---

## Human Input Channel

A key architectural feature is the **Human Input Channel** — a dedicated mechanism for human-agent collaboration.

Users can provide notes, observations, or strategic suggestions at any time. These inputs are:

- Stored separately from market data
- Interpreted at the beginning of the next cycle
- Recorded in the journal and dashboard with their own lifecycle
- Applied as directives when the request is explicit and safe to execute

The agent explicitly considers human input during its reasoning process. If a request influences the decision, the rationale explains how. If the request is blocked, unsafe, unsupported, or needs confirmation, the system records that outcome instead of silently ignoring it.

This preserves the system's autonomy while enabling **meaningful human oversight**.

### Resolver, Confirmations, and External Discovery

Human input is resolved through a typed resolver that converts free-form notes into structured outcomes:

| Outcome | Example |
|---|---|
| Immediate instruction | `Invest 1k on AAPL` |
| Pending confirmation | `Balance my portfolio please` |
| Confirmation response | `confirm CF-ABCD` / `reject CF-ABCD` |
| Modified confirmation | `confirm CF-ABCD but cap the rebalance at $2,000` |
| Scheduled or conditional action | `Buy AAPL tomorrow at market open` |
| Constraint update | `Don't buy fossil fuel companies` |
| Information request | `What's my current P&L?` |
| Advisory reply | Unsupported or ambiguous requests |

The resolver can use **NewsAPI** and the configured ticker provider to discover tradable public tickers for requests that do not name a ticker directly, such as:

- `Buy the company that acquired Cursor`
- `Invest 10k on SpaceX`
- `Buy an aircraft company and associated market shares`
- `Buy stocks exposed to USA-Iran war fluctuations`

External discovery is evidence-based. The LLM is not allowed to simulate web-search results; it can only interpret the candidates and source context provided by the discovery layer. When a candidate is indirect, proxy-like, or uncertain, the system requires confirmation before trading.

Confirmation and cancellation handling is stateful:

- Equivalent pending confirmations are deduplicated.
- Confirmed multi-step workflows are tracked through the instruction ledger.
- `Stop the portfolio balancing` can cancel queued rebalance instructions.
- `Cancel scheduled action <SCHEDULED_ACTION_ID>` cancels the scheduled action store directly when the ID is known.

### Live Dashboard

Each run can produce a browser-readable dashboard backed by `dashboard_data.js`. The HTML shell is stable, while the projection file is refreshed as run data changes, so the dashboard can be updated with a browser refresh or by its built-in polling.

The dashboard is organized into focused tabs:

- Overview
- Trades
- Human Messages
- Confirmations
- Scheduled & Conditional
- Constraints
- Agent Replies
- Diversification
- Evidence
- Technical Audit

The dashboard is designed for auditability and user readability:

- Human input has its own lifecycle timeline.
- Tables are scrollable and ordered newest-first.
- BUY, SELL, HOLD, and WAIT decisions are retained.
- Confirmation IDs are easy to copy.
- Tabs show notification dots only when they contain new content.
- Modals expose user-facing summaries first, then execution, portfolio/risk, market/technical data, evidence, and complete details.
- External sources collected by the resolver are cited in the Evidence tab and in relevant human-input details.

---

## Project Structure

```text
trading-agent-hackathon/
├── trading-agent/
│   │
│   ├── adapters/
│   │   ├── alpaca_broker.py            # Alpaca Broker, able to get Portfolio info
│   │   ├── alpaca_market_data.py       # Alpaca Marker Data, able to get tickers' prices
│   │   ├── news_provider.py            # News API Helper for currents available news
│   │   └── ticker_provider.py          # Available Tickers Provider based on the Market 
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
│   │   ├── actions.py                  # Trading Actions Definitons
│   │   ├── autonomous_selection.py     # Autonomous ticker discovery, scoring, and selection
│   │   ├── confidence_policy.py        # Confidence Policies on News, Prices (and others) Fidelity
│   │   ├── confirmation_store.py       # Pending confirmation persistence and deduplication
│   │   ├── conditional_order_store.py  # Conditional order persistence
│   │   ├── constraint_store.py         # Persistent user constraints
│   │   ├── data_hygiene.py             # Data Cleaning Functions Definition 
│   │   ├── entity_discovery.py         # Evidence-based external ticker discovery
│   │   ├── execution_policy.py         # Execution Policies on the Market
│   │   ├── evidence.py                 # Evidence model and source citation helpers
│   │   ├── human_input.py              # Human Input Class Definition
│   │   ├── human_instruction.py        # Human Input Handler Functions
│   │   ├── human_intent.py             # Human Intent Functions and Definitions
│   │   ├── instruction_ledger.py       # Idempotency and workflow status tracking
│   │   ├── llm_guardrails.py           # LLM Guardrails for Output Checking
│   │   ├── models.py                   # Base Custom Models
│   │   ├── news_query.py               # Base News Template Query and Search Parameters
│   │   ├── portfolio.py                # PortFolio Related Functions
│   │   ├── ports.py                    # Custom Inferfaces
│   │   ├── rebalance.py                # Sector Definitions and Related Tickers
│   │   ├── retry_policy.py             # Retry Policy definition
│   │   ├── risk_policy.py              # Risk Policy Definition
│   │   ├── scheduled_action_store.py   # Scheduled action persistence and cancellation
│   │   ├── sector_classifier.py        # Sector classification and ticker suggestions
│   │   ├── ticker_selection.py         # Ticker Selection Functions
│   │   ├── ticker_symbols.py           # Ticker normalization helpers
│   │   ├── ticker_universe.py          # Ticker Universe Definition based on Market
│   │   ├── typed_outcomes.py           # Structured resolver outcomes
│   │   ├── typed_resolver.py           # Active human input resolver
│   │   └── web_search.py               # Web-search adapter layer
│   │
│   ├── journal/
│   │   ├── dashboard.py                # Live dashboard shell and projection writer
│   │   ├── memory.py                   # Memory Handling
│   │   ├── run_manager.py              # Single Run Manager for Saving and Recalling
│   │   ├── store.py                    # Journal Storing and Recalling Model
│   │   └── terminal.py                 # Terminal Inferface API
│   │
│   ├── pipeline/
│   │   └── graph.py                    # Pipeline Visualization
│   │
|   ├── scripts/
│   │   ├── human_input.py              # Helper CLI to append human notes
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
│       ├── http_requests.py            # Https Utils to request external files
│       ├── llm_clients.py              # Availble Clients Config (OpenAI, Ollama)
│       ├── llm_metadata.py             # Availble Clients Metadata
│       ├── logger.py                   # Logger Info
│       └── portfolio.py                # Portfolio Availablity Checks
│
├── pyproject.toml                      # v0.1.0, alpaca-py>=0.30, project scripts
└── .env                                # API keys (not committed)
```

---

> Built with the goal of making autonomous trading systems **transparent**, **safe**, and **explainable** — one cycle at a time.
