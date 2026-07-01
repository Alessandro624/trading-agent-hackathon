from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from trading_agent.core.actions import Action, normalize_quantity_for_action, validate_action
from trading_agent.core.data_hygiene import clean_news_items

ConfidenceLabel = Literal["high", "medium", "low", "none"]
Sentiment = Literal["positive", "negative", "neutral", "mixed", "unknown"]
Trend = Literal["bullish", "bearish", "neutral"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TechnicalIndicators:
    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    confidence: ConfidenceLabel = "none"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.confidence not in {"high", "medium", "low", "none"}:
            raise ValueError("invalid technical indicators confidence")
        for name in ["sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_histogram"]:
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric when present")
        if self.rsi_14 is not None and not 0 <= self.rsi_14 <= 100:
            raise ValueError("rsi_14 must be between 0 and 100")
        if not isinstance(self.notes, list):
            raise ValueError("notes must be a list")

    def summary(self) -> str:
        parts: list[str] = []
        if self.rsi_14 is not None:
            if self.rsi_14 >= 70:
                parts.append("RSI overbought")
            elif self.rsi_14 <= 30:
                parts.append("RSI oversold")
            else:
                parts.append("RSI neutral")
        if self.macd is not None and self.macd_signal is not None:
            parts.append("MACD bullish" if self.macd > self.macd_signal else "MACD bearish")
        if self.sma_20 is not None and self.sma_50 is not None:
            parts.append("SMA20 above SMA50" if self.sma_20 > self.sma_50 else "SMA20 below SMA50")
        return "; ".join(parts) if parts else "No technical signal"


@dataclass(slots=True)
class MarketSnapshot:
    ticker: str
    timestamp: str
    price: float | None
    price_confidence: ConfidenceLabel
    news: list[dict[str, Any]]
    news_confidence: ConfidenceLabel
    data_sources: list[str]
    technical_indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    failures: list[str] = field(default_factory=list)
    guardrails_triggered: list[str] = field(default_factory=list)
    retry_count: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.ticker = self.ticker.upper()
        if self.price is not None and self.price <= 0:
            raise ValueError("price must be positive when present")
        if self.price_confidence not in {"high", "medium", "low", "none"}:
            raise ValueError("invalid price_confidence")
        if self.news_confidence not in {"high", "medium", "low", "none"}:
            raise ValueError("invalid news_confidence")
        self.news = clean_news_items(self.news)
        if not self.data_sources:
            raise ValueError("data_sources must not be empty")
        if not isinstance(self.technical_indicators, TechnicalIndicators):
            raise ValueError("technical_indicators must be TechnicalIndicators")
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")


@dataclass(slots=True)
class TechnicalOpinion:
    trend: Trend
    strength: float
    confidence: float
    evidence: list[str]
    risks: list[str]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.trend not in {"bullish", "bearish", "neutral"}:
            raise ValueError("technical trend must be bullish, bearish, or neutral")
        if not 0 <= self.strength <= 1:
            raise ValueError("technical strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("technical opinion confidence must be between 0 and 1")
        if not isinstance(self.evidence, list) or not isinstance(self.risks, list):
            raise ValueError("technical evidence and risks must be lists")


@dataclass(slots=True)
class NewsOpinion:
    sentiment: Sentiment
    relevance: float
    confidence: float
    evidence: list[str]
    risks: list[str]
    sources: list[str]
    summary: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.sentiment not in {"positive", "negative", "neutral", "mixed", "unknown"}:
            raise ValueError("news sentiment must be positive, negative, neutral, mixed, or unknown")
        if not 0 <= self.relevance <= 1:
            raise ValueError("news relevance must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("news opinion confidence must be between 0 and 1")
        if not isinstance(self.evidence, list) or not isinstance(self.risks, list) or not isinstance(self.sources, list):
            raise ValueError("news evidence, risks, and sources must be lists")
        if not self.summary:
            raise ValueError("news summary must not be empty")


@dataclass(slots=True)
class RiskAssessment:
    can_trade: bool
    max_quantity: int
    reasons: list[str]
    blocked_reason: str | None
    portfolio_context: str = ""
    max_buy_quantity: int = 0
    max_explicit_notional_buy_quantity: int = 0
    max_sell_quantity: int = 0
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False
    risk_flags: list[str] = field(default_factory=list)
    human_risk_profile: dict[str, Any] = field(default_factory=dict)
    current_quantity: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.max_quantity < 0:
            raise ValueError("risk max_quantity must be >= 0")
        if self.max_buy_quantity < 0 or self.max_explicit_notional_buy_quantity < 0 or self.max_sell_quantity < 0:
            raise ValueError("risk action quantities must be >= 0")
        if not isinstance(self.reasons, list):
            raise ValueError("risk reasons must be a list")
        if not self.can_trade and not self.blocked_reason:
            raise ValueError("blocked risk assessment must include blocked_reason")
        if not isinstance(self.portfolio_context, str):
            raise ValueError("risk portfolio_context must be a string")
        if not isinstance(self.take_profit_triggered, bool):
            raise ValueError("risk take_profit_triggered must be a bool")
        if not isinstance(self.risk_flags, list):
            raise ValueError("risk_flags must be a list")
        if not isinstance(self.human_risk_profile, dict):
            raise ValueError("human_risk_profile must be a dict")


@dataclass(slots=True)
class TradingDecision:
    ticker: str
    action: Action
    quantity: int
    confidence: float
    rationale: str
    used_data_sources: list[str]
    guardrails_triggered: list[str] = field(default_factory=list)
    reflection: str | None = None
    llm_metadata: dict[str, Any] = field(default_factory=dict)
    rationale_details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.ticker = self.ticker.upper()
        validate_action(self.action)
        if self.quantity < 0:
            raise ValueError("quantity must be >= 0")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.quantity = normalize_quantity_for_action(self.action, self.quantity)
        if not self.rationale:
            raise ValueError("rationale must not be empty")
        if not isinstance(self.used_data_sources, list):
            raise ValueError("used_data_sources must be a list")
        if not isinstance(self.llm_metadata, dict):
            raise ValueError("llm_metadata must be a dict")
        if not isinstance(self.rationale_details, dict):
            raise ValueError("rationale_details must be a dict")


@dataclass(slots=True)
class ExecutionResult:
    ticker: str
    attempted_action: Action
    status: Literal["skipped", "waiting", "submitted", "filled", "blocked", "rejected", "failed"]
    order_id: str | None
    message: str
    portfolio_after: dict[str, Any]
    retry_count: int = 0
    portfolio_before: dict[str, Any] = field(default_factory=dict)
    requested_quantity: int | None = None
    allowed_quantity: int | None = None
    risk_explanation: str | None = None
    current_price_at_order: float | None = None
    filled_avg_price: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.ticker = self.ticker.upper()
        validate_action(self.attempted_action, label="attempted_action")
        if self.status not in {"skipped", "waiting", "submitted", "filled", "blocked", "rejected", "failed"}:
            raise ValueError("invalid execution status")
        if not isinstance(self.portfolio_after, dict):
            raise ValueError("portfolio_after must be a dict")
        if not isinstance(self.portfolio_before, dict):
            raise ValueError("portfolio_before must be a dict")
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if self.requested_quantity is not None and self.requested_quantity < 0:
            raise ValueError("requested_quantity must be >= 0 when present")
        if self.allowed_quantity is not None and self.allowed_quantity < 0:
            raise ValueError("allowed_quantity must be >= 0 when present")
        if self.current_price_at_order is not None and self.current_price_at_order <= 0:
            raise ValueError("current_price_at_order must be positive when present")
        if self.filled_avg_price is not None and self.filled_avg_price <= 0:
            raise ValueError("filled_avg_price must be positive when present")
        if self.status == "filled" and not self.order_id:
            raise ValueError("filled executions must include order_id")


@dataclass(slots=True)
class JournalEntry:
    timestamp: str
    ticker: str
    action: Action
    rationale: str
    data_source: str
    confidence: float
    outcome: str
    cycle_summary: str
    market_snapshot: dict[str, Any]
    decision: dict[str, Any]
    execution_result: dict[str, Any] | None
    draft_decision: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = "none"
    llm_fallback_used: bool = False
    llm_fallback_provider: str | None = None
    llm_fallback_reason: str | None = None
    guardrails_triggered: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    human_input_used: list[str] = field(default_factory=list)
    entry_type: Literal["cycle"] = "cycle"
    instruction_id: str | None = None
    cancelled_by: str | None = None
    retry_count: int = 0
    failure_type: str | None = None
    reversal_of: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.ticker = self.ticker.upper()
        if self.entry_type != "cycle":
            raise ValueError("journal entry_type must be cycle")
        validate_action(self.action, label="journal action")
        if not 0 <= self.confidence <= 1:
            raise ValueError("journal confidence must be between 0 and 1")
        if not self.timestamp:
            raise ValueError("journal timestamp must not be empty")
        if not self.rationale:
            raise ValueError("journal rationale must not be empty")
        if not self.cycle_summary:
            raise ValueError("cycle_summary must not be empty")
        if not isinstance(self.market_snapshot, dict):
            raise ValueError("market_snapshot must be a dict")
        if not isinstance(self.decision, dict):
            raise ValueError("decision must be a dict")
        if self.draft_decision is not None and not isinstance(self.draft_decision, dict):
            raise ValueError("draft_decision must be a dict or None")
        if not isinstance(self.risk_assessment, dict):
            raise ValueError("risk_assessment must be a dict")
        if self.execution_result is not None and not isinstance(self.execution_result, dict):
            raise ValueError("execution_result must be a dict or None")
        if not isinstance(self.guardrails_triggered, list):
            raise ValueError("guardrails_triggered must be a list")
        if not isinstance(self.failures, list):
            raise ValueError("failures must be a list")
        if not isinstance(self.human_input_used, list):
            raise ValueError("human_input_used must be a list")
        if not isinstance(self.llm_fallback_used, bool):
            raise ValueError("llm_fallback_used must be a bool")
        if self.retry_count < 0:
            raise ValueError("journal retry_count must be >= 0")
        if self.instruction_id is not None and not isinstance(self.instruction_id, str):
            raise ValueError("instruction_id must be a string or None")
        if self.cancelled_by is not None and not isinstance(self.cancelled_by, str):
            raise ValueError("cancelled_by must be a string or None")
        if self.failure_type is not None and not isinstance(self.failure_type, str):
            raise ValueError("failure_type must be a string or None")
        if self.reversal_of is not None and not isinstance(self.reversal_of, str):
            raise ValueError("reversal_of must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_cycle_summary(
    snapshot: MarketSnapshot,
    decision: TradingDecision,
    execution: ExecutionResult | None,
) -> str:
    missing: list[str] = []
    if snapshot.price is None or snapshot.price_confidence in {"low", "none"}:
        missing.append("reliable price")
    if snapshot.news_confidence == "none":
        missing.append("news")
    if snapshot.technical_indicators.confidence == "none":
        missing.append("technical indicators")

    status = execution.status if execution else "not executed"
    if missing:
        return f"{snapshot.ticker}: {decision.action} because missing/degraded " f"{', '.join(missing)}; execution {status}."
    return f"{snapshot.ticker}: {decision.action} with confidence {decision.confidence:.2f}; " f"{snapshot.technical_indicators.summary()}; execution {status}."
