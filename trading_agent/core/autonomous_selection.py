from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from trading_agent.core.evidence import Evidence
from trading_agent.core.evidence import make_evidence
from trading_agent.core.portfolio import positions as portfolio_positions
from trading_agent.technicals.indicators import calculate_indicators

logger = logging.getLogger("trading_agent.autonomous_selection")


@dataclass(frozen=True)
class SelectionPolicy:
    loss_risk_weight: float = 3.0
    gain_review_weight: float = 1.0
    negative_momentum_weight: float = 2.0
    risk_news_weight: float = 0.4
    opportunity_momentum_weight: float = 3.0
    opportunity_news_weight: float = 0.5
    diversification_bonus: float = 0.05
    technical_trend_weight: float = 0.15
    technical_momentum_weight: float = 0.10
    llm_confidence_threshold: float = 0.6

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


class AutonomousSelectionCache:
    def __init__(
        self,
        *,
        market_ttl_seconds: float = 30.0,
        news_ttl_seconds: float = 600.0,
        discovery_ttl_seconds: float = 600.0,
        clock: Any = time.monotonic,
    ) -> None:
        self.market_ttl_seconds = max(0.0, float(market_ttl_seconds))
        self.news_ttl_seconds = max(0.0, float(news_ttl_seconds))
        self.discovery_ttl_seconds = max(0.0, float(discovery_ttl_seconds))
        self._clock = clock
        self._market: dict[str, tuple[float, dict[str, Any]]] = {}
        self._news: dict[str, tuple[float, tuple[Evidence, ...]]] = {}
        self._discovery: tuple[float, tuple[str, ...]] | None = None
        self._recent_selections: list[str] = []

    def get_market(self, ticker: str) -> dict[str, Any] | None:
        return self._get(self._market, ticker, self.market_ttl_seconds)

    def put_market(self, ticker: str, payload: dict[str, Any]) -> None:
        self._market[ticker] = (self._clock(), dict(payload))

    def get_news(self, ticker: str) -> tuple[Evidence, ...] | None:
        return self._get(self._news, ticker, self.news_ttl_seconds)

    def put_news(self, ticker: str, evidence: list[Evidence]) -> None:
        self._news[ticker] = (self._clock(), tuple(evidence))

    def get_discovery(self) -> list[str] | None:
        if self._discovery is None:
            return None
        created_at, symbols = self._discovery
        if self._clock() - created_at > self.discovery_ttl_seconds:
            self._discovery = None
            return None
        return list(symbols)

    def put_discovery(self, symbols: list[str]) -> None:
        self._discovery = (self._clock(), tuple(symbols))

    def recent_selections(self, limit: int = 3) -> list[str]:
        return list(self._recent_selections[-max(1, limit) :])

    def record_selection(self, ticker: str) -> None:
        self._recent_selections.append(str(ticker).upper())
        self._recent_selections = self._recent_selections[-10:]

    def _get(self, store: dict[str, tuple[float, Any]], ticker: str, ttl: float) -> Any | None:
        cached = store.get(ticker)
        if cached is None:
            return None
        created_at, value = cached
        if self._clock() - created_at > ttl:
            store.pop(ticker, None)
            return None
        return value


def discover_dynamic_candidates(
    news_provider: Any | None,
    ticker_provider: Any | None,
    *,
    cache: AutonomousSelectionCache | None = None,
    max_candidates: int = 6,
) -> list[str]:
    cache = cache or AutonomousSelectionCache()
    cached = cache.get_discovery()
    if cached is not None:
        return cached[: max(1, max_candidates)]

    limit = max(1, min(int(max_candidates), 10))
    symbols: list[str] = []

    market_candidates = getattr(ticker_provider, "get_market_candidates", None)
    if callable(market_candidates):
        try:
            for raw_symbol in market_candidates(limit=min(4, limit)) or []:
                symbol = str(raw_symbol or "").strip().upper()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        except Exception as error:
            logger.info("autonomous_discovery.movers.fail reason=%s", str(error)[:160])

    try:
        from trading_agent.core.entity_discovery import discover_candidates

        validator = None
        get_info = getattr(ticker_provider, "get_tickers_with_info", None)
        if callable(get_info):
            validator = lambda symbol: bool(get_info(symbol))
        discovered, _ = discover_candidates(
            "public companies stock market earnings acquisitions mergers and major business news",
            news_provider=news_provider,
            alpaca_validator=validator,
            max_discovered_tickers=max(1, limit - len(symbols)),
        )
        for candidate in discovered:
            symbol = str(candidate.get("ticker") or candidate.get("symbol") or "").strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    except Exception as error:
        logger.info("autonomous_discovery.news.fail reason=%s", str(error)[:160])

    bounded = symbols[:limit]
    cache.put_discovery(bounded)
    logger.info("autonomous_discovery.complete candidates=%s", ",".join(bounded) or "-")
    return bounded


@dataclass(frozen=True)
class TickerCandidateAssessment:
    ticker: str
    sources: tuple[str, ...] = ()
    held: bool = False
    quantity: float = 0.0
    market_value: float = 0.0
    avg_entry_price: float | None = None
    pnl_fraction: float | None = None
    price: float | None = None
    momentum: float | None = None
    indicators: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    news_score: float | None = None
    data_quality_flags: tuple[str, ...] = ()
    risk_urgency: float = 0.0
    opportunity: float = 0.0
    local_priority: float = 0.0
    technical_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sources": list(self.sources),
            "held": self.held,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "avg_entry_price": self.avg_entry_price,
            "pnl_fraction": self.pnl_fraction,
            "price": self.price,
            "momentum": self.momentum,
            "indicators": dict(self.indicators),
            "evidence": [item.to_dict() for item in self.evidence],
            "news_score": self.news_score,
            "data_quality_flags": list(self.data_quality_flags),
            "risk_urgency": self.risk_urgency,
            "opportunity": self.opportunity,
            "local_priority": self.local_priority,
            "technical_details": dict(self.technical_details),
        }


@dataclass(frozen=True)
class AutonomousSelection:
    ticker: str
    selection_source: str
    selection_mode: str
    confidence: float
    user_summary: str
    candidates: tuple[TickerCandidateAssessment, ...]
    rejections: tuple[dict[str, str], ...] = ()
    evidence_ids: tuple[str, ...] = ()
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)

    def to_dict(self) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in self.candidates:
            for item in candidate.evidence:
                if item.source_id not in seen:
                    seen.add(item.source_id)
                    evidence.append(item.to_dict())
        return {
            "ticker": self.ticker,
            "selection_source": self.selection_source,
            "selection_mode": self.selection_mode,
            "confidence": self.confidence,
            "user_summary": self.user_summary,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evidence": evidence,
            "rejections": [dict(item) for item in self.rejections],
            "evidence_ids": list(self.evidence_ids),
            "policy": self.policy.to_dict(),
        }


def score_candidate(
    candidate: TickerCandidateAssessment,
    policy: SelectionPolicy | None = None,
) -> TickerCandidateAssessment:
    policy = policy or SelectionPolicy()
    components: dict[str, float] = {}
    risk = 0.0
    opportunity = 0.0

    if candidate.held and candidate.pnl_fraction is not None:
        if candidate.pnl_fraction < 0:
            components["unrealized_loss"] = min(1.0, abs(candidate.pnl_fraction) * policy.loss_risk_weight)
            risk += components["unrealized_loss"]
        elif candidate.pnl_fraction >= 0.10:
            components["gain_review"] = min(0.35, candidate.pnl_fraction * policy.gain_review_weight)
            risk += components["gain_review"]
    if candidate.momentum is not None:
        if candidate.momentum < 0 and candidate.held:
            components["negative_momentum"] = min(0.5, abs(candidate.momentum) * policy.negative_momentum_weight)
            risk += components["negative_momentum"]
        elif candidate.momentum > 0:
            components["positive_momentum"] = min(0.5, candidate.momentum * policy.opportunity_momentum_weight)
            opportunity += components["positive_momentum"]
    if candidate.news_score is not None:
        if candidate.news_score < 0 and candidate.held:
            components["adverse_news"] = min(0.4, abs(candidate.news_score) * policy.risk_news_weight)
            risk += components["adverse_news"]
        elif candidate.news_score > 0:
            components["positive_news"] = min(0.5, candidate.news_score * policy.opportunity_news_weight)
            opportunity += components["positive_news"]
    price = _optional_float(candidate.price)
    sma_20 = _optional_float(candidate.indicators.get("sma_20"))
    sma_50 = _optional_float(candidate.indicators.get("sma_50"))
    macd_histogram = _optional_float(candidate.indicators.get("macd_histogram"))
    rsi = _optional_float(candidate.indicators.get("rsi_14"))
    if price is not None and sma_20 is not None and sma_50 is not None:
        if price > sma_20 > sma_50:
            components["positive_trend"] = policy.technical_trend_weight
            opportunity += components["positive_trend"]
        elif candidate.held and price < sma_20 < sma_50:
            components["negative_trend"] = policy.technical_trend_weight
            risk += components["negative_trend"]
    if macd_histogram is not None:
        if macd_histogram > 0:
            components["positive_macd"] = policy.technical_momentum_weight
            opportunity += components["positive_macd"]
        elif candidate.held and macd_histogram < 0:
            components["negative_macd"] = policy.technical_momentum_weight
            risk += components["negative_macd"]
    if rsi is not None and 50 <= rsi <= 70:
        components["constructive_rsi"] = 0.05
        opportunity += components["constructive_rsi"]
    if not candidate.held and (candidate.momentum is not None or candidate.news_score is not None):
        components["diversification_bonus"] = policy.diversification_bonus
        opportunity += policy.diversification_bonus

    risk = round(min(1.0, risk), 4)
    opportunity = round(min(1.0, opportunity), 4)
    return replace(
        candidate,
        risk_urgency=risk,
        opportunity=opportunity,
        local_priority=max(risk, opportunity),
        technical_details={**candidate.technical_details, "score_components": components},
    )


def assess_candidates(
    tickers: list[str],
    *,
    sources: dict[str, list[str]],
    portfolio: dict[str, Any] | None,
    market_data: Any,
    news_provider: Any | None,
    llm_client: Any | None = None,
    policy: SelectionPolicy | None = None,
    tavily_search: Any | None = None,
    cache: AutonomousSelectionCache | None = None,
    news_shortlist_size: int = 3,
) -> list[TickerCandidateAssessment]:
    policy = policy or SelectionPolicy()
    cache = cache or AutonomousSelectionCache()
    positions = portfolio_positions(portfolio)
    assessed: list[TickerCandidateAssessment] = []
    phase_started = time.perf_counter()
    logger.info("autonomous_selection.market.start candidates=%d", len(tickers))
    for ticker in tickers:
        candidate_started = time.perf_counter()
        position = positions.get(ticker, {})
        flags: list[str] = []
        market = cache.get_market(ticker)
        cache_hit = market is not None
        if market is None:
            try:
                price_payload = market_data.get_price(ticker) or {}
                closes = [float(value) for value in (market_data.get_closes(ticker, limit=60) or []) if float(value) > 0]
                momentum = (closes[-1] - closes[0]) / closes[0] if len(closes) >= 2 and closes[0] > 0 else None
                market = {
                    "price": _optional_float(price_payload.get("price")),
                    "momentum": momentum,
                    "indicators": asdict(calculate_indicators(closes)),
                }
                cache.put_market(ticker, market)
            except Exception as error:
                market = {"price": None, "momentum": None, "indicators": {}}
                flags.append("market_data_unavailable")

        quantity = _optional_float(position.get("qty")) or 0.0
        market_value = _optional_float(position.get("market_value")) or 0.0
        unrealized_pl = _optional_float(position.get("unrealized_pl"))
        cost_basis = market_value - unrealized_pl if unrealized_pl is not None else None
        pnl_fraction = unrealized_pl / cost_basis if unrealized_pl is not None and cost_basis and cost_basis > 0 else None
        assessed.append(
            score_candidate(
                TickerCandidateAssessment(
                    ticker=ticker,
                    sources=tuple(sources.get(ticker) or ()),
                    held=quantity > 0,
                    quantity=quantity,
                    market_value=market_value,
                    avg_entry_price=_optional_float(position.get("avg_entry_price")),
                    pnl_fraction=pnl_fraction,
                    price=market.get("price"),
                    momentum=market.get("momentum"),
                    indicators=dict(market.get("indicators") or {}),
                    data_quality_flags=tuple(flags),
                ),
                policy,
            )
        )
        if "market_data_unavailable" not in flags:
            logger.info(
                "autonomous_selection.market.candidate ticker=%s cache_hit=%s elapsed_ms=%d",
                ticker,
                cache_hit,
                int((time.perf_counter() - candidate_started) * 1000),
            )

    assessed = prune_unusable_candidates(assessed)
    shortlist_size = min(len(assessed), max(1, int(news_shortlist_size)))
    ranked_indexes = sorted(range(len(assessed)), key=lambda index: assessed[index].local_priority, reverse=True)
    shortlist_indexes = set(ranked_indexes[:shortlist_size])
    logger.info(
        "autonomous_selection.news.start shortlist=%s market_elapsed_ms=%d",
        ",".join(assessed[index].ticker for index in ranked_indexes[:shortlist_size]),
        int((time.perf_counter() - phase_started) * 1000),
    )
    enriched: list[TickerCandidateAssessment] = []
    for index, candidate in enumerate(assessed):
        if index not in shortlist_indexes:
            enriched.append(candidate)
            continue
        news_started = time.perf_counter()
        evidence = cache.get_news(candidate.ticker)
        cache_hit = evidence is not None
        news_flags = list(candidate.data_quality_flags)
        if evidence is None:
            collected: list[Evidence] = []
            try:
                search = getattr(news_provider, "search_market_news", None) or getattr(news_provider, "search_news", None)
                raw_news = search(candidate.ticker, limit=5) if callable(search) else []
                if isinstance(raw_news, dict):
                    raw_news = raw_news.get("items") or raw_news.get("articles") or []
                for item in raw_news or []:
                    if isinstance(item, dict):
                        collected.append(
                            make_evidence(
                                title=str(item.get("title") or ""),
                                url=str(item.get("url") or ""),
                                published_at=str(item.get("published_at") or item.get("publishedAt") or "") or None,
                                provider="newsapi",
                                query=candidate.ticker,
                                excerpt=str(item.get("summary") or item.get("description") or item.get("content") or ""),
                                confidence=0.7,
                            )
                        )
            except Exception as error:
                news_flags.append("news_unavailable")
                logger.info("autonomous_selection.news.fail ticker=%s reason=%s", candidate.ticker, str(error)[:160])
            if not collected:
                try:
                    if tavily_search is None:
                        from trading_agent.core.entity_discovery import _tavily_discovery

                        collected.extend(_tavily_discovery(candidate.ticker, os.getenv("TAVILY_API_KEY")))
                    else:
                        collected.extend(tavily_search(candidate.ticker) or [])
                except Exception:
                    news_flags.append("tavily_unavailable")
            cache.put_news(candidate.ticker, collected)
            evidence = tuple(collected)
        enriched.append(replace(candidate, evidence=tuple(evidence), data_quality_flags=tuple(dict.fromkeys(news_flags))))
        logger.info(
            "autonomous_selection.news.candidate ticker=%s cache_hit=%s sources=%d elapsed_ms=%d",
            candidate.ticker,
            cache_hit,
            len(evidence),
            int((time.perf_counter() - news_started) * 1000),
        )
    logger.info(
        "autonomous_selection.assessment.complete candidates=%d shortlist=%d elapsed_ms=%d",
        len(enriched),
        shortlist_size,
        int((time.perf_counter() - phase_started) * 1000),
    )
    return enriched


def prune_unusable_candidates(
    candidates: list[TickerCandidateAssessment],
) -> list[TickerCandidateAssessment]:
    usable: list[TickerCandidateAssessment] = []
    for candidate in candidates:
        indicator_values = candidate.indicators.values() if candidate.indicators else ()
        has_market_data = candidate.price is not None or candidate.momentum is not None or any(_optional_float(value) is not None for value in indicator_values)
        if candidate.held or has_market_data:
            usable.append(candidate)
        else:
            logger.info(
                "autonomous_selection.market.reject ticker=%s reason=no_usable_market_data",
                candidate.ticker,
            )
    return usable or candidates


def choose_cycle_ticker(
    *,
    active_ticker: str | None,
    tickers: list[str],
    sources: dict[str, list[str]],
    portfolio: dict[str, Any] | None,
    market_data: Any,
    news_provider: Any | None,
    llm_client: Any | None,
    policy: SelectionPolicy | None = None,
    cache: AutonomousSelectionCache | None = None,
    news_shortlist_size: int = 3,
) -> AutonomousSelection:
    policy = policy or SelectionPolicy()
    if active_ticker:
        candidate = TickerCandidateAssessment(ticker=active_ticker, sources=("human_instruction",))
        return AutonomousSelection(
            ticker=active_ticker,
            selection_source="human_instruction",
            selection_mode="instruction",
            confidence=1.0,
            user_summary=f"Selected {active_ticker} because it is targeted by the active instruction.",
            candidates=(candidate,),
            policy=policy,
        )
    assessed = assess_candidates(
        tickers,
        sources=sources,
        portfolio=portfolio,
        market_data=market_data,
        news_provider=news_provider,
        llm_client=llm_client,
        policy=policy,
        cache=cache,
        news_shortlist_size=news_shortlist_size,
    )
    selection = select_autonomously(
        assessed,
        llm_client=llm_client,
        policy=policy,
        recent_tickers=cache.recent_selections() if cache else None,
    )
    if cache:
        cache.record_selection(selection.ticker)
    return selection


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def select_locally(
    candidates: list[TickerCandidateAssessment],
    policy: SelectionPolicy | None = None,
    recent_tickers: list[str] | None = None,
) -> AutonomousSelection:
    if not candidates:
        raise ValueError("autonomous selection requires at least one candidate")
    policy = policy or SelectionPolicy()
    scored = [score_candidate(item, policy) for item in candidates]
    recent = [str(ticker).upper() for ticker in (recent_tickers or [])]
    ranked: list[TickerCandidateAssessment] = []
    for item in scored:
        occurrences = recent.count(item.ticker.upper())
        penalty = min(0.12, occurrences * 0.06)
        ranked.append(
            replace(
                item,
                local_priority=item.local_priority - penalty,
                technical_details={**item.technical_details, "recent_selection_penalty": penalty},
            )
        )
    winner = max(ranked, key=lambda item: item.local_priority)
    mode = "risk_review" if winner.risk_urgency > winner.opportunity else "opportunity_review"
    limitations = ", ".join(winner.data_quality_flags)
    summary = f"Selected {winner.ticker} for {mode.replace('_', ' ')} based on the strongest available " f"portfolio, market, and news signals."
    if limitations:
        summary += f" Some data was unavailable: {limitations}."
    rejections = tuple({"ticker": item.ticker, "reason": f"local priority {item.local_priority:.2f} below {winner.local_priority:.2f}"} for item in ranked if item.ticker != winner.ticker)
    return AutonomousSelection(
        ticker=winner.ticker,
        selection_source="local_fallback",
        selection_mode=mode,
        confidence=min(1.0, 0.5 + winner.local_priority / 2),
        user_summary=summary,
        candidates=tuple(ranked),
        rejections=rejections,
        policy=policy,
    )


def select_autonomously(
    candidates: list[TickerCandidateAssessment],
    *,
    llm_client: Any | None,
    policy: SelectionPolicy | None = None,
    recent_tickers: list[str] | None = None,
) -> AutonomousSelection:
    policy = policy or SelectionPolicy()
    local = select_locally(candidates, policy, recent_tickers=recent_tickers)
    complete_json = getattr(llm_client, "complete_json", None) if llm_client else None
    if not callable(complete_json):
        logger.info("autonomous_selection.llm.skip reason=unavailable")
        return local
    if len(local.candidates) <= 1:
        logger.info("autonomous_selection.llm.skip reason=single_candidate")
        return local
    sorted_by_priority = sorted(local.candidates, key=lambda c: c.local_priority, reverse=True)
    top, second = sorted_by_priority[0], sorted_by_priority[1]
    if top.local_priority - second.local_priority >= 0.20:
        logger.info(
            "autonomous_selection.llm.skip reason=clear_winner ticker=%s gap=%.2f",
            top.ticker,
            top.local_priority - second.local_priority,
        )
        return local
    prompt = (
        "Choose exactly one ticker to analyse next. Balance urgent supervision of held positions with strong new opportunities. "
        "Use only supplied candidates and evidence. Return JSON with selected_ticker, selection_mode, confidence, "
        "user_summary, rejections, and evidence_ids."
    )
    llm_started = time.perf_counter()
    logger.info("autonomous_selection.llm.start candidates=%d", len(local.candidates))
    try:
        raw = complete_json(prompt, json.dumps({"candidates": [item.to_dict() for item in local.candidates]}, default=str))
        resolved = raw if isinstance(raw, dict) else json.loads(str(raw))
    except Exception as error:
        logger.info(
            "autonomous_selection.llm.complete status=fallback elapsed_ms=%d reason=%s",
            int((time.perf_counter() - llm_started) * 1000),
            str(error)[:160],
        )
        return local
    allowed = {item.ticker for item in local.candidates}
    selected = str(resolved.get("selected_ticker") or "").upper()
    mode = str(resolved.get("selection_mode") or "")
    try:
        confidence = float(resolved.get("confidence"))
    except (TypeError, ValueError):
        return local
    if selected not in allowed or mode not in {"risk_review", "opportunity_review"} or confidence < policy.llm_confidence_threshold or confidence > 1:
        logger.info(
            "autonomous_selection.llm.complete status=fallback elapsed_ms=%d reason=invalid_selection " "selected=%s mode=%s confidence=%s allowed=%s",
            int((time.perf_counter() - llm_started) * 1000),
            selected or "-",
            mode or "-",
            confidence,
            ",".join(sorted(allowed)),
        )
        return local
    rejections = resolved.get("rejections") or []
    if not isinstance(rejections, list) or any(str(item.get("ticker") or "").upper() not in allowed for item in rejections if isinstance(item, dict)):
        logger.info(
            "autonomous_selection.llm.complete status=fallback elapsed_ms=%d reason=invalid_rejections",
            int((time.perf_counter() - llm_started) * 1000),
        )
        return local
    evidence_ids = resolved.get("evidence_ids") or []
    available_evidence = {item.source_id for candidate in local.candidates for item in candidate.evidence}
    if not isinstance(evidence_ids, list) or (available_evidence and not evidence_ids) or any(str(item) not in available_evidence for item in evidence_ids):
        logger.info(
            "autonomous_selection.llm.complete status=fallback elapsed_ms=%d reason=invalid_evidence",
            int((time.perf_counter() - llm_started) * 1000),
        )
        return local
    logger.info(
        "autonomous_selection.llm.complete status=accepted ticker=%s elapsed_ms=%d",
        selected,
        int((time.perf_counter() - llm_started) * 1000),
    )
    return AutonomousSelection(
        ticker=selected,
        selection_source="llm_autonomous",
        selection_mode=mode,
        confidence=confidence,
        user_summary=str(resolved.get("user_summary") or f"Selected {selected} for autonomous analysis."),
        candidates=local.candidates,
        rejections=tuple(dict(item) for item in rejections if isinstance(item, dict)),
        evidence_ids=tuple(str(item) for item in evidence_ids),
        policy=policy,
    )
