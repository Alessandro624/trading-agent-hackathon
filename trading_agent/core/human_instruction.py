from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.human_intent import parse_human_intent
from trading_agent.core.portfolio import positions as portfolio_positions

logger = logging.getLogger("trading_agent.human_instruction")


@dataclass(frozen=True)
class HumanInstruction:
    note: str
    target_ticker: str | None
    reason: str
    sequence_index: int = 1
    sequence_total: int = 1
    resolver_confidence: float | None = None
    resolver_rationale: str | None = None
    resolver_topic: str | None = None


def plan_human_instructions(
    notes: list[str],
    *,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None = None,
) -> list[HumanInstruction]:
    instructions: list[HumanInstruction] = []
    for note in notes:
        cleaned = clean_text(note, max_chars=2000)
        if not cleaned:
            continue
        instructions.extend(_instructions_for_note(cleaned, watchlist, portfolio, fallback_ticker, llm_client))
    return instructions


def _instructions_for_note(
    note: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    fallback_ticker: str | None,
    llm_client: Any | None,
) -> list[HumanInstruction]:
    intent = parse_human_intent([note])
    if "position_sweep" in intent.intents:
        resolved = _llm_position_sweep_instructions(note, watchlist, portfolio, llm_client)
        if resolved:
            return resolved
        if intent.impact_topic:
            logger.info(
                "human.resolver.semantic_required topic=%s reason=no_valid_llm_resolution note=%s",
                intent.impact_topic,
                _short_note(note),
            )
            return [
                HumanInstruction(
                    note,
                    None,
                    "human_context_requires_resolver",
                    resolver_topic=intent.impact_topic,
                )
            ]
        targets = _open_position_targets(watchlist, portfolio, fallback_ticker=fallback_ticker)
        if targets:
            total = len(targets)
            return [
                HumanInstruction(note, target, "human_position_sweep", index, total)
                for index, target in enumerate(targets, start=1)
            ]
    if intent.requested_action in {"BUY", "SELL"}:
        targets = intent.tickers or ([fallback_ticker.upper()] if fallback_ticker else [])
        return [HumanInstruction(note, target, "human_forced_action") for target in targets]
    if intent.tickers:
        return [HumanInstruction(note, intent.tickers[-1], "human_input")]
    return [HumanInstruction(note, None, "human_context")]


def _llm_position_sweep_instructions(
    note: str,
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    llm_client: Any | None,
) -> list[HumanInstruction]:
    complete_json = getattr(llm_client, "complete_json", None)
    if not callable(complete_json):
        return []
    sellable_targets = _open_position_targets(watchlist, portfolio)
    if not sellable_targets:
        logger.info("human.resolver.llm.skip reason=no_sellable_positions note=%s", _short_note(note))
        return []
    try:
        payload = {
            "human_note": note,
            "watchlist": _watchlist_symbols(watchlist),
            "sellable_open_position_tickers": sellable_targets,
            "portfolio_positions": portfolio_positions(portfolio),
            "instructions": (
                "Interpret the human note into JSON. For broad sell requests, return only target_tickers "
                "from sellable_open_position_tickers, ordered by likely relevance. Do not invent tickers."
            ),
        }
        logger.info(
            "human.resolver.llm.start sellable=%s watchlist=%s note=%s",
            ",".join(sellable_targets),
            ",".join(_watchlist_symbols(watchlist)),
            _short_note(note),
        )
        raw = complete_json(_RESOLVER_PROMPT, json.dumps(payload, default=str))
        resolved = _parse_resolver_json(raw)
        logger.info(
            "human.resolver.llm.raw intent_type=%s targets=%s confidence=%s topic=%s",
            resolved.get("intent_type"),
            ",".join(str(item) for item in resolved.get("target_tickers", [])),
            resolved.get("confidence"),
            clean_text(str(resolved.get("topic") or ""), max_chars=120),
        )
    except Exception as error:
        raw_preview = _raw_preview(locals().get("raw", ""))
        logger.warning("human.resolver.llm.fail reason=%s raw_preview=%s note=%s", error, raw_preview, _short_note(note))
        return []

    if str(resolved.get("intent_type", "")).lower() not in {"broad_sell", "conditional_sell", "position_sweep"}:
        logger.info("human.resolver.llm.reject reason=unsupported_intent intent_type=%s", resolved.get("intent_type"))
        return []
    confidence = _confidence(resolved.get("confidence"))
    if confidence < 0.6:
        logger.info("human.resolver.llm.reject reason=confidence_below_threshold confidence=%.2f", confidence)
        return []
    sellable = set(sellable_targets)
    targets = [
        str(ticker).upper()
        for ticker in resolved.get("target_tickers", [])
        if isinstance(ticker, str) and str(ticker).upper() in sellable
    ]
    targets = _dedupe(targets)
    if not targets:
        logger.info(
            "human.resolver.llm.reject reason=no_valid_sellable_targets requested=%s sellable=%s",
            ",".join(str(ticker) for ticker in resolved.get("target_tickers", [])),
            ",".join(sellable_targets),
        )
        return []
    total = len(targets)
    logger.info(
        "human.resolver.llm.accept targets=%s confidence=%.2f topic=%s",
        ",".join(targets),
        confidence,
        clean_text(str(resolved.get("topic") or ""), max_chars=120),
    )
    return [
        HumanInstruction(
            note,
            target,
            "human_llm_position_sweep",
            index,
            total,
            confidence,
            clean_text(str(resolved.get("rationale") or ""), max_chars=500) or None,
            clean_text(str(resolved.get("topic") or ""), max_chars=240) or None,
        )
        for index, target in enumerate(targets, start=1)
    ]


def _open_position_targets(
    watchlist: list[str] | str | None,
    portfolio: dict[str, Any] | None,
    *,
    fallback_ticker: str | None = None,
) -> list[str]:
    open_positions = portfolio_positions(portfolio)
    watchlist_symbols = _watchlist_symbols(watchlist)
    if watchlist_symbols:
        targets = [symbol for symbol in watchlist_symbols if open_positions.get(symbol, {}).get("qty", 0.0) > 0]
    else:
        targets = [symbol for symbol, position in open_positions.items() if position.get("qty", 0.0) > 0]
    return _deprioritize_fallback_ticker(targets, fallback_ticker)


def _watchlist_symbols(watchlist: list[str] | str | None) -> list[str]:
    if watchlist is None:
        return []
    raw_items = watchlist.split(",") if isinstance(watchlist, str) else list(watchlist)
    symbols: list[str] = []
    for item in raw_items:
        symbol = clean_text(str(item), max_chars=16).upper()
        if re.fullmatch(r"[A-Z]{1,5}", symbol) and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _deprioritize_fallback_ticker(targets: list[str], fallback_ticker: str | None) -> list[str]:
    fallback = (fallback_ticker or "").upper()
    if not fallback or len(targets) <= 1 or fallback not in targets:
        return targets
    return [target for target in targets if target != fallback] + [fallback]


def _short_note(note: str) -> str:
    return clean_text(note, max_chars=120)


def _parse_resolver_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("resolver_response_must_be_object")
    return parsed


def _raw_preview(raw: Any) -> str:
    text = clean_text(str(raw or ""), max_chars=160)
    return text if text else "<empty>"


_RESOLVER_PROMPT = """You resolve human trading instructions into strict JSON.
Return only JSON with:
{"intent_type":"broad_sell|conditional_sell|position_sweep|forced_buy|forced_sell|advisory","target_tickers":["AAPL"],"excluded_tickers":[],"topic":"...","confidence":0.0,"rationale":"...","requires_validation":true}
For broad sell requests, target_tickers must be selected only from sellable_open_position_tickers.
If unsure, return an empty target_tickers list with confidence below 0.6.
"""
