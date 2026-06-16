from __future__ import annotations

from trading_agent.core import TechnicalIndicators


def _ema(values: list[float], period: int) -> list[float]:
    """Calculate an exponential moving average used by MACD."""
    if not values:
        return []
    alpha = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append((value * alpha) + (result[-1] * (1 - alpha)))
    return result


def _rsi(values: list[float], period: int = 14) -> float | None:
    """Estimate momentum by comparing recent average gains and losses."""
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(values, values[1:]):
        change = current - previous
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(closes: list[float]) -> TechnicalIndicators:
    """Return SMA, RSI and MACD signals from validated close prices."""
    clean = [float(value) for value in closes if value > 0]
    notes: list[str] = []
    if len(clean) < 26:
        notes.append("not enough closes for full MACD/SMA50 confidence")
    sma_20 = sum(clean[-20:]) / 20 if len(clean) >= 20 else None
    sma_50 = sum(clean[-50:]) / 50 if len(clean) >= 50 else None
    rsi_14 = _rsi(clean)

    macd = macd_signal = macd_histogram = None
    if len(clean) >= 26:
        ema_12 = _ema(clean, 12)
        ema_26 = _ema(clean, 26)
        macd_line = [a - b for a, b in zip(ema_12[-len(ema_26) :], ema_26)]
        signal = _ema(macd_line, 9)
        macd = macd_line[-1]
        macd_signal = signal[-1]
        macd_histogram = macd - macd_signal

    confidence = "high" if len(clean) >= 50 else "medium" if len(clean) >= 26 else "none"
    return TechnicalIndicators(
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        confidence=confidence,
        notes=notes,
    )
