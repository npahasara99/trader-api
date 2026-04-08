"""Cheap swing-trade pre-scan scoring used before full structured planning."""

from __future__ import annotations

from .config import PlanningConfig
from .indicators import add_indicator_columns, bars_to_frame, latest_value


SECTOR_BENCHMARKS = {
    "technology": "XLK",
    "software": "XLK",
    "it services": "XLK",
    "networking": "XLK",
    "consumer electronics": "XLK",
    "semiconductors": "SMH",
    "financials": "XLF",
    "banks": "XLF",
    "payments": "XLF",
    "capital markets": "XLF",
    "health care": "XLV",
    "biopharma": "XLV",
    "biotech": "XLV",
    "medical devices": "XLV",
    "pharma": "XLV",
    "life sciences": "XLV",
    "communication services": "XLC",
    "media": "XLC",
    "internet platforms": "XLC",
    "telecom": "XLC",
    "consumer discretionary": "XLY",
    "retail": "XLY",
    "internet retail": "XLY",
    "restaurants": "XLY",
    "autos": "XLY",
    "consumer staples": "XLP",
    "energy": "XLE",
    "oil and gas": "XLE",
    "industrials": "XLI",
    "aerospace": "XLI",
    "machinery": "XLI",
    "electrical equipment": "XLI",
    "materials": "XLB",
    "real estate": "XLRE",
    "retail reit": "XLRE",
    "utilities": "XLU",
}


def _clip(value: float) -> float:
    return max(0.0, min(10.0, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def sector_benchmark_symbol_for_meta(meta: dict | None) -> str | None:
    meta = meta or {}
    industry = str(meta.get("industry") or "").lower()
    sector = str(meta.get("sector") or "").lower()
    return SECTOR_BENCHMARKS.get(industry) or SECTOR_BENCHMARKS.get(sector)


def _return_over(frame, periods: int) -> float | None:
    if frame.empty or len(frame) <= periods:
        return None
    start = _safe_float(frame["close"].iloc[-periods - 1], 0.0)
    end = _safe_float(frame["close"].iloc[-1], 0.0)
    if start <= 0:
        return None
    return (end / start) - 1.0


def _relative_edge(stock_ret: float | None, bench_ret: float | None) -> float | None:
    if stock_ret is None or bench_ret is None:
        return None
    return stock_ret - bench_ret


def build_pre_scan_profile(
    *,
    ticker: str,
    current_price: float | None,
    bars: list[dict],
    benchmark_bars: dict[str, list[dict]] | None,
    sector_benchmark_symbol: str | None,
    earnings_context: dict | None,
    config: PlanningConfig,
) -> dict:
    """Compute a cheap swing-oriented score before full planning."""

    frame = add_indicator_columns(
        bars_to_frame(bars),
        atr_window=config.atr_window,
        volume_window=config.volume_window,
    )
    if frame.empty or len(frame) < config.pre_scan_min_history_bars or current_price is None:
        return {
            "ticker": ticker,
            "pre_scan_score": 0.0,
            "pre_scan_reason_tags": ["insufficient_history"],
            "sector_relative_strength": None,
            "scan_shortlisted": False,
            "scan_rejection_reason": "insufficient_history",
        }

    benchmark_bars = benchmark_bars or {}
    benchmark_frames = {
        symbol: add_indicator_columns(bars_to_frame(rows), atr_window=config.atr_window, volume_window=config.volume_window)
        for symbol, rows in benchmark_bars.items()
        if rows
    }

    close = _safe_float(current_price)
    ema20 = latest_value(frame, "ema20")
    sma50 = latest_value(frame, "sma50")
    sma200 = latest_value(frame, "sma200")
    atr_pct = latest_value(frame, "atr_pct")
    avg_volume = latest_value(frame, "avg_volume")
    volume_ratio = latest_value(frame, "volume_ratio")
    ret_20 = _return_over(frame, 20)
    ret_60 = _return_over(frame, 60)

    highs_60 = frame["close"].tail(60).max()
    lows_252 = frame["close"].tail(min(252, len(frame))).min()
    highs_252 = frame["close"].tail(min(252, len(frame))).max()
    pullback_pct = ((float(highs_60) - close) / max(float(highs_60), 1e-9)) if highs_60 else 0.0
    position_52w = 0.5
    if highs_252 and highs_252 > lows_252:
        position_52w = (close - float(lows_252)) / max(float(highs_252) - float(lows_252), 1e-9)

    rs_1m_vals: list[float] = []
    rs_3m_vals: list[float] = []
    for symbol in ("SPY", "QQQ"):
        bench = benchmark_frames.get(symbol)
        if bench is None or bench.empty:
            continue
        edge_1m = _relative_edge(ret_20, _return_over(bench, 20))
        edge_3m = _relative_edge(ret_60, _return_over(bench, 60))
        if edge_1m is not None:
            rs_1m_vals.append(edge_1m)
        if edge_3m is not None:
            rs_3m_vals.append(edge_3m)

    sector_relative_strength = None
    if sector_benchmark_symbol:
        bench = benchmark_frames.get(sector_benchmark_symbol)
        if bench is not None and not bench.empty:
            sector_relative_strength = _relative_edge(ret_20, _return_over(bench, 20))

    trend_score = 4.2
    if ema20 is not None and close >= ema20:
        trend_score += 1.2
    if sma50 is not None and close >= sma50:
        trend_score += 1.35
    if sma200 is not None and close >= sma200:
        trend_score += 0.95
    if ema20 is not None and sma50 is not None and ema20 >= sma50:
        trend_score += 1.0
    if sma50 is not None and sma200 is not None and sma50 >= sma200:
        trend_score += 0.9
    if ret_20 is not None and ret_20 > 0:
        trend_score += 0.45
    if ret_60 is not None and ret_60 > 0:
        trend_score += 0.55
    trend_score = _clip(trend_score)

    relative_strength_score = 5.0
    if rs_1m_vals or rs_3m_vals:
        avg_edge = (
            (sum(rs_1m_vals) / max(len(rs_1m_vals), 1)) * 0.55
            + (sum(rs_3m_vals) / max(len(rs_3m_vals), 1)) * 0.45
        )
        relative_strength_score = _clip(5.0 + avg_edge * 60.0)

    sector_relative_score = 5.0
    if sector_relative_strength is not None:
        sector_relative_score = _clip(5.0 + sector_relative_strength * 70.0)

    pullback_score = 5.0
    dist_to_ema20 = None if ema20 is None else (close - ema20) / max(ema20, 1e-9)
    dist_to_sma50 = None if sma50 is None else (close - sma50) / max(sma50, 1e-9)
    if 0.03 <= pullback_pct <= 0.14:
        pullback_score += 1.8
    elif pullback_pct < 0.015:
        pullback_score -= 0.8
    elif pullback_pct > 0.2:
        pullback_score -= 1.3
    if dist_to_ema20 is not None and -0.04 <= dist_to_ema20 <= 0.03:
        pullback_score += 1.0
    elif dist_to_ema20 is not None and dist_to_ema20 > 0.08:
        pullback_score -= 1.1
    if dist_to_sma50 is not None and close >= sma50:
        pullback_score += 0.7
    if position_52w >= 0.35:
        pullback_score += 0.5
    if position_52w >= 0.9:
        pullback_score -= 0.4
    pullback_score = _clip(pullback_score)

    volatility_score = 5.5
    atr_pct_val = (_safe_float(atr_pct, -1.0) * 100.0) if atr_pct is not None else -1.0
    if atr_pct_val < 0:
        volatility_score = 5.0
    elif atr_pct_val < 1.0:
        volatility_score = 4.8
    elif atr_pct_val <= 5.5:
        volatility_score = 8.0
    elif atr_pct_val <= 7.5:
        volatility_score = 6.1
    else:
        volatility_score = 3.8

    liquidity_score = 4.0
    avg_dollar_volume = _safe_float(avg_volume) * close if avg_volume is not None else 0.0
    if avg_dollar_volume >= config.pre_scan_min_avg_dollar_volume:
        liquidity_score = 7.2
    if avg_dollar_volume >= config.pre_scan_min_avg_dollar_volume * 3.0:
        liquidity_score = 8.4
    elif avg_dollar_volume < config.pre_scan_min_avg_dollar_volume * 0.5:
        liquidity_score = 2.8

    volume_score = 5.0
    if volume_ratio is not None:
        if volume_ratio >= 1.2:
            volume_score = 6.7
        elif volume_ratio >= 0.85:
            volume_score = 5.6
        else:
            volume_score = 4.3

    earnings_score = 7.0
    days_to_earnings = (earnings_context or {}).get("days_to_earnings")
    if days_to_earnings is not None:
        dte = int(days_to_earnings)
        if dte <= config.earnings_hard_block_days:
            earnings_score = 1.2
        elif dte <= config.earnings_penalty_near_days:
            earnings_score = 2.8
        elif dte <= config.earnings_penalty_mid_days:
            earnings_score = 5.1

    weights = config.pre_scan_weights
    total_weight = sum(weights.values())
    pre_scan_score = (
        trend_score * weights["trend"]
        + relative_strength_score * weights["relative_strength"]
        + sector_relative_score * weights["sector_relative"]
        + pullback_score * weights["pullback"]
        + volatility_score * weights["volatility"]
        + volume_score * weights["volume"]
        + earnings_score * weights["earnings"]
        + liquidity_score * weights["liquidity"]
    ) / max(total_weight, 1e-9)
    pre_scan_score = round(_clip(pre_scan_score), 4)

    tags: list[str] = []
    if close >= _safe_float(ema20, close + 1.0):
        tags.append("above_ema20")
    if close >= _safe_float(sma50, close + 1.0):
        tags.append("above_sma50")
    if relative_strength_score >= 6.2:
        tags.append("relative_strength_supportive")
    elif relative_strength_score <= 4.4:
        tags.append("relative_strength_weak")
    if sector_relative_score >= 6.0:
        tags.append("sector_leader")
    if 0.03 <= pullback_pct <= 0.14:
        tags.append("constructive_pullback")
    elif pullback_pct > 0.2:
        tags.append("deep_pullback")
    if avg_dollar_volume >= config.pre_scan_min_avg_dollar_volume:
        tags.append("liquid")
    else:
        tags.append("liquidity_low")
    if atr_pct_val > 0:
        if 1.0 <= atr_pct_val <= 5.5:
            tags.append("atr_swing_friendly")
        elif atr_pct_val > 7.5:
            tags.append("atr_too_high")
    if days_to_earnings is None or int(days_to_earnings) > config.earnings_penalty_near_days:
        tags.append("no_near_earnings_blocker")
    else:
        tags.append("earnings_near")

    return {
        "ticker": ticker,
        "pre_scan_score": float(pre_scan_score),
        "pre_scan_reason_tags": tags[:8],
        "sector_relative_strength": None if sector_relative_strength is None else round(float(sector_relative_strength), 6),
        "scan_shortlisted": False,
        "scan_rejection_reason": None,
    }
