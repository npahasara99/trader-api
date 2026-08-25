"""Cheap swing-trade pre-scan scoring used before full structured planning."""

from __future__ import annotations

import math

from .config import PlanningConfig
from .indicators import add_indicator_columns, bars_to_frame, latest_value
from .setup_archetypes import score_setup_families


SECTOR_BENCHMARKS = {
    "technology": "XLK",
    "information technology": "XLK",
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


def _finite_float(value: object, default: float | None = None) -> float | None:
    parsed = _safe_float(value, float("nan"))
    return parsed if math.isfinite(parsed) else default


def classify_volatility(atr_pct_ratio: float | None, config: PlanningConfig) -> dict:
    """Classify ATR percentage without treating the preferred band as a hard gate."""

    if atr_pct_ratio is None:
        return {
            "atr_percent": None,
            "volatility_regime": "unknown",
            "volatility_suitability_score": 5.0,
        }
    atr_percent = max(0.0, float(atr_pct_ratio) * 100.0)
    if atr_percent < config.atr_pct_too_slow:
        regime, score = "too_slow", 3.2
    elif atr_percent < config.atr_pct_preferred_min:
        regime, score = "low_volatility", 5.4
    elif atr_percent <= config.atr_pct_preferred_max:
        regime, score = "preferred", 8.5
    elif atr_percent <= config.atr_pct_high_risk_max:
        regime, score = "high_risk", 6.0
    else:
        regime, score = "very_high_risk", 3.4
    return {
        "atr_percent": round(atr_percent, 4),
        "volatility_regime": regime,
        "volatility_suitability_score": score,
    }


def build_universe_suitability(
    *,
    current_price: float | None,
    frame,
    config: PlanningConfig,
) -> dict:
    """Evaluate basic price, share-volume, and history requirements."""

    price = _safe_float(current_price, 0.0)
    avg_volume = latest_value(frame, "avg_volume") if frame is not None and not frame.empty else None
    avg_volume_value = _safe_float(avg_volume, 0.0)
    history_bars = 0 if frame is None else len(frame)
    rejection_reasons: list[str] = []
    if price < config.min_price:
        rejection_reasons.append("price_below_minimum")
    if avg_volume_value < config.min_avg_daily_volume:
        rejection_reasons.append("average_daily_volume_below_minimum")
    if history_bars < config.min_history_bars:
        rejection_reasons.append("insufficient_history")

    liquidity_score = 5.0
    if price <= 0:
        liquidity_score = 0.0
    elif avg_volume_value >= config.min_avg_daily_volume * 5.0:
        liquidity_score = 9.0
    elif avg_volume_value >= config.min_avg_daily_volume * 2.0:
        liquidity_score = 8.0
    elif avg_volume_value >= config.min_avg_daily_volume:
        liquidity_score = 7.0
    elif avg_volume_value >= config.min_avg_daily_volume * 0.5:
        liquidity_score = 4.0
    else:
        liquidity_score = 2.0
    if price < config.min_price:
        liquidity_score = min(liquidity_score, 3.0)

    return {
        "universe_eligible": not rejection_reasons,
        "universe_rejection_reasons": rejection_reasons,
        "average_daily_volume": round(avg_volume_value, 2),
        "average_dollar_volume": round(avg_volume_value * price, 2),
        "history_bars": int(history_bars),
        "liquidity_score": round(liquidity_score, 3),
    }


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
    ema50 = latest_value(frame, "ema50")
    ema100 = latest_value(frame, "ema100")
    ema200 = latest_value(frame, "ema200")
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

    atr_value = _finite_float(latest_value(frame, "atr"), max(close * 0.02, 0.01)) or max(close * 0.02, 0.01)
    recent = frame.tail(min(10, len(frame))).copy()
    down_days = recent[recent["close"] < recent["close"].shift(1)]
    up_days = recent[recent["close"] > recent["close"].shift(1)]
    baseline_volume = max(_safe_float(frame["volume"].tail(20).mean(), 1.0), 1.0)
    down_volume_ratio = _safe_float(down_days["volume"].mean(), 0.0) / baseline_volume if not down_days.empty else 0.0
    up_volume_ratio = _safe_float(up_days["volume"].mean(), 0.0) / baseline_volume if not up_days.empty else 0.0
    pullback_volume_quality = 5.0
    if 0.0 < down_volume_ratio <= 0.9:
        pullback_volume_quality += 2.0
    elif down_volume_ratio >= 1.25:
        pullback_volume_quality -= 2.4
    if up_volume_ratio >= 1.15:
        pullback_volume_quality += 1.4
    elif 0.0 < up_volume_ratio < 0.85:
        pullback_volume_quality -= 0.6
    pullback_volume_quality = _clip(pullback_volume_quality)

    ema50_prior = _finite_float(frame["ema50"].iloc[-11]) if len(frame) >= 11 and "ema50" in frame else None
    ema50_rising = bool(ema50 is not None and ema50_prior is not None and float(ema50) > ema50_prior)
    bullish_stack = bool(
        ema20 is not None and ema50 is not None and ema100 is not None and ema200 is not None
        and float(ema20) >= float(ema50) >= float(ema100) >= float(ema200)
    )
    continuation_structure = 3.2
    continuation_structure += 2.0 if bullish_stack else 0.0
    continuation_structure += 1.3 if ema50_rising else 0.0
    continuation_structure += 1.0 if ret_20 is not None and ret_20 > 0.02 else 0.0
    continuation_structure += 1.0 if ret_60 is not None and ret_60 > 0.05 else 0.0
    continuation_structure += 0.8 if position_52w >= 0.72 else 0.0
    continuation_structure = _clip(continuation_structure)

    support_confluence = 3.5
    if dist_to_ema20 is not None and abs(dist_to_ema20) <= 0.035:
        support_confluence += 2.4
    if dist_to_sma50 is not None and abs(dist_to_sma50) <= 0.055:
        support_confluence += 1.8
    if ema50 is not None and close >= float(ema50):
        support_confluence += 1.0
    support_confluence = _clip(support_confluence)

    price_location_score = _clip(
        7.6 if 0.03 <= pullback_pct <= 0.12
        else 7.2 if 0.0 <= pullback_pct < 0.03 and position_52w < 0.96
        else 6.4 if 0.12 < pullback_pct <= 0.22
        else 4.0
    )
    deep_pullback_quality = _clip(
        8.4 if 0.12 <= pullback_pct <= 0.25
        else 6.2 if 0.08 <= pullback_pct < 0.12
        else 3.8 if pullback_pct < 0.08
        else 4.6
    )

    recent_high = _finite_float(frame["high"].tail(20).max(), close)
    recent_low = _finite_float(frame["low"].tail(20).min(), close)
    recent_range_atr = (float(recent_high) - float(recent_low)) / max(atr_value, 1e-9)
    base_quality = _clip(
        8.2 if recent_range_atr <= config.structure_base_max_atr_range and continuation_structure >= 6.0
        else 6.2 if recent_range_atr <= config.structure_base_max_atr_range * 1.35
        else 3.5
    )

    prior_breakout_high = None
    if len(frame) >= 25:
        prior_breakout_high = _finite_float(frame["high"].iloc[-25:-5].max())
    breakout_retest_quality = 3.0
    if prior_breakout_high is not None:
        recent_break = _finite_float(frame["close"].tail(5).max(), close) or close
        held_retest = close >= prior_breakout_high * 0.98
        if recent_break > prior_breakout_high and held_retest:
            breakout_retest_quality = 8.4
        elif close >= prior_breakout_high * 0.985:
            breakout_retest_quality = 6.2

    five_day_return = _return_over(frame, min(5, max(len(frame) - 1, 1))) or 0.0
    reversal_quality = 3.0
    if position_52w <= 0.35:
        reversal_quality += 2.0
    if five_day_return > 0.015:
        reversal_quality += 1.8
    if up_volume_ratio >= 1.15:
        reversal_quality += 1.4
    if close < _safe_float(sma50, close):
        reversal_quality += 0.5
    reversal_quality = _clip(reversal_quality)

    confirmation_quality = _clip(
        5.0
        + (1.8 if five_day_return > 0.01 else -0.5)
        + (1.2 if up_volume_ratio >= 1.15 else 0.0)
        - (1.6 if down_volume_ratio >= 1.25 else 0.0)
    )
    target_quality = _clip(7.0 - (2.2 if position_52w >= 0.97 else 0.0) - (1.0 if pullback_pct > 0.25 else 0.0))

    volatility = classify_volatility(atr_pct, config)
    volatility_score = float(volatility["volatility_suitability_score"])
    atr_pct_val = volatility["atr_percent"] if volatility["atr_percent"] is not None else -1.0

    universe = build_universe_suitability(current_price=current_price, frame=frame, config=config)
    liquidity_score = float(universe["liquidity_score"])
    avg_dollar_volume = _safe_float(avg_volume) * close if avg_volume is not None else 0.0
    if avg_dollar_volume < config.pre_scan_min_avg_dollar_volume:
        liquidity_score = min(liquidity_score, 4.0)

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

    lane_components = {
        "trend_strength": trend_score,
        "pullback_quality": pullback_score,
        "deep_pullback_quality": deep_pullback_quality,
        "price_location": price_location_score,
        "relative_strength": relative_strength_score,
        "pullback_volume": pullback_volume_quality,
        "support_confluence": support_confluence,
        "continuation_structure": continuation_structure,
        "confirmation": confirmation_quality,
        "target_quality": target_quality,
        "base_quality": base_quality,
        "breakout_retest_quality": breakout_retest_quality,
        "reversal_quality": reversal_quality,
        "volatility": volatility_score,
        "liquidity": liquidity_score,
        "earnings": earnings_score,
    }
    lane_profile = score_setup_families(
        lane_components,
        weights_by_family=config.setup_family_score_weights,
    )

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
    legacy_pre_scan_score = round(_clip(pre_scan_score), 4)
    family_score = max(lane_profile["setup_lane_scores"].values(), default=legacy_pre_scan_score)
    operational_score = (volatility_score + liquidity_score + earnings_score) / 3.0
    pre_scan_score = round(_clip(family_score * 0.82 + operational_score * 0.18), 4)

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
        if config.atr_pct_preferred_min <= atr_pct_val <= config.atr_pct_preferred_max:
            tags.append("atr_swing_friendly")
        elif atr_pct_val > config.atr_pct_high_risk_max:
            tags.append("atr_too_high")
    if days_to_earnings is None or int(days_to_earnings) > config.earnings_penalty_near_days:
        tags.append("no_near_earnings_blocker")
    else:
        tags.append("earnings_near")

    scan_rejections = list(universe["universe_rejection_reasons"])
    return {
        "ticker": ticker,
        "pre_scan_score": float(pre_scan_score),
        "pre_scan_reason_tags": tags[:8],
        "legacy_pre_scan_score": legacy_pre_scan_score,
        "setup_family": lane_profile["setup_family"],
        "setup_family_score": family_score,
        "setup_lane_qualified": bool(family_score >= config.setup_lane_min_score),
        "setup_lane_scores": lane_profile["setup_lane_scores"],
        "setup_lane_components": {key: round(float(value), 4) for key, value in lane_components.items()},
        "setup_lane_contributions": lane_profile["setup_lane_contributions"],
        "alternative_setup_families": lane_profile["alternative_setup_families"][:3],
        "pullback_volume_quality": pullback_volume_quality,
        "continuation_structure_score": continuation_structure,
        "base_quality_score": base_quality,
        "breakout_retest_quality_score": breakout_retest_quality,
        "reversal_quality_score": reversal_quality,
        "down_volume_ratio": round(down_volume_ratio, 4),
        "up_volume_ratio": round(up_volume_ratio, 4),
        "sector_relative_strength": None if sector_relative_strength is None else round(float(sector_relative_strength), 6),
        "scan_shortlisted": False,
        "scan_rejection_reason": ",".join(scan_rejections) if scan_rejections else None,
        "universe_suitability": universe,
        **volatility,
    }
