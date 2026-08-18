from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PivotPoint:
    date: str
    price: float
    pivot_type: str


@dataclass
class StructureSummary:
    trend_state: str
    structure_state: str
    swing_highs: list[PivotPoint]
    swing_lows: list[PivotPoint]
    breakout_level: float | None
    prior_breakout_retest_zone: dict | None
    consolidation_range: dict | None
    gap_zone: dict | None
    structure_flags: list[str]
    ema_structure: dict


def find_pivots(frame: pd.DataFrame, *, lookback: int = 4, max_points: int = 6) -> tuple[list[PivotPoint], list[PivotPoint]]:
    if frame.empty or len(frame) < (lookback * 2 + 3):
        return [], []

    highs: list[PivotPoint] = []
    lows: list[PivotPoint] = []
    highs_series = frame["high"].tolist()
    lows_series = frame["low"].tolist()
    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()

    for idx in range(lookback, len(frame) - lookback):
        hi = highs_series[idx]
        lo = lows_series[idx]
        hi_window = highs_series[idx - lookback : idx + lookback + 1]
        lo_window = lows_series[idx - lookback : idx + lookback + 1]
        if hi == max(hi_window):
            highs.append(PivotPoint(date=dates[idx], price=float(hi), pivot_type="high"))
        if lo == min(lo_window):
            lows.append(PivotPoint(date=dates[idx], price=float(lo), pivot_type="low"))

    return highs[-max_points:], lows[-max_points:]


def classify_structure_state(
    frame: pd.DataFrame,
    swing_highs: list[PivotPoint],
    swing_lows: list[PivotPoint],
    *,
    extended_from_ema20_pct: float = 0.08,
    parabolic_from_ema20_pct: float = 0.12,
    base_max_atr_range: float = 3.2,
) -> tuple[str, str, list[str], dict]:
    """Return a rich structure state plus the backward-compatible trend state."""

    flags: list[str] = []
    if frame.empty:
        return "range", "range", flags, {}

    close = float(frame["close"].iloc[-1])
    def _value(column: str) -> float | None:
        if column not in frame.columns or pd.isna(frame[column].iloc[-1]):
            return None
        return float(frame[column].iloc[-1])

    ema20 = _value("ema20")
    ema50 = _value("ema50") or _value("sma50")
    ema100 = _value("ema100") or _value("sma100")
    ema200 = _value("ema200") or _value("sma200")
    atr_val = _value("atr") or max(close * 0.01, 0.01)
    volume_ratio = _value("volume_ratio") or 1.0

    slopes = {
        period: (_value(f"ema{period}_slope_pct") or 0.0)
        for period in (20, 50, 100, 200)
    }
    ema_values = [ema20, ema50, ema100, ema200]
    available_emas = [value for value in ema_values if value is not None]
    above_count = sum(1 for value in available_emas if close >= value)
    below_count = len(available_emas) - above_count
    bullish_stack = all(value is not None for value in ema_values) and bool(ema20 >= ema50 >= ema100 >= ema200)
    bearish_stack = all(value is not None for value in ema_values) and bool(ema20 <= ema50 <= ema100 <= ema200)
    distance_ema20 = ((close - ema20) / max(ema20, 1e-9)) if ema20 is not None else 0.0

    higher_highs = len(swing_highs) >= 2 and swing_highs[-1].price > swing_highs[-2].price
    higher_lows = len(swing_lows) >= 2 and swing_lows[-1].price > swing_lows[-2].price
    lower_highs = len(swing_highs) >= 2 and swing_highs[-1].price < swing_highs[-2].price
    lower_lows = len(swing_lows) >= 2 and swing_lows[-1].price < swing_lows[-2].price

    if higher_highs:
        flags.append("higher_highs")
    if higher_lows:
        flags.append("higher_lows")
    if lower_highs:
        flags.append("lower_highs")
    if lower_lows:
        flags.append("lower_lows")
    if bullish_stack:
        flags.append("bullish_ema_stack")
    if bearish_stack:
        flags.append("bearish_ema_stack")

    prior_high = float(frame["high"].iloc[-21:-1].max()) if len(frame) >= 22 else None
    breakout = prior_high is not None and close > prior_high and distance_ema20 <= parabolic_from_ema20_pct
    recent = frame.tail(min(20, len(frame)))
    base_range_atr = (float(recent["high"].max()) - float(recent["low"].min())) / max(atr_val, 1e-9)
    base_building = base_range_atr <= base_max_atr_range and abs(slopes[20]) <= 0.012
    heavy_selloff = volume_ratio >= 1.25 and len(frame) >= 2 and close < float(frame["close"].iloc[-2])
    recent_reclaim = (
        ema20 is not None
        and len(frame) >= 2
        and float(frame["close"].iloc[-2]) < float(frame["ema20"].iloc[-2])
        and close >= ema20
    )

    if distance_ema20 >= parabolic_from_ema20_pct or (
        distance_ema20 >= extended_from_ema20_pct and above_count >= 3
    ):
        state, legacy = "extended", "uptrend"
        flags.append("extended_from_ema20")
    elif breakout and (bullish_stack or slopes[20] > 0):
        state, legacy = "breakout", "uptrend"
        flags.append("breakout_above_recent_high")
    elif below_count == len(available_emas) and heavy_selloff and slopes[20] < 0:
        state, legacy = "structural_breakdown", "downtrend"
        flags.extend(["below_all_major_emas", "heavy_selloff"])
    elif below_count >= 3 and (lower_lows or slopes[20] < -0.01):
        state, legacy = "trend_damage", "weak_breakdown_risk"
        flags.append("below_multiple_major_emas")
    elif below_count >= 2 and recent_reclaim:
        state, legacy = "reversal_attempt", "weak_breakdown_risk"
        flags.append("short_term_reclaim")
    elif bullish_stack and ema50 is not None and close >= ema50 and distance_ema20 <= 0.035 and not lower_lows:
        state, legacy = "healthy_pullback", "pullback_in_uptrend"
        flags.append("pullback_near_rising_ema_support")
    elif ema200 is not None and close >= ema200 and ema100 is not None and slopes[100] >= -0.005 and close < (ema50 or close + 1):
        state, legacy = "deep_pullback", "pullback_in_uptrend"
        flags.append("deep_pullback_above_long_term_support")
    elif base_building:
        state, legacy = "base_building", "range"
        flags.append("compressed_base")
    elif lower_highs and lower_lows:
        state, legacy = "structural_breakdown", "downtrend"
    else:
        state, legacy = "range", "range"
        flags.append("range_bound")

    ema_structure = {
        "bullish_stack": bool(bullish_stack),
        "bearish_stack": bool(bearish_stack),
        "above_ema_count": int(above_count),
        "below_ema_count": int(below_count),
        "distance_from_ema20_pct": round(distance_ema20 * 100.0, 4),
        "slopes_pct_5bar": {f"ema{period}": round(slopes[period] * 100.0, 4) for period in slopes},
    }
    return state, legacy, flags, ema_structure


def detect_consolidation(frame: pd.DataFrame, *, window: int, atr_mult: float) -> dict | None:
    if frame.empty or len(frame) < window:
        return None
    recent = frame.tail(window)
    atr_val = recent["atr"].iloc[-1] if "atr" in recent.columns else None
    if pd.isna(atr_val) or atr_val is None:
        return None

    high = float(recent["high"].max())
    low = float(recent["low"].min())
    if (high - low) <= float(atr_val) * atr_mult:
        return {
            "lower": low,
            "upper": high,
            "source_tags": ["consolidation"],
        }
    return None


def detect_gap_zone(frame: pd.DataFrame) -> dict | None:
    if frame.empty or len(frame) < 3:
        return None
    recent = frame.tail(15).reset_index(drop=True)
    best_gap = None
    best_size = 0.0
    for idx in range(1, len(recent)):
        prev_high = float(recent.loc[idx - 1, "high"])
        prev_low = float(recent.loc[idx - 1, "low"])
        cur_low = float(recent.loc[idx, "low"])
        cur_high = float(recent.loc[idx, "high"])
        if cur_low > prev_high:
            size = cur_low - prev_high
            if size > best_size:
                best_gap = {"lower": prev_high, "upper": cur_low, "source_tags": ["gap_up"]}
                best_size = size
        elif cur_high < prev_low:
            size = prev_low - cur_high
            if size > best_size:
                best_gap = {"lower": cur_high, "upper": prev_low, "source_tags": ["gap_down"]}
                best_size = size
    return best_gap


def summarize_structure(
    frame: pd.DataFrame,
    *,
    pivot_lookback: int,
    pivot_max_points: int,
    consolidation_window: int,
    consolidation_range_atr_mult: float,
    extended_from_ema20_pct: float = 0.08,
    parabolic_from_ema20_pct: float = 0.12,
    base_max_atr_range: float = 3.2,
) -> StructureSummary:
    highs, lows = find_pivots(frame, lookback=pivot_lookback, max_points=pivot_max_points)
    structure_state, trend_state, flags, ema_structure = classify_structure_state(
        frame,
        highs,
        lows,
        extended_from_ema20_pct=extended_from_ema20_pct,
        parabolic_from_ema20_pct=parabolic_from_ema20_pct,
        base_max_atr_range=base_max_atr_range,
    )

    breakout_level = highs[-1].price if highs else None
    prior_breakout_retest_zone = None
    if breakout_level is not None and len(lows) >= 1:
        last_low = lows[-1].price
        if last_low <= breakout_level:
            lower = min(last_low, breakout_level)
            upper = max(last_low, breakout_level)
            prior_breakout_retest_zone = {
                "lower": float(lower),
                "upper": float(upper),
                "source_tags": ["breakout_retest"],
            }

    consolidation = detect_consolidation(
        frame,
        window=consolidation_window,
        atr_mult=consolidation_range_atr_mult,
    )
    gap_zone = detect_gap_zone(frame)

    return StructureSummary(
        trend_state=trend_state,
        structure_state=structure_state,
        swing_highs=highs,
        swing_lows=lows,
        breakout_level=(float(breakout_level) if breakout_level is not None else None),
        prior_breakout_retest_zone=prior_breakout_retest_zone,
        consolidation_range=consolidation,
        gap_zone=gap_zone,
        structure_flags=flags,
        ema_structure=ema_structure,
    )
