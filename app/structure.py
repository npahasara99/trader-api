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
    swing_highs: list[PivotPoint]
    swing_lows: list[PivotPoint]
    breakout_level: float | None
    prior_breakout_retest_zone: dict | None
    consolidation_range: dict | None
    gap_zone: dict | None
    structure_flags: list[str]


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


def _classify_trend(frame: pd.DataFrame, swing_highs: list[PivotPoint], swing_lows: list[PivotPoint]) -> tuple[str, list[str]]:
    flags: list[str] = []
    if frame.empty:
        return "range", flags

    close = float(frame["close"].iloc[-1])
    ema20 = frame["ema20"].iloc[-1] if "ema20" in frame.columns else None
    sma50 = frame["sma50"].iloc[-1] if "sma50" in frame.columns else None
    sma200 = frame["sma200"].iloc[-1] if "sma200" in frame.columns else None

    higher_highs = len(swing_highs) >= 2 and swing_highs[-1].price > swing_highs[-2].price
    higher_lows = len(swing_lows) >= 2 and swing_lows[-1].price > swing_lows[-2].price
    lower_highs = len(swing_highs) >= 2 and swing_highs[-1].price < swing_highs[-2].price
    lower_lows = len(swing_lows) >= 2 and swing_lows[-1].price < swing_lows[-2].price

    if higher_highs and higher_lows and ema20 is not None and sma50 is not None and close >= ema20 >= sma50:
        flags.append("higher_highs")
        flags.append("higher_lows")
        return "uptrend", flags

    if ema20 is not None and sma50 is not None and close >= sma50 and lower_lows is False and len(swing_lows) >= 1:
        last_low = swing_lows[-1].price
        if close <= max(float(ema20), float(sma50)) * 1.02 and close >= last_low:
            flags.append("pullback_near_support")
            return "pullback_in_uptrend", flags

    if lower_highs and lower_lows and sma50 is not None and sma200 is not None and close < sma50 <= sma200:
        flags.append("lower_highs")
        flags.append("lower_lows")
        return "downtrend", flags

    if lower_lows or (sma50 is not None and close < sma50 and ema20 is not None and close < ema20):
        flags.append("breakdown_risk")
        return "weak_breakdown_risk", flags

    flags.append("range_bound")
    return "range", flags


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


def summarize_structure(frame: pd.DataFrame, *, pivot_lookback: int, pivot_max_points: int, consolidation_window: int, consolidation_range_atr_mult: float) -> StructureSummary:
    highs, lows = find_pivots(frame, lookback=pivot_lookback, max_points=pivot_max_points)
    trend_state, flags = _classify_trend(frame, highs, lows)

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
        swing_highs=highs,
        swing_lows=lows,
        breakout_level=(float(breakout_level) if breakout_level is not None else None),
        prior_breakout_retest_zone=prior_breakout_retest_zone,
        consolidation_range=consolidation,
        gap_zone=gap_zone,
        structure_flags=flags,
    )
