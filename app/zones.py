from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import PlanningConfig
from .structure import StructureSummary


def _zone(lower: float, upper: float, tags: list[str]) -> dict:
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    return {"lower": lo, "upper": hi, "source_tags": sorted(set(tags))}


def fibonacci_levels(frame: pd.DataFrame, structure: StructureSummary) -> dict[str, float | None]:
    if frame.empty:
        return {"fib_382": None, "fib_500": None, "fib_618": None}

    recent = frame.tail(90)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    if structure.trend_state in {"uptrend", "pullback_in_uptrend"}:
        top = swing_high
        base = swing_low
    else:
        top = swing_high
        base = swing_low

    move = top - base
    if move <= 0:
        return {"fib_382": None, "fib_500": None, "fib_618": None}

    return {
        "fib_382": float(top - move * 0.382),
        "fib_500": float(top - move * 0.5),
        "fib_618": float(top - move * 0.618),
    }


def _volume_congestion_zone(frame: pd.DataFrame, atr_val: float | None) -> dict | None:
    if frame.empty or atr_val is None or atr_val <= 0:
        return None
    recent = frame.tail(80)
    low = float(recent["low"].min())
    high = float(recent["high"].max())
    if high <= low:
        return None

    bins = np.linspace(low, high, 13)
    weights: defaultdict[int, float] = defaultdict(float)
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    for price, vol in zip(typical.tolist(), recent["volume"].fillna(0.0).tolist()):
        idx = int(np.digitize(price, bins) - 1)
        weights[max(0, min(idx, len(bins) - 2))] += float(vol)
    if not weights:
        return None

    best_idx = max(weights.items(), key=lambda item: item[1])[0]
    return _zone(float(bins[best_idx]), float(bins[best_idx + 1]), ["volume_congestion"])


def build_support_resistance_zones(frame: pd.DataFrame, structure: StructureSummary, fibs: dict[str, float | None], config: PlanningConfig) -> dict[str, dict | None]:
    if frame.empty:
        return {
            "support_zone_1": None,
            "support_zone_2": None,
            "resistance_zone_1": None,
            "resistance_zone_2": None,
        }

    close = float(frame["close"].iloc[-1])
    atr_val = frame["atr"].iloc[-1] if "atr" in frame.columns else None
    atr_val = float(atr_val) if atr_val is not None and not pd.isna(atr_val) else max(close * 0.02, 0.01)
    zone_pad = atr_val * config.atr_zone_width_mult

    supports: list[dict] = []
    resistances: list[dict] = []

    for pivot in structure.swing_lows:
        supports.append(_zone(pivot.price - zone_pad, pivot.price + zone_pad, ["pivot_low"]))
    for pivot in structure.swing_highs:
        resistances.append(_zone(pivot.price - zone_pad, pivot.price + zone_pad, ["pivot_high"]))

    for ma_tag in ["ema20", "sma50", "sma100", "sma200"]:
        if ma_tag in frame.columns:
            val = frame[ma_tag].iloc[-1]
            if val is not None and not pd.isna(val):
                zone = _zone(float(val) - zone_pad, float(val) + zone_pad, [ma_tag])
                if float(val) <= close:
                    supports.append(zone)
                else:
                    resistances.append(zone)

    for fib_tag, fib_price in fibs.items():
        if fib_price is None:
            continue
        zone = _zone(float(fib_price) - zone_pad, float(fib_price) + zone_pad, [fib_tag])
        if fib_price <= close:
            supports.append(zone)
        else:
            resistances.append(zone)

    if structure.prior_breakout_retest_zone:
        supports.append(structure.prior_breakout_retest_zone)
    if structure.consolidation_range:
        shelf = structure.consolidation_range
        midpoint = (float(shelf["lower"]) + float(shelf["upper"])) / 2.0
        if midpoint <= close:
            supports.append(shelf)
        else:
            resistances.append(shelf)
    if structure.gap_zone:
        gap_mid = (float(structure.gap_zone["lower"]) + float(structure.gap_zone["upper"])) / 2.0
        if gap_mid <= close:
            supports.append(structure.gap_zone)
        else:
            resistances.append(structure.gap_zone)

    congestion = _volume_congestion_zone(frame, atr_val)
    if congestion:
        congestion_mid = (float(congestion["lower"]) + float(congestion["upper"])) / 2.0
        if congestion_mid <= close:
            supports.append(congestion)
        else:
            resistances.append(congestion)

    supports = sorted(supports, key=lambda z: abs(close - ((z["lower"] + z["upper"]) / 2.0)))
    resistances = sorted(resistances, key=lambda z: abs(close - ((z["lower"] + z["upper"]) / 2.0)))

    def _dedupe(zones: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for zone in zones:
            mid = (zone["lower"] + zone["upper"]) / 2.0
            if any(abs(mid - ((k["lower"] + k["upper"]) / 2.0)) <= zone_pad * 0.6 for k in kept):
                continue
            kept.append(zone)
        return kept

    supports = _dedupe(supports)
    resistances = _dedupe(resistances)

    return {
        "support_zone_1": supports[0] if len(supports) >= 1 else None,
        "support_zone_2": supports[1] if len(supports) >= 2 else None,
        "resistance_zone_1": resistances[0] if len(resistances) >= 1 else None,
        "resistance_zone_2": resistances[1] if len(resistances) >= 2 else None,
    }
