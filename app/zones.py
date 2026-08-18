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


def _merge_and_rank_zones(
    zones: list[dict],
    *,
    frame: pd.DataFrame,
    close: float,
    atr_val: float,
    side: str,
) -> list[dict]:
    """Merge overlapping evidence and rank levels by confluence and reactions."""

    if not zones:
        return []
    merge_tolerance = max(atr_val * 0.45, close * 0.003)
    ordered = sorted(zones, key=lambda item: (float(item["lower"]) + float(item["upper"])) / 2.0)
    merged: list[dict] = []
    for candidate in ordered:
        midpoint = (float(candidate["lower"]) + float(candidate["upper"])) / 2.0
        match = next(
            (
                zone
                for zone in merged
                if abs(midpoint - ((float(zone["lower"]) + float(zone["upper"])) / 2.0)) <= merge_tolerance
            ),
            None,
        )
        if match is None:
            merged.append(dict(candidate))
            continue
        match["lower"] = min(float(match["lower"]), float(candidate["lower"]))
        match["upper"] = max(float(match["upper"]), float(candidate["upper"]))
        match["source_tags"] = sorted(set(match.get("source_tags", [])) | set(candidate.get("source_tags", [])))

    recent = frame.tail(min(120, len(frame)))
    ranked: list[dict] = []
    for zone in merged:
        lower = float(zone["lower"])
        upper = float(zone["upper"])
        midpoint = (lower + upper) / 2.0
        reaction_tolerance = max(atr_val * 0.3, close * 0.0025)
        if side == "support":
            reactions = ((recent["low"] >= lower - reaction_tolerance) & (recent["low"] <= upper + reaction_tolerance)).sum()
        else:
            reactions = ((recent["high"] >= lower - reaction_tolerance) & (recent["high"] <= upper + reaction_tolerance)).sum()
        confluence = len(zone.get("source_tags", []))
        distance_pct = abs(close - midpoint) / max(close, 1e-9) * 100.0
        proximity = max(0.0, 1.5 - distance_pct / 2.0)
        strength_score = min(10.0, 3.0 + confluence * 1.15 + min(int(reactions), 6) * 0.55 + proximity)
        strength = "major" if strength_score >= 8.0 else "strong" if strength_score >= 6.5 else "moderate" if strength_score >= 5.0 else "weak"
        ranked.append(
            {
                **zone,
                "midpoint": round(midpoint, 6),
                "side": side,
                "strength_score": round(strength_score, 3),
                "strength": strength,
                "reaction_count": int(reactions),
                "confluence_count": int(confluence),
                "distance_from_price_pct": round(distance_pct, 4),
            }
        )

    ranked.sort(
        key=lambda zone: (
            abs(close - float(zone["midpoint"])),
            -float(zone["strength_score"]),
        )
    )
    for index, zone in enumerate(ranked, start=1):
        zone["rank"] = index
    return ranked


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


def build_support_resistance_zones(frame: pd.DataFrame, structure: StructureSummary, fibs: dict[str, float | None], config: PlanningConfig) -> dict:
    if frame.empty:
        return {
            "support_zone_1": None,
            "support_zone_2": None,
            "resistance_zone_1": None,
            "resistance_zone_2": None,
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
            "major_resistance_cluster": [],
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

    for ma_tag in ["ema20", "ema50", "ema100", "ema200", "sma50", "sma100", "sma200"]:
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

    psychological_step = 100.0 if close >= 500 else 50.0 if close >= 200 else 10.0 if close >= 50 else 5.0
    psychological_anchor = round(close / psychological_step) * psychological_step
    for level in (psychological_anchor - psychological_step, psychological_anchor, psychological_anchor + psychological_step):
        if level <= 0:
            continue
        zone = _zone(level - zone_pad * 0.35, level + zone_pad * 0.35, ["psychological_level"])
        if level <= close:
            supports.append(zone)
        else:
            resistances.append(zone)

    supports = sorted(supports, key=lambda z: abs(close - ((z["lower"] + z["upper"]) / 2.0)))
    resistances = sorted(resistances, key=lambda z: abs(close - ((z["lower"] + z["upper"]) / 2.0)))

    supports = _merge_and_rank_zones(supports, frame=frame, close=close, atr_val=atr_val, side="support")
    resistances = _merge_and_rank_zones(resistances, frame=frame, close=close, atr_val=atr_val, side="resistance")
    major_cluster = [
        level
        for level in resistances
        if level["strength_score"] >= 7.0 and level["confluence_count"] >= 2
    ][:3]

    return {
        "support_zone_1": supports[0] if len(supports) >= 1 else None,
        "support_zone_2": supports[1] if len(supports) >= 2 else None,
        "resistance_zone_1": resistances[0] if len(resistances) >= 1 else None,
        "resistance_zone_2": resistances[1] if len(resistances) >= 2 else None,
        "support_levels": supports,
        "resistance_levels": resistances,
        "nearest_support": supports[0]["midpoint"] if supports else None,
        "nearest_resistance": resistances[0]["midpoint"] if resistances else None,
        "major_resistance_cluster": major_cluster,
    }
