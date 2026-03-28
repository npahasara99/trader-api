from __future__ import annotations

import math

from .config import PlanningConfig


def _zone_mid(zone: dict | None) -> float | None:
    if not zone:
        return None
    lower = zone.get("lower")
    upper = zone.get("upper")
    if lower is None or upper is None:
        return None
    return float(lower + upper) / 2.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_entry_candidates(
    *,
    current_price: float,
    trend_state: str,
    support_zone_1: dict | None,
    support_zone_2: dict | None,
    resistance_zone_1: dict | None,
    fib_levels: dict[str, float | None],
    moving_averages: dict[str, float | None],
    atr: float,
    volume_context: dict,
    config: PlanningConfig,
) -> list[dict]:
    atr = max(float(atr or 0.0), max(current_price * 0.01, 0.01))
    candidates: list[dict] = []

    support1_mid = _zone_mid(support_zone_1)
    support2_mid = _zone_mid(support_zone_2)
    resistance1_mid = _zone_mid(resistance_zone_1)
    ema20 = moving_averages.get("ema20")
    sma50 = moving_averages.get("sma50")
    fib_382 = fib_levels.get("fib_382")
    fib_500 = fib_levels.get("fib_500")
    fib_618 = fib_levels.get("fib_618")

    immediate_price = current_price
    if support1_mid is not None:
        immediate_price = min(current_price, support1_mid + atr * config.immediate_entry_atr_mult)
    immediate_score = 5.0
    if trend_state == "uptrend":
        immediate_score += 1.0
    if volume_context.get("selloff_volume_state") == "heavy_distribution":
        immediate_score -= 2.0
    if resistance1_mid is not None and resistance1_mid <= current_price + atr:
        immediate_score -= 1.0
    candidates.append(
        {
            "type": "immediate",
            "price": round(immediate_price, 6),
            "zone_ref": "current_price",
            "confluence_score": round(immediate_score, 3),
            "requires_confirmation": trend_state in {"weak_breakdown_risk", "range"},
            "confirmation_trigger": "Need reversal close above prior day high or reclaim of EMA20" if trend_state != "uptrend" else "Optional: hold above support zone 1",
        }
    )

    pullback_base = support1_mid or ema20 or fib_382 or current_price - atr * 0.5
    if fib_500 is not None and support1_mid is not None:
        pullback_base = (pullback_base + fib_500) / 2.0
    pullback_price = min(current_price, float(pullback_base) + atr * config.pullback_buffer_atr_mult)
    pullback_score = 6.2
    if support_zone_1:
        pullback_score += len(support_zone_1.get("source_tags", [])) * 0.35
    if fib_500 is not None:
        pullback_score += 0.5
    if sma50 is not None and abs(pullback_price - float(sma50)) <= atr * 0.4:
        pullback_score += 0.6
    candidates.append(
        {
            "type": "pullback",
            "price": round(pullback_price, 6),
            "zone_ref": "support_zone_1",
            "confluence_score": round(pullback_score, 3),
            "requires_confirmation": trend_state != "uptrend" or volume_context.get("reversal_volume_state") in {"weak_bounce", "no_confirmation"},
            "confirmation_trigger": "Wait for stabilization inside support zone 1 and improving reversal volume",
        }
    )

    deeper_base = support2_mid or fib_618 or sma50 or pullback_price - atr * 0.6
    deeper_price = min(current_price, float(deeper_base) + atr * config.deeper_pullback_buffer_atr_mult)
    deeper_score = 5.4
    if support_zone_2:
        deeper_score += len(support_zone_2.get("source_tags", [])) * 0.45
    if fib_618 is not None:
        deeper_score += 0.6
    deeper_requires_confirmation = True
    if trend_state == "uptrend" and support_zone_2:
        deeper_score += 0.4
    candidates.append(
        {
            "type": "deeper_pullback",
            "price": round(deeper_price, 6),
            "zone_ref": "support_zone_2",
            "confluence_score": round(deeper_score, 3),
            "requires_confirmation": deeper_requires_confirmation,
            "confirmation_trigger": "Only enter after bounce confirmation; avoid knife-catching through support zone 2",
        }
    )

    return candidates


def choose_preferred_entry(
    *,
    current_price: float,
    candidates: list[dict],
    trend_state: str,
    support_zone_1: dict | None,
    volume_context: dict,
    config: PlanningConfig,
) -> dict:
    best = None
    best_score = -10_000.0
    support_mid = _zone_mid(support_zone_1)

    for cand in candidates:
        price = float(cand["price"])
        distance_pct = (current_price - price) / max(current_price, 1e-9)
        score = float(cand.get("confluence_score", 0.0))

        if price > current_price:
            score -= 3.0
        if distance_pct < -0.005:
            score -= 3.0
        if distance_pct > config.deep_entry_distance_pct:
            score -= 2.8
        elif distance_pct > config.max_entry_distance_pct:
            score -= 1.0

        if support_mid is not None and price > support_mid + max(current_price * 0.005, 0.01):
            score -= 0.9
        if trend_state in {"weak_breakdown_risk", "downtrend"}:
            score -= 1.1
        if volume_context.get("selloff_volume_state") == "heavy_distribution":
            score -= 0.9
        if trend_state == "uptrend" and cand["type"] == "pullback":
            score += 0.7
        if trend_state == "uptrend" and cand["type"] == "immediate":
            score += 0.3
        if cand.get("requires_confirmation"):
            score -= 0.4

        if score > best_score:
            best_score = score
            best = {
                **cand,
                "entry_quality_score": round(_clip(score, 0.0, 10.0), 3),
                "entry_distance_from_current_price_pct": round(distance_pct * 100.0, 3),
                "entry_confluence_score": round(float(cand.get("confluence_score", 0.0)), 3),
            }

    if best is None:
        return {
            "preferred_entry": round(current_price, 6),
            "preferred_entry_type": "immediate",
            "entry_quality_score": 0.0,
            "entry_distance_from_current_price_pct": 0.0,
            "entry_confluence_score": 0.0,
            "entry_requires_confirmation": True,
            "confirmation_trigger": "No valid candidate generated",
        }

    return {
        "preferred_entry": float(best["price"]),
        "preferred_entry_type": str(best["type"]),
        "entry_quality_score": float(best["entry_quality_score"]),
        "entry_distance_from_current_price_pct": float(best["entry_distance_from_current_price_pct"]),
        "entry_confluence_score": float(best["entry_confluence_score"]),
        "entry_requires_confirmation": bool(best.get("requires_confirmation", False)),
        "confirmation_trigger": str(best.get("confirmation_trigger") or ""),
    }
