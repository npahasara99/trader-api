from __future__ import annotations

"""Deterministic price-zone and confirmation-state evaluation."""

import pandas as pd

from .config import PlanningConfig


def _zone_bounds(zone: dict | None) -> tuple[float | None, float | None]:
    if not zone:
        return None, None
    lower = zone.get("lower")
    upper = zone.get("upper")
    if lower is None or upper is None:
        return None, None
    return float(min(lower, upper)), float(max(lower, upper))


def build_confirmation_plan(
    *,
    current_price: float,
    preferred_entry: float,
    support_zone_1: dict | None,
    resistance_zone_1: dict | None,
    moving_averages: dict[str, float | None],
    structure_state: str,
    frame: pd.DataFrame,
    atr: float,
    invalidation_level: float | None,
    volume_context: dict,
    requires_confirmation: bool,
    config: PlanningConfig,
) -> dict:
    """Build a numeric trigger and evaluate price location separately from confirmation."""

    price = float(current_price)
    atr_value = max(float(atr or 0.0), price * 0.005)
    zone_low, zone_high = _zone_bounds(support_zone_1)
    if zone_low is None or zone_high is None:
        half_width = atr_value * config.confirmation_entry_zone_atr
        zone_low = float(preferred_entry) - half_width
        zone_high = float(preferred_entry) + half_width

    prior_high = None
    if frame is not None and len(frame) >= 2:
        prior_high = float(frame["high"].iloc[-2])
    ema20 = moving_averages.get("ema20")
    resistance_low, resistance_high = _zone_bounds(resistance_zone_1)
    buffer = atr_value * config.confirmation_trigger_buffer_atr

    candidates: list[tuple[float, str]] = []
    if prior_high is not None:
        candidates.append((prior_high + buffer, "reclaim prior session high"))
    if ema20 is not None and float(ema20) >= zone_low:
        candidates.append((float(ema20) + buffer, "reclaim EMA20"))
    if structure_state == "breakout" and resistance_high is not None:
        candidates.append((resistance_high + buffer, "clear ranked resistance"))
    elif structure_state in {"reversal_attempt", "trend_damage", "structural_breakdown"} and resistance_low is not None:
        candidates.append((resistance_low + buffer, "repair through nearest resistance"))

    candidates = [item for item in candidates if item[0] > zone_low]
    if candidates:
        if structure_state in {"reversal_attempt", "trend_damage", "structural_breakdown", "breakout"}:
            trigger_price, trigger_reason = max(candidates, key=lambda item: item[0])
        else:
            above_zone = [item for item in candidates if item[0] >= zone_high]
            trigger_price, trigger_reason = min(above_zone or candidates, key=lambda item: item[0])
    else:
        trigger_price = zone_high + buffer
        trigger_reason = "hold above the preferred price zone"

    trigger_price = round(float(trigger_price), 6)
    price_confirmed = price >= trigger_price
    volume_confirmed = volume_context.get("reversal_volume_state") == "confirmed_bounce"
    heavy_distribution = volume_context.get("selloff_volume_state") == "heavy_distribution"
    damaged = structure_state in {"trend_damage", "structural_breakdown"}
    confirmation_required = bool(requires_confirmation or structure_state not in {"breakout"})

    missed_buffer = max(atr_value * config.confirmation_missed_atr, trigger_price * config.confirmation_missed_pct)
    if invalidation_level is not None and price <= float(invalidation_level):
        entry_status = "invalidated"
    elif structure_state == "extended":
        entry_status = "extended"
    elif price > trigger_price + missed_buffer:
        entry_status = "missed"
    elif price_confirmed and not heavy_distribution and not damaged:
        entry_status = "confirmed"
    elif zone_low <= price <= zone_high:
        entry_status = "in_price_zone"
    elif price < zone_low:
        entry_status = "too_early"
    else:
        entry_status = "awaiting_confirmation"

    if entry_status == "confirmed":
        confirmation_state = "confirmed"
        confirmation_score = 9.0 if volume_confirmed else 7.8
    elif entry_status == "in_price_zone":
        confirmation_state = "awaiting_confirmation"
        confirmation_score = 6.0
    elif entry_status == "awaiting_confirmation":
        confirmation_state = "awaiting_confirmation"
        confirmation_score = 5.0
    elif entry_status == "too_early":
        confirmation_state = "too_early"
        confirmation_score = 3.5
    elif entry_status in {"extended", "missed"}:
        confirmation_state = entry_status
        confirmation_score = 2.0
    else:
        confirmation_state = "invalidated"
        confirmation_score = 0.0

    if damaged and price_confirmed:
        confirmation_state = "awaiting_confirmation"
        confirmation_score = min(confirmation_score, 4.5)

    return {
        "preferred_entry_low": round(zone_low, 6),
        "preferred_entry_high": round(zone_high, 6),
        "confirmation_trigger_price": trigger_price,
        "confirmation_reason": trigger_reason,
        "confirmation_state": confirmation_state,
        "entry_status": entry_status,
        "confirmation_required": confirmation_required,
        "price_confirmed": bool(price_confirmed),
        "volume_confirmed": bool(volume_confirmed),
        "confirmation_score": round(confirmation_score, 3),
    }
