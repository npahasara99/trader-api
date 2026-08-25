from __future__ import annotations

"""Deterministic price-zone and confirmation-state evaluation."""

import pandas as pd

from .config import PlanningConfig
from .setup_archetypes import BASE_BREAKOUT, BREAKOUT_RETEST, MOMENTUM_CONTINUATION, family_policy, normalize_setup_family


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
    setup_family: str | None = None,
    breakout_level: float | None = None,
    consolidation_range: dict | None = None,
    retest_zone: dict | None = None,
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
    family = normalize_setup_family(setup_family)
    policy = family_policy(family)

    near_candidates: list[tuple[float, str]] = []
    if prior_high is not None:
        near_candidates.append((prior_high + buffer, "reclaim prior session high"))
    if ema20 is not None and float(ema20) >= zone_low:
        near_candidates.append((float(ema20) + buffer, "reclaim EMA20"))
    near_candidates = [item for item in near_candidates if item[0] > zone_low]
    future_near = [item for item in near_candidates if item[0] > price]
    near_level = min(future_near or near_candidates, key=lambda item: item[0]) if near_candidates else None

    primary_candidates = list(near_candidates)
    if resistance_low is not None:
        primary_candidates.append((resistance_low + buffer, "clear nearest ranked resistance"))
    consolidation_low, consolidation_high = _zone_bounds(consolidation_range)
    retest_low, retest_high = _zone_bounds(retest_zone)
    if family in {MOMENTUM_CONTINUATION, BASE_BREAKOUT} and consolidation_high is not None:
        primary_candidates.append((consolidation_high + buffer, "clear short consolidation resistance"))
    if family == BREAKOUT_RETEST and retest_high is not None:
        primary_candidates.append((retest_high + buffer, "hold and reclaim the breakout retest zone"))
    if breakout_level is not None and family in {MOMENTUM_CONTINUATION, BREAKOUT_RETEST, BASE_BREAKOUT}:
        primary_candidates.append((float(breakout_level) + buffer, "clear the active breakout level"))
    primary_candidates = [item for item in primary_candidates if item[0] > zone_low]
    if structure_state == "breakout" and resistance_high is not None:
        trigger_price, trigger_reason = resistance_high + buffer, "clear ranked resistance"
    elif primary_candidates:
        # The first executable reclaim is the primary trigger. Longer-term
        # repair belongs in the strong/major tiers, not the starter entry.
        above_zone = [item for item in primary_candidates if item[0] >= zone_high]
        trigger_price, trigger_reason = min(above_zone or primary_candidates, key=lambda item: item[0])
    else:
        trigger_price = zone_high + buffer
        trigger_reason = "hold above the preferred price zone"

    trigger_price = round(float(trigger_price), 6)
    primary_level = {"price": trigger_price, "reason": trigger_reason}
    near_confirmation = None
    if near_level is not None and float(near_level[0]) < trigger_price - max(buffer * 0.25, 1e-9):
        near_confirmation = {"price": round(float(near_level[0]), 6), "reason": near_level[1]}

    strong_confirmation = None
    if resistance_high is not None and float(resistance_high) + buffer > trigger_price + buffer * 0.5:
        strong_confirmation = {
            "price": round(float(resistance_high) + buffer, 6),
            "reason": "hold above the upper edge of ranked resistance",
        }

    major_candidates: list[tuple[float, str]] = []
    for key, reason in (
        ("ema50", "reclaim EMA50 trend structure"),
        ("ema100", "reclaim EMA100 trend structure"),
        ("ema200", "reclaim EMA200 trend structure"),
    ):
        value = moving_averages.get(key)
        if value is not None and float(value) > trigger_price + buffer:
            major_candidates.append((float(value) + buffer, reason))
    if resistance_high is not None and float(resistance_high) + buffer > trigger_price + buffer:
        major_candidates.append((float(resistance_high) + buffer, "repair the broader resistance regime"))
    major_trend_repair = None
    if major_candidates:
        major_price, major_reason = max(major_candidates, key=lambda item: item[0])
        major_trend_repair = {"price": round(major_price, 6), "reason": major_reason}

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

    if damaged and price_confirmed and not volume_confirmed:
        confirmation_state = "awaiting_confirmation"
        confirmation_score = min(confirmation_score, 4.5)

    confirmation_requirements = [policy["confirmation_style"]]
    if policy["requires_strong_volume"]:
        confirmation_requirements.append("volume_expansion_required")
    if family == BREAKOUT_RETEST:
        confirmation_requirements.append("former_resistance_must_hold_as_support")

    return {
        "preferred_entry_low": round(zone_low, 6),
        "preferred_entry_high": round(zone_high, 6),
        "confirmation_trigger_price": trigger_price,
        "near_confirmation": near_confirmation,
        "primary_entry_trigger": primary_level,
        "strong_confirmation": strong_confirmation,
        "major_trend_repair": major_trend_repair,
        "confirmation_levels": {
            "near_confirmation": near_confirmation,
            "primary_entry_trigger": primary_level,
            "strong_confirmation": strong_confirmation,
            "major_trend_repair": major_trend_repair,
        },
        "confirmation_reason": trigger_reason,
        "confirmation_state": confirmation_state,
        "entry_status": entry_status,
        "confirmation_required": confirmation_required,
        "price_confirmed": bool(price_confirmed),
        "volume_confirmed": bool(volume_confirmed),
        "confirmation_score": round(confirmation_score, 3),
        "confirmation_style": policy["confirmation_style"],
        "confirmation_requirements": confirmation_requirements,
        "setup_family": policy["setup_family"],
    }
