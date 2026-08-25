from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from .config import PlanningConfig
from .setup_archetypes import (
    BASE_BREAKOUT,
    BREAKOUT_RETEST,
    DEEP_PULLBACK,
    REVERSAL_ATTEMPT,
    build_runner_plan,
    family_policy,
    normalize_setup_family,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _zone_mid(zone: dict | None) -> float | None:
    if not zone:
        return None
    lower = zone.get("lower")
    upper = zone.get("upper")
    if lower is None or upper is None:
        return None
    return (float(lower) + float(upper)) / 2.0


def _zone_lower(zone: dict | None) -> float | None:
    if not zone or zone.get("lower") is None:
        return None
    return float(zone["lower"])


def _zone_upper(zone: dict | None) -> float | None:
    if not zone or zone.get("upper") is None:
        return None
    return float(zone["upper"])


def _zone_actionable_target(zone: dict | None, atr: float, tp_aggressiveness: str | None = None) -> float | None:
    lower = _zone_lower(zone)
    upper = _zone_upper(zone)
    if lower is None or upper is None:
        return None
    width = max(upper - lower, 0.0)
    normalized = (tp_aggressiveness or "").strip().lower()
    slice_fraction = 0.25
    if normalized == "moderate_to_high":
        slice_fraction = 0.4
    elif normalized == "high":
        slice_fraction = 0.48
    elif normalized == "conservative":
        slice_fraction = 0.18
    actionable_offset = min(width * slice_fraction, atr * 0.45)
    return lower + actionable_offset


def _is_repair_setup(trend_state: str | None) -> bool:
    return (trend_state or "").strip().lower() in {
        "weak_breakdown_risk",
        "deep_pullback",
        "reversal_attempt",
        "trend_damage",
        "structural_breakdown",
    }


def _stop_caps(*, trend_state: str | None, config: PlanningConfig) -> tuple[float, float]:
    if _is_repair_setup(trend_state):
        return config.max_stop_width_pct_repair, config.max_stop_width_atr_repair
    return config.max_stop_width_pct_default, config.max_stop_width_atr_default


def _tp1_caps(*, trend_state: str | None, config: PlanningConfig) -> tuple[float, float]:
    if _is_repair_setup(trend_state):
        return config.max_tp1_distance_pct_repair, config.max_tp1_distance_atr_repair
    return config.max_tp1_distance_pct_default, config.max_tp1_distance_atr_default


def _trend_reachability_factor(trend_state: str | None) -> float:
    normalized = (trend_state or "").strip().lower()
    if normalized == "uptrend":
        return 1.08
    if normalized == "pullback_in_uptrend":
        return 1.0
    if normalized == "range":
        return 0.88
    if normalized == "weak_breakdown_risk":
        return 0.72
    if normalized in {"healthy_pullback", "breakout"}:
        return 1.05
    if normalized in {"deep_pullback", "reversal_attempt"}:
        return 0.78
    if normalized in {"trend_damage", "structural_breakdown"}:
        return 0.65
    return 0.84


def _scenario_stop_mult(*, sl_tolerance: str | None, config: PlanningConfig) -> float:
    normalized = (sl_tolerance or "").strip().lower()
    if normalized == "tight":
        return config.scenario_stop_tight_mult
    if normalized == "tight_to_moderate":
        return config.scenario_stop_tight_to_moderate_mult
    if normalized == "moderate_to_wide":
        return config.scenario_stop_moderate_to_wide_mult
    return 1.0


def _scenario_tp_mult(*, tp_aggressiveness: str | None, config: PlanningConfig) -> float:
    normalized = (tp_aggressiveness or "").strip().lower()
    if normalized == "high":
        return config.scenario_tp_aggressive_mult
    if normalized == "moderate_to_high":
        return config.scenario_tp_moderate_high_mult
    if normalized == "conservative":
        return config.scenario_tp_conservative_mult
    return 1.0


def _format_pct(distance: float, base: float) -> float:
    return float(round(distance / max(base, 1e-9) * 100.0, 3))


def _add_trading_days(start: datetime, trading_days: int) -> datetime:
    result = start
    remaining = max(0, int(trading_days))
    while remaining:
        result += timedelta(days=1)
        if result.weekday() < 5:
            remaining -= 1
    return result


def build_stop_loss(
    *,
    preferred_entry: float,
    support_zone_1: dict | None,
    support_zone_2: dict | None,
    recent_swing_low: float | None,
    atr: float,
    current_price: float,
    trend_state: str | None,
    sl_tolerance: str | None = None,
    setup_scenario: str | None = None,
    setup_family: str | None = None,
    invalidation_zone: dict | None = None,
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(current_price * 0.01, 0.01))
    buffer = atr * config.stop_buffer_atr_mult
    max_valid_stop = preferred_entry - max(atr * 0.35, current_price * 0.0035)
    family = normalize_setup_family(setup_family)
    policy = family_policy(family)
    max_width_pct, max_width_atr = _stop_caps(trend_state=family or trend_state, config=config)
    stop_mult = _scenario_stop_mult(sl_tolerance=sl_tolerance, config=config)
    max_width_pct *= stop_mult
    max_width_atr *= stop_mult

    candidates: list[dict] = []

    family_zone = _zone_lower(invalidation_zone)
    if family_zone is not None:
        level = family_zone - buffer * (0.75 if family in {BREAKOUT_RETEST, BASE_BREAKOUT} else 1.0)
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": f"{policy['stop_style']} and ATR buffer", "priority": 3})

    support_1 = _zone_lower(support_zone_1)
    if support_1 is not None:
        level = support_1 - buffer - current_price * config.stop_below_zone_buffer_pct
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below support_zone_1 and ATR buffer", "priority": 2})

    support_2 = _zone_lower(support_zone_2)
    if support_2 is not None:
        level = support_2 - buffer * 0.8
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below support_zone_2 and ATR buffer", "priority": 1})

    if recent_swing_low is not None:
        level = float(recent_swing_low) - buffer
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below recent swing low and ATR buffer", "priority": 2})

    fallback_stop = preferred_entry - atr * 1.6
    if not candidates:
        candidates.append({"price": fallback_stop, "basis": "fallback ATR invalidation", "priority": 0})

    for candidate in candidates:
        width = max(preferred_entry - float(candidate["price"]), 0.0)
        candidate["width"] = width
        candidate["width_pct"] = width / max(preferred_entry, 1e-9)
        candidate["width_atr"] = width / max(atr, 1e-9)
        candidate["within_cap"] = (
            candidate["width_pct"] <= max_width_pct and candidate["width_atr"] <= max_width_atr
        )

    valid_candidates = [candidate for candidate in candidates if candidate["within_cap"]]
    candidate_pool = candidates if family in {DEEP_PULLBACK, REVERSAL_ATTEMPT} else (valid_candidates or candidates)
    if family in {DEEP_PULLBACK, REVERSAL_ATTEMPT}:
        # Recovery setups fail at the deeper thesis level; if that level is too
        # wide, report untradeable geometry rather than inventing a tight stop.
        selected = min(candidate_pool, key=lambda item: (item["price"], -item.get("priority", 0)))
    else:
        selected = max(candidate_pool, key=lambda item: (item.get("priority", 0), item["price"]))

    invalidation_level = float(selected["price"])
    invalidation_reason = str(selected["basis"])
    stop_loss = invalidation_level
    suggested_stop: float | None = invalidation_level
    stop_generation_reason = str(selected["basis"])
    swing_realism_flag = "realistic"
    risk_width_flag = "ok"

    if not selected["within_cap"]:
        if family in {DEEP_PULLBACK, REVERSAL_ATTEMPT}:
            suggested_stop = None
            swing_realism_flag = "flagged"
            risk_width_flag = "too_wide_for_swing"
            stop_generation_reason = f"{selected['basis']}; no technically valid executable stop inside the swing-risk envelope"
        else:
            allowed_width = min(preferred_entry * max_width_pct, atr * max_width_atr)
            capped_stop = preferred_entry - max(allowed_width, atr * 0.95)
            capped_stop = min(capped_stop, max_valid_stop)
            if capped_stop > stop_loss:
                stop_loss = capped_stop
                suggested_stop = capped_stop
                swing_realism_flag = "compressed"
                risk_width_flag = "capped_for_swing"
                scenario_suffix = f"; scenario={setup_scenario}" if setup_scenario else ""
                stop_generation_reason = f"{selected['basis']}; compressed to swing-risk envelope{scenario_suffix}"
            else:
                swing_realism_flag = "flagged"
                risk_width_flag = "too_wide_for_swing"
                scenario_suffix = f"; scenario={setup_scenario}" if setup_scenario else ""
                stop_generation_reason = f"{selected['basis']}; broad structure exceeds normal swing width{scenario_suffix}"

    if stop_loss >= preferred_entry:
        stop_loss = fallback_stop
        suggested_stop = fallback_stop
        swing_realism_flag = "compressed"
        risk_width_flag = "fallback_atr_stop"
        stop_generation_reason = "fallback ATR invalidation after invalid structural stop"

    stop_width = max(preferred_entry - stop_loss, 0.0)
    stop_distance_pct = _format_pct(stop_width, preferred_entry)
    stop_width_atr = float(round(stop_width / max(atr, 1e-9), 3))
    stop_too_tight = stop_width < atr * 0.9
    invalidation_width = max(preferred_entry - invalidation_level, 0.0)
    executable_stop_technically_valid = bool(selected["within_cap"] and abs(stop_loss - invalidation_level) <= 1e-9)
    trade_geometry_status = (
        "valid" if executable_stop_technically_valid else "valid_setup_but_untradeable_geometry"
    )

    return {
        "stop_loss": float(round(stop_loss, 6)),
        "stop_basis": str(selected["basis"]),
        "stop_distance_pct": stop_distance_pct,
        "stop_width_pct": stop_distance_pct,
        "stop_width_atr": stop_width_atr,
        "stop_too_tight_flag": bool(stop_too_tight),
        "swing_realism_flag": swing_realism_flag,
        "risk_width_flag": risk_width_flag,
        "stop_generation_reason": stop_generation_reason,
        "invalidation_level": float(round(invalidation_level, 6)),
        "invalidation_reason": invalidation_reason,
        "suggested_stop": None if suggested_stop is None else float(round(suggested_stop, 6)),
        "invalidation_width_pct": _format_pct(invalidation_width, preferred_entry),
        "invalidation_width_atr": float(round(invalidation_width / max(atr, 1e-9), 3)),
        "executable_stop_technically_valid": executable_stop_technically_valid,
        "trade_geometry_status": trade_geometry_status,
        "stop_style": policy["stop_style"],
    }


def build_take_profits(
    *,
    preferred_entry: float,
    stop_loss: float,
    resistance_zone_1: dict | None,
    resistance_zone_2: dict | None,
    recent_swing_high: float | None,
    atr: float,
    hold_days_hint: int,
    trend_state: str,
    tp_aggressiveness: str | None = None,
    expected_move_profile: str | None = None,
    price_location_context: str | None = None,
    config: PlanningConfig,
    ranked_resistance_levels: list[dict] | None = None,
    setup_family: str | None = None,
) -> dict:
    atr = max(float(atr or 0.0), max(preferred_entry * 0.01, 0.01))
    risk_per_share = max(preferred_entry - stop_loss, atr * 0.6)
    hold_days = max(1, int(hold_days_hint or config.max_hold_days_min))
    reachability_factor = _trend_reachability_factor(trend_state)
    reachable_move = atr * math.sqrt(hold_days) * config.hold_window_reachability_factor * reachability_factor
    family = normalize_setup_family(setup_family)
    policy = family_policy(family)
    max_tp1_pct, max_tp1_atr = _tp1_caps(trend_state=family or trend_state, config=config)
    tp_mult = _scenario_tp_mult(tp_aggressiveness=tp_aggressiveness, config=config)
    max_tp1_pct *= tp_mult
    max_tp1_atr *= tp_mult
    if (price_location_context or "").strip().lower() == "extended_near_high":
        max_tp1_pct *= 0.92
        max_tp1_atr *= 0.92
    if (expected_move_profile or "").strip().lower() in {"repair_bounce_not_full_recovery", "limited_rebound_only"}:
        max_tp1_pct *= 0.88
        max_tp1_atr *= 0.9
    max_tp1_distance = min(preferred_entry * max_tp1_pct, atr * max_tp1_atr, reachable_move)
    min_tp1_distance = max(atr * 0.45, risk_per_share * 0.65)

    candidates: list[dict] = []
    resistance_1_target = _zone_actionable_target(resistance_zone_1, atr, tp_aggressiveness=tp_aggressiveness)
    if resistance_1_target is not None and resistance_1_target > preferred_entry:
        candidates.append({"price": resistance_1_target, "basis": "tp1 near first actionable resistance slice"})
    if recent_swing_high is not None and float(recent_swing_high) > preferred_entry:
        candidates.append({"price": float(recent_swing_high), "basis": "tp1 near recent swing high"})
    for level in ranked_resistance_levels or []:
        actionable = _zone_actionable_target(level, atr, tp_aggressiveness=tp_aggressiveness)
        if actionable is not None and actionable > preferred_entry:
            candidates.append(
                {
                    "price": actionable,
                    "basis": f"ranked resistance level {level.get('rank', '?')} ({level.get('strength', 'unrated')})",
                }
            )

    structural_candidates = list(candidates)
    fallback_tp1 = preferred_entry + max(risk_per_share * 1.05, atr * config.tp1_atr_mult)
    fallback_candidate = {"price": fallback_tp1, "basis": "tp1 via ATR/risk multiple"}
    candidates.append(fallback_candidate)

    eligible_structural = [
        candidate for candidate in structural_candidates if candidate["price"] >= preferred_entry + min_tp1_distance
    ]
    raw_tp1_candidate = min(eligible_structural, key=lambda item: item["price"]) if eligible_structural else fallback_candidate
    raw_tp1 = float(raw_tp1_candidate["price"])
    raw_tp1_distance = max(raw_tp1 - preferred_entry, 0.0)

    tp1_distance_cap = max(atr * 0.35, max_tp1_distance)
    tp1_distance = min(raw_tp1_distance, tp1_distance_cap)
    tp1 = preferred_entry + tp1_distance

    target_reachability_flag = "reachable"
    tp1_generation_reason = str(raw_tp1_candidate["basis"])
    if raw_tp1_distance > tp1_distance_cap + 1e-9:
        target_reachability_flag = "capped_to_hold_window"
        profile_suffix = f"; profile={expected_move_profile}" if expected_move_profile else ""
        tp1_generation_reason = f"{raw_tp1_candidate['basis']}; compressed to first reachable swing target{profile_suffix}"
    elif raw_tp1_distance < min_tp1_distance:
        target_reachability_flag = "minimal_target"
        tp1_generation_reason = f"{raw_tp1_candidate['basis']}; nearest reasonable target remains close"

    resistance_candidates = sorted(
        {
            float(level["midpoint"] if level.get("midpoint") is not None else _zone_mid(level))
            for level in (ranked_resistance_levels or [])
            if _zone_mid(level) is not None and float(_zone_mid(level)) > tp1
        }
    )
    resistance_2_mid = _zone_mid(resistance_zone_2)
    if resistance_2_mid is not None and resistance_2_mid > tp1:
        resistance_candidates.append(float(resistance_2_mid))
        resistance_candidates = sorted(set(resistance_candidates))

    tp2_cap = preferred_entry + max(reachable_move * 1.18, tp1_distance + atr * 0.55)
    raw_tp2 = resistance_candidates[0] if resistance_candidates else tp1 + max(atr * 1.0, risk_per_share * 0.7)
    tp2 = min(raw_tp2, tp2_cap)
    if tp2 <= tp1:
        tp2 = tp1 + max(atr * 0.5, risk_per_share * 0.35)
    tp2_reason = "next ranked resistance" if resistance_candidates else "measured ATR swing extension"
    if raw_tp2 > tp2_cap:
        tp2_reason += "; capped to hold-window reachability"

    tp3_cap = preferred_entry + max(reachable_move * 1.42, (tp2 - preferred_entry) + atr * 0.5)
    higher_resistances = [level for level in resistance_candidates if level > tp2 + atr * 0.15]
    raw_tp3 = higher_resistances[0] if higher_resistances else tp2 + max(atr * 0.9, risk_per_share * 0.55)
    tp3 = min(raw_tp3, tp3_cap)
    if tp3 <= tp2:
        tp3 = tp2 + max(atr * 0.45, risk_per_share * 0.3)
    tp3_reason = "higher ranked resistance" if higher_resistances else "secondary ATR swing extension"
    if raw_tp3 > tp3_cap:
        tp3_reason += "; capped to extended hold-window reachability"

    stretch_cap = preferred_entry + min(reachable_move * 1.75, atr * 8.0)
    stretch_target = max(tp3 + atr * 0.45, stretch_cap)
    stretch_reason = "stretch objective beyond the base 2-10 day expectation; requires sustained trend confirmation"
    tp_final = stretch_target

    rr_divisor = max(preferred_entry - stop_loss, 1e-9)
    rr1 = (tp1 - preferred_entry) / rr_divisor
    rr2 = (tp2 - preferred_entry) / rr_divisor
    rr3 = (tp3 - preferred_entry) / rr_divisor
    rr_final = (stretch_target - preferred_entry) / rr_divisor
    optimistic_flag = (tp_final - preferred_entry) > reachable_move * 1.35 and trend_state != "uptrend"
    reachability_score = _clip(
        10.0 * min(1.0, tp1_distance_cap / max(raw_tp1_distance, 1e-9)),
        0.0,
        10.0,
    )

    basis_parts = [
        tp1_generation_reason,
        "tp2 near next resistance shelf" if resistance_zone_2 else "tp2 via measured swing extension",
        tp3_reason,
        stretch_reason,
    ]

    runner = build_runner_plan(
        setup_family=family,
        tp1=tp1,
        extension_target=stretch_target,
        config=config,
    )
    return {
        "take_profit_1": float(round(tp1, 6)),
        "take_profit_2": float(round(tp2, 6)),
        "take_profit_3": float(round(tp3, 6)),
        "stretch_target": float(round(stretch_target, 6)),
        "take_profit_final": float(round(tp_final, 6)),
        "tp_basis": "; ".join(basis_parts),
        "expected_reward_risk_to_tp1": float(round(rr1, 3)),
        "expected_reward_risk_to_tp2": float(round(rr2, 3)),
        "expected_reward_risk_to_tp3": float(round(rr3, 3)),
        "expected_reward_risk_to_final": float(round(rr_final, 3)),
        "tp_too_optimistic_flag": bool(optimistic_flag),
        "tp1_distance_pct": _format_pct(tp1 - preferred_entry, preferred_entry),
        "tp1_distance_atr": float(round((tp1 - preferred_entry) / max(atr, 1e-9), 3)),
        "tp1_atr_distance": float(round((tp1 - preferred_entry) / max(atr, 1e-9), 3)),
        "tp2_atr_distance": float(round((tp2 - preferred_entry) / max(atr, 1e-9), 3)),
        "tp3_atr_distance": float(round((tp3 - preferred_entry) / max(atr, 1e-9), 3)),
        "hold_window_reachability_score": float(round(reachability_score, 3)),
        "target_reachability_flag": target_reachability_flag,
        "tp1_generation_reason": tp1_generation_reason,
        "tp1_reason": tp1_generation_reason,
        "tp2_reason": tp2_reason,
        "tp3_reason": tp3_reason,
        "stretch_target_reason": stretch_reason,
        "target_realism_score": float(round(reachability_score, 3)),
        "target_style": policy["target_style"],
        "runner_plan": runner,
        **runner,
    }


def estimate_hold_window(
    *,
    preferred_entry: float,
    take_profit_1: float,
    atr: float,
    recent_swing_bars: int | None,
    historical_hold_days: int | None,
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(preferred_entry * 0.01, 0.01))
    distance = max(0.0, take_profit_1 - preferred_entry)
    atr_days = int(round(distance / max(atr, 1e-9)))
    base = atr_days + 4
    if recent_swing_bars is not None:
        base = int(round((base * 0.6) + (recent_swing_bars * 0.4)))
    if historical_hold_days is not None:
        base = int(round((base * 0.7) + (historical_hold_days * 0.3)))

    hold_days = max(config.max_hold_days_min, min(config.max_hold_days_max, base))
    max_hold_date = _add_trading_days(datetime.now(timezone.utc), hold_days)
    return {
        "expected_hold_days": int(hold_days),
        "max_hold_days": int(hold_days),
        "max_hold_date": max_hold_date,
    }
