from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import PlanningConfig


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
    return (trend_state or "").strip().lower() == "weak_breakdown_risk"


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
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(current_price * 0.01, 0.01))
    buffer = atr * config.stop_buffer_atr_mult
    max_valid_stop = preferred_entry - max(atr * 0.35, current_price * 0.0035)
    max_width_pct, max_width_atr = _stop_caps(trend_state=trend_state, config=config)
    stop_mult = _scenario_stop_mult(sl_tolerance=sl_tolerance, config=config)
    max_width_pct *= stop_mult
    max_width_atr *= stop_mult

    candidates: list[dict] = []

    support_1 = _zone_lower(support_zone_1)
    if support_1 is not None:
        level = support_1 - buffer - current_price * config.stop_below_zone_buffer_pct
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below support_zone_1 and ATR buffer"})

    support_2 = _zone_lower(support_zone_2)
    if support_2 is not None:
        level = support_2 - buffer * 0.8
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below support_zone_2 and ATR buffer"})

    if recent_swing_low is not None:
        level = float(recent_swing_low) - buffer
        if level < max_valid_stop:
            candidates.append({"price": level, "basis": "below recent swing low and ATR buffer"})

    fallback_stop = preferred_entry - atr * 1.6
    if not candidates:
        candidates.append({"price": fallback_stop, "basis": "fallback ATR invalidation"})

    for candidate in candidates:
        width = max(preferred_entry - float(candidate["price"]), 0.0)
        candidate["width"] = width
        candidate["width_pct"] = width / max(preferred_entry, 1e-9)
        candidate["width_atr"] = width / max(atr, 1e-9)
        candidate["within_cap"] = (
            candidate["width_pct"] <= max_width_pct and candidate["width_atr"] <= max_width_atr
        )

    valid_candidates = [candidate for candidate in candidates if candidate["within_cap"]]
    selected = max(valid_candidates or candidates, key=lambda item: item["price"])

    stop_loss = float(selected["price"])
    stop_generation_reason = str(selected["basis"])
    swing_realism_flag = "realistic"
    risk_width_flag = "ok"

    if not selected["within_cap"]:
        allowed_width = min(preferred_entry * max_width_pct, atr * max_width_atr)
        capped_stop = preferred_entry - max(allowed_width, atr * 0.95)
        capped_stop = min(capped_stop, max_valid_stop)
        if capped_stop > stop_loss:
            stop_loss = capped_stop
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
        swing_realism_flag = "compressed"
        risk_width_flag = "fallback_atr_stop"
        stop_generation_reason = "fallback ATR invalidation after invalid structural stop"

    stop_width = max(preferred_entry - stop_loss, 0.0)
    stop_distance_pct = _format_pct(stop_width, preferred_entry)
    stop_width_atr = float(round(stop_width / max(atr, 1e-9), 3))
    stop_too_tight = stop_width < atr * 0.9

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
) -> dict:
    atr = max(float(atr or 0.0), max(preferred_entry * 0.01, 0.01))
    risk_per_share = max(preferred_entry - stop_loss, atr * 0.6)
    hold_days = max(1, int(hold_days_hint or config.max_hold_days_min))
    reachability_factor = _trend_reachability_factor(trend_state)
    reachable_move = atr * hold_days * config.hold_window_reachability_factor * reachability_factor
    max_tp1_pct, max_tp1_atr = _tp1_caps(trend_state=trend_state, config=config)
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

    fallback_tp1 = preferred_entry + max(risk_per_share * 1.05, atr * config.tp1_atr_mult)
    candidates.append({"price": fallback_tp1, "basis": "tp1 via ATR/risk multiple"})

    eligible = [candidate for candidate in candidates if candidate["price"] >= preferred_entry + min_tp1_distance]
    raw_tp1_candidate = min(eligible or candidates, key=lambda item: item["price"])
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

    resistance_2_mid = _zone_mid(resistance_zone_2)
    tp2 = resistance_2_mid or (tp1 + max(atr * 1.0, risk_per_share * 0.8))
    if tp2 <= tp1:
        tp2 = tp1 + max(atr * 0.9, risk_per_share * 0.55)

    trend_bonus = 1.25 if trend_state == "uptrend" else 1.0
    final_cap = preferred_entry + min(reachable_move * 1.35 * trend_bonus, atr * 8.0)
    tp_final = max(tp2, final_cap)
    if tp_final <= tp2:
        tp_final = tp2 + max(atr * 0.5, risk_per_share * 0.35)

    rr_divisor = max(preferred_entry - stop_loss, 1e-9)
    rr1 = (tp1 - preferred_entry) / rr_divisor
    rr2 = (tp2 - preferred_entry) / rr_divisor
    rr_final = (tp_final - preferred_entry) / rr_divisor
    optimistic_flag = (tp_final - preferred_entry) > reachable_move * 1.35 and trend_state != "uptrend"
    reachability_score = _clip(
        10.0 * min(1.0, tp1_distance_cap / max(raw_tp1_distance, 1e-9)),
        0.0,
        10.0,
    )

    basis_parts = [
        tp1_generation_reason,
        "tp2 near next resistance shelf" if resistance_zone_2 else "tp2 via measured swing extension",
        "final target bounded by trend and ATR reachability",
    ]

    return {
        "take_profit_1": float(round(tp1, 6)),
        "take_profit_2": float(round(tp2, 6)),
        "take_profit_final": float(round(tp_final, 6)),
        "tp_basis": "; ".join(basis_parts),
        "expected_reward_risk_to_tp1": float(round(rr1, 3)),
        "expected_reward_risk_to_tp2": float(round(rr2, 3)),
        "expected_reward_risk_to_final": float(round(rr_final, 3)),
        "tp_too_optimistic_flag": bool(optimistic_flag),
        "tp1_distance_pct": _format_pct(tp1 - preferred_entry, preferred_entry),
        "tp1_distance_atr": float(round((tp1 - preferred_entry) / max(atr, 1e-9), 3)),
        "hold_window_reachability_score": float(round(reachability_score, 3)),
        "target_reachability_flag": target_reachability_flag,
        "tp1_generation_reason": tp1_generation_reason,
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
    max_hold_date = datetime.now(timezone.utc) + timedelta(days=hold_days)
    return {
        "max_hold_days": int(hold_days),
        "max_hold_date": max_hold_date,
    }
