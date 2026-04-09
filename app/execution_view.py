"""Trader-readable execution framing built from structured planner fields."""

from __future__ import annotations

from .config import PlanningConfig
from .monitoring import build_zone_display


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _zone_width(zone: dict | None) -> float:
    if not zone:
        return 0.0
    return max(0.0, _safe_float(zone.get("upper")) - _safe_float(zone.get("lower")))


def _overlap_ratio(zone_a: dict | None, zone_b: dict | None) -> float:
    if not zone_a or not zone_b:
        return 0.0
    lower = max(_safe_float(zone_a.get("lower")), _safe_float(zone_b.get("lower")))
    upper = min(_safe_float(zone_a.get("upper")), _safe_float(zone_b.get("upper")))
    overlap = max(0.0, upper - lower)
    base = min(_zone_width(zone_a), _zone_width(zone_b))
    if base <= 0:
        return 0.0
    return overlap / base


def _build_display_payload(zone: dict | None, *, current_price: float | None, label: str) -> dict | None:
    if not zone:
        return None
    display = build_zone_display(zone, current_price=current_price, zone_label=label)
    if display["display"] is None:
        return None
    source_tags = list(zone.get("source_tags", []))
    return {
        "lower": float(zone["lower"]),
        "upper": float(zone["upper"]),
        "display": display["display"],
        "source_tags": source_tags,
    }


def _zone_quality(zone: dict | None, *, current_price: float, config: PlanningConfig) -> str:
    if not zone:
        return "messy"
    width_pct = _zone_width(zone) / max(current_price, 0.01)
    if width_pct <= config.execution_zone_max_width_pct * 0.45:
        return "clean"
    if width_pct <= config.execution_zone_max_width_pct:
        return "moderate"
    return "messy"


def _tighten_zone(
    zone: dict | None,
    *,
    current_price: float,
    atr: float,
    config: PlanningConfig,
    focus: str,
    anchor: float | None = None,
) -> dict | None:
    if not zone:
        return None
    lower = _safe_float(zone.get("lower"))
    upper = _safe_float(zone.get("upper"))
    if upper <= lower:
        return None

    width = upper - lower
    min_width = max(current_price * config.execution_zone_min_width_pct, atr * 0.18, 0.01)
    max_width = max(min_width, min(current_price * config.execution_zone_max_width_pct, atr * 1.1 if atr > 0 else current_price * config.execution_zone_max_width_pct))

    if focus == "breakout":
        target_width = max(min_width, min(max_width, width * config.execution_breakout_zone_fraction))
        if anchor is None:
            anchor = upper
        new_upper = upper
        new_lower = max(lower, anchor - target_width)
        if new_upper - new_lower < min_width:
            new_lower = max(lower, new_upper - min_width)
    elif focus == "pullback":
        target_width = max(min_width, min(max_width, width * config.execution_pullback_zone_fraction))
        if anchor is None:
            anchor = max(lower, min(upper, lower + width * 0.62))
        new_upper = min(upper, anchor + target_width * 0.4)
        new_lower = max(lower, new_upper - target_width)
        if new_upper - new_lower < min_width:
            new_upper = min(upper, new_lower + min_width)
    else:
        target_width = max(min_width, min(max_width, width * config.execution_deeper_zone_fraction))
        if anchor is None:
            anchor = max(lower, min(upper, lower + width * 0.28))
        new_lower = max(lower, anchor - target_width * 0.2)
        new_upper = min(upper, new_lower + target_width)
        if new_upper - new_lower < min_width:
            new_lower = max(lower, new_upper - min_width)

    tightened_lower = _clamp(new_lower, lower, upper)
    tightened_upper = _clamp(new_upper, tightened_lower + 0.01, upper)
    tightened = {
        "lower": round(tightened_lower, 6),
        "upper": round(tightened_upper, 6),
        "source_tags": list(zone.get("source_tags", [])),
    }
    return tightened if tightened["upper"] > tightened["lower"] else None


def _separate_deeper_zone(
    *,
    pullback_zone: dict | None,
    deeper_zone: dict | None,
    current_price: float,
    atr: float,
    config: PlanningConfig,
) -> dict | None:
    if not deeper_zone:
        return None
    if not pullback_zone:
        return deeper_zone

    overlap = _overlap_ratio(pullback_zone, deeper_zone)
    gap_floor = max(current_price * config.execution_zone_min_width_pct * 0.4, atr * 0.12, 0.02)
    deeper_lower = _safe_float(deeper_zone.get("lower"))
    deeper_upper = _safe_float(deeper_zone.get("upper"))
    pullback_lower = _safe_float(pullback_zone.get("lower"))

    if overlap > config.execution_zone_overlap_max_pct or deeper_upper >= pullback_lower:
        width = deeper_upper - deeper_lower
        max_width = max(
            current_price * config.execution_zone_max_width_pct * 0.75,
            atr * 0.8 if atr > 0 else current_price * config.execution_zone_max_width_pct * 0.75,
            current_price * config.execution_zone_min_width_pct,
        )
        target_width = min(width, max_width)
        new_upper = min(deeper_upper, pullback_lower - gap_floor)
        new_lower = max(deeper_lower, new_upper - target_width)
        if new_upper > new_lower:
            deeper_zone = {
                "lower": round(new_lower, 6),
                "upper": round(new_upper, 6),
                "source_tags": list(deeper_zone.get("source_tags", [])),
            }

    if _safe_float(deeper_zone.get("upper")) <= _safe_float(deeper_zone.get("lower")):
        return None
    return deeper_zone


def _current_price_location(
    *,
    current_price: float,
    pullback_zone: dict | None,
    breakout_point: dict | None,
    trend_state: str,
    entry_requires_confirmation: bool,
    atr: float,
    config: PlanningConfig,
) -> str:
    trigger_buffer = max(current_price * config.execution_near_trigger_buffer_pct, atr * 0.2, 0.02)

    if pullback_zone and current_price < _safe_float(pullback_zone.get("lower")):
        if trend_state in {"weak_breakdown_risk", "downtrend"}:
            return "structure_below_trigger"
        return "below_support"

    if breakout_point:
        breakout_lower = _safe_float(breakout_point.get("lower"))
        breakout_upper = _safe_float(breakout_point.get("upper"))
        if current_price > breakout_upper:
            distance_pct = (current_price - breakout_upper) / max(current_price, 0.01)
            if entry_requires_confirmation:
                return "above_first_trigger_not_confirmed"
            if distance_pct >= config.execution_extended_above_trigger_pct:
                return "extended_above_trigger"
            return "post_breakout_retest"
        if current_price >= breakout_lower - trigger_buffer:
            return "near_resistance"

    if pullback_zone:
        support_upper = _safe_float(pullback_zone.get("upper"))
        support_lower = _safe_float(pullback_zone.get("lower"))
        if support_lower - trigger_buffer <= current_price <= support_upper + trigger_buffer:
            return "near_support"

    if pullback_zone and breakout_point:
        return "mid_range"
    if breakout_point:
        return "near_resistance"
    if pullback_zone:
        return "near_support"
    return "mid_range"


def build_chart_execution_view(row, *, config: PlanningConfig) -> dict | None:
    """Frame the current setup in trader execution terms using existing zones."""

    current_price = _safe_float(getattr(row, "current_price", None), 0.0)
    if current_price <= 0:
        return None

    final_action = str(getattr(row, "final_action", None) or "").upper()
    trend_state = str(getattr(row, "trend_state", None) or "")
    preferred_entry = _safe_float(getattr(row, "preferred_entry", None), current_price)
    preferred_entry_type = str(getattr(row, "preferred_entry_type", None) or "")
    entry_quality = _safe_float(getattr(row, "entry_quality_score", None))
    entry_requires_confirmation = bool(getattr(row, "entry_requires_confirmation", False))
    confirmation_trigger = str(getattr(row, "confirmation_trigger", None) or "")
    reward_risk = getattr(row, "reward_risk", None) or {}
    rr1 = _safe_float(reward_risk.get("tp1"))
    support_zone_1 = getattr(row, "support_zone_1", None)
    support_zone_2 = getattr(row, "support_zone_2", None)
    resistance_zone_1 = getattr(row, "resistance_zone_1", None)
    resistance_zone_2 = getattr(row, "resistance_zone_2", None)
    volume_context = getattr(row, "volume_context", None) or {}
    atr = _safe_float(getattr(row, "atr", None))

    breakout_anchor = None
    if resistance_zone_2:
        breakout_anchor = min(_safe_float(resistance_zone_2.get("lower")), _safe_float(resistance_zone_1.get("upper", current_price)) if resistance_zone_1 else _safe_float(resistance_zone_2.get("lower")))
    breakout_point_raw = _tighten_zone(
        resistance_zone_1,
        current_price=current_price,
        atr=atr,
        config=config,
        focus="breakout",
        anchor=breakout_anchor,
    )
    pullback_anchor = preferred_entry if preferred_entry_type in {"pullback", "immediate"} else None
    pullback_zone_raw = _tighten_zone(
        support_zone_1,
        current_price=current_price,
        atr=atr,
        config=config,
        focus="pullback",
        anchor=pullback_anchor,
    )
    deeper_anchor = preferred_entry if preferred_entry_type == "deeper_pullback" else None
    deeper_pullback_raw = _tighten_zone(
        support_zone_2,
        current_price=current_price,
        atr=atr,
        config=config,
        focus="deeper",
        anchor=deeper_anchor,
    )
    deeper_pullback_raw = _separate_deeper_zone(
        pullback_zone=pullback_zone_raw,
        deeper_zone=deeper_pullback_raw,
        current_price=current_price,
        atr=atr,
        config=config,
    )

    breakout_point = _build_display_payload(breakout_point_raw, current_price=current_price, label="Breakout Point")
    pullback_entry_zone = _build_display_payload(pullback_zone_raw, current_price=current_price, label="Pullback Entry Zone")
    deeper_pullback_zone = _build_display_payload(deeper_pullback_raw, current_price=current_price, label="Deeper Pullback Zone")
    location = _current_price_location(
        current_price=current_price,
        pullback_zone=pullback_zone_raw,
        breakout_point=breakout_point_raw,
        trend_state=trend_state,
        entry_requires_confirmation=entry_requires_confirmation,
        atr=atr,
        config=config,
    )

    near_resistance = location in {"near_resistance", "post_breakout_retest", "above_first_trigger_not_confirmed", "extended_above_trigger"}
    near_support = location == "near_support"
    weak_reversal = str(volume_context.get("reversal_volume_state") or "") in {"weak_bounce", "no_confirmation"}
    weak_structure = trend_state in {"weak_breakdown_risk", "downtrend"}

    if weak_structure:
        trade_shape = "structure_repair_needed"
    elif near_resistance and breakout_point and pullback_entry_zone:
        trade_shape = "breakout_or_pullback"
    elif near_resistance:
        trade_shape = "extended_near_resistance"
    elif near_support or preferred_entry_type in {"pullback", "deeper_pullback"}:
        trade_shape = "pullback_candidate"
    elif breakout_point and trend_state in {"uptrend", "pullback_in_uptrend"}:
        trade_shape = "breakout_candidate"
    else:
        trade_shape = "no_clear_setup"

    if weak_structure:
        breakout_point_type = "repair_trigger"
    elif location in {"post_breakout_retest", "above_first_trigger_not_confirmed", "extended_above_trigger"}:
        breakout_point_type = "reclaim_trigger"
    else:
        breakout_point_type = "breakout_trigger"

    qualities = [
        _zone_quality(breakout_point_raw, current_price=current_price, config=config) if breakout_point_raw else "clean",
        _zone_quality(pullback_zone_raw, current_price=current_price, config=config) if pullback_zone_raw else "clean",
        _zone_quality(deeper_pullback_raw, current_price=current_price, config=config) if deeper_pullback_raw else "clean",
    ]
    execution_zone_quality = "messy" if "messy" in qualities else "moderate" if "moderate" in qualities else "clean"

    if final_action == "BUY" and near_support and not near_resistance and not entry_requires_confirmation and rr1 >= config.min_reward_risk_for_buy:
        enter_now = "yes"
    elif weak_structure or final_action == "AVOID":
        enter_now = "no"
    elif entry_requires_confirmation or near_resistance:
        enter_now = "only_on_confirmation"
    else:
        enter_now = "no"

    if trade_shape == "breakout_candidate":
        execution_bias = "breakout_preferred"
    elif trade_shape in {"pullback_candidate", "support_retest_setup"}:
        execution_bias = "pullback_preferred"
    elif trade_shape == "structure_repair_needed":
        execution_bias = "wait_for_repair"
    elif near_resistance:
        execution_bias = "avoid_chasing"
    else:
        execution_bias = "wait_for_confirmation"

    if enter_now == "yes":
        enter_now_reason = "Current price sits close enough to first support that immediate risk/reward is still actionable."
    elif enter_now == "only_on_confirmation":
        enter_now_reason = (
            f"Current price needs confirmation before it becomes attractive. {confirmation_trigger}".strip()
        )
    else:
        if weak_structure:
            enter_now_reason = "Structure still needs repair, so this should not be treated as an immediate execution setup."
        elif near_resistance:
            enter_now_reason = "Price is too close to the upper trigger area to justify chasing without a cleaner breakout or retest."
        else:
            enter_now_reason = "A better execution point likely comes from a pullback into support rather than entering at the current level."

    breakout_reason = None
    breakout_point_source = None
    if breakout_point:
        breakout_point_source = ",".join(breakout_point.get("source_tags", [])) or "resistance_cluster"
        if breakout_point_type == "repair_trigger":
            breakout_reason = "This is the reclaim area that would signal structure repair rather than a normal breakout."
        elif breakout_point_type == "reclaim_trigger":
            breakout_reason = "This is the nearest reclaim band to watch after price has already pushed above the first trigger."
        else:
            breakout_reason = "This is the nearest actionable resistance cluster to watch for breakout confirmation."

    pullback_reason = None
    pullback_zone_source = None
    if pullback_entry_zone:
        pullback_zone_source = ",".join(pullback_entry_zone.get("source_tags", [])) or "support_cluster"
        pullback_reason = "This is the cleaner first pullback shelf for improved swing-trade risk/reward."

    deeper_pullback_reason = None
    deeper_pullback_zone_source = None
    if deeper_pullback_zone:
        deeper_pullback_zone_source = ",".join(deeper_pullback_zone.get("source_tags", [])) or "deeper_support_cluster"
        deeper_pullback_reason = "This is the next lower support band if the first pullback zone fails to hold."

    summary_parts: list[str] = []
    if location == "near_resistance":
        summary_parts.append("Price is near recent resistance, so chasing is less attractive")
    elif location == "near_support":
        summary_parts.append("Price is closer to first support than resistance")
    elif location == "mid_range":
        summary_parts.append("Price is sitting in the middle of the active range")
    elif location == "post_breakout_retest":
        summary_parts.append("Price is above the first trigger and needs to hold that move on a retest")
    elif location == "above_first_trigger_not_confirmed":
        summary_parts.append("Price is above the first trigger zone, but the move is not cleanly confirmed yet")
    elif location == "extended_above_trigger":
        summary_parts.append("Price is stretched above the first trigger zone, so better execution likely comes on a retest")
    elif location == "structure_below_trigger":
        summary_parts.append("Price is below the first support trigger, so structure repair is still needed")
    elif location == "below_support":
        summary_parts.append("Price is below the first support zone")

    if breakout_point:
        if breakout_point_type == "repair_trigger":
            summary_parts.append(f"watch for a reclaim through {breakout_point['display']}")
        elif trade_shape in {"breakout_candidate", "breakout_or_pullback", "extended_near_resistance"} or location in {"post_breakout_retest", "above_first_trigger_not_confirmed", "extended_above_trigger"}:
            summary_parts.append(f"prefer breakout confirmation above {breakout_point['display']}")
    if pullback_entry_zone:
        summary_parts.append(f"preferred pullback support is {pullback_entry_zone['display']}")
    if deeper_pullback_zone:
        summary_parts.append(f"if that fails, deeper support sits near {deeper_pullback_zone['display']}")
    if weak_structure:
        summary_parts.append("before treating this as a clean swing setup")
    elif weak_reversal or entry_requires_confirmation:
        summary_parts.append("and confirmation is still needed before active entry")

    chart_execution_summary = ". ".join(part.rstrip(".") for part in summary_parts if part).strip()
    chart_execution_summary = f"{chart_execution_summary}." if chart_execution_summary else "The chart does not currently offer a clean swing execution pattern."

    return {
        "trade_shape": trade_shape,
        "enter_now": enter_now,
        "enter_now_reason": enter_now_reason,
        "breakout_point": breakout_point,
        "breakout_point_type": breakout_point_type if breakout_point else None,
        "breakout_point_source": breakout_point_source,
        "breakout_reason": breakout_reason,
        "pullback_entry_zone": pullback_entry_zone,
        "pullback_zone_source": pullback_zone_source,
        "pullback_reason": pullback_reason,
        "deeper_pullback_zone": deeper_pullback_zone,
        "deeper_pullback_zone_source": deeper_pullback_zone_source,
        "deeper_pullback_reason": deeper_pullback_reason,
        "current_price_location": location,
        "execution_bias": execution_bias,
        "execution_zone_quality": execution_zone_quality,
        "chart_execution_summary": chart_execution_summary,
    }
