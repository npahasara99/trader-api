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


def _level_zone(level: float | None, *, current_price: float, atr: float, config: PlanningConfig, tags: list[str]) -> dict | None:
    if level is None:
        return None
    half_width = max(current_price * config.execution_zone_min_width_pct * 0.5, atr * 0.12, 0.02)
    return {
        "lower": round(float(level) - half_width, 6),
        "upper": round(float(level) + half_width, 6),
        "source_tags": list(tags),
    }


def _zone_mid(zone: dict | None) -> float | None:
    if not zone:
        return None
    return (_safe_float(zone.get("lower")) + _safe_float(zone.get("upper"))) / 2.0


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
    focus_cap_map = {
        "breakout": config.execution_breakout_zone_max_width_pct,
        "pullback": config.execution_pullback_zone_max_width_pct,
        "deeper": config.execution_deeper_zone_max_width_pct,
    }
    max_width = max(
        min_width,
        min(
            current_price * focus_cap_map.get(focus, config.execution_zone_max_width_pct),
            atr * 0.95 if atr > 0 else current_price * focus_cap_map.get(focus, config.execution_zone_max_width_pct),
        ),
    )

    if focus == "breakout":
        target_width = max(min_width, min(max_width, width * config.execution_breakout_zone_fraction))
        if anchor is None:
            anchor = upper
        new_upper = min(upper, anchor + target_width * 0.15)
        new_lower = max(lower, new_upper - target_width)
        if new_upper - new_lower < min_width:
            new_lower = max(lower, new_upper - min_width)
    elif focus == "pullback":
        target_width = max(min_width, min(max_width, width * config.execution_pullback_zone_fraction))
        if anchor is None:
            anchor = max(lower, min(upper, lower + width * 0.62))
        new_upper = min(upper, anchor + target_width * 0.35)
        new_lower = max(lower, new_upper - target_width)
        if new_upper - new_lower < min_width:
            new_upper = min(upper, new_lower + min_width)
    else:
        target_width = max(min_width, min(max_width, width * config.execution_deeper_zone_fraction))
        if anchor is None:
            anchor = max(lower, min(upper, lower + width * 0.22))
        new_lower = max(lower, anchor - target_width * 0.18)
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
    gap_floor = max(current_price * config.execution_deeper_zone_min_gap_pct, atr * 0.14, 0.02)
    deeper_lower = _safe_float(deeper_zone.get("lower"))
    deeper_upper = _safe_float(deeper_zone.get("upper"))
    pullback_lower = _safe_float(pullback_zone.get("lower"))

    if overlap > config.execution_zone_overlap_max_pct or deeper_upper >= pullback_lower:
        width = deeper_upper - deeper_lower
        max_width = max(
            current_price * config.execution_deeper_zone_max_width_pct,
            atr * 0.7 if atr > 0 else current_price * config.execution_deeper_zone_max_width_pct,
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

    overlap_after = _overlap_ratio(pullback_zone, deeper_zone)
    if overlap_after > config.execution_deeper_zone_drop_overlap_pct:
        return None
    gap_after = _safe_float(pullback_zone.get("lower")) - _safe_float(deeper_zone.get("upper"))
    if gap_after < gap_floor * 0.35:
        return None
    if _safe_float(deeper_zone.get("upper")) <= _safe_float(deeper_zone.get("lower")):
        return None
    return deeper_zone


def _active_range_metrics(
    *,
    current_price: float,
    pullback_zone: dict | None,
    deeper_zone: dict | None,
    breakout_zone: dict | None,
    resistance_zone_2: dict | None,
    support_zone_2: dict | None,
    config: PlanningConfig,
) -> dict:
    active_low = None
    active_high = None

    if deeper_zone:
        active_low = _safe_float(deeper_zone.get("lower"))
    elif support_zone_2:
        active_low = _safe_float(support_zone_2.get("lower"))
    elif pullback_zone:
        active_low = _safe_float(pullback_zone.get("lower"))

    if resistance_zone_2:
        active_high = _safe_float(resistance_zone_2.get("upper"))
    elif breakout_zone:
        active_high = _safe_float(breakout_zone.get("upper"))

    range_position_pct = None
    distance_to_local_high_pct = None
    distance_to_local_low_pct = None
    is_near_recent_high = False
    is_near_recent_low = False

    if active_low is not None and active_high is not None and active_high > active_low:
        range_position_pct = round((current_price - active_low) / (active_high - active_low), 4)
        distance_to_local_high_pct = round((active_high - current_price) / max(current_price, 0.01), 4)
        distance_to_local_low_pct = round((current_price - active_low) / max(current_price, 0.01), 4)
        is_near_recent_high = bool(
            range_position_pct >= 1.0 - config.execution_range_near_high_pct
            or distance_to_local_high_pct <= config.execution_range_near_high_pct * 0.5
        )
        is_near_recent_low = bool(
            range_position_pct <= config.execution_range_near_low_pct
            or distance_to_local_low_pct <= config.execution_range_near_low_pct * 0.5
        )

    return {
        "active_range_low": None if active_low is None else round(active_low, 6),
        "active_range_high": None if active_high is None else round(active_high, 6),
        "range_position_pct": None if range_position_pct is None else round(range_position_pct, 4),
        "distance_to_local_high_pct": distance_to_local_high_pct,
        "distance_to_local_low_pct": distance_to_local_low_pct,
        "is_near_recent_high": is_near_recent_high,
        "is_near_recent_low": is_near_recent_low,
    }


def _current_price_location(
    *,
    current_price: float,
    pullback_zone: dict | None,
    first_trigger_zone: dict | None,
    breakout_point: dict | None,
    trend_state: str,
    entry_requires_confirmation: bool,
    atr: float,
    config: PlanningConfig,
    range_metrics: dict,
) -> str:
    trigger_buffer = max(current_price * config.execution_near_trigger_buffer_pct, atr * 0.2, 0.02)
    constructive_trend = trend_state in {"uptrend", "pullback_in_uptrend"}

    if pullback_zone and current_price < _safe_float(pullback_zone.get("lower")):
        if trend_state in {"weak_breakdown_risk", "downtrend"}:
            return "structure_below_trigger"
        return "below_support"

    if first_trigger_zone:
        first_upper = _safe_float(first_trigger_zone.get("upper"))
        if current_price > first_upper:
            if breakout_point and breakout_point is not first_trigger_zone:
                breakout_lower = _safe_float(breakout_point.get("lower"))
                breakout_upper = _safe_float(breakout_point.get("upper"))
                if current_price < breakout_lower:
                    if constructive_trend and range_metrics.get("is_near_recent_high"):
                        return "continuation_near_range_high"
                    return "above_first_trigger_not_confirmed" if entry_requires_confirmation else "post_breakout_retest"
                if current_price > breakout_upper:
                    distance_pct = (current_price - breakout_upper) / max(current_price, 0.01)
                    if entry_requires_confirmation:
                        return "above_first_trigger_not_confirmed"
                    if distance_pct >= config.execution_extended_above_trigger_pct:
                        return "extended_above_trigger"
                    return "post_breakout_retest"
                if constructive_trend and range_metrics.get("is_near_recent_high"):
                    return "continuation_near_range_high"
                return "near_resistance"
            distance_pct = (current_price - first_upper) / max(current_price, 0.01)
            if entry_requires_confirmation:
                return "above_first_trigger_not_confirmed"
            if distance_pct >= config.execution_extended_above_trigger_pct:
                return "extended_above_trigger"
            if constructive_trend and range_metrics.get("is_near_recent_high"):
                return "continuation_near_range_high"
            return "post_breakout_retest"

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

    if range_metrics.get("is_near_recent_high"):
        return "continuation_near_range_high" if constructive_trend else "near_resistance"
    if range_metrics.get("is_near_recent_low"):
        return "near_support"
    if pullback_zone and breakout_point:
        return "mid_range"
    if breakout_point:
        return "near_resistance"
    if pullback_zone:
        return "near_support"
    return "mid_range"


def _select_active_breakout_zone(
    *,
    current_price: float,
    atr: float,
    config: PlanningConfig,
    prior_trigger_status: str | None,
    prior_trigger_raw: dict | None,
    current_execution_anchor_raw: dict | None,
    breakout_point_raw: dict | None,
    resistance_zone_2: dict | None,
    consolidation_range: dict | None,
    range_metrics: dict,
    weak_structure: bool,
) -> tuple[dict | None, str]:
    if prior_trigger_status != "context_only":
        return breakout_point_raw, "existing_regime"

    current_anchor_upper = _safe_float(current_execution_anchor_raw.get("upper")) if current_execution_anchor_raw else current_price
    prior_upper = _safe_float(prior_trigger_raw.get("upper")) if prior_trigger_raw else 0.0
    min_trigger_floor = max(current_anchor_upper, current_price * (1.0 + config.execution_near_trigger_buffer_pct), prior_upper)

    candidates: list[tuple[float, dict, str]] = []

    if resistance_zone_2:
        candidate = _tighten_zone(
            resistance_zone_2,
            current_price=current_price,
            atr=atr,
            config=config,
            focus="breakout",
            anchor=_safe_float(resistance_zone_2.get("upper")),
        )
        if candidate and _safe_float(candidate.get("upper")) > min_trigger_floor:
            candidates.append((_safe_float(candidate.get("lower")), candidate, "resistance_zone_2"))

    if consolidation_range:
        consolidation_high = _safe_float(consolidation_range.get("upper"))
        if consolidation_high > min_trigger_floor:
            candidate = _level_zone(
                consolidation_high,
                current_price=current_price,
                atr=atr,
                config=config,
                tags=list(consolidation_range.get("source_tags", [])) + ["active_range_high"],
            )
            if candidate:
                candidates.append((_safe_float(candidate.get("lower")), candidate, "active_range_high"))

    active_range_high = _safe_float(range_metrics.get("active_range_high"))
    if active_range_high > min_trigger_floor:
        candidate = _level_zone(
            active_range_high,
            current_price=current_price,
            atr=atr,
            config=config,
            tags=["active_range_high"],
        )
        if candidate:
            candidates.append((_safe_float(candidate.get("lower")), candidate, "active_range_high"))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], candidates[0][2]

    if weak_structure:
        return None, "no_clean_repair_trigger"
    return None, "no_clean_active_trigger"


def build_chart_execution_view(row, *, config: PlanningConfig) -> dict | None:
    """Frame the current setup in trader execution terms using existing zones."""

    current_price = _safe_float(getattr(row, "current_price", None), 0.0)
    if current_price <= 0:
        return None

    final_action = str(getattr(row, "final_action", None) or "").upper()
    trend_state = str(getattr(row, "trend_state", None) or "")
    preferred_entry = _safe_float(getattr(row, "preferred_entry", None), current_price)
    preferred_entry_type = str(getattr(row, "preferred_entry_type", None) or "")
    entry_requires_confirmation = bool(getattr(row, "entry_requires_confirmation", False))
    confirmation_trigger = str(getattr(row, "confirmation_trigger", None) or "")
    reward_risk = getattr(row, "reward_risk", None) or {}
    rr1 = _safe_float(reward_risk.get("tp1"))
    support_zone_1 = getattr(row, "support_zone_1", None)
    support_zone_2 = getattr(row, "support_zone_2", None)
    resistance_zone_1 = getattr(row, "resistance_zone_1", None)
    resistance_zone_2 = getattr(row, "resistance_zone_2", None)
    breakout_level = getattr(row, "breakout_level", None)
    prior_breakout_retest_zone = getattr(row, "prior_breakout_retest_zone", None)
    consolidation_range = getattr(row, "consolidation_range", None)
    volume_context = getattr(row, "volume_context", None) or {}
    atr = _safe_float(getattr(row, "atr", None))
    weak_structure = trend_state in {"weak_breakdown_risk", "downtrend"}
    constructive_trend = trend_state in {"uptrend", "pullback_in_uptrend"}

    above_first_resistance = bool(
        resistance_zone_1 and current_price > _safe_float(resistance_zone_1.get("upper")) + max(current_price * config.execution_near_trigger_buffer_pct, atr * 0.08, 0.02)
    )

    breakout_anchor = None
    breakout_source_zone = resistance_zone_1
    first_trigger_raw = _tighten_zone(
        resistance_zone_1,
        current_price=current_price,
        atr=atr,
        config=config,
        focus="breakout",
        anchor=None,
    )
    if above_first_resistance and resistance_zone_2:
        breakout_source_zone = resistance_zone_2
        breakout_anchor = _safe_float(resistance_zone_2.get("lower"))
    elif resistance_zone_2:
        breakout_anchor = min(
            _safe_float(resistance_zone_2.get("lower")),
            _safe_float(resistance_zone_1.get("upper", current_price)) if resistance_zone_1 else _safe_float(resistance_zone_2.get("lower")),
        )

    breakout_point_raw = _tighten_zone(
        breakout_source_zone,
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

    range_metrics = _active_range_metrics(
        current_price=current_price,
        pullback_zone=pullback_zone_raw,
        deeper_zone=deeper_pullback_raw,
        breakout_zone=breakout_point_raw,
        resistance_zone_2=resistance_zone_2,
        support_zone_2=support_zone_2,
        config=config,
    )

    pullback_entry_zone = _build_display_payload(pullback_zone_raw, current_price=current_price, label="Pullback Entry Zone")
    deeper_pullback_zone = _build_display_payload(deeper_pullback_raw, current_price=current_price, label="Deeper Pullback Zone")
    deeper_pullback_available = deeper_pullback_zone is not None

    initial_location = _current_price_location(
        current_price=current_price,
        pullback_zone=pullback_zone_raw,
        first_trigger_zone=first_trigger_raw,
        breakout_point=breakout_point_raw,
        trend_state=trend_state,
        entry_requires_confirmation=entry_requires_confirmation,
        atr=atr,
        config=config,
        range_metrics=range_metrics,
    )

    weak_reversal = str(volume_context.get("reversal_volume_state") or "") in {"weak_bounce", "no_confirmation"}
    near_recent_high = bool(range_metrics.get("is_near_recent_high"))
    range_position = range_metrics.get("range_position_pct")
    prior_trigger_raw = None
    if weak_structure and prior_breakout_retest_zone:
        prior_trigger_raw = _tighten_zone(
            prior_breakout_retest_zone,
            current_price=current_price,
            atr=atr,
            config=config,
            focus="breakout",
            anchor=_safe_float(prior_breakout_retest_zone.get("upper")),
        )
    elif first_trigger_raw:
        prior_trigger_raw = first_trigger_raw
    elif breakout_level is not None:
        prior_trigger_raw = _level_zone(
            breakout_level,
            current_price=current_price,
            atr=atr,
            config=config,
            tags=["breakout_level"],
        )

    prior_trigger_type = None
    if prior_trigger_raw:
        if weak_structure:
            prior_trigger_type = "repair_trigger"
        elif above_first_resistance or initial_location in {"post_breakout_retest", "above_first_trigger_not_confirmed", "continuation_near_range_high", "extended_above_trigger"}:
            prior_trigger_type = "reclaim_trigger"
        else:
            prior_trigger_type = "breakout_trigger"

    prior_trigger_status = None
    if prior_trigger_raw:
        prior_upper = _safe_float(prior_trigger_raw.get("upper"))
        prior_lower = _safe_float(prior_trigger_raw.get("lower"))
        range_high = _safe_float(range_metrics.get("active_range_high"))
        if current_price > prior_upper * (1.0 + config.execution_reanchor_above_prior_trigger_pct):
            prior_trigger_status = "context_only"
        elif range_high > 0 and current_price > range_high * (1.0 + config.execution_reanchor_above_range_pct):
            prior_trigger_status = "context_only"
        elif current_price < prior_lower - max(current_price * config.execution_zone_min_width_pct, atr * 0.3, 0.02):
            prior_trigger_status = "stale"
        else:
            prior_trigger_status = "active"

    current_execution_anchor_raw = None
    current_execution_anchor_type = None
    if weak_structure and pullback_zone_raw:
        current_execution_anchor_raw = pullback_zone_raw
        current_execution_anchor_type = "repair_band"
    elif weak_structure and breakout_point_raw:
        current_execution_anchor_raw = breakout_point_raw
        current_execution_anchor_type = "repair_band"
    elif prior_trigger_status == "context_only" and pullback_zone_raw:
        current_execution_anchor_raw = pullback_zone_raw
        current_execution_anchor_type = "continuation_support"
    elif preferred_entry_type in {"pullback", "deeper_pullback", "immediate"} and pullback_zone_raw:
        current_execution_anchor_raw = pullback_zone_raw
        current_execution_anchor_type = "pullback_support"
    elif breakout_point_raw:
        current_execution_anchor_raw = breakout_point_raw
        current_execution_anchor_type = "resistance_trigger"

    breakout_point_raw, active_breakout_source = _select_active_breakout_zone(
        current_price=current_price,
        atr=atr,
        config=config,
        prior_trigger_status=prior_trigger_status,
        prior_trigger_raw=prior_trigger_raw,
        current_execution_anchor_raw=current_execution_anchor_raw,
        breakout_point_raw=breakout_point_raw,
        resistance_zone_2=resistance_zone_2,
        consolidation_range=consolidation_range,
        range_metrics=range_metrics,
        weak_structure=weak_structure,
    )
    breakout_point = _build_display_payload(breakout_point_raw, current_price=current_price, label="Breakout Point")

    location = _current_price_location(
        current_price=current_price,
        pullback_zone=pullback_zone_raw,
        first_trigger_zone=first_trigger_raw,
        breakout_point=breakout_point_raw,
        trend_state=trend_state,
        entry_requires_confirmation=entry_requires_confirmation,
        atr=atr,
        config=config,
        range_metrics=range_metrics,
    )
    repair_anchor_buffer = max(current_price * config.execution_zone_min_width_pct, atr * 0.18, 0.02)
    if weak_structure:
        if prior_trigger_status == "context_only":
            location = "repair_reclaimed_but_not_clean"
        else:
            location = "repair_band_still_active"
            if current_execution_anchor_raw:
                anchor_lower = _safe_float(current_execution_anchor_raw.get("lower"))
                anchor_upper = _safe_float(current_execution_anchor_raw.get("upper"))
                if not (anchor_lower - repair_anchor_buffer <= current_price <= anchor_upper + repair_anchor_buffer):
                    location = "repair_band_still_active"
    elif prior_trigger_status == "context_only" and location in {"post_breakout_retest", "above_first_trigger_not_confirmed"}:
        location = "continuation_above_old_trigger" if not range_metrics.get("is_near_recent_high") else "continuation_near_range_high"
    elif constructive_trend and location == "above_first_trigger_not_confirmed":
        location = "continuation_near_range_high" if range_metrics.get("is_near_recent_high") else "near_resistance"

    if weak_structure:
        trade_shape = "structure_repair_needed"
    elif location in {"post_breakout_retest", "above_first_trigger_not_confirmed"}:
        trade_shape = "post_breakout_retest" if not weak_reversal else "continuation_pullback_preferred"
    elif location == "extended_above_trigger":
        trade_shape = "extended_after_breakout"
    elif location in {"continuation_near_range_high", "continuation_above_old_trigger"}:
        trade_shape = "continuation_pullback_preferred"
    elif location == "near_resistance" and constructive_trend and pullback_entry_zone and breakout_point:
        trade_shape = "continuation_pullback_preferred" if prior_trigger_status == "context_only" else "breakout_or_pullback"
    elif location == "near_resistance":
        trade_shape = "near_resistance_wait"
    elif location == "near_support" and constructive_trend:
        trade_shape = "pullback_candidate"
    elif breakout_point and constructive_trend and range_position is not None and range_position >= 0.62:
        trade_shape = "fresh_breakout_candidate"
    else:
        trade_shape = "no_clear_setup"

    if not breakout_point_raw:
        breakout_point_type = "none"
    elif weak_structure:
        breakout_point_type = "repair_trigger"
    elif prior_trigger_status == "context_only" or location in {"post_breakout_retest", "above_first_trigger_not_confirmed", "extended_above_trigger", "continuation_above_old_trigger"}:
        breakout_point_type = "reclaim_trigger"
    else:
        breakout_point_type = "breakout_trigger"

    if trade_shape == "fresh_breakout_candidate" and breakout_point_raw:
        current_execution_anchor_raw = breakout_point_raw
        current_execution_anchor_type = "resistance_trigger"
    elif trade_shape == "pullback_candidate" and pullback_zone_raw:
        current_execution_anchor_raw = pullback_zone_raw
        current_execution_anchor_type = "pullback_support"
    elif trade_shape in {"continuation_pullback_preferred", "post_breakout_retest", "extended_after_breakout", "near_resistance_wait", "breakout_or_pullback"} and pullback_zone_raw:
        current_execution_anchor_raw = pullback_zone_raw
        current_execution_anchor_type = "continuation_support" if prior_trigger_status == "context_only" or trade_shape in {"continuation_pullback_preferred", "post_breakout_retest", "extended_after_breakout"} else "pullback_support"
    elif not current_execution_anchor_raw and breakout_point_raw:
        current_execution_anchor_raw = breakout_point_raw
        current_execution_anchor_type = "reclaim_band"

    qualities = [
        _zone_quality(breakout_point_raw, current_price=current_price, config=config) if breakout_point_raw else "clean",
        _zone_quality(pullback_zone_raw, current_price=current_price, config=config) if pullback_zone_raw else "clean",
        _zone_quality(deeper_pullback_raw, current_price=current_price, config=config) if deeper_pullback_raw else "clean",
    ]
    execution_zone_quality = "messy" if "messy" in qualities else "moderate" if "moderate" in qualities else "clean"

    if final_action == "BUY" and location == "near_support" and not entry_requires_confirmation and rr1 >= config.min_reward_risk_for_buy:
        enter_now = "yes"
    elif weak_structure or final_action == "AVOID":
        enter_now = "no"
    elif location in {"near_resistance", "post_breakout_retest", "above_first_trigger_not_confirmed", "extended_above_trigger", "continuation_above_old_trigger", "continuation_near_range_high", "repair_reclaimed_but_not_clean", "repair_band_still_active"} or entry_requires_confirmation:
        enter_now = "only_on_confirmation"
    else:
        enter_now = "no"

    if trade_shape in {"fresh_breakout_candidate"}:
        execution_bias = "breakout_preferred"
    elif trade_shape in {"pullback_candidate", "continuation_pullback_preferred", "post_breakout_retest"}:
        execution_bias = "pullback_preferred"
    elif trade_shape == "structure_repair_needed":
        execution_bias = "wait_for_repair"
    elif trade_shape in {"breakout_or_pullback", "near_resistance_wait", "extended_after_breakout"}:
        execution_bias = "avoid_chasing"
    else:
        execution_bias = "wait_for_confirmation"

    if enter_now == "yes":
        enter_now_reason = "Current price sits close enough to first support that immediate risk/reward is still actionable."
    elif enter_now == "only_on_confirmation":
        if trade_shape in {"post_breakout_retest", "continuation_pullback_preferred", "extended_after_breakout"}:
            enter_now_reason = "Price already ran through an earlier trigger, so confirmation or a cleaner reset is still needed."
        elif trade_shape == "structure_repair_needed" and prior_trigger_status == "context_only":
            enter_now_reason = "The earlier repair trigger has already been reclaimed, but the current structure is still not clean enough for immediate entry."
        elif trade_shape == "structure_repair_needed" and location == "repair_band_still_active":
            enter_now_reason = "Price is still trading around the active repair band, so this remains a repair-monitoring setup rather than a normal continuation entry."
        else:
            enter_now_reason = f"Current price needs confirmation before it becomes attractive. {confirmation_trigger}".strip()
    else:
        if weak_structure:
            enter_now_reason = "Structure still needs repair, so this should not be treated as an immediate execution setup."
        elif trade_shape in {"near_resistance_wait", "breakout_or_pullback"}:
            enter_now_reason = "Price is near the upper part of the range, so chasing is less attractive than waiting for breakout confirmation or a pullback."
        else:
            enter_now_reason = "A better execution point likely comes from a pullback into support rather than entering at the current level."

    breakout_reason = None
    breakout_point_source = None
    if breakout_point:
        breakout_point_source = active_breakout_source or ",".join(breakout_point.get("source_tags", [])) or "resistance_cluster"
        if breakout_point_type == "repair_trigger":
            breakout_reason = "This is the reclaim area that would signal structure repair rather than a normal breakout."
        elif breakout_point_type == "reclaim_trigger":
            breakout_reason = "This is the next reclaim or continuation band after price already moved through an earlier trigger."
        else:
            breakout_reason = "This is the tightest nearby breakout band that would confirm a fresh move through resistance."
    elif prior_trigger_status == "context_only":
        breakout_reason = "The earlier trigger is now context only, and there is no clean current breakout or continuation trigger in the active regime."

    prior_trigger_anchor = _build_display_payload(prior_trigger_raw, current_price=current_price, label="Prior Trigger Anchor")
    current_execution_anchor = _build_display_payload(current_execution_anchor_raw, current_price=current_price, label="Current Execution Anchor")

    pullback_reason = None
    pullback_zone_source = None
    if pullback_entry_zone:
        pullback_zone_source = ",".join(pullback_entry_zone.get("source_tags", [])) or "support_cluster"
        if trade_shape in {"post_breakout_retest", "continuation_pullback_preferred", "extended_after_breakout"}:
            pullback_reason = "This is the cleaner reset area to watch after price already moved through an earlier trigger."
        else:
            pullback_reason = "This is the cleaner first pullback shelf for improved swing-trade risk/reward."

    deeper_pullback_reason = None
    deeper_pullback_zone_source = None
    if deeper_pullback_zone:
        deeper_pullback_zone_source = ",".join(deeper_pullback_zone.get("source_tags", [])) or "deeper_support_cluster"
        deeper_pullback_reason = "This is the next lower support band if the first pullback zone fails to hold."

    summary_parts: list[str] = []
    if trade_shape in {"breakout_or_pullback", "near_resistance_wait"}:
        summary_parts.append("Price is near the upper end of the recent range, so chasing is less attractive")
    elif trade_shape in {"post_breakout_retest", "continuation_pullback_preferred"}:
        summary_parts.append("Price is already above the earlier trigger area, so this is no longer a fresh breakout setup")
    elif trade_shape == "extended_after_breakout":
        summary_parts.append("Price has already stretched beyond the earlier trigger area and likely needs a reset")
    elif trade_shape == "pullback_candidate":
        summary_parts.append("Price is sitting close enough to support to frame this as a pullback setup")
    elif trade_shape == "structure_repair_needed":
        summary_parts.append("Structure still needs repair before this becomes a clean swing setup")
    else:
        if location == "mid_range":
            summary_parts.append("Price is sitting in the middle of the active range")
        elif location == "near_support":
            summary_parts.append("Price is closer to support than resistance")
        elif location == "repair_band_still_active":
            summary_parts.append("Price is still trading around the active repair band")
        elif location == "repair_reclaimed_but_not_clean":
            summary_parts.append("The earlier repair trigger has been reclaimed, but the setup still is not clean")

    if prior_trigger_anchor and prior_trigger_status == "context_only":
        summary_parts.append(f"the earlier trigger at {prior_trigger_anchor['display']} is now context only")
    elif prior_trigger_anchor and prior_trigger_status == "stale":
        summary_parts.append(f"the earlier trigger at {prior_trigger_anchor['display']} is now stale context")

    if breakout_point:
        if breakout_point_type == "repair_trigger":
            summary_parts.append(f"watch for reclaim through {breakout_point['display']}")
        elif trade_shape in {"fresh_breakout_candidate", "breakout_or_pullback", "near_resistance_wait"}:
            summary_parts.append(f"prefer confirmed breakout above {breakout_point['display']}")
        elif trade_shape in {"post_breakout_retest", "continuation_pullback_preferred", "extended_after_breakout"}:
            summary_parts.append(f"treat {breakout_point['display']} as the continuation/reclaim band")
    elif prior_trigger_status == "context_only":
        summary_parts.append("there is no clean fresh breakout trigger in the active regime right now")
    if current_execution_anchor:
        if current_execution_anchor_type in {"continuation_support", "pullback_support"}:
            if trade_shape in {"post_breakout_retest", "continuation_pullback_preferred", "extended_after_breakout"}:
                summary_parts.append(f"the active execution area is now {current_execution_anchor['display']}")
            else:
                summary_parts.append(f"preferred pullback support is {current_execution_anchor['display']}")
        elif current_execution_anchor_type in {"repair_band", "reclaim_band"}:
            if location == "repair_band_still_active":
                summary_parts.append(f"watch the active repair band around {current_execution_anchor['display']}")
            else:
                summary_parts.append(f"watch the current execution area around {current_execution_anchor['display']}")
    elif pullback_entry_zone:
        summary_parts.append(f"preferred pullback support is {pullback_entry_zone['display']}")
    if deeper_pullback_zone:
        summary_parts.append(f"if that fails, deeper support sits near {deeper_pullback_zone['display']}")

    chart_execution_summary = ". ".join(part.rstrip(".") for part in summary_parts if part).strip()
    chart_execution_summary = f"{chart_execution_summary}." if chart_execution_summary else "The chart does not currently offer a clean swing execution pattern."

    return {
        "trade_shape": trade_shape,
        "enter_now": enter_now,
        "enter_now_reason": enter_now_reason,
        "breakout_point": breakout_point,
        "breakout_point_type": breakout_point_type,
        "breakout_point_source": breakout_point_source,
        "breakout_reason": breakout_reason,
        "prior_trigger_anchor": prior_trigger_anchor,
        "prior_trigger_anchor_type": prior_trigger_type,
        "prior_trigger_anchor_display": None if not prior_trigger_anchor else prior_trigger_anchor["display"],
        "prior_trigger_anchor_status": prior_trigger_status,
        "current_execution_anchor": current_execution_anchor,
        "current_execution_anchor_type": current_execution_anchor_type,
        "current_execution_anchor_display": None if not current_execution_anchor else current_execution_anchor["display"],
        "current_execution_anchor_status": "active" if current_execution_anchor else None,
        "pullback_entry_zone": pullback_entry_zone,
        "pullback_zone_source": pullback_zone_source,
        "pullback_reason": pullback_reason,
        "deeper_pullback_zone": deeper_pullback_zone,
        "deeper_pullback_available": deeper_pullback_available,
        "deeper_pullback_zone_source": deeper_pullback_zone_source,
        "deeper_pullback_reason": deeper_pullback_reason,
        "current_price_location": location,
        "execution_bias": execution_bias,
        "execution_zone_quality": execution_zone_quality,
        "active_range_high": range_metrics["active_range_high"],
        "active_range_low": range_metrics["active_range_low"],
        "range_position_pct": range_metrics["range_position_pct"],
        "distance_to_local_high_pct": range_metrics["distance_to_local_high_pct"],
        "distance_to_local_low_pct": range_metrics["distance_to_local_low_pct"],
        "is_near_recent_high": range_metrics["is_near_recent_high"],
        "is_near_recent_low": range_metrics["is_near_recent_low"],
        "chart_execution_summary": chart_execution_summary,
    }
