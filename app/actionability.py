"""Deterministic prioritization for WAIT setups that may become tradable soon."""

from __future__ import annotations

from .config import PlanningConfig


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clip(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def _distance_to_zone_pct(current_price: float, zone: dict | None) -> float | None:
    if current_price <= 0 or not isinstance(zone, dict):
        return None
    lower = _safe_float(zone.get("lower"), -1.0)
    upper = _safe_float(zone.get("upper"), -1.0)
    if lower <= 0 or upper <= 0 or upper < lower:
        return None
    if lower <= current_price <= upper:
        return 0.0
    if current_price < lower:
        return (lower - current_price) / current_price
    return (current_price - upper) / current_price


def _closest_trigger(row, execution_view: dict) -> tuple[str | None, float | None]:
    current_price = _safe_float(getattr(row, "current_price", None), -1.0)
    if current_price <= 0:
        return None, None

    candidates: list[tuple[str, float]] = []
    current_anchor = execution_view.get("current_execution_anchor")
    current_anchor_type = str(execution_view.get("current_execution_anchor_type") or "")
    breakout_point = execution_view.get("breakout_point")
    breakout_type = str(execution_view.get("breakout_point_type") or "")
    pullback_zone = execution_view.get("pullback_entry_zone")

    anchor_distance = _distance_to_zone_pct(current_price, current_anchor)
    if anchor_distance is not None:
        if current_anchor_type in {"repair_band", "repair_band_still_active"}:
            trigger_type = "repair"
        elif current_anchor_type in {"reclaim_band"}:
            trigger_type = "reclaim"
        else:
            trigger_type = "pullback"
        candidates.append((trigger_type, anchor_distance))

    pullback_distance = _distance_to_zone_pct(current_price, pullback_zone)
    if pullback_distance is not None:
        candidates.append(("pullback", pullback_distance))

    breakout_distance = _distance_to_zone_pct(current_price, breakout_point)
    if breakout_distance is not None:
        if breakout_type == "repair_trigger":
            trigger_type = "repair"
        elif breakout_type == "reclaim_trigger":
            trigger_type = "reclaim"
        else:
            trigger_type = "breakout"
        candidates.append((trigger_type, breakout_distance))

    if bool(getattr(row, "entry_requires_confirmation", False)):
        candidates.append(("confirmation", min([c[1] for c in candidates], default=0.018)))

    if not candidates:
        return None, None
    trigger_type, distance = min(candidates, key=lambda item: item[1])
    return trigger_type, distance


def _estimate_days_to_action(row, *, closest_trigger_distance_pct: float | None, execution_view: dict, config: PlanningConfig) -> float:
    atr = _safe_float(getattr(row, "atr", None), -1.0)
    current_price = _safe_float(getattr(row, "current_price", None), -1.0)
    trade_shape = str(execution_view.get("trade_shape") or "")
    execution_bias = str(execution_view.get("execution_bias") or "")

    if closest_trigger_distance_pct is None:
        base_days = float(getattr(row, "monitor_window_days", None) or config.wait_monitor_days_other)
    else:
        if atr > 0 and current_price > 0:
            target_distance = closest_trigger_distance_pct * current_price
            base_days = target_distance / max(atr, 0.01)
        else:
            base_days = closest_trigger_distance_pct * 100.0 / 2.2

    if bool(getattr(row, "entry_requires_confirmation", False)):
        base_days += 0.7
    if trade_shape == "structure_repair_needed":
        base_days += 2.0
    elif execution_bias in {"avoid_chasing", "pullback_preferred"} and closest_trigger_distance_pct is not None and closest_trigger_distance_pct > 0.015:
        base_days += 0.8
    if str(getattr(row, "trend_state", None) or "") == "weak_breakdown_risk":
        base_days += 1.0

    window = int(getattr(row, "monitor_window_days", None) or config.wait_monitor_days_other)
    return round(_clip(base_days, low=0.5, high=max(float(window) + 2.0, 6.0)), 2)


def build_actionability_soon(row, *, config: PlanningConfig) -> dict | None:
    """Prioritize WAIT setups by how close they are to becoming actionable.

    This layer is intentionally narrower than swing-trade suitability. It helps
    reduce watch overload by separating WAIT names that deserve active
    monitoring now from those that can remain in the background.
    """

    final_action = str(getattr(row, "final_action", None) or "").upper()
    if final_action != "WAIT":
        return None

    execution_view = getattr(row, "chart_execution_view", None) or {}
    suitability = getattr(row, "swing_trade_suitability", None) or {}
    current_price_location = str(execution_view.get("current_price_location") or "")
    trade_shape = str(execution_view.get("trade_shape") or "")
    execution_bias = str(execution_view.get("execution_bias") or "")
    trend_state = str(getattr(row, "trend_state", None) or "")
    setup_scenario = str(getattr(row, "setup_scenario", None) or "")
    news_regime_alignment = str(getattr(row, "news_regime_alignment", None) or "neutral")
    watch_priority = str(getattr(row, "watch_priority", None) or "").lower()
    watchlist_tier = str(getattr(row, "watchlist_tier", None) or "").lower()
    monitorable_setup = bool(getattr(row, "monitorable_setup", False))
    market_regime = str(getattr(row, "market_regime", None) or "neutral").lower()
    suitability_score = _safe_float(suitability.get("suitability_score"))

    closest_trigger_type, closest_trigger_distance_pct = _closest_trigger(row, execution_view)
    trigger_distance_pct = closest_trigger_distance_pct if closest_trigger_distance_pct is not None else 0.08
    trigger_proximity_score = _clip(10.0 - (trigger_distance_pct * 100.0 * 1.7))
    if closest_trigger_distance_pct == 0.0:
        trigger_proximity_score = min(10.0, trigger_proximity_score + 0.9)

    confirmation_readiness_score = 6.3
    if bool(getattr(row, "entry_requires_confirmation", False)):
        confirmation_readiness_score -= 1.2
    if getattr(row, "confirmation_trigger", None):
        confirmation_readiness_score += 0.6
    if current_price_location in {"continuation_near_range_high", "near_resistance"}:
        confirmation_readiness_score -= 0.4
    if trade_shape == "structure_repair_needed":
        confirmation_readiness_score -= 1.6
    confirmation_readiness_score = _clip(confirmation_readiness_score)

    structure_readiness_score = 5.8
    if trend_state == "pullback_in_uptrend":
        structure_readiness_score = 8.1
    elif trade_shape in {"continuation_pullback_preferred", "breakout_or_pullback", "near_resistance_wait"}:
        structure_readiness_score = 6.9
    elif trade_shape == "post_breakout_retest":
        structure_readiness_score = 6.2
    elif trade_shape == "structure_repair_needed" or trend_state == "weak_breakdown_risk":
        structure_readiness_score = 3.9

    if watch_priority == "high":
        structure_readiness_score += 0.5
    elif watchlist_tier == "primary":
        structure_readiness_score += 0.35
    if setup_scenario in {"strong_continuation_pullback", "supported_high_range_continuation", "range_rebound_candidate"}:
        structure_readiness_score += 0.45
    elif setup_scenario in {"structure_still_damaged", "extension_needs_reset", "conflicted_setup"}:
        structure_readiness_score -= 0.55
    if news_regime_alignment == "aligned_bullish":
        structure_readiness_score += 0.35
    elif news_regime_alignment in {"aligned_bearish", "conflicted"}:
        structure_readiness_score -= 0.45
    structure_readiness_score = _clip(structure_readiness_score)

    days_to_action_estimate = _estimate_days_to_action(
        row,
        closest_trigger_distance_pct=closest_trigger_distance_pct,
        execution_view=execution_view,
        config=config,
    )
    monitor_window_days = int(getattr(row, "monitor_window_days", None) or config.wait_monitor_days_other)
    timing_readiness_score = _clip(
        9.2
        - (days_to_action_estimate * 1.05)
        + (0.5 if monitor_window_days <= 5 else 0.0)
        - (0.5 if market_regime == "risk_off" and trade_shape == "structure_repair_needed" else 0.0)
    )

    support_bonus = 0.0
    if watch_priority == "high":
        support_bonus += 0.75
    elif watch_priority == "medium":
        support_bonus += 0.35
    if watchlist_tier == "primary":
        support_bonus += 0.65
    elif watchlist_tier == "secondary":
        support_bonus += 0.25
    support_bonus += max(0.0, min(1.25, (suitability_score - 5.0) * 0.18))

    actionability_score = round(
        (
            trigger_proximity_score * 1.25
            + timing_readiness_score * 1.05
            + confirmation_readiness_score * 0.95
            + structure_readiness_score * 1.15
            + (4.5 + support_bonus) * 0.75
        )
        / 5.15,
        4,
    )

    key_reasons: list[str] = []
    not_ready_reasons: list[str] = []

    if monitorable_setup:
        key_reasons.append("Setup is already monitorable under the current plan")
    if watch_priority == "high" or watchlist_tier == "primary":
        key_reasons.append("Watch priority is elevated relative to other WAIT setups")
    if closest_trigger_distance_pct is not None and closest_trigger_distance_pct <= 0.015:
        key_reasons.append("Active execution area is nearby")
    if trade_shape in {"continuation_pullback_preferred", "breakout_or_pullback", "near_resistance_wait"}:
        key_reasons.append("Structure is close enough that timing is the main blocker")
    if trade_shape == "structure_repair_needed":
        not_ready_reasons.append("Structure still needs repair before it deserves close monitoring")
    if execution_bias in {"avoid_chasing", "pullback_preferred"} and current_price_location in {"continuation_near_range_high", "near_resistance"}:
        not_ready_reasons.append("Current price is not an attractive chase and likely needs a reset")
    if closest_trigger_distance_pct is not None and closest_trigger_distance_pct > 0.03:
        not_ready_reasons.append("Price is still a meaningful distance from the most relevant trigger area")
    if bool(getattr(row, "entry_requires_confirmation", False)):
        not_ready_reasons.append("Confirmation is still missing")

    ready_soon = bool(
        monitorable_setup
        and actionability_score >= config.actionability_ready_soon_threshold
        and structure_readiness_score >= config.actionability_ready_soon_min_structure
        and trigger_proximity_score >= config.actionability_ready_soon_min_trigger
        and timing_readiness_score >= config.actionability_ready_soon_min_timing
        and trade_shape != "structure_repair_needed"
        and (watch_priority == "high" or watchlist_tier == "primary")
    )
    if ready_soon:
        actionability_label = "ready_soon"
        active_watch = True
        watch_urgency = "high"
    elif monitorable_setup and actionability_score >= config.actionability_monitor_threshold:
        actionability_label = "monitor"
        active_watch = True
        watch_urgency = "medium"
    else:
        actionability_label = "background"
        active_watch = False
        watch_urgency = "low"

    if actionability_label == "ready_soon":
        actionability_summary = (
            "This WAIT setup is close to becoming actionable. Structure is constructive enough and the active execution area is nearby, so it deserves active monitoring."
        )
    elif actionability_label == "monitor":
        actionability_summary = (
            "This WAIT setup is valid, but it likely needs a better reset or more confirmation before it becomes tradable."
        )
    else:
        actionability_summary = (
            "This WAIT setup is still too early or too weak to monitor closely right now. Keep it in the background unless structure improves."
        )

    return {
        "actionability_score": actionability_score,
        "actionability_label": actionability_label,
        "active_watch": bool(active_watch),
        "watch_urgency": watch_urgency,
        "days_to_action_estimate": days_to_action_estimate,
        "closest_trigger_type": closest_trigger_type,
        "closest_trigger_distance_pct": None if closest_trigger_distance_pct is None else round(closest_trigger_distance_pct, 4),
        "trigger_proximity_score": round(trigger_proximity_score, 3),
        "timing_readiness_score": round(timing_readiness_score, 3),
        "confirmation_readiness_score": round(confirmation_readiness_score, 3),
        "structure_readiness_score": round(structure_readiness_score, 3),
        "key_reasons": key_reasons[:5],
        "not_ready_reasons": not_ready_reasons[:5],
        "actionability_summary": actionability_summary,
    }
