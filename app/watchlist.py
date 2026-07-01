"""Mutually exclusive watchlist prioritization for planned rows."""

from __future__ import annotations

from .config import PlanningConfig


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_watchlist_profile(row, *, config: PlanningConfig) -> dict:
    """Summarize whether a setup belongs on the primary watchlist, secondary
    watchlist, or nowhere right now.

    This layer is presentation-oriented and intentionally separate from both the
    action classifier and swing-trade suitability scoring.
    """

    final_action = str(getattr(row, "final_action", None) or "").upper()
    trend_state = str(getattr(row, "trend_state", None) or "")
    watch_priority = str(getattr(row, "watch_priority", None) or "").lower()
    monitorable_setup = bool(getattr(row, "monitorable_setup", False))
    composite_score = _safe_float(getattr(row, "composite_score", None))
    relative_strength_score = _safe_float(getattr(row, "relative_strength_score", None))
    avoid_reason = getattr(row, "avoid_reason", None)
    setup_scenario = str(getattr(row, "setup_scenario", None) or "")
    news_regime_alignment = str(getattr(row, "news_regime_alignment", None) or "neutral")

    suitability = getattr(row, "swing_trade_suitability", None) or {}
    suitability_score = _safe_float(suitability.get("suitability_score"))
    suitability_label = str(suitability.get("suitability_label") or "")
    suitable_for_long_swing = bool(suitability.get("suitable_for_long_swing"))
    suitable_for_watchlist_only = bool(suitability.get("suitable_for_watchlist_only"))
    not_suitable_reason = suitability.get("not_suitable_reason")

    constructive_traits = list(getattr(row, "constructive_traits", None) or [])
    constructive_enough = bool(
        trend_state == "pullback_in_uptrend"
        or relative_strength_score >= config.watchlist_primary_min_relative_strength_score
        or len(constructive_traits) >= 3
        or setup_scenario in {"strong_continuation_pullback", "supported_high_range_continuation", "range_rebound_candidate", "rebound_repair_candidate"}
    )
    primary_ready = bool(
        final_action in {"BUY", "WAIT"}
        and (final_action == "BUY" or monitorable_setup)
        and (suitable_for_watchlist_only or suitable_for_long_swing)
        and suitability_score >= config.watchlist_primary_min_suitability_score
        and suitability_label in {"medium", "high"}
        and composite_score >= config.watchlist_primary_min_composite_score
        and constructive_enough
        and not avoid_reason
        and not not_suitable_reason
        and (final_action == "BUY" or watch_priority == "high")
        and news_regime_alignment != "aligned_bearish"
    )

    secondary_ready = bool(
        final_action == "WAIT"
        and monitorable_setup
        and (suitable_for_watchlist_only or suitable_for_long_swing)
        and suitability_score >= config.watchlist_secondary_min_suitability_score
        and not not_suitable_reason
    )

    if primary_ready:
        watchlist_tier = "primary"
        watchlist_bucket = "high_priority_watchlist"
        is_primary_watchlist_candidate = True
        is_secondary_watchlist_candidate = False
        watchlist_reason = (
            "Constructive enough to monitor closely, but still waiting on better confirmation or timing."
        )
        watchlist_summary = (
            "This is a primary watchlist name. The setup is constructive enough to monitor closely, "
            "with supportive structure and relative strength, but it still needs confirmation."
        )
    elif secondary_ready:
        watchlist_tier = "secondary"
        watchlist_bucket = "secondary_watchlist"
        is_primary_watchlist_candidate = False
        is_secondary_watchlist_candidate = True
        watchlist_reason = (
            "Still monitorable, but quality is weaker and the setup needs more repair before it deserves top attention."
        )
        watchlist_summary = (
            "This is a secondary watchlist name. The setup is still monitorable, but quality is weaker and "
            "confirmation or structure repair is still needed."
        )
    else:
        watchlist_tier = "none"
        watchlist_bucket = "avoid"
        is_primary_watchlist_candidate = False
        is_secondary_watchlist_candidate = False
        watchlist_reason = str(
            not_suitable_reason
            or avoid_reason
            or "This setup is not worth watchlist attention right now because structure and confirmation are too weak."
        )
        watchlist_summary = (
            "This setup is not a watchlist priority right now because structure, confirmation, or payoff quality is too weak."
        )

    return {
        "watchlist_tier": watchlist_tier,
        "watchlist_bucket": watchlist_bucket,
        "watchlist_summary": watchlist_summary,
        "watchlist_reason": watchlist_reason,
        "is_primary_watchlist_candidate": bool(is_primary_watchlist_candidate),
        "is_secondary_watchlist_candidate": bool(is_secondary_watchlist_candidate),
    }
