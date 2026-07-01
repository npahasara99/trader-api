"""Deterministic swing-trade suitability scoring for planned rows."""

from __future__ import annotations

from .config import PlanningConfig


def _clip(value: float) -> float:
    return max(0.0, min(10.0, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _label(score: float, config: PlanningConfig) -> str:
    if score >= config.suitability_high_threshold:
        return "high"
    if score >= config.suitability_medium_threshold:
        return "medium"
    if score >= config.suitability_low_threshold:
        return "low"
    return "unsuitable"


def build_swing_trade_suitability(row, *, config: PlanningConfig) -> dict:
    """Score whether a planned stock is a practical swing-trade candidate.

    This layer is intentionally separate from BUY / WAIT / AVOID classification.
    It answers whether the name is structurally and practically suitable for
    swing trading right now or soon, even if timing is not yet actionable.
    """

    trend_state = str(getattr(row, "trend_state", None) or "")
    market_regime = str(getattr(row, "market_regime", None) or "neutral")
    final_action = str(getattr(row, "final_action", None) or "")
    composite = _safe_float(getattr(row, "composite_score", None))
    rr = getattr(row, "reward_risk", None) or {}
    earnings = getattr(row, "earnings", None) or {}
    volume_context = getattr(row, "volume_context", None) or {}
    price_location_context = str(getattr(row, "price_location_context", None) or "")
    setup_scenario = str(getattr(row, "setup_scenario", None) or "")
    news_regime_alignment = str(getattr(row, "news_regime_alignment", None) or "neutral")
    macro_alignment_score = _safe_float(getattr(row, "macro_alignment_score", None), 5.0)

    trend_suitability = _safe_float(getattr(row, "trend_quality_score", None))
    if trend_state == "pullback_in_uptrend":
        trend_suitability = max(trend_suitability, 7.2)
    elif trend_state == "weak_breakdown_risk":
        trend_suitability = min(trend_suitability, 5.0)
    elif trend_state == "downtrend":
        trend_suitability = min(trend_suitability, 2.2)

    structure_suitability = _clip(
        (_safe_float(getattr(row, "support_quality_score", None)) * 0.7)
        + (1.2 if getattr(row, "support_zone_1", None) else 0.0)
        + (0.8 if getattr(row, "support_zone_2", None) else 0.0)
        + (0.6 if getattr(row, "stop_basis", None) else 0.0)
    )
    entry_suitability = _clip(
        _safe_float(getattr(row, "entry_quality_score", None))
        - (1.0 if bool(getattr(row, "entry_requires_confirmation", False)) else 0.0)
        - (0.6 if abs(_safe_float(getattr(row, "entry_distance_from_current_price_pct", None))) > 4.5 else 0.0)
    )
    reward_risk_suitability = _clip(
        (_safe_float(rr.get("tp1")) * 3.2) * 0.65
        + (_safe_float(rr.get("tp2")) * 1.2) * 0.35
    )

    atr_pct = _safe_float(getattr(row, "atr_pct", None), -1.0) * 100.0
    if atr_pct < 0:
        volatility_suitability = 5.0
    elif atr_pct < 1.0:
        volatility_suitability = 4.8
    elif atr_pct <= 4.5:
        volatility_suitability = 8.0
    elif atr_pct <= 6.5:
        volatility_suitability = 6.3
    else:
        volatility_suitability = 4.1
    if bool(getattr(row, "stop_too_tight_flag", False)):
        volatility_suitability = max(0.0, volatility_suitability - 1.2)

    volume_confirmation_suitability = _safe_float(getattr(row, "volume_confirmation_score", None))
    relative_strength_suitability = _safe_float(getattr(row, "relative_strength_score", None))
    event_risk_suitability = _safe_float(getattr(row, "earnings_risk_score", None))

    timing_suitability = 6.2
    max_hold_days = getattr(row, "max_hold_days", None)
    monitor_window_days = getattr(row, "monitor_window_days", None)
    if max_hold_days is not None:
        hold_days = int(max_hold_days)
        if hold_days <= 5:
            timing_suitability = 6.8
        elif hold_days <= 15:
            timing_suitability = 7.6
        elif hold_days <= 25:
            timing_suitability = 6.6
        else:
            timing_suitability = 5.2
    if monitor_window_days is not None:
        timing_suitability += 0.5 if int(monitor_window_days) <= 6 else 0.0
    if market_regime == "risk_off" and trend_state in {"weak_breakdown_risk", "downtrend"}:
        timing_suitability -= 1.0
    if price_location_context in {"near_high_but_supported", "mid_range_constructive", "reversal_from_low"}:
        timing_suitability += 0.45
    elif price_location_context in {"extended_near_high", "weak_near_low", "damaged_mid_range"}:
        timing_suitability -= 0.65
    timing_suitability = _clip(timing_suitability)

    contextual_bonus = 0.0
    if setup_scenario in {"strong_continuation_pullback", "supported_high_range_continuation", "range_rebound_candidate", "rebound_repair_candidate"}:
        contextual_bonus += 0.45
    elif setup_scenario in {"extension_needs_reset", "conflicted_setup", "structure_still_damaged"}:
        contextual_bonus -= 0.65
    if news_regime_alignment == "aligned_bullish":
        contextual_bonus += 0.35
    elif news_regime_alignment in {"aligned_bearish", "conflicted"}:
        contextual_bonus -= 0.45
    contextual_bonus += (macro_alignment_score - 5.0) * 0.08

    suitability_score = round(
        (
            trend_suitability * 1.15
            + structure_suitability * 1.0
            + entry_suitability * 1.1
            + reward_risk_suitability * 1.1
            + volatility_suitability * 0.8
            + volume_confirmation_suitability * 0.85
            + relative_strength_suitability * 1.0
            + event_risk_suitability * 0.75
            + timing_suitability * 0.75
            + (5.0 + contextual_bonus) * 0.9
        )
        / 9.4,
        4,
    )
    suitability_label = _label(suitability_score, config)

    key_strengths: list[str] = []
    key_weaknesses: list[str] = []
    suitability_reasons: list[str] = []
    disqualifiers: list[str] = []

    if trend_state == "pullback_in_uptrend":
        key_strengths.append("Constructive pullback in uptrend")
    elif trend_state == "weak_breakdown_risk":
        key_weaknesses.append("Structure still needs repair")
    elif trend_state == "downtrend":
        disqualifiers.append("Trend is in downtrend")

    if relative_strength_suitability >= 6.0:
        key_strengths.append("Relative strength is supportive versus benchmarks")
    elif relative_strength_suitability < 4.8:
        key_weaknesses.append("Relative strength is weak versus benchmarks")

    if structure_suitability >= 6.2:
        key_strengths.append("Support and invalidation zones are clearly tradable")
    elif structure_suitability < 4.6:
        key_weaknesses.append("Trade structure is not very clear")

    if reward_risk_suitability >= 6.0:
        key_strengths.append("Reward/risk is acceptable for a swing setup")
    elif reward_risk_suitability < 4.8:
        key_weaknesses.append("Reward/risk is weak for a swing trade")

    reversal_state = str(volume_context.get("reversal_volume_state") or "unknown")
    selloff_state = str(volume_context.get("selloff_volume_state") or "unknown")
    if reversal_state == "confirmed_bounce":
        key_strengths.append("Volume confirmation is supportive")
    elif reversal_state in {"weak_bounce", "no_confirmation"}:
        key_weaknesses.append("Bounce confirmation is still weak")
    if selloff_state == "heavy_distribution":
        disqualifiers.append("Selling pressure still looks distributive")

    expected_return = getattr(row, "expected_return", None)
    if expected_return is not None and _safe_float(expected_return) <= 0:
        disqualifiers.append("Expected return is not positive")
    prob_tp = getattr(row, "prob_tp", None)
    prob_sl = getattr(row, "prob_sl", None)
    if prob_tp is not None and prob_sl is not None and _safe_float(prob_sl) >= _safe_float(prob_tp):
        disqualifiers.append("Downside probability is not better than upside probability")

    days_to_earnings = earnings.get("days_to_earnings")
    if days_to_earnings is not None and int(days_to_earnings) <= config.earnings_hard_block_days:
        disqualifiers.append("Event risk is too close for a normal swing trade")

    if final_action == "WAIT":
        suitability_reasons.append("The stock is monitorable even though it is not buy-ready yet.")
    elif final_action == "BUY":
        suitability_reasons.append("The stock is both suitable and currently actionable.")
    elif final_action == "AVOID":
        suitability_reasons.append("The stock is currently unattractive for swing positioning.")

    if suitability_label in {"high", "medium"}:
        suitable_for_long_swing = True
        suitable_for_watchlist_only = final_action != "BUY"
        not_suitable_reason = None
    elif suitability_label == "low":
        suitable_for_long_swing = False
        suitable_for_watchlist_only = bool(final_action == "WAIT" or bool(getattr(row, "monitorable_setup", False)))
        not_suitable_reason = None if suitable_for_watchlist_only else "Swing structure is weak enough that it should not be prioritized."
    else:
        suitable_for_long_swing = False
        suitable_for_watchlist_only = False
        not_suitable_reason = disqualifiers[0] if disqualifiers else "The stock is not currently a practical swing-trade candidate."

    if suitability_label == "high":
        suitability_summary = "This stock has high swing-trade suitability. Structure, relative strength, and trade geometry are supportive even if timing may still matter."
    elif suitability_label == "medium":
        suitability_summary = "This stock has medium swing-trade suitability. The setup is tradable or watchlist-worthy, but notable caveats still need respect."
    elif suitability_label == "low":
        suitability_summary = "This stock has low swing-trade suitability. Some elements are usable, but the setup is not especially attractive right now."
    else:
        suitability_summary = "This stock is currently unsuitable for swing trading. Structural and practical drawbacks outweigh the strengths."

    return {
        "suitability_score": suitability_score,
        "suitability_label": suitability_label,
        "suitability_summary": suitability_summary,
        "trend_suitability": round(_clip(trend_suitability), 3),
        "structure_suitability": round(_clip(structure_suitability), 3),
        "entry_suitability": round(_clip(entry_suitability), 3),
        "reward_risk_suitability": round(_clip(reward_risk_suitability), 3),
        "volatility_suitability": round(_clip(volatility_suitability), 3),
        "volume_confirmation_suitability": round(_clip(volume_confirmation_suitability), 3),
        "relative_strength_suitability": round(_clip(relative_strength_suitability), 3),
        "event_risk_suitability": round(_clip(event_risk_suitability), 3),
        "timing_suitability": round(_clip(timing_suitability), 3),
        "key_strengths": key_strengths[:5],
        "key_weaknesses": key_weaknesses[:5],
        "suitability_reasons": suitability_reasons[:4],
        "disqualifiers": disqualifiers[:5],
        "suitable_for_long_swing": bool(suitable_for_long_swing),
        "suitable_for_watchlist_only": bool(suitable_for_watchlist_only),
        "not_suitable_reason": not_suitable_reason,
    }
