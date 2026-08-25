from __future__ import annotations

from .config import PlanningConfig
from .setup_archetypes import score_setup_families


def _clip_score(value: float) -> float:
    return max(0.0, min(10.0, value))


def score_price_location(
    *,
    current_price: float,
    frame,
    structure_state: str,
    support_zone_1: dict | None,
    resistance_zone_1: dict | None,
    atr: float,
    config: PlanningConfig,
) -> dict:
    """Score whether the current price offers practical swing entry geometry."""

    price = max(float(current_price), 1e-9)
    atr_value = max(float(atr or 0.0), price * 0.005)
    ema20 = None
    if frame is not None and not frame.empty and "ema20" in frame.columns:
        value = frame["ema20"].iloc[-1]
        ema20 = None if value != value else float(value)
    distance_ema20 = 0.0 if ema20 is None else (price - ema20) / max(ema20, 1e-9)

    recent = frame.tail(min(60, len(frame))) if frame is not None and not frame.empty else None
    range_position = 0.5
    consecutive_green = 0
    if recent is not None and not recent.empty:
        low = float(recent["low"].min())
        high = float(recent["high"].max())
        range_position = 0.5 if high <= low else max(0.0, min(1.0, (price - low) / (high - low)))
        for is_green in reversed((recent["close"] > recent["open"]).tolist()):
            if not is_green:
                break
            consecutive_green += 1

    def _distance_atr(zone: dict | None, side: str) -> float | None:
        if not zone:
            return None
        lower = float(zone.get("lower", price))
        upper = float(zone.get("upper", price))
        if lower <= price <= upper:
            return 0.0
        distance = price - upper if side == "support" else lower - price
        return max(0.0, distance) / atr_value

    support_distance = _distance_atr(support_zone_1, "support")
    resistance_distance = _distance_atr(resistance_zone_1, "resistance")
    score = 5.0
    reasons: list[str] = []

    if support_distance is not None and support_distance <= config.price_location_near_support_atr:
        score += 2.0
        reasons.append("near_ranked_support")
    if -0.025 <= distance_ema20 <= 0.035:
        score += 1.6
        reasons.append("near_ema20")
    elif distance_ema20 >= config.structure_extended_from_ema20_pct:
        score -= 3.0
        reasons.append("extended_above_ema20")
    if resistance_distance is not None and resistance_distance <= config.price_location_near_resistance_atr:
        if structure_state != "breakout":
            score -= 1.7
            reasons.append("directly_below_resistance")
    if consecutive_green >= 5:
        score -= 1.8
        reasons.append("five_or_more_green_sessions")
    elif consecutive_green >= 3:
        score -= 0.7
        reasons.append("multiple_green_sessions")
    if structure_state == "healthy_pullback" and 0.25 <= range_position <= 0.8:
        score += 1.0
        reasons.append("constructive_pullback_location")
    elif structure_state == "breakout" and distance_ema20 <= config.structure_extended_from_ema20_pct:
        score += 0.7
        reasons.append("supported_breakout_location")
    elif structure_state in {"trend_damage", "structural_breakdown"}:
        score = min(score, 3.8 if structure_state == "trend_damage" else 2.5)
        reasons.append("damaged_structure_limits_location_quality")
    elif structure_state == "extended":
        score = min(score, 2.5)
        reasons.append("extension_requires_reset")

    score = _clip_score(score)
    if structure_state == "extended" or distance_ema20 >= config.structure_parabolic_from_ema20_pct:
        category = "extended"
    elif score >= 8.5:
        category = "excellent"
    elif score >= 7.0:
        category = "good"
    elif score >= 5.0:
        category = "neutral"
    else:
        category = "poor"
    return {
        "price_location_score": round(score, 3),
        "price_location_category": category,
        "price_location_reasons": reasons,
        "local_range_position": round(range_position, 4),
        "consecutive_green_sessions": consecutive_green,
    }


def score_setup(
    *,
    trend_state: str,
    support_zone_1: dict | None,
    atr_pct: float | None,
    volume_context: dict,
    relative_strength: dict,
    earnings: dict,
    reward_risk: dict,
    entry_quality_score: float,
    history_stats: dict | None,
    llm_quality_score: float,
    context: dict | None,
    sector_relative_strength: float | None,
    config: PlanningConfig,
    structure_state: str | None = None,
    liquidity_score: float = 5.0,
    volatility_suitability_score: float | None = None,
    price_location_score: float = 5.0,
    target_realism_score: float = 5.0,
    confirmation_score: float = 5.0,
    setup_family: str | None = None,
    pre_scan_profile: dict | None = None,
) -> dict:
    rich_state = str(structure_state or "")
    rich_trend_scores = {
        "healthy_pullback": 8.7,
        "breakout": 8.2,
        "base_building": 6.2,
        "deep_pullback": 5.6,
        "reversal_attempt": 4.8,
        "range": 5.0,
        "extended": 4.3,
        "trend_damage": 2.8,
        "structural_breakdown": 1.2,
    }
    trend_quality = rich_trend_scores.get(
        rich_state,
        8.2 if trend_state == "uptrend" else 6.8 if trend_state == "pullback_in_uptrend" else 4.8 if trend_state == "range" else 2.4,
    )
    pullback_quality = 7.8 if trend_state == "pullback_in_uptrend" else 6.8 if trend_state == "uptrend" else 4.5 if trend_state == "range" else 2.5
    support_quality = 4.5
    if support_zone_1:
        ranked_strength = support_zone_1.get("strength_score")
        if ranked_strength is not None:
            support_quality = max(support_quality, float(ranked_strength))
        support_quality += min(2.0, len(support_zone_1.get("source_tags", [])) * 0.55)

    volatility_quality = float(volatility_suitability_score) if volatility_suitability_score is not None else 6.0
    if volatility_suitability_score is None and atr_pct is not None:
        atr_pct_val = float(atr_pct) * 100.0
        if atr_pct_val < 1.0:
            volatility_quality = 5.3
        elif atr_pct_val <= 4.5:
            volatility_quality = 8.0
        elif atr_pct_val <= 6.5:
            volatility_quality = 6.8
        else:
            volatility_quality = 4.3

    rel_strength_score = 5.0
    rs_spy = relative_strength.get("vs_spy")
    rs_qqq = relative_strength.get("vs_qqq")
    if rs_spy is not None and rs_qqq is not None:
        avg_rs = (float(rs_spy) + float(rs_qqq)) / 2.0
        rel_strength_score = _clip_score(5.0 + avg_rs * 55.0)

    volume_score = 5.0
    if volume_context.get("selloff_volume_state") == "light_pullback":
        volume_score += 2.0
    elif volume_context.get("selloff_volume_state") == "heavy_distribution":
        volume_score -= 2.4
    if volume_context.get("reversal_volume_state") == "confirmed_bounce":
        volume_score += 1.6
    elif volume_context.get("reversal_volume_state") == "weak_bounce":
        volume_score -= 1.0

    earnings_score = 7.0
    days_to_earnings = earnings.get("days_to_earnings")
    if days_to_earnings is not None:
        days = int(days_to_earnings)
        if days <= config.earnings_hard_block_days:
            earnings_score = 1.5
        elif days <= config.earnings_penalty_near_days:
            earnings_score = 3.0
        elif days <= config.earnings_penalty_mid_days:
            earnings_score = 5.0
    if earnings.get("earnings_risk_flag"):
        earnings_score -= 1.0

    reward_risk_score = _clip_score(float(reward_risk.get("tp1", 0.0)) * 3.2)
    hist_score = 5.0
    if history_stats:
        samples = int(history_stats.get("samples", 0))
        if samples > 0:
            win = float(history_stats.get("win_rate", 0.5))
            avg_ret = float(history_stats.get("avg_return", 0.0))
            hist_score = _clip_score(5.0 + ((win - 0.5) * 8.0) + (avg_ret * 100.0 * 0.4))

    context = context or {}
    price_location_context = str(context.get("price_location_context") or "")
    setup_scenario = str(context.get("setup_scenario") or "")
    chart_news_alignment = str(context.get("chart_news_alignment") or "news_neutral")
    macro_alignment_score = float(context.get("macro_alignment_score") or 5.0)
    catalyst_strength_score = float(context.get("catalyst_strength_score") or 0.0)
    scenario_confidence = float(context.get("scenario_confidence") or 0.5)

    context_score = 5.2
    if price_location_context in {"near_high_but_supported", "mid_range_constructive", "reversal_from_low"}:
        context_score += 1.4
    elif price_location_context in {"extended_near_high", "weak_near_low", "damaged_mid_range"}:
        context_score -= 1.4
    if setup_scenario in {"strong_continuation_pullback", "supported_high_range_continuation", "range_rebound_candidate", "rebound_repair_candidate"}:
        context_score += 0.8
    elif setup_scenario in {"extension_needs_reset", "conflicted_setup", "structure_still_damaged"}:
        context_score -= 0.9

    catalyst_score = _clip_score(4.8 + (catalyst_strength_score - 5.0) * 0.7)
    if chart_news_alignment in {"news_supports_continuation", "news_supports_rebound", "aligned_bullish"}:
        catalyst_score += 1.0
    elif chart_news_alignment in {"news_conflicts_with_chart", "aligned_bearish"}:
        catalyst_score -= 1.25

    macro_score = _clip_score(macro_alignment_score)
    if sector_relative_strength is not None:
        macro_score = _clip_score(macro_score + max(-1.0, min(1.0, float(sector_relative_strength) * 16.0)))

    scenario_score = _clip_score((scenario_confidence * 10.0) + (0.8 if setup_scenario.startswith("strong_") else 0.0))

    scores = {
        "trend_quality_score": _clip_score(trend_quality),
        "pullback_quality_score": _clip_score(pullback_quality),
        "support_quality_score": _clip_score(support_quality),
        "volatility_quality_score": _clip_score(volatility_quality),
        "relative_strength_score": _clip_score(rel_strength_score),
        "volume_confirmation_score": _clip_score(volume_score),
        "earnings_risk_score": _clip_score(earnings_score),
        "reward_risk_score": _clip_score(reward_risk_score),
        "historical_analogue_score": _clip_score(hist_score),
        "entry_quality_score": _clip_score(entry_quality_score),
        "llm_quality_score": _clip_score(llm_quality_score),
        "context_score": _clip_score(context_score),
        "catalyst_score": _clip_score(catalyst_score),
        "macro_score": _clip_score(macro_score),
        "scenario_score": _clip_score(scenario_score),
        "liquidity_score": _clip_score(liquidity_score),
        "price_location_score": _clip_score(price_location_score),
        "target_realism_score": _clip_score(target_realism_score),
        "confirmation_score": _clip_score(confirmation_score),
    }

    pre_scan_profile = pre_scan_profile or {}
    prescan_components = pre_scan_profile.get("setup_lane_components") or {}
    family_components = {
        "trend_strength": scores["trend_quality_score"],
        "pullback_quality": scores["pullback_quality_score"],
        "deep_pullback_quality": float(prescan_components.get("deep_pullback_quality", scores["pullback_quality_score"])),
        "price_location": scores["price_location_score"],
        "relative_strength": scores["relative_strength_score"],
        "pullback_volume": float(prescan_components.get("pullback_volume", scores["volume_confirmation_score"])),
        "support_confluence": scores["support_quality_score"],
        "continuation_structure": float(
            prescan_components.get(
                "continuation_structure",
                (scores["trend_quality_score"] + scores["relative_strength_score"]) / 2.0,
            )
        ),
        "confirmation": scores["confirmation_score"],
        "target_quality": scores["target_realism_score"],
        "base_quality": float(prescan_components.get("base_quality", scores["context_score"])),
        "breakout_retest_quality": float(prescan_components.get("breakout_retest_quality", scores["support_quality_score"])),
        "reversal_quality": float(prescan_components.get("reversal_quality", scores["confirmation_score"])),
        "volatility": scores["volatility_quality_score"],
        "liquidity": scores["liquidity_score"],
        "earnings": scores["earnings_risk_score"],
    }
    family_profile = score_setup_families(
        family_components,
        weights_by_family=config.setup_family_score_weights,
    )
    selected_family = setup_family or family_profile["setup_family"]
    setup_family_score = float((family_profile["setup_lane_scores"] or {}).get(selected_family, 0.0))

    total_weight = sum(config.score_weights.values())
    composite = 0.0
    for key, weight in config.score_weights.items():
        metric_key = f"{key}_score" if not key.endswith("_score") else key
        if metric_key not in scores:
            continue
        composite += scores[metric_key] * weight
    generic_composite = composite / max(total_weight, 1e-9)
    blend = max(0.0, min(float(config.setup_family_raw_score_blend), 0.6))
    composite_score = generic_composite * (1.0 - blend) + setup_family_score * blend
    scores["composite_score"] = round(_clip_score(composite_score), 4)
    scores["trend_score"] = scores["trend_quality_score"]
    scores["volatility_suitability_score"] = scores["volatility_quality_score"]
    scores["support_confluence_score"] = scores["support_quality_score"]
    scores["trend_strength_score"] = family_components["trend_strength"]
    scores["pullback_volume_quality"] = family_components["pullback_volume"]
    scores["continuation_structure_score"] = family_components["continuation_structure"]
    scores["target_quality_score"] = family_components["target_quality"]
    scores["setup_family_score"] = round(setup_family_score, 4)
    scores["setup_family_components"] = {key: round(float(value), 4) for key, value in family_components.items()}
    scores["setup_family_scores"] = family_profile["setup_lane_scores"]
    scores["setup_family_weights"] = dict(config.setup_family_score_weights.get(selected_family) or {})
    return scores
