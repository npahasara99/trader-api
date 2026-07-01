from __future__ import annotations

from .config import PlanningConfig


def _clip_score(value: float) -> float:
    return max(0.0, min(10.0, value))


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
) -> dict:
    trend_quality = 8.2 if trend_state == "uptrend" else 6.8 if trend_state == "pullback_in_uptrend" else 4.8 if trend_state == "range" else 2.4
    pullback_quality = 7.8 if trend_state == "pullback_in_uptrend" else 6.8 if trend_state == "uptrend" else 4.5 if trend_state == "range" else 2.5
    support_quality = 4.5
    if support_zone_1:
        support_quality += min(3.5, len(support_zone_1.get("source_tags", [])) * 1.0)

    volatility_quality = 6.0
    if atr_pct is not None:
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
    }

    total_weight = sum(config.score_weights.values())
    composite = 0.0
    for key, weight in config.score_weights.items():
        metric_key = f"{key}_score" if not key.endswith("_score") else key
        if metric_key not in scores:
            continue
        composite += scores[metric_key] * weight
    composite_score = (composite / max(total_weight, 1e-9))
    scores["composite_score"] = round(_clip_score(composite_score), 4)
    return scores
