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
