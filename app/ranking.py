"""Post-plan ranking helpers for immediate and watchlist swing setups."""

from __future__ import annotations

from .config import PlanningConfig


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_ranking_profile(row, *, config: PlanningConfig) -> dict:
    final_action = str(getattr(row, "final_action", None) or "").upper()
    watchlist_tier = str(getattr(row, "watchlist_tier", None) or "").lower()
    market_regime = str(getattr(row, "market_regime", None) or "neutral")
    trend_state = str(getattr(row, "trend_state", None) or "")

    composite = _safe_float(getattr(row, "composite_score", None))
    entry_quality = _safe_float(getattr(row, "entry_quality_score", None))
    rr = getattr(row, "reward_risk", None) or {}
    rr1 = _safe_float(rr.get("tp1"))
    expected_return = _safe_float(getattr(row, "expected_return", None))
    p_tp = _safe_float(getattr(row, "prob_tp", None))
    p_sl = _safe_float(getattr(row, "prob_sl", None))
    confidence = _safe_float(getattr(row, "confidence", None))
    suitability = getattr(row, "swing_trade_suitability", None) or {}
    suitability_score = _safe_float(suitability.get("suitability_score"))
    pre_scan_score = _safe_float(getattr(row, "pre_scan_score", None))
    sector_relative = _safe_float(getattr(row, "sector_relative_strength", None))

    p_edge = p_tp - p_sl
    immediate_w = config.immediate_rank_weights
    watch_w = config.watchlist_rank_weights

    immediate_rank_score = (
        composite * immediate_w["composite"]
        + entry_quality * immediate_w["entry_quality"]
        + rr1 * immediate_w["reward_risk_tp1"]
        + expected_return * immediate_w["expected_return"]
        + p_edge * immediate_w["probability_edge"]
        + confidence * immediate_w["confidence"]
        + suitability_score * immediate_w["suitability"]
        + pre_scan_score * immediate_w["pre_scan"]
        + sector_relative * immediate_w["sector_relative"]
    )
    if final_action == "BUY":
        immediate_rank_score += 3.0
    elif final_action == "WAIT":
        immediate_rank_score -= 2.0
    else:
        immediate_rank_score -= 5.0
    if market_regime == "risk_off" and trend_state in {"weak_breakdown_risk", "downtrend"}:
        immediate_rank_score -= 1.2

    watchlist_rank_score = (
        composite * watch_w["composite"]
        + entry_quality * watch_w["entry_quality"]
        + rr1 * watch_w["reward_risk_tp1"]
        + expected_return * watch_w["expected_return"]
        + p_edge * watch_w["probability_edge"]
        + confidence * watch_w["confidence"]
        + suitability_score * watch_w["suitability"]
        + pre_scan_score * watch_w["pre_scan"]
        + sector_relative * watch_w["sector_relative"]
    )
    if watchlist_tier == "primary":
        watchlist_rank_score += 2.4
    elif watchlist_tier == "secondary":
        watchlist_rank_score += 0.9
    else:
        watchlist_rank_score -= 3.2
    if final_action == "WAIT":
        watchlist_rank_score += 1.0
    elif final_action == "AVOID":
        watchlist_rank_score -= 2.8

    if final_action == "BUY":
        ranking_bucket = "best_immediate_setups"
        scanner_rank_score = immediate_rank_score
    elif final_action == "WAIT" and watchlist_tier in {"primary", "secondary"}:
        ranking_bucket = "best_watchlist_setups"
        scanner_rank_score = watchlist_rank_score
    else:
        ranking_bucket = "rejected_or_low_priority"
        scanner_rank_score = max(immediate_rank_score, watchlist_rank_score) - 3.5

    return {
        "immediate_rank_score": round(float(immediate_rank_score), 4),
        "watchlist_rank_score": round(float(watchlist_rank_score), 4),
        "scanner_rank_score": round(float(scanner_rank_score), 4),
        "ranking_bucket": ranking_bucket,
    }
