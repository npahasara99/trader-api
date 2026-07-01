from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd

from .config import PlanningConfig


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def _pct_distance(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    base = max(abs(float(b)), 1e-9)
    return round(((float(a) - float(b)) / base) * 100.0, 4)


def _range_position(values: pd.Series, current_price: float) -> tuple[float | None, float | None, float | None]:
    if values.empty:
        return None, None, None
    low = _safe_float(values.min())
    high = _safe_float(values.max())
    if low is None or high is None:
        return None, None, None
    if high <= low:
        return 0.5, 0.0, 0.0
    position = (current_price - low) / max(high - low, 1e-9)
    return (
        round(_clip(position, 0.0, 1.0), 4),
        round(((high - current_price) / max(current_price, 1e-9)) * 100.0, 4),
        round(((current_price - low) / max(current_price, 1e-9)) * 100.0, 4),
    )


def _latest_text_blob(news_items: list[dict]) -> str:
    bits: list[str] = []
    for item in news_items:
        bits.append(str(item.get("headline") or ""))
        bits.append(str(item.get("summary") or ""))
    return " ".join(bits).lower()


def _news_recency_score(news_items: list[dict]) -> float:
    if not news_items:
        return 0.0
    now = datetime.now(timezone.utc)
    best = 0.0
    for item in news_items:
        dt_raw = item.get("datetime")
        if not dt_raw:
            continue
        try:
            parsed = datetime.fromisoformat(str(dt_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        age_hours = max(0.0, (now - parsed).total_seconds() / 3600.0)
        if age_hours <= 24:
            best = max(best, 9.5)
        elif age_hours <= 72:
            best = max(best, 7.2)
        elif age_hours <= 168:
            best = max(best, 5.4)
        else:
            best = max(best, 3.6)
    return round(best or 3.2, 3)


def _classify_catalysts(*, news_items: list[dict], news_score: int, earnings: dict) -> dict:
    text = _latest_text_blob(news_items)
    signal_map = {
        "earnings_beat": ["beat", "tops estimates", "above estimates", "strong quarter"],
        "earnings_miss": ["miss", "below estimates", "weak quarter", "soft quarter"],
        "guidance_raise": ["raises guidance", "raised guidance", "guidance increase", "boosted outlook"],
        "guidance_cut": ["cuts guidance", "cut guidance", "lowered outlook", "trimmed outlook"],
        "analyst_upgrade_cluster": ["upgrade", "upgraded", "price target raised", "raised target"],
        "analyst_downgrade_cluster": ["downgrade", "downgraded", "price target cut", "cut target"],
        "product_catalyst": ["launch", "partnership", "contract", "deal", "ai", "new product"],
        "regulatory_tailwind": ["approval", "cleared", "wins approval"],
        "regulatory_headwind": ["probe", "lawsuit", "investigation", "ban", "recall"],
        "sector_tailwind": ["sector rally", "industry demand", "tailwind", "strong demand"],
        "sector_headwind": ["sector slowdown", "weak demand", "headwind", "slowdown"],
        "macro_linked_catalyst": ["rates", "inflation", "yield", "oil", "fed", "tariff"],
    }

    active_signals: list[str] = []
    raw_strength = 0.0
    for signal, keywords in signal_map.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > 0:
            active_signals.append(signal)
            raw_strength += min(2.0, 0.9 + hits * 0.35)

    days_to_earnings = earnings.get("days_to_earnings")
    if days_to_earnings is not None and int(days_to_earnings) <= 14:
        active_signals.append("macro_linked_catalyst")
        raw_strength += 0.5

    recency_score = _news_recency_score(news_items)
    catalyst_strength = _clip(abs(float(news_score)) * 0.6 + raw_strength + (recency_score * 0.2), 0.0, 10.0)

    if news_score >= 3:
        directional_bias = "bullish"
    elif news_score <= -3:
        directional_bias = "bearish"
    elif active_signals:
        directional_bias = "mixed"
    else:
        directional_bias = "neutral"

    if not active_signals:
        active_signals = ["no_meaningful_catalyst"]

    return {
        "catalyst_signals": active_signals[:6],
        "news_directional_bias": directional_bias,
        "catalyst_strength_score": round(catalyst_strength, 3),
        "catalyst_recency_score": recency_score,
    }


def _macro_sensitivity_tag(*, sector: str | None, industry: str | None) -> str:
    sector_value = (sector or "").strip().lower()
    industry_value = (industry or "").strip().lower()
    if "semiconductor" in industry_value:
        return "semis_high_beta"
    if sector_value == "technology":
        return "rates_sensitive_growth"
    if sector_value == "financials":
        return "financials_rates_linked"
    if sector_value == "health care":
        return "healthcare_defensive"
    if sector_value == "energy":
        return "commodity_linked"
    if sector_value == "utilities":
        return "defensive"
    if sector_value == "consumer staples":
        return "defensive"
    if sector_value == "communication services" and "telecom" in industry_value:
        return "telecom_defensive_yield"
    if sector_value == "industrials":
        return "industrials_macro_sensitive"
    if sector_value == "materials":
        return "cyclical"
    return "macro_neutral"


def _macro_alignment(
    *,
    market_regime: str,
    macro_sensitivity_tag: str,
    sector_relative_strength: float | None,
) -> tuple[float, str, str]:
    score = 5.0
    normalized_regime = (market_regime or "neutral").strip().lower()
    if normalized_regime == "risk_on":
        if macro_sensitivity_tag in {"semis_high_beta", "rates_sensitive_growth", "cyclical", "industrials_macro_sensitive", "commodity_linked"}:
            score += 1.4
        elif macro_sensitivity_tag in {"defensive", "healthcare_defensive", "telecom_defensive_yield"}:
            score -= 0.4
    elif normalized_regime == "risk_off":
        if macro_sensitivity_tag in {"defensive", "healthcare_defensive", "telecom_defensive_yield"}:
            score += 1.0
        elif macro_sensitivity_tag in {"semis_high_beta", "rates_sensitive_growth", "cyclical"}:
            score -= 1.2

    if sector_relative_strength is not None:
        score += _clip(float(sector_relative_strength) * 18.0, -1.4, 1.4)

    score = _clip(score, 0.0, 10.0)
    if score >= 6.2:
        label = "macro_supports_extension"
        sector_regime = "sector_tailwind_active"
    elif score <= 4.0:
        label = "macro_conflicts_with_setup"
        sector_regime = "sector_headwind_active"
    else:
        label = "macro_neutral"
        sector_regime = "sector_neutral"
    return round(score, 3), label, sector_regime


def build_market_context(
    *,
    ticker: str,
    current_price: float,
    frame: pd.DataFrame,
    trend_state: str,
    moving_averages: dict[str, float | None],
    atr: float,
    volume_context: dict,
    relative_strength: dict,
    market_regime: str,
    news_items: list[dict] | None,
    news_score: int,
    earnings: dict,
    ticker_meta: dict | None,
    sector_relative_strength: float | None,
    config: PlanningConfig,
) -> dict:
    news_items = list(news_items or [])
    current_price = float(current_price)
    close_series = frame["close"].dropna() if "close" in frame.columns else pd.Series(dtype=float)
    local_window = close_series.tail(config.context_local_range_window)
    one_month = close_series.tail(config.context_range_window_1m)
    three_month = close_series.tail(config.context_range_window_3m)
    twelve_month = close_series.tail(config.context_range_window_12m)
    expansion_window = close_series.tail(config.context_expansion_window)

    range_position_1m, distance_to_1m_high_pct, distance_to_1m_low_pct = _range_position(one_month, current_price)
    range_position_3m, distance_to_3m_high_pct, distance_to_3m_low_pct = _range_position(three_month, current_price)
    range_position_12m, distance_to_12m_high_pct, distance_to_12m_low_pct = _range_position(twelve_month, current_price)
    local_range_position, _, _ = _range_position(local_window, current_price)

    ema20 = _safe_float(moving_averages.get("ema20"))
    sma50 = _safe_float(moving_averages.get("sma50"))
    sma100 = _safe_float(moving_averages.get("sma100"))
    sma200 = _safe_float(moving_averages.get("sma200"))

    distance_from_ema20_pct = _pct_distance(current_price, ema20)
    distance_from_sma50_pct = _pct_distance(current_price, sma50)
    distance_from_sma100_pct = _pct_distance(current_price, sma100)
    distance_from_sma200_pct = _pct_distance(current_price, sma200)

    atr = max(float(atr or 0.0), max(current_price * 0.01, 0.01))
    expansion_range = (_safe_float(expansion_window.max(), current_price) or current_price) - (_safe_float(expansion_window.min(), current_price) or current_price)
    expansion_atr = expansion_range / max(atr, 1e-9)
    if expansion_atr >= config.context_expansion_range_atr:
        recent_expansion_state = "expanded"
    elif expansion_atr <= config.context_compression_range_atr:
        recent_expansion_state = "compressed"
    else:
        recent_expansion_state = "balanced"
    recent_compression_state = "tight" if expansion_atr <= config.context_compression_range_atr else "normal"

    extension_from_ema20 = abs(distance_from_ema20_pct or 0.0) / 100.0
    if (
        local_range_position is not None
        and local_range_position >= config.context_near_high_position
        and extension_from_ema20 >= config.context_extended_from_ema20_pct
    ):
        breakout_extension_state = "post_breakout_extension"
    elif local_range_position is not None and local_range_position >= config.context_near_high_position:
        breakout_extension_state = "supported_near_high"
    elif local_range_position is not None and local_range_position <= config.context_near_low_position:
        breakout_extension_state = "testing_lower_range"
    else:
        breakout_extension_state = "inside_range"

    historical_range_context = "mid_range_constructive"
    price_location_context = "mid_range_constructive"
    if range_position_12m is not None and range_position_12m <= config.context_near_low_position:
        if trend_state == "weak_breakdown_risk":
            historical_range_context = "deep_repair_zone"
        else:
            historical_range_context = "deep_in_lower_range"
    elif range_position_12m is not None and range_position_12m >= config.context_near_high_position:
        historical_range_context = "upper_historical_range"

    if local_range_position is not None and local_range_position >= config.context_near_high_position:
        if trend_state in {"uptrend", "pullback_in_uptrend"} and extension_from_ema20 < config.context_extended_from_ema20_pct:
            price_location_context = "near_high_but_supported"
        else:
            price_location_context = "extended_near_high"
    elif local_range_position is not None and local_range_position <= config.context_near_low_position:
        if trend_state == "weak_breakdown_risk":
            price_location_context = "weak_near_low"
        elif volume_context.get("reversal_volume_state") == "confirmed_bounce":
            price_location_context = "reversal_from_low"
        else:
            price_location_context = "deep_in_lower_range"
    elif trend_state == "weak_breakdown_risk":
        price_location_context = "damaged_mid_range"
    else:
        price_location_context = "mid_range_constructive"

    catalyst = _classify_catalysts(news_items=news_items, news_score=news_score, earnings=earnings)
    rs_spy = _safe_float(relative_strength.get("vs_spy"), 0.0) or 0.0
    rs_qqq = _safe_float(relative_strength.get("vs_qqq"), 0.0) or 0.0
    avg_rs = (rs_spy + rs_qqq) / 2.0

    sector = (ticker_meta or {}).get("sector")
    industry = (ticker_meta or {}).get("industry")
    macro_tag = _macro_sensitivity_tag(sector=sector, industry=industry)
    macro_alignment_score, macro_context_label, sector_regime = _macro_alignment(
        market_regime=market_regime,
        macro_sensitivity_tag=macro_tag,
        sector_relative_strength=sector_relative_strength,
    )

    news_directional_bias = str(catalyst["news_directional_bias"])
    chart_news_alignment = "news_neutral"
    news_supports_continuation = False
    news_supports_rebound = False
    news_conflicts_with_chart = False
    if news_directional_bias == "bullish":
        if trend_state in {"uptrend", "pullback_in_uptrend"} and avg_rs >= 0:
            chart_news_alignment = "news_supports_continuation"
            news_supports_continuation = True
        elif price_location_context in {"weak_near_low", "deep_in_lower_range", "deep_repair_zone", "reversal_from_low"}:
            chart_news_alignment = "news_supports_rebound"
            news_supports_rebound = True
        else:
            chart_news_alignment = "aligned_bullish"
    elif news_directional_bias == "bearish":
        if trend_state in {"uptrend", "pullback_in_uptrend", "range"}:
            chart_news_alignment = "news_conflicts_with_chart"
            news_conflicts_with_chart = True
        else:
            chart_news_alignment = "aligned_bearish"
    else:
        chart_news_alignment = "news_neutral"

    setup_type = "constructive_pullback"
    setup_scenario = "balanced_setup"
    continuation_vs_reversion_bias = "balanced"
    tp_aggressiveness = "moderate"
    sl_tolerance = "moderate"
    expected_move_profile = "first_resistance_test"

    if trend_state == "uptrend" and price_location_context == "near_high_but_supported":
        if catalyst["catalyst_strength_score"] >= config.context_strong_catalyst_score or macro_alignment_score >= config.context_macro_alignment_supportive:
            setup_type = "momentum_expansion_candidate"
            setup_scenario = "supported_high_range_continuation"
            continuation_vs_reversion_bias = "continuation_favored"
            tp_aggressiveness = "moderate_to_high"
            sl_tolerance = "tight_to_moderate"
            expected_move_profile = "breakout_can_extend_if_confirmed"
        else:
            setup_type = "continuation_breakout"
            setup_scenario = "controlled_high_range_continuation"
            continuation_vs_reversion_bias = "continuation_favored"
            tp_aggressiveness = "moderate"
            sl_tolerance = "tight_to_moderate"
            expected_move_profile = "continuation_can_grind_higher"
    elif trend_state == "pullback_in_uptrend":
        setup_type = "constructive_pullback"
        setup_scenario = "strong_continuation_pullback" if avg_rs >= 0 and price_location_context != "extended_near_high" else "pullback_needs_confirmation"
        continuation_vs_reversion_bias = "continuation_favored"
        tp_aggressiveness = "moderate_to_high" if macro_alignment_score >= config.context_macro_alignment_supportive else "moderate"
        sl_tolerance = "tight_to_moderate"
        expected_move_profile = "continuation_can_resume_from_support"
    elif trend_state == "range" and price_location_context in {"reversal_from_low", "deep_in_lower_range"} and news_directional_bias == "bullish":
        setup_type = "range_rebound"
        setup_scenario = "range_rebound_candidate"
        continuation_vs_reversion_bias = "rebound_candidate"
        tp_aggressiveness = "moderate"
        sl_tolerance = "moderate"
        expected_move_profile = "bounce_to_first_resistance"
    elif trend_state == "weak_breakdown_risk":
        if price_location_context in {"weak_near_low", "deep_repair_zone"} and (news_supports_rebound or macro_alignment_score >= config.context_macro_alignment_supportive):
            setup_type = "deep_rebound_attempt"
            setup_scenario = "rebound_repair_candidate"
            continuation_vs_reversion_bias = "rebound_candidate"
            tp_aggressiveness = "conservative"
            sl_tolerance = "moderate_to_wide"
            expected_move_profile = "repair_bounce_not_full_recovery"
        else:
            setup_type = "repair_after_breakdown"
            setup_scenario = "structure_still_damaged"
            continuation_vs_reversion_bias = "mean_reversion_favored"
            tp_aggressiveness = "conservative"
            sl_tolerance = "moderate_to_wide"
            expected_move_profile = "repair_needs_acceptance_first"
    elif price_location_context == "extended_near_high":
        setup_type = "overextended_breakout"
        setup_scenario = "extension_needs_reset"
        continuation_vs_reversion_bias = "mean_reversion_favored"
        tp_aggressiveness = "conservative"
        sl_tolerance = "tight"
        expected_move_profile = "extension_needs_reset"
    elif avg_rs < 0 and news_directional_bias == "bearish":
        setup_type = "weak_rally"
        setup_scenario = "conflicted_setup"
        continuation_vs_reversion_bias = "mean_reversion_favored"
        tp_aggressiveness = "conservative"
        sl_tolerance = "tight"
        expected_move_profile = "limited_rebound_only"

    if news_supports_continuation and setup_type in {"momentum_expansion_candidate", "continuation_breakout", "constructive_pullback"}:
        news_regime_alignment = "aligned_bullish"
    elif news_supports_rebound and setup_type in {"deep_rebound_attempt", "range_rebound"}:
        news_regime_alignment = "aligned_bullish"
    elif news_conflicts_with_chart or macro_context_label == "macro_conflicts_with_setup":
        news_regime_alignment = "conflicted"
    elif news_directional_bias == "bearish":
        news_regime_alignment = "aligned_bearish"
    else:
        news_regime_alignment = "neutral"

    confidence_components = [
        0.45,
        min(0.14, catalyst["catalyst_strength_score"] / 100.0 * 1.5),
        min(0.12, max(macro_alignment_score - 5.0, 0.0) * 0.025),
        0.08 if continuation_vs_reversion_bias != "balanced" else 0.0,
        0.05 if price_location_context not in {"extended_near_high", "weak_near_low"} else -0.04,
    ]
    scenario_confidence = round(_clip(sum(confidence_components), 0.25, 0.88), 3)

    location_context_summary = (
        f"{ticker} is trading in a {price_location_context} state with local range position "
        f"{'unknown' if local_range_position is None else f'{local_range_position:.2f}'} and 12m range position "
        f"{'unknown' if range_position_12m is None else f'{range_position_12m:.2f}'}."
    )
    setup_context_summary = (
        f"{setup_scenario.replace('_', ' ')} with {continuation_vs_reversion_bias.replace('_', ' ')} bias; "
        f"news/regime alignment is {news_regime_alignment.replace('_', ' ')} and macro context is {macro_context_label.replace('_', ' ')}."
    )
    scenario_rationale = (
        f"Trend={trend_state}; location={price_location_context}; catalyst_bias={news_directional_bias}; "
        f"macro={macro_context_label}; setup={setup_scenario}."
    )

    return {
        "range_position_1m": range_position_1m,
        "range_position_3m": range_position_3m,
        "range_position_12m": range_position_12m,
        "distance_to_1m_high_pct": distance_to_1m_high_pct,
        "distance_to_1m_low_pct": distance_to_1m_low_pct,
        "distance_to_3m_high_pct": distance_to_3m_high_pct,
        "distance_to_3m_low_pct": distance_to_3m_low_pct,
        "distance_to_12m_high_pct": distance_to_12m_high_pct,
        "distance_to_12m_low_pct": distance_to_12m_low_pct,
        "distance_from_ema20_pct": distance_from_ema20_pct,
        "distance_from_sma50_pct": distance_from_sma50_pct,
        "distance_from_sma100_pct": distance_from_sma100_pct,
        "distance_from_sma200_pct": distance_from_sma200_pct,
        "recent_expansion_state": recent_expansion_state,
        "recent_compression_state": recent_compression_state,
        "breakout_extension_state": breakout_extension_state,
        "local_range_position": local_range_position,
        "historical_range_context": historical_range_context,
        "price_location_context": price_location_context,
        "setup_type": setup_type,
        "catalyst_signals": catalyst["catalyst_signals"],
        "news_directional_bias": news_directional_bias,
        "catalyst_strength_score": catalyst["catalyst_strength_score"],
        "catalyst_recency_score": catalyst["catalyst_recency_score"],
        "chart_news_alignment": chart_news_alignment,
        "news_supports_continuation": news_supports_continuation,
        "news_supports_rebound": news_supports_rebound,
        "news_conflicts_with_chart": news_conflicts_with_chart,
        "news_neutral": chart_news_alignment == "news_neutral",
        "sector_regime": sector_regime,
        "macro_sensitivity_tag": macro_tag,
        "macro_alignment_score": macro_alignment_score,
        "macro_context_label": macro_context_label,
        "setup_scenario": setup_scenario,
        "continuation_vs_reversion_bias": continuation_vs_reversion_bias,
        "news_regime_alignment": news_regime_alignment,
        "tp_aggressiveness": tp_aggressiveness,
        "sl_tolerance": sl_tolerance,
        "expected_move_profile": expected_move_profile,
        "scenario_confidence": scenario_confidence,
        "scenario_rationale": scenario_rationale,
        "setup_context_summary": setup_context_summary,
        "location_context_summary": location_context_summary,
        "sector": sector,
        "industry": industry,
    }
