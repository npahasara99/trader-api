from __future__ import annotations

from typing import Callable

import pandas as pd

from .config import DEFAULT_PLANNING_CONFIG, PlanningConfig
from .chart_context import build_chart_context
from .confirmation import build_confirmation_plan
from .context_scenarios import build_market_context
from .entry_engine import build_entry_candidates, choose_preferred_entry
from .indicators import add_indicator_columns, bars_to_frame, latest_value
from .llm_reasoning import review_setup
from .risk_engine import build_stop_loss, build_take_profits, estimate_hold_window
from .scenario_engine import generate_execution_scenarios
from .scanner import build_universe_suitability, classify_volatility
from .scoring import score_price_location, score_setup
from .structure import summarize_structure
from .zones import build_support_resistance_zones, fibonacci_levels


DailyBarsLoader = Callable[[str], list[dict]]


def _score_relative_strength(stock_frame: pd.DataFrame, benchmark_frame: pd.DataFrame) -> float | None:
    if stock_frame.empty or benchmark_frame.empty or len(stock_frame) < 40 or len(benchmark_frame) < 40:
        return None
    stock_ret = float(stock_frame["close"].iloc[-1] / stock_frame["close"].iloc[-40] - 1.0)
    bench_ret = float(benchmark_frame["close"].iloc[-1] / benchmark_frame["close"].iloc[-40] - 1.0)
    return round(stock_ret - bench_ret, 6)


def _volume_context(frame: pd.DataFrame) -> dict:
    if frame.empty or len(frame) < 25:
        return {"selloff_volume_state": "unknown", "reversal_volume_state": "unknown"}

    recent = frame.tail(8).copy()
    avg_vol = float(frame["volume"].tail(20).replace(0, pd.NA).dropna().mean() or 0.0)
    avg_vol = max(avg_vol, 1.0)
    down_days = recent[recent["close"] < recent["close"].shift(1)]
    up_days = recent[recent["close"] > recent["close"].shift(1)]

    down_ratio = float((down_days["volume"].mean() if not down_days.empty else 0.0) / avg_vol)
    up_ratio = float((up_days["volume"].mean() if not up_days.empty else 0.0) / avg_vol)

    selloff_state = "normal_pullback"
    if down_ratio >= 1.25:
        selloff_state = "heavy_distribution"
    elif 0.0 < down_ratio <= 0.9:
        selloff_state = "light_pullback"

    reversal_state = "no_confirmation"
    if up_ratio >= 1.15 and float(recent["close"].iloc[-1]) > float(recent["close"].iloc[0]):
        reversal_state = "confirmed_bounce"
    elif up_ratio > 0.0 and up_ratio < 0.95:
        reversal_state = "weak_bounce"

    return {
        "selloff_volume_state": selloff_state,
        "reversal_volume_state": reversal_state,
        "down_volume_ratio": round(down_ratio, 3),
        "up_volume_ratio": round(up_ratio, 3),
        "abnormal_participation": bool(max(down_ratio, up_ratio) >= 1.5),
    }


def _earnings_payload(earnings_context: dict | None) -> dict:
    ctx = earnings_context or {}
    days = ctx.get("days_to_earnings")
    return {
        "days_to_earnings": days,
        "earnings_risk_flag": bool(days is not None and int(days) <= 10),
        "avg_post_earnings_move_pct": ctx.get("avg_post_earnings_move_pct"),
        "post_earnings_up_rate": ctx.get("post_earnings_up_rate"),
        "reaction_samples": ctx.get("reaction_samples"),
        "avg_surprise_percent": ctx.get("avg_surprise_percent"),
        "price_position_52w": ctx.get("price_position_52w"),
    }


def generate_structured_plan(
    *,
    ticker: str,
    current_price: float,
    bars: list[dict],
    timeframe_bars: dict[str, list[dict]] | None = None,
    news_items: list[dict] | None,
    news_score: int,
    earnings_score: int,
    earnings_context: dict | None,
    market_regime: str,
    buy_threshold: int,
    avoid_threshold: int,
    history_stats: dict | None = None,
    benchmark_bars: dict[str, list[dict]] | None = None,
    ticker_meta: dict | None = None,
    sector_relative_strength: float | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_style: str | None = None,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> dict:
    frame = add_indicator_columns(bars_to_frame(bars), atr_window=config.atr_window, volume_window=config.volume_window)
    if frame.empty:
        raise ValueError(f"No daily bars available for {ticker}")

    structure = summarize_structure(
        frame,
        pivot_lookback=config.pivot_lookback,
        pivot_max_points=config.pivot_max_points,
        consolidation_window=config.consolidation_window,
        consolidation_range_atr_mult=config.consolidation_range_atr_mult,
        extended_from_ema20_pct=config.structure_extended_from_ema20_pct,
        parabolic_from_ema20_pct=config.structure_parabolic_from_ema20_pct,
        base_max_atr_range=config.structure_base_max_atr_range,
    )
    fibs = fibonacci_levels(frame, structure)
    zones = build_support_resistance_zones(frame, structure, fibs, config)

    atr_val = latest_value(frame, "atr") or max(current_price * 0.02, 0.01)
    atr_pct = latest_value(frame, "atr_pct")
    volatility = classify_volatility(atr_pct, config)
    universe = build_universe_suitability(current_price=current_price, frame=frame, config=config)
    moving_averages = {
        "ema20": latest_value(frame, "ema20"),
        "ema50": latest_value(frame, "ema50"),
        "ema100": latest_value(frame, "ema100"),
        "ema200": latest_value(frame, "ema200"),
        "sma50": latest_value(frame, "sma50"),
        "sma100": latest_value(frame, "sma100"),
        "sma200": latest_value(frame, "sma200"),
    }
    volume_context = _volume_context(frame)
    price_location = score_price_location(
        current_price=current_price,
        frame=frame,
        structure_state=structure.structure_state,
        support_zone_1=zones["support_zone_1"],
        resistance_zone_1=zones["resistance_zone_1"],
        atr=atr_val,
        config=config,
    )

    benchmark_bars = benchmark_bars or {}
    rs = {
        "vs_spy": None,
        "vs_qqq": None,
    }
    if benchmark_bars.get("SPY"):
        rs["vs_spy"] = _score_relative_strength(frame, add_indicator_columns(bars_to_frame(benchmark_bars["SPY"])))
    if benchmark_bars.get("QQQ"):
        rs["vs_qqq"] = _score_relative_strength(frame, add_indicator_columns(bars_to_frame(benchmark_bars["QQQ"])))

    entries = build_entry_candidates(
        current_price=current_price,
        trend_state=structure.trend_state,
        support_zone_1=zones["support_zone_1"],
        support_zone_2=zones["support_zone_2"],
        resistance_zone_1=zones["resistance_zone_1"],
        fib_levels=fibs,
        moving_averages=moving_averages,
        atr=atr_val,
        volume_context=volume_context,
        config=config,
    )
    preferred = choose_preferred_entry(
        current_price=current_price,
        candidates=entries,
        trend_state=structure.trend_state,
        support_zone_1=zones["support_zone_1"],
        volume_context=volume_context,
        config=config,
    )
    earnings = _earnings_payload(earnings_context)
    context = build_market_context(
        ticker=ticker,
        current_price=current_price,
        frame=frame,
        trend_state=structure.trend_state,
        moving_averages=moving_averages,
        atr=atr_val,
        volume_context=volume_context,
        relative_strength=rs,
        market_regime=market_regime,
        news_items=news_items,
        news_score=news_score,
        earnings=earnings,
        ticker_meta=ticker_meta,
        sector_relative_strength=sector_relative_strength,
        config=config,
    )
    normalized_timeframes = dict(timeframe_bars or {})
    normalized_timeframes["daily"] = bars
    chart_context = build_chart_context(
        normalized_timeframes,
        current_price=current_price,
        config=config,
    )

    recent_swing_low = structure.swing_lows[-1].price if structure.swing_lows else None
    recent_swing_high = structure.swing_highs[-1].price if structure.swing_highs else None
    stop = build_stop_loss(
        preferred_entry=preferred["preferred_entry"],
        support_zone_1=zones["support_zone_1"],
        support_zone_2=zones["support_zone_2"],
        recent_swing_low=recent_swing_low,
        atr=atr_val,
        current_price=current_price,
        trend_state=structure.structure_state,
        sl_tolerance=context["sl_tolerance"],
        setup_scenario=context["setup_scenario"],
        config=config,
    )
    confirmation = build_confirmation_plan(
        current_price=current_price,
        preferred_entry=preferred["preferred_entry"],
        support_zone_1=zones["support_zone_1"],
        resistance_zone_1=zones["resistance_zone_1"],
        moving_averages=moving_averages,
        structure_state=structure.structure_state,
        frame=frame,
        atr=atr_val,
        invalidation_level=stop["invalidation_level"],
        volume_context=volume_context,
        requires_confirmation=preferred["entry_requires_confirmation"],
        config=config,
    )
    effective_entry_requires_confirmation = bool(
        confirmation["confirmation_required"] and confirmation["confirmation_state"] != "confirmed"
    )
    historical_hold_days = None
    if history_stats and history_stats.get("samples"):
        historical_hold_days = 12
    hold = estimate_hold_window(
        preferred_entry=preferred["preferred_entry"],
        take_profit_1=zones["resistance_zone_1"]["upper"] if zones["resistance_zone_1"] else current_price + atr_val * config.tp1_atr_mult,
        atr=atr_val,
        recent_swing_bars=12,
        historical_hold_days=historical_hold_days,
        config=config,
    )
    targets = build_take_profits(
        preferred_entry=preferred["preferred_entry"],
        stop_loss=stop["stop_loss"],
        resistance_zone_1=zones["resistance_zone_1"],
        resistance_zone_2=zones["resistance_zone_2"],
        recent_swing_high=recent_swing_high,
        atr=atr_val,
        hold_days_hint=hold["max_hold_days"],
        trend_state=structure.structure_state,
        tp_aggressiveness=context["tp_aggressiveness"],
        expected_move_profile=context["expected_move_profile"],
        price_location_context=context["price_location_context"],
        config=config,
        ranked_resistance_levels=zones["resistance_levels"],
    )
    final_hold = estimate_hold_window(
        preferred_entry=preferred["preferred_entry"],
        take_profit_1=targets["take_profit_1"],
        atr=atr_val,
        recent_swing_bars=12,
        historical_hold_days=historical_hold_days,
        config=config,
    )
    if abs(final_hold["max_hold_days"] - hold["max_hold_days"]) >= 2:
        hold = final_hold
        targets = build_take_profits(
            preferred_entry=preferred["preferred_entry"],
            stop_loss=stop["stop_loss"],
            resistance_zone_1=zones["resistance_zone_1"],
            resistance_zone_2=zones["resistance_zone_2"],
            recent_swing_high=recent_swing_high,
            atr=atr_val,
            hold_days_hint=hold["max_hold_days"],
            trend_state=structure.structure_state,
            tp_aggressiveness=context["tp_aggressiveness"],
            expected_move_profile=context["expected_move_profile"],
            price_location_context=context["price_location_context"],
            config=config,
            ranked_resistance_levels=zones["resistance_levels"],
        )
    else:
        hold = final_hold
    if stop["stop_loss"] >= preferred["preferred_entry"]:
        raise ValueError(f"Invalid long stop placement for {ticker}: stop >= entry")
    if targets["take_profit_1"] <= preferred["preferred_entry"]:
        raise ValueError(f"Invalid long target placement for {ticker}: tp1 <= entry")

    stop_realism = stop["swing_realism_flag"]
    target_realism = targets["target_reachability_flag"]
    if stop_realism != "realistic" and target_realism != "reachable":
        level_geometry_flag = "compressed_stop_and_target"
    elif stop_realism != "realistic":
        level_geometry_flag = "compressed_stop"
    elif target_realism != "reachable":
        level_geometry_flag = "compressed_target"
    else:
        level_geometry_flag = "balanced"

    reward_risk = {
        "tp1": targets["expected_reward_risk_to_tp1"],
        "tp2": targets["expected_reward_risk_to_tp2"],
        "tp3": targets["expected_reward_risk_to_tp3"],
        "final": targets["expected_reward_risk_to_final"],
    }
    relative_strength_values = [value for value in rs.values() if value is not None]
    average_relative_strength = (
        sum(float(value) for value in relative_strength_values) / len(relative_strength_values)
        if relative_strength_values
        else 0.0
    )
    scenario_bundle = generate_execution_scenarios(
        chart_context=chart_context,
        current_price=current_price,
        atr=atr_val,
        support_zone_1=zones["support_zone_1"],
        support_zone_2=zones["support_zone_2"],
        resistance_zone_1=zones["resistance_zone_1"],
        resistance_zone_2=zones["resistance_zone_2"],
        trend_state=structure.trend_state,
        relative_strength_score=5.0 + average_relative_strength * 20.0,
        macro_alignment_score=context.get("macro_alignment_score"),
        news_regime_alignment=context.get("news_regime_alignment"),
        config=config,
    )

    preliminary_payload = {
        "ticker": ticker,
        "trend_state": structure.trend_state,
        "market_regime": market_regime,
        "buy_threshold": buy_threshold,
        "entry_quality_score": preferred["entry_quality_score"],
        "entry_requires_confirmation": effective_entry_requires_confirmation,
        "confirmation_trigger": preferred["confirmation_trigger"],
        "confirmation_trigger_price": confirmation["confirmation_trigger_price"],
        "confirmation_state": confirmation["confirmation_state"],
        "entry_status": confirmation["entry_status"],
        "structure_state": structure.structure_state,
        "executable_stop_technically_valid": stop["executable_stop_technically_valid"],
        "universe_eligible": universe["universe_eligible"],
        "volume_context": volume_context,
        "reward_risk": reward_risk,
        "earnings": earnings,
        "price_location_context": context["price_location_context"],
        "setup_type": context["setup_type"],
        "setup_scenario": context["setup_scenario"],
        "continuation_vs_reversion_bias": context["continuation_vs_reversion_bias"],
        "news_regime_alignment": context["news_regime_alignment"],
        "chart_news_alignment": context["chart_news_alignment"],
        "macro_alignment_score": context["macro_alignment_score"],
        "tp_aggressiveness": context["tp_aggressiveness"],
        "sl_tolerance": context["sl_tolerance"],
        "expected_move_profile": context["expected_move_profile"],
        "scenario_confidence": context["scenario_confidence"],
        "scenario_rationale": context["scenario_rationale"],
        "chart_context": chart_context,
        "execution_scenarios": scenario_bundle["execution_scenarios"],
        "preferred_scenario": scenario_bundle["preferred_scenario"],
        "execution_action": scenario_bundle["execution_action"],
        "stop_too_tight_flag": stop["stop_too_tight_flag"],
        "tp_too_optimistic_flag": targets["tp_too_optimistic_flag"],
        "composite_score": 0.0,
    }
    llm_review = review_setup(
        payload=preliminary_payload,
        config=config,
        provider=llm_provider,
        model=llm_model,
        style=llm_style,
    )

    scores = score_setup(
        trend_state=structure.trend_state,
        support_zone_1=zones["support_zone_1"],
        atr_pct=atr_pct,
        volume_context=volume_context,
        relative_strength=rs,
        earnings=earnings,
        reward_risk=reward_risk,
        entry_quality_score=preferred["entry_quality_score"],
        history_stats=history_stats,
        llm_quality_score=float(llm_review["llm_quality_score"]),
        context=context,
        sector_relative_strength=sector_relative_strength,
        config=config,
        structure_state=structure.structure_state,
        liquidity_score=universe["liquidity_score"],
        volatility_suitability_score=volatility["volatility_suitability_score"],
        price_location_score=price_location["price_location_score"],
        target_realism_score=targets["target_realism_score"],
        confirmation_score=confirmation["confirmation_score"],
    )

    composite_payload = {
        **preliminary_payload,
        "support_quality_score": scores["support_quality_score"],
        "relative_strength_score": scores["relative_strength_score"],
        "volume_confirmation_score": scores["volume_confirmation_score"],
        "earnings_risk_score": scores["earnings_risk_score"],
        "composite_score": scores["composite_score"],
        "context_score": scores["context_score"],
        "catalyst_score": scores["catalyst_score"],
        "macro_score": scores["macro_score"],
        "scenario_score": scores["scenario_score"],
    }
    llm_review = review_setup(
        payload=composite_payload,
        config=config,
        provider=llm_provider,
        model=llm_model,
        style=llm_style,
    )
    scores["llm_quality_score"] = float(llm_review["llm_quality_score"])
    scores = score_setup(
        trend_state=structure.trend_state,
        support_zone_1=zones["support_zone_1"],
        atr_pct=atr_pct,
        volume_context=volume_context,
        relative_strength=rs,
        earnings=earnings,
        reward_risk=reward_risk,
        entry_quality_score=preferred["entry_quality_score"],
        history_stats=history_stats,
        llm_quality_score=float(llm_review["llm_quality_score"]),
        context=context,
        sector_relative_strength=sector_relative_strength,
        config=config,
        structure_state=structure.structure_state,
        liquidity_score=universe["liquidity_score"],
        volatility_suitability_score=volatility["volatility_suitability_score"],
        price_location_score=price_location["price_location_score"],
        target_realism_score=targets["target_realism_score"],
        confirmation_score=confirmation["confirmation_score"],
    )

    # The reasoning layer may select only among eligible deterministic
    # scenarios. It cannot supply or alter any numeric scenario level.
    reviewed_preference = str(llm_review.get("preferred_scenario") or scenario_bundle["preferred_scenario"])
    reviewed_candidate = scenario_bundle["execution_scenarios"].get(reviewed_preference)
    if reviewed_preference == "none" or (reviewed_candidate and reviewed_candidate.get("eligible")):
        scenario_bundle["preferred_scenario"] = reviewed_preference
    selected_preference = scenario_bundle["preferred_scenario"]
    if selected_preference == "enter_now":
        scenario_bundle["execution_action"] = "BUY_NOW"
    elif selected_preference == "pullback":
        scenario_bundle["execution_action"] = "WAIT_FOR_PULLBACK"
    elif selected_preference == "breakout":
        breakout_candidate = scenario_bundle["execution_scenarios"]["breakout"]
        scenario_bundle["execution_action"] = "BUY_NOW" if breakout_candidate.get("activated") else "WAIT_FOR_BREAKOUT"
    elif selected_preference == "repair":
        scenario_bundle["execution_action"] = "WAIT_FOR_REPAIR"
    else:
        scenario_bundle["execution_action"] = "MONITOR"

    if confirmation["entry_status"] != "confirmed" and scenario_bundle["execution_action"] == "BUY_NOW":
        scenario_bundle["execution_action"] = (
            "WAIT_FOR_BREAKOUT" if structure.structure_state == "breakout" else "WAIT_FOR_PULLBACK"
        )

    signal_score = int(news_score + earnings_score)
    strategy_action = str(llm_review["llm_action"])
    setup_downgrade_reasons: list[str] = []
    if not stop["executable_stop_technically_valid"]:
        setup_downgrade_reasons.append("technical_invalidation_exceeds_swing_risk_limit")
    if not universe["universe_eligible"]:
        setup_downgrade_reasons.extend(universe["universe_rejection_reasons"])
    if structure.structure_state in {"structural_breakdown", "trend_damage"}:
        setup_downgrade_reasons.append(f"structure_state_{structure.structure_state}")
    if confirmation["entry_status"] != "confirmed":
        setup_downgrade_reasons.append(f"entry_{confirmation['entry_status']}")
    if strategy_action == "BUY" and setup_downgrade_reasons:
        strategy_action = (
            "AVOID"
            if structure.structure_state == "structural_breakdown" or not universe["universe_eligible"]
            else "WAIT"
        )
    if not stop["executable_stop_technically_valid"] and scenario_bundle["execution_action"] == "BUY_NOW":
        scenario_bundle["execution_action"] = "MONITOR"

    plan = {
        "ticker": ticker,
        "current_price": float(current_price),
        "trend_state": structure.trend_state,
        "structure_state": structure.structure_state,
        "enhanced_trend_state": structure.structure_state,
        "ema_structure": structure.ema_structure,
        "universe_suitability": universe,
        "universe_eligible": universe["universe_eligible"],
        "universe_rejection_reasons": universe["universe_rejection_reasons"],
        "average_daily_volume": universe["average_daily_volume"],
        "liquidity_score": scores["liquidity_score"],
        "support_zone_1": zones["support_zone_1"],
        "support_zone_2": zones["support_zone_2"],
        "resistance_zone_1": zones["resistance_zone_1"],
        "resistance_zone_2": zones["resistance_zone_2"],
        "support_levels": zones["support_levels"],
        "resistance_levels": zones["resistance_levels"],
        "nearest_support": zones["nearest_support"],
        "nearest_resistance": zones["nearest_resistance"],
        "major_resistance_cluster": zones["major_resistance_cluster"],
        "atr": round(float(atr_val), 6),
        "atr_pct": None if atr_pct is None else round(float(atr_pct), 6),
        **volatility,
        "ema20": moving_averages["ema20"],
        "ema50": moving_averages["ema50"],
        "ema100": moving_averages["ema100"],
        "ema200": moving_averages["ema200"],
        **price_location,
        "fib_levels": fibs,
        "moving_averages": moving_averages,
        "volume_context": volume_context,
        "relative_strength": rs,
        "earnings": earnings,
        "entry_candidates": entries,
        "preferred_entry": preferred["preferred_entry"],
        "preferred_entry_type": preferred["preferred_entry_type"],
        "entry_quality_score": preferred["entry_quality_score"],
        "entry_distance_from_current_price_pct": preferred["entry_distance_from_current_price_pct"],
        "entry_confluence_score": preferred["entry_confluence_score"],
        "entry_requires_confirmation": effective_entry_requires_confirmation,
        "confirmation_trigger": preferred["confirmation_trigger"],
        **confirmation,
        "stop_loss": stop["stop_loss"],
        "suggested_stop": stop["suggested_stop"],
        "invalidation_level": stop["invalidation_level"],
        "invalidation_reason": stop["invalidation_reason"],
        "invalidation_width_pct": stop["invalidation_width_pct"],
        "invalidation_width_atr": stop["invalidation_width_atr"],
        "executable_stop_technically_valid": stop["executable_stop_technically_valid"],
        "stop_basis": stop["stop_basis"],
        "stop_distance_pct": stop["stop_distance_pct"],
        "stop_width_pct": stop["stop_width_pct"],
        "stop_width_atr": stop["stop_width_atr"],
        "stop_too_tight_flag": stop["stop_too_tight_flag"],
        "take_profit_1": targets["take_profit_1"],
        "take_profit_2": targets["take_profit_2"],
        "take_profit_3": targets["take_profit_3"],
        "stretch_target": targets["stretch_target"],
        "take_profit_final": targets["take_profit_final"],
        "tp1_distance_pct": targets["tp1_distance_pct"],
        "tp1_distance_atr": targets["tp1_distance_atr"],
        "tp1_atr_distance": targets["tp1_atr_distance"],
        "tp2_atr_distance": targets["tp2_atr_distance"],
        "tp3_atr_distance": targets["tp3_atr_distance"],
        "tp1_reason": targets["tp1_reason"],
        "tp2_reason": targets["tp2_reason"],
        "tp3_reason": targets["tp3_reason"],
        "stretch_target_reason": targets["stretch_target_reason"],
        "tp_basis": targets["tp_basis"],
        "reward_risk": reward_risk,
        "tp_too_optimistic_flag": targets["tp_too_optimistic_flag"],
        "hold_window_reachability_score": targets["hold_window_reachability_score"],
        "target_realism_score": scores["target_realism_score"],
        "swing_realism_flag": stop["swing_realism_flag"],
        "risk_width_flag": stop["risk_width_flag"],
        "target_reachability_flag": targets["target_reachability_flag"],
        "level_geometry_flag": level_geometry_flag,
        "stop_generation_reason": stop["stop_generation_reason"],
        "tp1_generation_reason": targets["tp1_generation_reason"],
        "max_hold_days": hold["max_hold_days"],
        "expected_hold_days": hold["expected_hold_days"],
        "max_hold_date": hold["max_hold_date"],
        "trend_quality_score": scores["trend_quality_score"],
        "trend_score": scores["trend_score"],
        "pullback_quality_score": scores["pullback_quality_score"],
        "support_quality_score": scores["support_quality_score"],
        "support_confluence_score": scores["support_confluence_score"],
        "volatility_quality_score": scores["volatility_quality_score"],
        "volatility_suitability_score": scores["volatility_suitability_score"],
        "confirmation_score": scores["confirmation_score"],
        "relative_strength_score": scores["relative_strength_score"],
        "volume_confirmation_score": scores["volume_confirmation_score"],
        "earnings_risk_score": scores["earnings_risk_score"],
        "reward_risk_score": scores["reward_risk_score"],
        "historical_analogue_score": scores["historical_analogue_score"],
        "llm_quality_score": scores["llm_quality_score"],
        "context_score": scores["context_score"],
        "catalyst_score": scores["catalyst_score"],
        "macro_score": scores["macro_score"],
        "scenario_score": scores["scenario_score"],
        "composite_score": scores["composite_score"],
        "component_scores": {
            key: scores[key]
            for key in (
                "liquidity_score",
                "volatility_suitability_score",
                "trend_score",
                "price_location_score",
                "support_confluence_score",
                "target_realism_score",
                "confirmation_score",
            )
        },
        "setup_downgrade_reasons": setup_downgrade_reasons,
        **context,
        "llm_review": {k: v for k, v in llm_review.items() if k not in {"prompt_preview", "provider", "model", "style", "llm_quality_score"}},
        "strategy_action": strategy_action,
        "chart_context": chart_context,
        "timeframe_context": chart_context.get("timeframes") or {},
        "daily_trend": chart_context.get("daily_trend"),
        "four_hour_trend": chart_context.get("four_hour_trend"),
        "one_hour_trend": chart_context.get("one_hour_trend"),
        "thirty_minute_trend": chart_context.get("thirty_minute_trend"),
        "multi_timeframe_alignment_score": chart_context.get("multi_timeframe_alignment_score"),
        "preferred_trade_shape": chart_context.get("preferred_trade_shape"),
        "execution_scenarios": scenario_bundle["execution_scenarios"],
        "enter_now_scenario": scenario_bundle["enter_now_scenario"],
        "pullback_scenario": scenario_bundle["pullback_scenario"],
        "breakout_scenario": scenario_bundle["breakout_scenario"],
        "repair_scenario": scenario_bundle["repair_scenario"],
        "preferred_scenario": scenario_bundle["preferred_scenario"],
        "execution_action": scenario_bundle["execution_action"],
        "execution_scenario_confidence": scenario_bundle["scenario_confidence"],
        "scenario_selection_reason": scenario_bundle["scenario_selection_reason"],
        "pullback_entry_zone": scenario_bundle["pullback_entry_zone"],
        "breakout_trigger_zone": scenario_bundle["breakout_trigger_zone"],
        "repair_trigger_zone": scenario_bundle["repair_trigger_zone"],
        "live_scenario_status": "plan_generated",
        "replan_needed": False,
        "signal_score": signal_score,
        "strategy_reason": (
            f"trend={structure.trend_state}; setup={context['setup_scenario']}; entry={preferred['preferred_entry_type']}; rr1={reward_risk['tp1']:.2f}; "
            f"earnings_days={earnings['days_to_earnings']}; composite={scores['composite_score']:.2f}; llm={llm_review['llm_action']}"
        ),
        "risk_tuning_reason": llm_review["risk_tuning_reason"],
        "market_regime": market_regime,
        "buy_threshold": buy_threshold,
        "avoid_threshold": avoid_threshold,
        "structure_flags": structure.structure_flags,
        "breakout_level": structure.breakout_level,
        "prior_breakout_retest_zone": structure.prior_breakout_retest_zone,
        "consolidation_range": structure.consolidation_range,
        "gap_zone": structure.gap_zone,
        "recent_swing_highs": [pivot.__dict__ for pivot in structure.swing_highs],
        "recent_swing_lows": [pivot.__dict__ for pivot in structure.swing_lows],
    }
    return plan
