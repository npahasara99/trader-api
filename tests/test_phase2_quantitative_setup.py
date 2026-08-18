from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.chart_context import build_chart_context
from app.config import DEFAULT_PLANNING_CONFIG
from app.confirmation import build_confirmation_plan
from app.indicators import add_indicator_columns, bars_to_frame
from app.planner import generate_structured_plan
from app.risk_engine import build_stop_loss, build_take_profits
from app.scanner import build_universe_suitability, classify_volatility
from app.scoring import score_price_location
from app.structure import summarize_structure
from app.zones import build_support_resistance_zones, fibonacci_levels


def _bars_from_closes(closes: list[float], *, volumes: list[float] | None = None) -> list[dict]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    for index, close in enumerate(closes):
        prior = closes[index - 1] if index else close
        volume = volumes[index] if volumes else 2_000_000.0
        result.append(
            {
                "date": start + timedelta(days=index),
                "open": prior,
                "high": max(prior, close) + 0.6,
                "low": min(prior, close) - 0.6,
                "close": close,
                "volume": volume,
            }
        )
    return result


def _structure_for(closes: list[float], *, volumes: list[float] | None = None):
    frame = add_indicator_columns(bars_to_frame(_bars_from_closes(closes, volumes=volumes)))
    structure = summarize_structure(
        frame,
        pivot_lookback=DEFAULT_PLANNING_CONFIG.pivot_lookback,
        pivot_max_points=DEFAULT_PLANNING_CONFIG.pivot_max_points,
        consolidation_window=DEFAULT_PLANNING_CONFIG.consolidation_window,
        consolidation_range_atr_mult=DEFAULT_PLANNING_CONFIG.consolidation_range_atr_mult,
        extended_from_ema20_pct=DEFAULT_PLANNING_CONFIG.structure_extended_from_ema20_pct,
        parabolic_from_ema20_pct=DEFAULT_PLANNING_CONFIG.structure_parabolic_from_ema20_pct,
        base_max_atr_range=DEFAULT_PLANNING_CONFIG.structure_base_max_atr_range,
    )
    return frame, structure


def test_healthy_uptrend_pullback_has_rich_state_and_good_location():
    closes = [80.0 + index * 0.15 for index in range(230)]
    closes += [114.7, 115.0, 115.3, 115.6, 115.9, 115.5, 115.1, 114.8, 114.6, 114.5]
    frame, structure = _structure_for(closes)
    fibs = fibonacci_levels(frame, structure)
    zones = build_support_resistance_zones(frame, structure, fibs, DEFAULT_PLANNING_CONFIG)
    location = score_price_location(
        current_price=closes[-1],
        frame=frame,
        structure_state=structure.structure_state,
        support_zone_1=zones["support_zone_1"],
        resistance_zone_1=zones["resistance_zone_1"],
        atr=float(frame["atr"].iloc[-1]),
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert structure.structure_state == "healthy_pullback"
    assert structure.trend_state == "pullback_in_uptrend"
    assert location["price_location_score"] >= 7.0
    assert location["price_location_category"] in {"good", "excellent"}
    assert all(f"ema{period}" in frame.columns for period in (20, 50, 100, 200))
    assert all(f"sma{period}" in frame.columns for period in (50, 100, 200))


def test_oversold_style_selloff_is_structural_breakdown_not_pullback():
    closes = [100.0 + index * 0.08 for index in range(220)]
    closes += [117.0, 115.0, 112.0, 108.0, 104.0, 100.0, 96.0, 92.0, 88.0, 84.0, 80.0, 76.0, 72.0, 68.0, 64.0, 60.0, 56.0, 52.0, 48.0, 44.0]
    volumes = [2_000_000.0] * (len(closes) - 1) + [6_000_000.0]
    bars = _bars_from_closes(closes, volumes=volumes)
    _, structure = _structure_for(closes, volumes=volumes)
    plan = generate_structured_plan(
        ticker="BREAKDOWN",
        current_price=closes[-1],
        bars=bars,
        timeframe_bars={},
        news_items=[],
        news_score=0,
        earnings_score=0,
        earnings_context={},
        market_regime="neutral",
        buy_threshold=6,
        avoid_threshold=-4,
    )

    assert structure.structure_state == "structural_breakdown"
    assert structure.trend_state == "downtrend"
    assert "below_all_major_emas" in structure.structure_flags
    assert plan["strategy_action"] != "BUY"
    assert plan["entry_status"] != "confirmed"


def test_extended_parabolic_stock_is_not_a_good_location():
    closes = [70.0 + index * 0.1 for index in range(230)]
    closes += [95.0, 99.0, 104.0, 110.0, 117.0, 125.0, 134.0, 144.0, 155.0, 167.0]
    frame, structure = _structure_for(closes)
    location = score_price_location(
        current_price=closes[-1],
        frame=frame,
        structure_state=structure.structure_state,
        support_zone_1=None,
        resistance_zone_1=None,
        atr=float(frame["atr"].iloc[-1]),
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert structure.structure_state == "extended"
    assert location["price_location_category"] == "extended"
    assert location["price_location_score"] <= 2.5


def test_price_in_support_zone_remains_below_confirmation_trigger():
    frame = add_indicator_columns(bars_to_frame(_bars_from_closes([98.0, 99.0, 100.0, 100.5])))
    plan = build_confirmation_plan(
        current_price=100.0,
        preferred_entry=100.0,
        support_zone_1={"lower": 99.0, "upper": 101.0},
        resistance_zone_1={"lower": 104.0, "upper": 105.0},
        moving_averages={"ema20": None},
        structure_state="healthy_pullback",
        frame=frame,
        atr=2.0,
        invalidation_level=97.0,
        volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "no_confirmation"},
        requires_confirmation=True,
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert plan["entry_status"] == "in_price_zone"
    assert plan["confirmation_state"] == "awaiting_confirmation"
    assert plan["confirmation_trigger_price"] > 100.0


def test_price_crossing_numeric_trigger_becomes_confirmed():
    frame = add_indicator_columns(bars_to_frame(_bars_from_closes([98.0, 99.0, 100.0, 100.5])))
    base = dict(
        preferred_entry=100.0,
        support_zone_1={"lower": 99.0, "upper": 101.0},
        resistance_zone_1={"lower": 104.0, "upper": 105.0},
        moving_averages={"ema20": None},
        structure_state="healthy_pullback",
        frame=frame,
        atr=2.0,
        invalidation_level=97.0,
        volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "confirmed_bounce"},
        requires_confirmation=True,
        config=DEFAULT_PLANNING_CONFIG,
    )
    waiting = build_confirmation_plan(current_price=100.0, **base)
    confirmed = build_confirmation_plan(current_price=waiting["confirmation_trigger_price"] + 0.01, **base)

    assert confirmed["entry_status"] == "confirmed"
    assert confirmed["confirmation_state"] == "confirmed"
    assert confirmed["price_confirmed"] is True


def test_wide_true_invalidation_downgrades_executable_stop():
    stop = build_stop_loss(
        preferred_entry=100.0,
        support_zone_1={"lower": 71.0, "upper": 73.0, "source_tags": ["pivot_low"]},
        support_zone_2={"lower": 66.0, "upper": 68.0, "source_tags": ["ema200"]},
        recent_swing_low=69.5,
        atr=3.0,
        current_price=101.0,
        trend_state="healthy_pullback",
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert stop["invalidation_level"] < stop["suggested_stop"]
    assert stop["executable_stop_technically_valid"] is False
    assert stop["risk_width_flag"] == "capped_for_swing"


def test_far_target_is_capped_to_two_to_ten_day_reachability():
    targets = build_take_profits(
        preferred_entry=100.0,
        stop_loss=95.0,
        resistance_zone_1={"lower": 145.0, "upper": 150.0, "source_tags": ["pivot_high"]},
        resistance_zone_2={"lower": 155.0, "upper": 160.0, "source_tags": ["gap_fill"]},
        recent_swing_high=148.0,
        atr=2.0,
        hold_days_hint=10,
        trend_state="healthy_pullback",
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert targets["target_reachability_flag"] == "capped_to_hold_window"
    assert targets["tp1_atr_distance"] < 5.0
    assert targets["take_profit_1"] < 145.0
    assert targets["target_realism_score"] < 10.0


def test_missing_four_hour_and_intraday_data_does_not_crash():
    daily = _bars_from_closes([80.0 + index * 0.15 for index in range(220)])
    context = build_chart_context(daily_bars=daily, current_price=daily[-1]["close"])

    assert context["available"] is True
    assert context["available_timeframes"] == ["daily"]
    assert context["four_hour_trend"] is None
    assert context["multi_timeframe_alignment_score"] is not None
    assert {"four_hour", "hourly", "thirty_minute"}.issubset(set(context["missing_timeframes"]))


def test_universe_and_volatility_thresholds_are_config_driven():
    frame = add_indicator_columns(bars_to_frame(_bars_from_closes([25.0 + index * 0.02 for index in range(220)])))
    profile = build_universe_suitability(current_price=29.0, frame=frame, config=DEFAULT_PLANNING_CONFIG)
    volatility = classify_volatility(0.025, DEFAULT_PLANNING_CONFIG)

    assert profile["universe_eligible"] is True
    assert profile["liquidity_score"] >= 7.0
    assert volatility["volatility_regime"] == "preferred"
    assert volatility["atr_percent"] == 2.5
