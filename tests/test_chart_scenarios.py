from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.chart_context import build_chart_context
from app.planner import generate_structured_plan
from app.scenario_engine import evaluate_live_scenario_status, generate_execution_scenarios


SUPPORT = {"lower": 99.0, "upper": 101.0, "display": "99.00 to 101.00"}
SECONDARY_SUPPORT = {"lower": 94.0, "upper": 96.0, "display": "94.00 to 96.00"}
RESISTANCE = {"lower": 109.0, "upper": 111.0, "display": "109.00 to 111.00"}
MAJOR_RESISTANCE = {"lower": 118.0, "upper": 120.0, "display": "118.00 to 120.00"}


def _trend_bars(count=120, *, step=0.2, hours=False):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delta = timedelta(hours=1) if hours else timedelta(days=1)
    bars = []
    for index in range(count):
        close = 90.0 + index * step + (index % 7 - 3) * 0.08
        bars.append(
            {
                "date": start + delta * index,
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1_000_000 + index * 1000,
            }
        )
    return bars


def _context(**overrides):
    payload = {
        "dominant_trend": "uptrend",
        "current_structure": "constructive_trend",
        "preferred_trade_shape": "pullback_preferred",
        "extension_state": "balanced",
        "breakout_state": "inside_range",
        "short_term_reversal_state": "none",
        "rsi": 54.0,
        "volume_state": "normal",
        "nearest_support_zone": SUPPORT,
        "secondary_support_zone": SECONDARY_SUPPORT,
        "nearest_resistance_zone": RESISTANCE,
        "major_resistance_zone": MAJOR_RESISTANCE,
        "breakout_trigger_zone": RESISTANCE,
    }
    payload.update(overrides)
    return payload


def _generate(context, *, price=102.0, trend="uptrend", rs=6.0, macro=6.0):
    return generate_execution_scenarios(
        chart_context=context,
        current_price=price,
        atr=2.0,
        support_zone_1=SUPPORT,
        support_zone_2=SECONDARY_SUPPORT,
        resistance_zone_1=RESISTANCE,
        resistance_zone_2=MAJOR_RESISTANCE,
        trend_state=trend,
        relative_strength_score=rs,
        macro_alignment_score=macro,
        news_regime_alignment="aligned_bullish",
    )


def test_chart_context_degrades_when_intraday_timeframes_are_missing():
    daily = _trend_bars(90, step=0.25)

    context = build_chart_context(daily_bars=daily, current_price=daily[-1]["close"])

    assert context["available"] is True
    assert context["available_timeframes"] == ["daily"]
    assert set(context["missing_timeframes"]) == {"hourly", "thirty_minute"}
    assert context["timeframes"]["daily"]["available"] is True


def test_structured_plan_exposes_multi_timeframe_scenarios_without_llm_levels():
    daily = _trend_bars(180, step=0.18)
    hourly = _trend_bars(100, step=0.025, hours=True)
    plan = generate_structured_plan(
        ticker="TEST",
        current_price=daily[-1]["close"],
        bars=daily,
        timeframe_bars={"hourly": hourly, "thirty_minute": hourly},
        news_items=[],
        news_score=0,
        earnings_score=0,
        earnings_context={},
        market_regime="neutral",
        buy_threshold=4,
        avoid_threshold=-4,
    )

    assert plan["chart_context"]["available_timeframes"] == ["daily", "hourly", "thirty_minute"]
    assert set(plan["execution_scenarios"]) == {"enter_now", "pullback", "breakout", "repair"}
    assert plan["preferred_scenario"] in {"enter_now", "pullback", "breakout", "repair", "none"}
    assert plan["execution_action"] in {
        "BUY_NOW",
        "WAIT_FOR_PULLBACK",
        "WAIT_FOR_BREAKOUT",
        "WAIT_FOR_REPAIR",
        "MONITOR",
        "AVOID",
    }
    selected = plan["execution_scenarios"].get(plan["preferred_scenario"])
    if selected:
        assert selected["entry_price"] is not None
        assert selected["stop_loss"] < selected["entry_price"] < selected["take_profit_1"]


def test_uptrend_near_support_prefers_pullback_scenario():
    result = _generate(_context(), price=101.5)
    assert result["preferred_scenario"] == "pullback"
    assert result["execution_action"] == "WAIT_FOR_PULLBACK"
    assert result["pullback_scenario"]["eligible"] is True


def test_confirmed_breakout_activates_breakout_scenario():
    result = _generate(
        _context(preferred_trade_shape="breakout_preferred", breakout_state="confirmed_breakout"),
        price=112.0,
    )
    assert result["preferred_scenario"] == "breakout"
    assert result["breakout_scenario"]["activated"] is True
    assert result["execution_action"] == "BUY_NOW"


def test_mid_range_price_waits_instead_of_forcing_enter_now():
    result = _generate(_context(preferred_trade_shape="no_clean_trade"), price=105.0)
    assert result["enter_now_scenario"]["eligible"] is False
    assert result["execution_action"] in {"WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "MONITOR"}


def test_high_range_strength_keeps_breakout_alternative_valid():
    result = _generate(
        _context(preferred_trade_shape="breakout_preferred", extension_state="extended", rsi=66.0),
        price=108.0,
        rs=7.2,
        macro=6.5,
    )
    assert result["breakout_scenario"]["eligible"] is True
    assert result["preferred_scenario"] == "breakout"


def test_high_range_overextension_avoids_chase_and_prefers_pullback():
    result = _generate(
        _context(preferred_trade_shape="continuation_pullback", extension_state="overextended", rsi=78.0),
        price=108.0,
        rs=4.0,
        macro=4.0,
    )
    assert result["enter_now_scenario"]["eligible"] is False
    assert result["preferred_scenario"] == "pullback"


def test_weak_stock_near_lows_without_reversal_is_avoid():
    result = _generate(
        _context(
            dominant_trend="downtrend",
            current_structure="damaged_structure",
            preferred_trade_shape="no_clean_trade",
            short_term_reversal_state="none",
        ),
        price=96.5,
        trend="downtrend",
    )
    assert result["repair_scenario"]["eligible"] is False
    assert result["execution_action"] == "AVOID"


def test_genuine_repair_requires_wait_for_repair():
    result = _generate(
        _context(
            dominant_trend="downtrend",
            current_structure="constructive_recovery",
            preferred_trade_shape="repair_trade",
            short_term_reversal_state="confirmed",
        ),
        price=103.0,
        trend="weak_breakdown_risk",
    )
    assert result["repair_scenario"]["eligible"] is True
    assert result["preferred_scenario"] == "repair"
    assert result["execution_action"] == "WAIT_FOR_REPAIR"


def test_live_price_above_tp1_requires_replan():
    plan = _generate(_context(), price=101.5)
    tp1 = plan["pullback_scenario"]["take_profit_1"]
    status = evaluate_live_scenario_status(plan, float(tp1) + 0.01)
    assert status["live_scenario_status"] == "tp1_hit_replan"
    assert status["replan_needed"] is True


def test_live_price_below_scenario_stop_invalidates_plan():
    plan = _generate(_context(), price=101.5)
    stop = plan["pullback_scenario"]["stop_loss"]
    status = evaluate_live_scenario_status(plan, float(stop) - 0.01)
    assert status["live_scenario_status"] == "scenario_invalidated"
    assert status["replan_needed"] is True
