from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.candidate_discovery import build_multilane_candidate_order, run_adaptive_batches
from app.config import DEFAULT_PLANNING_CONFIG
from app.confirmation import build_confirmation_plan
from app.live_monitor.learning import aggregate_observations, similar_case_score
from app.opportunity_ranking import build_portfolio_snapshot, rank_daily_opportunities
from app.risk_engine import build_stop_loss, build_take_profits
from app.scanner import build_pre_scan_profile
from app.setup_archetypes import (
    BASE_BREAKOUT,
    BREAKOUT_RETEST,
    DEEP_PULLBACK,
    HEALTHY_PULLBACK,
    MOMENTUM_CONTINUATION,
    REVERSAL_ATTEMPT,
    SETUP_FAMILIES,
    evaluate_runner_state,
    score_setup_families,
)


def _bars(closes: list[float], volumes: list[float] | None = None) -> list[dict]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        rows.append(
            {
                "date": start + timedelta(days=index),
                "open": previous,
                "high": max(previous, close) + 0.35,
                "low": min(previous, close) - 0.35,
                "close": close,
                "volume": (volumes or [2_000_000.0] * len(closes))[index],
            }
        )
    return rows


@pytest.mark.parametrize(
    ("expected", "overrides"),
    [
        (HEALTHY_PULLBACK, {"trend_strength": 9, "pullback_quality": 10, "price_location": 9, "relative_strength": 9, "pullback_volume": 9, "support_confluence": 9, "continuation_structure": 8}),
        (MOMENTUM_CONTINUATION, {"trend_strength": 10, "relative_strength": 10, "continuation_structure": 10, "base_quality": 6, "pullback_volume": 9, "confirmation": 9, "price_location": 8}),
        (BREAKOUT_RETEST, {"breakout_retest_quality": 10, "support_confluence": 10, "trend_strength": 9, "relative_strength": 9, "pullback_volume": 9, "confirmation": 9}),
        (BASE_BREAKOUT, {"base_quality": 10, "trend_strength": 9, "continuation_structure": 9, "relative_strength": 8, "confirmation": 10}),
        (DEEP_PULLBACK, {"deep_pullback_quality": 10, "support_confluence": 10, "reversal_quality": 8, "trend_strength": 7, "pullback_volume": 9, "confirmation": 8}),
        (REVERSAL_ATTEMPT, {"reversal_quality": 10, "support_confluence": 9, "confirmation": 10, "pullback_volume": 9, "price_location": 9, "target_quality": 9}),
    ],
)
def test_each_archetype_has_an_independent_scoring_lane(expected: str, overrides: dict):
    components = {name: 1.0 for weights in DEFAULT_PLANNING_CONFIG.setup_family_score_weights.values() for name in weights}
    components.update({"volatility": 6.0, "liquidity": 7.0})
    components.update(overrides)

    result = score_setup_families(
        components,
        weights_by_family=DEFAULT_PLANNING_CONFIG.setup_family_score_weights,
    )

    assert result["setup_family"] == expected
    assert result["setup_lane_scores"][expected] >= DEFAULT_PLANNING_CONFIG.setup_lane_min_score


def test_healthy_pullback_discovery_does_not_require_oversold_rsi():
    closes = [75.0 + index * 0.17 for index in range(220)] + [112.5, 112.9, 113.2, 113.5, 113.1, 112.8, 112.5, 112.3]
    profile = build_pre_scan_profile(
        ticker="HEALTHY",
        current_price=closes[-1],
        bars=_bars(closes),
        benchmark_bars={},
        sector_benchmark_symbol=None,
        earnings_context={},
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert profile["setup_lane_scores"][HEALTHY_PULLBACK] >= DEFAULT_PLANNING_CONFIG.setup_lane_min_score
    assert "rsi" not in profile["setup_lane_components"]


def test_momentum_lane_discovers_stock_one_to_three_percent_below_high():
    closes = [60.0 + index * 0.24 for index in range(220)] + [113.0, 114.0, 115.0, 116.0, 117.0, 116.7, 116.2, 115.1]
    profile = build_pre_scan_profile(
        ticker="MOMO",
        current_price=closes[-1],
        bars=_bars(closes),
        benchmark_bars={},
        sector_benchmark_symbol=None,
        earnings_context={},
        config=DEFAULT_PLANNING_CONFIG,
    )
    distance_from_high = (max(closes[-20:]) - closes[-1]) / max(closes[-20:])

    assert 0.01 <= distance_from_high <= 0.03
    assert profile["setup_lane_scores"][MOMENTUM_CONTINUATION] >= DEFAULT_PLANNING_CONFIG.setup_lane_min_score


def _lane_candidate(ticker: str, family: str, score: float, raw: float) -> dict:
    return {
        "ticker": ticker,
        "setup_family": family,
        "setup_lane_scores": {name: score if name == family else 1.0 for name in SETUP_FAMILIES},
        "pre_scan_score": raw,
    }


def test_lane_reservations_prevent_one_strategy_from_consuming_discovery():
    candidates = [_lane_candidate(f"H{index}", HEALTHY_PULLBACK, 9.5, 10 - index / 10) for index in range(12)]
    candidates += [_lane_candidate(family[:3].upper(), family, 8.0, 6.0) for family in SETUP_FAMILIES[1:]]
    metadata = {item["ticker"]: {"sector": "Technology"} for item in candidates}

    ordered = build_multilane_candidate_order(
        candidates,
        metadata_by_ticker=metadata,
        initial_limit=6,
        min_per_sector=1,
        minimum_by_family={family: 1 for family in SETUP_FAMILIES},
        minimum_family_score=DEFAULT_PLANNING_CONFIG.setup_lane_min_score,
    )

    assert {item["setup_family"] for item in ordered[:6]} == set(SETUP_FAMILIES)


def test_adaptive_expansion_preserves_multilane_breadth():
    candidates = [_lane_candidate(f"{index}-{family}", family, 8.0, 9.0 - index / 10) for index, family in enumerate(SETUP_FAMILIES)]
    order = build_multilane_candidate_order(
        candidates,
        metadata_by_ticker={item["ticker"]: {"sector": "Mixed"} for item in candidates},
        initial_limit=3,
        min_per_sector=1,
        minimum_by_family={family: 1 for family in SETUP_FAMILIES},
        minimum_family_score=5.4,
    )
    rows, history = run_adaptive_batches(
        order,
        initial_limit=3,
        batch_size=3,
        maximum_limit=6,
        target_actionable=1,
        adaptive=True,
        analyze_batch=lambda batch: list(batch),
        count_actionable=lambda _rows: 0,
    )

    assert {item["setup_family"] for item in rows} == set(SETUP_FAMILIES)
    assert set(history[-1]["cumulative_setup_family_counts"]) == set(SETUP_FAMILIES)


def _rank_row(ticker: str, family: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker, current_price=100.0, last=100.0, preferred_entry=100.0,
        preferred_entry_low=99.0, preferred_entry_high=101.0, confirmation_trigger_price=103.0,
        stop_loss=95.0, invalidation_level=94.5, take_profit_1=112.0, take_profit_2=118.0,
        take_profit_3=123.0, stretch_target=128.0, atr=2.0, enhanced_trend_state="healthy_pullback",
        setup_type=family, setup_family=family, setup_family_score=score, final_action="WAIT",
        entry_status="in_price_zone", confirmation_state="awaiting_confirmation", price_confirmed=False,
        volume_confirmed=False, executable_stop_technically_valid=True, tp_too_optimistic_flag=False,
        universe_eligible=True, universe_rejection_reasons=[], trend_score=score, price_location_score=score,
        support_confluence_score=score, multi_timeframe_alignment_score=score, relative_strength_score=score,
        volatility_suitability_score=score, liquidity_score=score, target_realism_score=score,
        reward_risk_score=score, catalyst_score=score, macro_score=score, confirmation_score=5.0,
        hold_window_reachability_score=score, volume_confirmation_score=4.0, sector_relative_strength=0.02,
        reward_risk={"tp1": 2.4, "tp2": 3.6}, signal_score=5, runner_eligible=family not in {DEEP_PULLBACK, REVERSAL_ATTEMPT},
        tp1_partial_profit_min_pct=0.25, tp1_partial_profit_max_pct=0.5, runner_state="awaiting_tp1_breakout",
    )


def test_final_ranking_is_global_quality_not_forced_family_diversification():
    rows = [
        _rank_row("HP1", HEALTHY_PULLBACK, 9.7),
        _rank_row("HP2", HEALTHY_PULLBACK, 9.4),
        _rank_row("REV", REVERSAL_ATTEMPT, 7.0),
    ]
    metadata = {row.ticker: {"ticker": row.ticker, "sector": "Mixed", "industry": "Mixed"} for row in rows}
    result = rank_daily_opportunities(
        rows,
        metadata_by_ticker=metadata,
        market_regime="neutral",
        portfolio=build_portfolio_snapshot([], metadata_by_ticker=metadata, max_positions=10, trading_budget=10_000.0),
        best_setups_count=3,
        best_trades_max=0,
        next_to_trigger_count=3,
    )

    assert [item["ticker"] for item in result["best_setups"][:2]] == ["HP1", "HP2"]
    assert set(result["best_by_setup_family"]) == {HEALTHY_PULLBACK, REVERSAL_ATTEMPT}


def test_family_confirmation_stop_target_and_runner_policies_are_explicit():
    frame = pd.DataFrame({"high": [99.0, 100.0, 101.0], "low": [96.0, 97.0, 98.0], "close": [98.0, 99.0, 100.0], "volume": [1_000_000] * 3})
    confirmation = build_confirmation_plan(
        current_price=100.0, preferred_entry=100.0,
        support_zone_1={"lower": 99.0, "upper": 101.0}, resistance_zone_1={"lower": 104.0, "upper": 106.0},
        moving_averages={"ema20": 100.5}, structure_state="breakout", frame=frame, atr=2.0,
        invalidation_level=96.0, volume_context={"selloff_volume_state": "normal", "reversal_volume_state": "no_confirmation"},
        requires_confirmation=True, setup_family=BASE_BREAKOUT,
        consolidation_range={"lower": 99.0, "upper": 103.5}, config=DEFAULT_PLANNING_CONFIG,
    )
    stop = build_stop_loss(
        preferred_entry=100.0, support_zone_1={"lower": 90.0, "upper": 92.0},
        support_zone_2={"lower": 78.0, "upper": 80.0}, recent_swing_low=88.0, atr=2.5,
        current_price=100.0, trend_state="deep_pullback", setup_family=DEEP_PULLBACK,
        invalidation_zone={"lower": 70.0, "upper": 73.0}, config=DEFAULT_PLANNING_CONFIG,
    )
    continuation_targets = build_take_profits(
        preferred_entry=100.0, stop_loss=95.0, resistance_zone_1={"lower": 108.0, "upper": 110.0},
        resistance_zone_2={"lower": 114.0, "upper": 116.0}, recent_swing_high=109.0, atr=2.0,
        hold_days_hint=8, trend_state="uptrend", setup_family=MOMENTUM_CONTINUATION, config=DEFAULT_PLANNING_CONFIG,
    )
    recovery_targets = build_take_profits(
        preferred_entry=100.0, stop_loss=92.0, resistance_zone_1={"lower": 106.0, "upper": 108.0},
        resistance_zone_2=None, recent_swing_high=107.0, atr=2.5, hold_days_hint=8,
        trend_state="deep_pullback", setup_family=DEEP_PULLBACK, config=DEFAULT_PLANNING_CONFIG,
    )

    assert confirmation["confirmation_style"] == "base_resistance_break"
    assert "volume_expansion_required" in confirmation["confirmation_requirements"]
    assert stop["invalidation_level"] < 75.0
    assert stop["trade_geometry_status"] == "valid_setup_but_untradeable_geometry"
    assert stop["suggested_stop"] is None
    assert continuation_targets["runner_eligible"] is True
    assert continuation_targets["tp1_partial_profit_min_pct"] == 0.25
    assert recovery_targets["runner_eligible"] is False


def test_live_runner_distinguishes_tp1_rejection_from_confirmed_extension():
    rejected = evaluate_runner_state(
        setup_family=MOMENTUM_CONTINUATION, tp1=110.0, open_price=109.0,
        high=112.0, low=108.5, close=109.4, relative_volume=1.4,
    )
    confirmed = evaluate_runner_state(
        setup_family=MOMENTUM_CONTINUATION, tp1=110.0, open_price=109.5,
        high=112.0, low=109.2, close=111.2, relative_volume=1.3,
    )

    assert rejected["runner_state"] == "tp1_reached_breakout_rejected"
    assert confirmed["runner_state"] == "trend_extension_runner"


def test_learning_aggregates_and_weights_setup_family_separately():
    stats = aggregate_observations([
        {"setup_type": "pullback", "setup_family": HEALTHY_PULLBACK, "outcome": "TP1_REACHED", "runner_state": "trend_extension_runner", "runner_extension_atr": 1.4},
        {"setup_type": "pullback", "setup_family": MOMENTUM_CONTINUATION, "outcome": "FALSE_BREAKOUT"},
    ])
    same_family = similar_case_score(
        {"setup_type": "pullback", "setup_family": HEALTHY_PULLBACK},
        {"setup_type": "pullback", "setup_family": HEALTHY_PULLBACK},
    )
    other_family = similar_case_score(
        {"setup_type": "pullback", "setup_family": HEALTHY_PULLBACK},
        {"setup_type": "pullback", "setup_family": MOMENTUM_CONTINUATION},
    )

    assert set(stats["setup_family_stats"]) == {HEALTHY_PULLBACK, MOMENTUM_CONTINUATION}
    assert stats["average_runner_extension_atr"] == 1.4
    assert "trend_extension_runner" in stats["runner_state_stats"]
    assert same_family["similarity_score"] > other_family["similarity_score"]
    assert same_family["similarity_contributions"]["setup_family"] == 2.5
