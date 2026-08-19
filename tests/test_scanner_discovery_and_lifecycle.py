from types import SimpleNamespace

import pandas as pd

from app.candidate_discovery import (
    build_sector_aware_candidate_order,
    classify_best_setup_quality,
    classify_search_exhaustiveness,
    run_adaptive_batches,
    validate_sp500_universe,
)
from app.chart_context import derive_structure_layers
from app.config import DEFAULT_PLANNING_CONFIG
from app.confirmation import build_confirmation_plan
from app.opportunity_ranking import build_daily_actionability_profile
from app.setup_lifecycle import build_setup_lifecycle
from app.universe import _load_fallback


def _actionability_row(*, current: float, trigger: float, tp1: float, confirmed: bool):
    return SimpleNamespace(
        current_price=current,
        last=current,
        preferred_entry=100.0,
        preferred_entry_low=99.0,
        preferred_entry_high=101.0,
        confirmation_trigger_price=trigger,
        stop_loss=94.0,
        invalidation_level=93.0,
        take_profit_1=tp1,
        atr=2.0,
        entry_status="confirmed" if confirmed else "awaiting_confirmation",
        confirmation_state="confirmed" if confirmed else "awaiting_confirmation",
        price_confirmed=confirmed,
        volume_confirmed=confirmed,
        executable_stop_technically_valid=True,
        confirmation_score=9.0 if confirmed else 5.0,
        hold_window_reachability_score=8.0,
        volume_confirmation_score=9.0 if confirmed else 4.0,
        liquidity_score=9.0,
        sector_relative_strength=0.02,
    )


def test_fallback_is_a_broad_multi_sector_sp500_universe():
    snapshot = _load_fallback()
    validation = validate_sp500_universe(
        universe_size=len(snapshot.tickers),
        sector_filter=None,
        industry_filter=None,
        minimum_broad_size=450,
    )

    assert len(snapshot.tickers) >= 500
    assert validation["valid"] is True
    assert len({item["sector"] for item in snapshot.metadata_by_ticker.values()}) >= 10


def test_suspiciously_small_broad_universe_is_reported_not_silently_accepted():
    validation = validate_sp500_universe(
        universe_size=100,
        sector_filter=None,
        industry_filter=None,
        minimum_broad_size=450,
    )

    assert validation["status"] == "UNIVERSE_VALIDATION_FAILED"
    assert "100 symbols" in validation["warning"]


def test_sector_aware_discovery_reserves_non_tech_deep_analysis_access():
    ranked = [
        {"ticker": f"T{index}", "pre_scan_score": 10.0 - index / 100.0}
        for index in range(20)
    ] + [
        {"ticker": "JPM", "pre_scan_score": 7.0},
        {"ticker": "CAT", "pre_scan_score": 6.9},
        {"ticker": "LLY", "pre_scan_score": 6.8},
    ]
    metadata = {
        **{item["ticker"]: {"sector": "Information Technology"} for item in ranked[:20]},
        "JPM": {"sector": "Financials"},
        "CAT": {"sector": "Industrials"},
        "LLY": {"sector": "Health Care"},
    }

    ordered = build_sector_aware_candidate_order(
        ranked,
        metadata_by_ticker=metadata,
        initial_limit=8,
        min_per_sector=1,
    )

    assert {item["ticker"] for item in ordered[:8]} >= {"JPM", "CAT", "LLY"}
    # Discovery access changes, but the original scores are never modified.
    assert next(item for item in ordered if item["ticker"] == "JPM")["pre_scan_score"] == 7.0


def test_adaptive_analysis_expands_until_strict_target_without_threshold_changes():
    candidates = [{"ticker": f"S{index}"} for index in range(12)]

    def analyze(batch):
        return [{"ticker": item["ticker"], "strict_actionable": item["ticker"] == "S7"} for item in batch]

    rows, history = run_adaptive_batches(
        candidates,
        initial_limit=4,
        batch_size=2,
        maximum_limit=10,
        target_actionable=1,
        adaptive=True,
        analyze_batch=analyze,
        count_actionable=lambda values: sum(bool(item["strict_actionable"]) for item in values),
    )

    assert len(rows) == 8
    assert [item["deep_analyzed"] for item in history] == [4, 6, 8]
    assert history[-1]["actionable_count"] == 1


def test_zero_actionable_expands_to_configured_max_and_reports_exhaustiveness():
    candidates = [{"ticker": f"S{index}"} for index in range(10)]
    rows, history = run_adaptive_batches(
        candidates,
        initial_limit=4,
        batch_size=3,
        maximum_limit=10,
        target_actionable=1,
        adaptive=True,
        analyze_batch=lambda batch: list(batch),
        count_actionable=lambda _rows: 0,
    )

    assert len(rows) == 10
    assert history[-1]["actionable_count"] == 0
    assert classify_search_exhaustiveness(analyzed=10, viable=10, initial_limit=4, maximum_limit=10) == "exhaustive"


def test_price_below_unreached_trigger_is_not_missed_even_with_weak_current_rr():
    row = _actionability_row(current=100.0, trigger=107.0, tp1=101.0, confirmed=False)
    profile = build_daily_actionability_profile(row, market_regime="neutral")

    assert profile["actionability_state"] == "awaiting_confirmation"
    assert "poor_current_rr" in profile["actionability_negative"]


def test_price_materially_beyond_confirmed_entry_with_poor_rr_is_missed():
    row = _actionability_row(current=107.0, trigger=100.0, tp1=108.0, confirmed=True)
    profile = build_daily_actionability_profile(row, market_regime="neutral")

    assert profile["actionability_state"] in {"missed", "extended"}
    assert profile["actionability_penalties"]


def test_actionability_penalties_preserve_score_differentiation():
    mildly_late = build_daily_actionability_profile(
        _actionability_row(current=104.0, trigger=100.0, tp1=111.0, confirmed=True),
        market_regime="neutral",
    )
    severely_late = build_daily_actionability_profile(
        _actionability_row(current=109.0, trigger=100.0, tp1=110.0, confirmed=True),
        market_regime="neutral",
    )

    assert mildly_late["actionability_raw"] > 0
    assert severely_late["actionability_raw"] > 0
    assert mildly_late["actionability_score"] != severely_late["actionability_score"]


def test_tiered_confirmation_keeps_major_repair_out_of_primary_trigger():
    frame = pd.DataFrame(
        {
            "high": [99.0, 100.0, 101.0],
            "low": [96.0, 97.0, 98.0],
            "close": [98.0, 99.0, 100.0],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    result = build_confirmation_plan(
        current_price=100.0,
        preferred_entry=100.0,
        support_zone_1={"lower": 99.0, "upper": 101.0},
        resistance_zone_1={"lower": 104.0, "upper": 106.0},
        moving_averages={"ema20": 100.5, "ema50": 112.0, "ema100": 118.0, "ema200": 125.0},
        structure_state="trend_damage",
        frame=frame,
        atr=2.0,
        invalidation_level=96.0,
        volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "no_confirmation"},
        requires_confirmation=True,
        config=DEFAULT_PLANNING_CONFIG,
    )

    assert result["primary_entry_trigger"]["price"] < result["major_trend_repair"]["price"]
    assert result["confirmation_trigger_price"] == result["primary_entry_trigger"]["price"]


def test_three_structure_layers_preserve_broader_trend_and_execution_base():
    layers = derive_structure_layers(
        {
            "daily": {"available": True, "trend": "uptrend", "structure_state": "healthy_pullback"},
            "four_hour": {"available": True, "trend": "weak_breakdown_risk", "structure_state": "deep_pullback"},
            "hourly": {"available": True, "trend": "range", "structure_state": "reversal_attempt"},
            "thirty_minute": {"available": True, "trend": "range", "structure_state": "base_building", "compression_state": "compressed"},
        }
    )

    assert layers == {
        "broader_structure": "uptrend",
        "setup_type_layer": "deep_pullback",
        "execution_structure": "base_building",
    }


def test_broken_prior_setup_is_invalidated_and_replaced_by_new_signature():
    result = build_setup_lifecycle(
        ticker="AVGO",
        current_price=95.0,
        structure_state="reversal_attempt",
        entry_status="awaiting_confirmation",
        invalidation_level=91.0,
        primary_trigger=99.0,
        previous_setup={
            "setup_id": "AVGO-old",
            "setup_created_at": "2026-01-01T00:00:00+00:00",
            "invalidation_level": 100.0,
            "primary_entry_trigger": {"price": 105.0},
        },
    )

    assert result["replaced_setup"]["setup_status"] == "invalidated"
    assert result["replaced_setup"]["prior_primary_trigger"] == {"price": 105.0}
    assert result["setup_id"] != "AVGO-old"
    assert result["setup_status"] == "awaiting_confirmation"


def test_scan_quality_labels_best_of_weak_scan_without_calling_it_strong():
    assert classify_best_setup_quality([{"grade": "B", "raw_setup_score": 6.7}]) == "weak_scan"
    assert classify_best_setup_quality([]) == "no_quality_setups"
