from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.live_monitor.chart_data import build_chart_bundle
from app.live_monitor.chart_levels import (
    check_data_consistency,
    derive_chart_level_candidates,
    detect_stale_plan,
    validate_chart_levels,
)
from app.live_monitor.chart_review import review_chart_packet
from app.live_monitor.config import LiveMonitorConfig
from app.live_monitor.engine import evaluate_monitor
from app.live_monitor.service import LiveMonitorService
from app.models import ChartSnapshot, DailyBar
from dashboard.lightweight_chart import LIGHTWEIGHT_CHARTS_VERSION, build_lightweight_chart_html


NOW = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
CONFIG = LiveMonitorConfig(chart_review_on_add=False, stale_data_seconds=600)


def reaction_bars(*, start: datetime = NOW - timedelta(hours=20)) -> list[dict]:
    closes = [92.0, 92.5, 93.1, 94.6, 93.3, 92.8, 93.4, 95.4, 93.8, 93.1, 93.7, 94.8, 93.9]
    return [
        {
            "date": start + timedelta(minutes=30 * index),
            "open": close - 0.15,
            "high": close + 0.25,
            "low": close - 0.35,
            "close": close,
            "volume": 100_000 + index * 2_000,
            "source": "test",
        }
        for index, close in enumerate(closes)
    ]


def planner_levels() -> dict:
    return {
        "primary_entry_trigger": 107.91,
        "invalidation_level": 89.5,
        "suggested_stop": 90.0,
        "optional_support_level": 92.0,
        "atr": 2.0,
        "tp1": 112.0,
        "tp2": 116.0,
        "tp3": 120.0,
    }


def test_intc_like_distant_trigger_is_reclassified_as_major_repair():
    candidates = derive_chart_level_candidates(
        current_price=93.0,
        atr=2.0,
        planner_levels=planner_levels(),
        structure_bars=reaction_bars(),
        execution_bars=reaction_bars(start=NOW - timedelta(hours=8)),
        config=CONFIG,
    )
    assert candidates["planner_primary_reclassified_as_major_repair"] is True
    assert candidates["major_trend_repair"] == pytest.approx(107.91)
    assert 93.0 < candidates["primary_entry_trigger"] < 100.0


def test_planner_chart_agreement_remains_supported():
    levels = {**planner_levels(), "primary_entry_trigger": 95.65, "major_trend_repair": 107.91}
    candidates = derive_chart_level_candidates(
        current_price=93.0,
        atr=2.0,
        planner_levels=levels,
        structure_bars=reaction_bars(),
        execution_bars=[],
        config=CONFIG,
    )
    validation = validate_chart_levels(
        current_price=93.0,
        atr=2.0,
        proposed_levels={"primary_entry_trigger": candidates["primary_entry_trigger"]},
        planner_levels=levels,
        candidate_evidence=candidates,
        structure_bars=reaction_bars(),
    )
    assert validation["accepted_levels"]["primary_entry_trigger"] == pytest.approx(candidates["primary_entry_trigger"])


def test_unsupported_llm_level_is_rejected():
    candidates = derive_chart_level_candidates(
        current_price=93.0,
        atr=2.0,
        planner_levels=planner_levels(),
        structure_bars=reaction_bars(),
        execution_bars=[],
        config=CONFIG,
    )
    validation = validate_chart_levels(
        current_price=93.0,
        atr=2.0,
        proposed_levels={"primary_entry_trigger": 99.75},
        planner_levels=planner_levels(),
        candidate_evidence=candidates,
        structure_bars=reaction_bars(),
    )
    assert "primary_entry_trigger" in validation["rejected_levels"]
    assert "no_pivot_or_reaction_evidence" in validation["rejected_levels"]["primary_entry_trigger"]


def test_invalid_target_order_is_rejected():
    levels = {**planner_levels(), "primary_entry_trigger": 95.65}
    candidates = derive_chart_level_candidates(
        current_price=93.0,
        atr=2.0,
        planner_levels=levels,
        structure_bars=reaction_bars(),
        execution_bars=[],
        config=CONFIG,
    )
    validation = validate_chart_levels(
        current_price=93.0,
        atr=2.0,
        proposed_levels={"primary_entry_trigger": 95.65, "tp1": 95.0, "tp2": 94.0},
        planner_levels=levels,
        candidate_evidence=candidates,
        structure_bars=reaction_bars(),
    )
    assert "tp1" in validation["rejected_levels"]
    assert "tp2" in validation["rejected_levels"]


def test_stale_plan_detects_failed_support_and_distant_trigger():
    stale = detect_stale_plan(
        current_price=90.0,
        levels=planner_levels(),
        atr=2.0,
        setup_created_at=NOW - timedelta(days=2),
        structure_bars=reaction_bars(),
        plan_reference_price=102.5,
        config=CONFIG,
    )
    assert stale["status"] == "PLAN_STALE"
    assert "SUPPORT_FAILED" in stale["reasons"]
    assert "PRICE_DRIFT" in stale["reasons"]
    assert "PRIMARY_TRIGGER_SANITY_WARNING" in stale["warnings"]


def _monitor_bars(prices: list[float], *, minutes: int) -> list[dict]:
    return [
        {
            "date": NOW - timedelta(minutes=(len(prices) - index) * minutes),
            "open": price - 0.1,
            "high": price + 0.1,
            "low": price - 0.3,
            "close": price,
            "volume": 100.0 if index < len(prices) - 1 else 170.0,
            "bar_complete": True,
        }
        for index, price in enumerate(prices)
    ]


def test_rr_is_planned_before_trigger_and_executable_only_after_confirmation():
    levels = {
        "primary_entry_trigger": 100.0,
        "invalidation_level": 94.5,
        "suggested_stop": 95.0,
        "atr": 2.0,
        "tp1": 110.0,
    }
    waiting = evaluate_monitor(
        previous_state="WATCHING",
        levels=levels,
        bars_1m=_monitor_bars([98.0] * 21, minutes=1),
        bars_5m=_monitor_bars([98.0] * 21, minutes=5),
        setup_valid=True,
        now=NOW,
        config=CONFIG,
    )
    assert waiting["planned_rr_at_primary_trigger"] == pytest.approx(2.0)
    assert waiting["current_executable_rr"] is None
    assert waiting["manual_order_plan"] is None
    confirmed = evaluate_monitor(
        previous_state="CONFIRMING",
        levels=levels,
        bars_1m=_monitor_bars([99.8] * 20 + [100.4], minutes=1),
        bars_5m=_monitor_bars([99.0] * 20 + [100.5], minutes=5),
        setup_valid=True,
        now=NOW,
        config=CONFIG,
    )
    assert confirmed["state"] in {"APPROVED", "STRONGLY_CONFIRMED"}
    assert confirmed["current_executable_rr"] is not None
    assert confirmed["manual_order_plan"]["execution"] == "MANUAL_ONLY"


def test_chart_data_mismatch_blocks_automated_recommendation():
    mismatch = check_data_consistency(planner_price=100.0, monitor_price=100.2, chart_close=112.0, atr=2.0)
    assert mismatch["status"] == "CHART_DATA_MISMATCH"


def test_chart_bundle_never_contains_future_candles(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'chart.db'}")
    factory = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    with factory() as db:
        db.add(
            DailyBar(
                symbol="INTC",
                bar_date=date(2026, 8, 18),
                open=92.0,
                high=94.0,
                low=91.5,
                close=93.0,
                volume=1_000_000,
                source="test",
            )
        )
        db.commit()

        requested_timeframes: list[str] = []

        def loader(_ticker, timeframe, _lookback):
            requested_timeframes.append(timeframe)
            return [
                {"date": NOW - timedelta(minutes=5), "open": 92, "high": 94, "low": 91, "close": 93, "volume": 100, "source": "test"},
                {"date": NOW + timedelta(minutes=5), "open": 93, "high": 96, "low": 92, "close": 95, "volume": 200, "source": "test"},
            ] if timeframe != "daily" else []

        bundle = build_chart_bundle(
            db,
            ticker="INTC",
            levels=planner_levels(),
            level_sources={},
            bars_loader=loader,
            decision_time_boundary=NOW,
            max_bars=180,
        )
    assert "hourly" in requested_timeframes
    assert "one_hour" not in requested_timeframes
    for payload in bundle["timeframes"].values():
        assert all(datetime.fromisoformat(bar["time"]) <= NOW for bar in payload["bars"])


def test_chart_llm_failure_returns_deterministic_fallback():
    packet = {
        "review_type": "CHART_STRUCTURE_REVIEW",
        "ticker": "INTC",
        "current_price": 93.0,
        "atr": 2.0,
        "planner_levels": planner_levels(),
        "structure_bars": reaction_bars(),
        "execution_bars": [],
        "image_paths": ["missing.png"],
    }

    def fail(_packet):
        raise RuntimeError("image provider unavailable")

    result = review_chart_packet(packet, provider=fail, config=CONFIG)
    assert result["status"] == "UNAVAILABLE"
    assert result["provider_error"].startswith("RuntimeError")
    assert result["proposed_levels"]["major_trend_repair"] == pytest.approx(107.91)


def test_lightweight_chart_uses_api_payload_and_current_v5_bundle():
    timeframe = {
        "bars": [
            {"timestamp": 1_700_000_000, "open": 92, "high": 94, "low": 91, "close": 93, "volume": 100},
        ],
        "indicators": {},
    }
    markup = build_lightweight_chart_html(
        ticker="INTC",
        title="Execution",
        timeframe_payload=timeframe,
        levels=[{"name": "primary_entry_trigger", "label": "Primary", "price": 95.0, "source": "PLANNER", "color": "#fff"}],
    )
    assert f"lightweight-charts@{LIGHTWEIGHT_CHARTS_VERSION}" in markup
    assert "addSeries(LightweightCharts.CandlestickSeries" in markup
    assert "Canonical API OHLCV" in markup


def test_chart_levels_remain_proposals_until_user_accepts(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'service.db'}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module

    monkeypatch.setattr(service_module, "SessionLocal", factory)

    def provider(_packet):
        return {
            "chart_assessment": {
                "broader_structure": "repair",
                "setup_type": "reversal_attempt",
                "execution_structure": "local_base",
                "setup_quality": "conditional",
                "setup_stale": True,
            },
            "levels": {
                "near_confirmation": {"price": 94.85, "reason": "local reaction"},
                "primary_entry_trigger": {"price": 95.65, "reason": "local pivot"},
                "strong_confirmation": {"price": None, "reason": "not clean"},
                "major_trend_repair": {"price": 107.91, "reason": "old structural trigger"},
                "invalidation_level": {"price": 89.5, "reason": "planner invalidation"},
            },
            "targets": {
                "tp1": {"price": 112.0, "reason": "planner target"},
                "tp2": {"price": 116.0, "reason": "planner target"},
                "tp3": {"price": 120.0, "reason": "planner target"},
            },
            "planner_comparison": {
                "agrees_with_primary_trigger": False,
                "planner_trigger_issue": "old trigger is major repair",
                "recommended_action": "MODIFY_LEVELS",
            },
            "decision": "MODIFY_LEVELS",
            "confidence": 0.8,
            "positive_factors": ["local pivot evidence"],
            "risk_factors": ["broader repair incomplete"],
            "reason_summary": "Use the nearby pivot as primary and retain the old trigger as major repair.",
        }

    service = LiveMonitorService(
        config=replace(CONFIG, chart_review_on_add=False, chart_snapshot_dir=str(tmp_path / "snapshots")),
        bars_loader=lambda *_args: reaction_bars(start=datetime.now(timezone.utc) - timedelta(hours=6)),
        chart_review_provider=provider,
    )
    plan = {
        "ticker": "INTC",
        "current_price": 93.0,
        "primary_entry_trigger": {"price": 107.91},
        "invalidation_level": 89.5,
        "suggested_stop": 90.0,
        "atr": 2.0,
        "take_profit_1": 112.0,
        "take_profit_2": 116.0,
        "take_profit_3": 120.0,
        "raw_setup_score": 7.0,
        "setup_type": "reversal_attempt",
    }
    added = service.add_monitor("INTC", planner_payload=plan)
    watch_id = added["id"]
    review = service.run_chart_review(watch_id)
    before = service.get_monitor(watch_id)
    assert review["status"] == "DISAGREEMENT"
    assert before["active_levels"]["primary_entry_trigger"] == pytest.approx(107.91)
    assert before["validated_chart_levels"]["primary_entry_trigger"] == pytest.approx(95.65)
    with factory() as db:
        snapshots = db.query(ChartSnapshot).all()
        assert {snapshot.timeframe for snapshot in snapshots} == {"daily", "structure", "execution"}
        assert all(snapshot.image_data_base64 for snapshot in snapshots)
        assert all(snapshot.market_snapshot_id == before["market_snapshot_id"] for snapshot in snapshots)
        assert all(
            (snapshot.decision_time_boundary if snapshot.decision_time_boundary.tzinfo else snapshot.decision_time_boundary.replace(tzinfo=timezone.utc))
            <= datetime.now(timezone.utc)
            for snapshot in snapshots
        )
    accepted = service.apply_chart_level_decision(watch_id, decision="ACCEPT_VALIDATED")
    assert accepted["active_levels"]["primary_entry_trigger"] == pytest.approx(95.65)
    assert accepted["active_levels"]["major_trend_repair"] == pytest.approx(107.91)
    assert accepted["level_sources"]["primary_entry_trigger"] == "VALIDATED_CHART_LLM"


def test_backend_evaluates_without_dashboard_and_resume_does_not_fetch(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'continuous.db'}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module

    monkeypatch.setattr(service_module, "SessionLocal", factory)

    def loader(_ticker, timeframe, _lookback):
        now = datetime.now(timezone.utc)
        minutes = 1 if timeframe == "one_minute" else 5
        return [
            {
                "date": now - timedelta(minutes=(20 - index) * minutes),
                "open": 97.9,
                "high": 98.2,
                "low": 97.7,
                "close": 98.0,
                "volume": 100.0,
                "bar_complete": True,
            }
            for index in range(21)
        ]

    service = LiveMonitorService(
        config=replace(CONFIG, chart_review_on_add=False),
        bars_loader=loader,
    )
    plan = {
        "ticker": "TEST",
        "current_price": 98.0,
        "primary_entry_trigger": {"price": 100.0},
        "invalidation_level": 94.5,
        "suggested_stop": 95.0,
        "atr": 2.0,
        "take_profit_1": 110.0,
    }
    watch_id = service.add_monitor("TEST", planner_payload=plan)["id"]
    service.evaluate_all()
    evaluated = service.get_monitor(watch_id)
    assert evaluated["last_polled_at"] is not None
    last_polled = evaluated["last_polled_at"]
    service.control(watch_id, "pause")
    service.control(watch_id, "resume")
    resumed = service.get_monitor(watch_id)
    assert resumed["state"] == "WATCHING"
    assert resumed["last_polled_at"] == last_polled
