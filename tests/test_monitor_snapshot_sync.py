from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.live_monitor.chart_levels import validate_level_semantics
from app.live_monitor.config import LiveMonitorConfig
from app.live_monitor.service import LiveMonitorService
from app.models import MarketSnapshot, MonitorEvent, MonitorSetup


NOW = datetime.now(timezone.utc).replace(microsecond=0)
CONFIG = LiveMonitorConfig(
    chart_review_on_add=False,
    plan_price_drift_pct=0.03,
    plan_price_drift_atr=1.5,
    source_plan_max_age_minutes=120,
)


def _bars(price: float, timeframe: str) -> list[dict]:
    minutes = {"one_minute": 1, "five_minute": 5, "thirty_minute": 30, "hourly": 60, "daily": 390}[timeframe]
    count = 80 if timeframe == "daily" else 30
    return [
        {
            "date": NOW - timedelta(minutes=(count - index - 1) * minutes),
            "open": price - 0.15,
            "high": price + 0.30,
            "low": price - 0.35,
            "close": price,
            "volume": 100_000 + index * 100,
            "source": "sync-test",
            "bar_complete": True,
        }
        for index in range(count)
    ]


def _plan(price: float, *, planned_at: datetime | None = None) -> dict:
    return {
        "ticker": "INTC",
        "current_price": price,
        "planned_at": planned_at or NOW,
        "primary_entry_trigger": {"price": price + 2.0},
        "near_confirmation": price + 1.0,
        "strong_confirmation": price + 3.0,
        "major_trend_repair": price + 7.0,
        "nearest_support": price - 1.0,
        "invalidation_level": price - 4.0,
        "suggested_stop": price - 3.5,
        "atr": 2.0,
        "take_profit_1": price + 6.0,
        "take_profit_2": price + 8.0,
        "take_profit_3": price + 10.0,
        "raw_setup_score": 8.0,
        "valid_setup": True,
        "setup_type": "healthy_pullback",
    }


@pytest.fixture()
def sync_service(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'sync.db'}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.baseline as baseline_module
    import app.live_monitor.service as service_module

    state = {"price": 93.0, "force_refresh_calls": 0}

    def loader(_ticker, timeframe, _lookback, *, force_refresh=False):
        if force_refresh:
            state["force_refresh_calls"] += 1
        return _bars(float(state["price"]), timeframe)

    def generated_plan(**kwargs):
        return _plan(float(kwargs["current_price"]), planned_at=NOW)

    monkeypatch.setattr(service_module, "SessionLocal", factory)
    monkeypatch.setattr(baseline_module, "generate_structured_plan", generated_plan)
    service = LiveMonitorService(
        config=replace(CONFIG, chart_snapshot_dir=str(tmp_path / "chart_snapshots")),
        bars_loader=loader,
    )
    return service, factory, state


def test_intc_like_stale_scanner_plan_is_replanned_from_fresh_snapshot(sync_service):
    service, factory, state = sync_service
    old = {
        **_plan(102.50, planned_at=NOW - timedelta(hours=3)),
        "primary_entry_trigger": {"price": 107.91},
        "nearest_support": 102.40,
        "take_profit_1": 121.59,
        "take_profit_2": 125.36,
        "take_profit_3": 126.64,
    }
    result = service.add_monitor("INTC", source="best_setups", planner_payload=old)

    assert state["force_refresh_calls"] == 5
    assert result["plan_reference_price"] == pytest.approx(93.0)
    assert result["active_levels"]["primary_entry_trigger"] == pytest.approx(95.0)
    assert result["active_levels"]["primary_entry_trigger"] != pytest.approx(107.91)
    validation = result["planner_baseline"]["source_plan_validation"]
    assert {"PRICE_DRIFT", "ATR_DRIFT", "SUPPORT_FAILED"}.issubset(validation["reasons"])
    assert result["planner_baseline"]["monitor_plan_source"] == "fresh_canonical_replan"
    assert result["market_snapshot_id"] == result["planner_baseline"]["market_snapshot_id"]
    with factory() as db:
        assert db.query(MarketSnapshot).count() == 1
        event = db.query(MonitorEvent).filter(MonitorEvent.event_type == "MONITOR_CREATED_FROM_FRESH_SNAPSHOT").one()
        assert event.setup_id == result["current_setup_id"]


def test_fresh_scanner_plan_can_be_reused_after_snapshot_validation(sync_service):
    service, _factory, _state = sync_service
    result = service.add_monitor("INTC", source="best_setups", planner_payload=_plan(93.0, planned_at=NOW))
    assert result["planner_baseline"]["monitor_plan_source"] == "validated_scanner_context"
    assert result["active_levels"]["primary_entry_trigger"] == pytest.approx(95.0)
    assert result["planner_baseline"]["source_plan_validation"]["fresh"] is True


def test_reanalyze_forces_new_snapshot_and_preserves_old_setup(sync_service):
    service, factory, state = sync_service
    before = service.add_monitor("INTC", planner_payload=_plan(93.0))
    state["price"] = 96.0
    after = service.reanalyze(before["id"])

    assert after["current_setup_id"] != before["current_setup_id"]
    assert after["market_snapshot_id"] != before["market_snapshot_id"]
    assert after["plan_reference_price"] == pytest.approx(96.0)
    assert len(after["setup_history"]) == 2
    with factory() as db:
        old_setup = db.get(MonitorSetup, before["current_setup_id"])
        assert old_setup.status == "replaced"
        assert old_setup.replaced_by_setup_id == after["current_setup_id"]
        assert db.query(MonitorEvent).filter(MonitorEvent.event_type == "OLD_PLAN_REPLACED").count() == 1


def test_planner_and_chart_use_same_persisted_snapshot(sync_service):
    service, _factory, _state = sync_service
    monitor = service.add_monitor("INTC", planner_payload=_plan(93.0))
    chart = service.chart_bundle(monitor["id"])
    assert chart["snapshot_ids_match"] is True
    assert chart["market_snapshot_id"] == monitor["market_snapshot_id"]
    assert chart["reference_price"] == pytest.approx(monitor["plan_reference_price"])


def test_market_data_mismatch_blocks_monitor_creation(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mismatch.db'}")
    factory = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module
    monkeypatch.setattr(service_module, "SessionLocal", factory)

    def mismatch_loader(_ticker, timeframe, _lookback, *, force_refresh=False):
        price = 112.0 if timeframe == "five_minute" else 93.0
        return _bars(price, timeframe)

    service = LiveMonitorService(config=CONFIG, bars_loader=mismatch_loader)
    with pytest.raises(ValueError, match="MARKET_DATA_MISMATCH"):
        service.add_monitor("INTC", planner_payload=_plan(93.0))


def test_stale_runtime_plan_hides_old_levels_and_disables_executable_rr(sync_service):
    service, _factory, _state = sync_service
    monitor = service.add_monitor("INTC", planner_payload=_plan(93.0))
    one = _bars(84.0, "one_minute")
    five = _bars(84.0, "five_minute")
    result = service.evaluate_watch(monitor["id"], bars_1m=one, bars_5m=five, now=NOW)
    assert result["plan_stale"] is True
    assert "PRICE_DRIFT" in result["plan_stale_reasons"]
    assert result["active_levels"] == {}
    assert result["historical_stale_levels"]["primary_entry_trigger"] == pytest.approx(95.0)
    assert result["primary_trigger"] is None
    assert result["current_executable_rr"] is None
    assert result["action_required"] == "REANALYZE_REQUIRED"


def test_support_and_target_semantics_are_machine_readable():
    result = validate_level_semantics(
        current_price=93.0,
        levels={
            "primary_entry_trigger": 95.0,
            "optional_support_level": 102.40,
            "invalidation_level": 89.0,
            "atr": 2.0,
            "tp1": 121.59,
        },
        config=replace(CONFIG, chart_max_target_atr=5.0),
    )
    assert "OLD_SUPPORT_LOST" in result["warnings"]
    assert result["reclassified_levels"]["optional_support_level"] == "RESISTANCE_OR_HISTORICAL_SUPPORT"
    assert "TARGET_REACHABILITY_WARNING" in result["warnings"]
    assert result["target_diagnostics"]["tp1"]["reachable_2_10_days"] is False
