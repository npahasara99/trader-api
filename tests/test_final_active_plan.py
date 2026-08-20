from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.live_monitor.chart_levels import reconcile_levels
from app.live_monitor.config import LiveMonitorConfig
from app.live_monitor.final_plan import finalize_active_plan, validate_final_plan
from app.live_monitor.service import LiveMonitorService


NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)
CONFIG = LiveMonitorConfig(chart_review_on_add=False, stale_data_seconds=600, minimum_current_rr=0.05)


def _bars(price: float, *, minutes: int, latest_volume: float = 200.0) -> list[dict]:
    return [
        {
            "date": NOW - timedelta(minutes=(20 - index) * minutes),
            "open": price - 0.15,
            "high": price + 0.10,
            "low": price - 0.25,
            "close": price,
            "volume": latest_volume if index == 20 else 100.0,
            "bar_complete": True,
            "source": "final-plan-test",
        }
        for index in range(21)
    ]


def _plan(*, entry: float = 100.0, tp1: float = 102.0) -> dict:
    return {
        "ticker": "INTC",
        "current_price": 99.0,
        "primary_entry_trigger": {"price": entry},
        "invalidation_level": 94.0,
        "suggested_stop": 95.0,
        "atr": 2.0,
        "take_profit_1": tp1,
        "take_profit_2": 104.0,
        "take_profit_3": 106.0,
        "major_trend_repair": 108.0,
        "raw_setup_score": 8.0,
        "valid_setup": True,
        "setup_type": "healthy_pullback",
    }


def test_intc_geometry_with_tp1_below_entry_is_hard_invalid():
    result = validate_final_plan(
        levels={
            "primary_entry_trigger": 99.62,
            "invalidation_level": 87.83,
            "suggested_stop": 87.83,
            "tp1": 99.00,
            "tp2": 100.80,
            "tp3": 108.12,
            "major_trend_repair": 102.34,
            "atr": 2.0,
        },
        current_price=93.0,
        market_snapshot_id="snapshot-intc",
        config=CONFIG,
    )
    assert result["status"] == "INVALID"
    assert result["code"] == "PLAN_GEOMETRY_INVALID"
    assert "TP1_NOT_ABOVE_ENTRY" in {row["code"] for row in result["hard_failures"]}


def test_entry_change_regenerates_targets_and_never_keeps_old_tp1_below_entry():
    result = finalize_active_plan(
        setup_id="setup-1",
        levels={
            "primary_entry_trigger": 103.0,
            "invalidation_level": 94.0,
            "suggested_stop": 95.0,
            "atr": 2.0,
            "tp1": 102.0,
            "tp2": 104.0,
            "tp3": 106.0,
            "stretch_target": 108.0,
        },
        sources={"primary_entry_trigger": "VALIDATED_CHART_LLM", "tp1": "PLANNER", "tp2": "PLANNER", "tp3": "PLANNER"},
        current_price=99.0,
        market_snapshot_id="snapshot-1",
        config=CONFIG,
        reconciliation_status="LLM_CORRECTION_ACCEPTED",
        entry_changed=True,
        change_source="VALIDATED_CHART_LLM",
    )
    assert result["target_regeneration"]["status"] == "REGENERATED"
    assert result["flat_levels"]["tp1"] == pytest.approx(104.0)
    assert result["flat_levels"]["tp1"] > result["flat_levels"]["primary_entry_trigger"]
    assert result["level_sources"]["tp1"] == "VALIDATED_CHART_LLM"


def test_rejected_llm_primary_requires_review_instead_of_silent_planner_fallback():
    result = reconcile_levels(
        planner_levels={"primary_entry_trigger": 100.0, "invalidation_level": 94.0, "suggested_stop": 95.0, "tp1": 105.0},
        proposed_levels={"primary_entry_trigger": 95.0},
        validation={
            "accepted_levels": {},
            "rejected_levels": {"primary_entry_trigger": ["no_pivot_or_reaction_evidence"]},
        },
        manual_overrides={},
    )
    assert result["final_active_levels"]["primary_entry_trigger"] == pytest.approx(100.0)
    assert result["status"] == "MANUAL_REVIEW_REQUIRED"
    assert result["reconciliation_status"] == "MANUAL_REVIEW_REQUIRED"
    assert result["activation_blocked"] is True
    assert result["rejected_level_disagreements"]["primary_entry_trigger"]["llm_proposed_level"] == pytest.approx(95.0)


@pytest.fixture()
def final_plan_service(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'final-plan.sqlite'}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module

    monkeypatch.setattr(service_module, "SessionLocal", factory)

    def loader(_ticker, timeframe, _lookback, **_kwargs):
        minutes = {"one_minute": 1, "five_minute": 5, "thirty_minute": 30, "hourly": 60, "daily": 390}.get(timeframe, 5)
        return _bars(99.0, minutes=minutes)

    return LiveMonitorService(
        config=replace(
            CONFIG,
            chart_snapshot_dir=str(tmp_path / "snapshots"),
            plan_price_drift_pct=0.10,
            plan_price_drift_atr=10.0,
        ),
        bars_loader=loader,
    )


def test_invalid_planner_geometry_never_enters_active_monitor_state(final_plan_service):
    detail = final_plan_service.add_monitor("INTC", planner_payload=_plan(entry=100.0, tp1=99.0))
    assert detail["state"] == "PLAN_GEOMETRY_INVALID"
    assert detail["plan_integrity_status"] == "INVALID"
    assert detail["final_active_plan"]["validation"]["activation_allowed"] is False


def test_unexpected_chart_review_failure_is_persisted_not_silenced(final_plan_service):
    detail = final_plan_service.add_monitor("INTC", planner_payload=_plan())
    final_plan_service._record_chart_review_failure(
        detail["id"], review_type="CHART_LEVEL_REVIEW", error=RuntimeError("provider transport failed")
    )
    refreshed = final_plan_service.get_monitor(detail["id"])
    assert refreshed["chart_analysis_status"] == "VALIDATION_FAILED"
    assert refreshed["chart_reviews"][0]["status"] == "VALIDATION_FAILED"
    assert refreshed["chart_reviews"][0]["model"] == CONFIG.chart_review_model
    assert "provider transport failed" in refreshed["chart_reviews"][0]["reason_summary"]


def test_manual_entry_change_shares_final_plan_id_across_chart_rr_and_order(final_plan_service):
    detail = final_plan_service.add_monitor("INTC", planner_payload=_plan())
    updated = final_plan_service.edit_levels(detail["id"], {"primary_entry_trigger": 103.0})
    plan_id = updated["final_active_plan_id"]
    assert updated["active_levels"]["tp1"] == pytest.approx(104.0)
    assert updated["level_sources"]["primary_entry_trigger"] == "MANUAL"
    chart = final_plan_service.chart_bundle(detail["id"])
    assert chart["final_active_plan_id"] == plan_id
    assert next(row for row in chart["levels"] if row["name"] == "primary_entry_trigger")["source"] == "MANUAL"

    one = _bars(99.0, minutes=1)
    five = _bars(99.0, minutes=5)
    for prior in (one[-2], five[-2]):
        prior.update({"open": 102.65, "high": 102.90, "low": 102.55, "close": 102.80, "volume": 120.0})
    for latest in (one[-1], five[-1]):
        latest.update({"open": 103.05, "high": 103.30, "low": 102.95, "close": 103.20, "volume": 200.0})
    evaluated = final_plan_service.evaluate_watch(detail["id"], bars_1m=one, bars_5m=five, now=NOW)
    assert evaluated["latest_evaluation"]["final_active_plan_id"] == plan_id
    manual_plan = evaluated["latest_evaluation"].get("manual_order_plan")
    assert manual_plan is not None, evaluated["latest_evaluation"]
    assert manual_plan["final_active_plan_id"] == plan_id
