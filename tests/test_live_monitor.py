from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.live_monitor.advisor import build_advisory_packet, review_advisory_packet
from app.live_monitor.config import LiveMonitorConfig
from app.live_monitor.engine import evaluate_monitor
from app.live_monitor.learning import hierarchical_weights
from app.live_monitor.service import LiveMonitorService
from app.models import ConfirmationAttempt, LearningProposal, ManualMonitorTrade, MonitorRuleVersion, RecommendationOutcome, ShadowRuleEvaluation


NOW = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
CONFIG = LiveMonitorConfig(stale_data_seconds=300, auto_llm_min_setup_score=0.0)
LEVELS = {
    "primary_entry_trigger": 100.0,
    "invalidation_level": 94.5,
    "suggested_stop": 95.0,
    "atr": 2.0,
    "tp1": 110.0,
    "tp2": 114.0,
    "tp3": 118.0,
}


def bars(prices: list[float], *, latest_volume: float = 100.0, timeframe_minutes: int = 1, rejection: bool = False):
    output = []
    for index, price in enumerate(prices):
        volume = latest_volume if index == len(prices) - 1 else 100.0
        high = price + 0.08
        low = price - 0.25
        opened = price - 0.1
        close = price
        if rejection and index == len(prices) - 1:
            opened, high, low, close = 100.4, 101.2, 99.5, 99.7
        output.append(
            {
                "date": NOW - timedelta(minutes=(len(prices) - index) * timeframe_minutes) if timeframe_minutes == 5 else NOW - timedelta(minutes=(len(prices) - 1 - index) * timeframe_minutes),
                "open": opened,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return output


def evaluate(previous: str, one: list[dict], five: list[dict], **kwargs):
    return evaluate_monitor(
        previous_state=previous,
        levels=kwargs.pop("levels", LEVELS),
        bars_1m=one,
        bars_5m=five,
        setup_valid=kwargs.pop("setup_valid", True),
        now=NOW,
        config=kwargs.pop("config", CONFIG),
        prior_attempt_count=kwargs.pop("prior_attempt_count", 0),
    )


def test_trigger_sweep_is_rejected_not_approved():
    armed = evaluate("WATCHING", bars([97.8, 98.12, 100.12]), bars([99.2] * 21))
    assert armed["state"] == "ARMED"
    rejected = evaluate("CONFIRMING", bars([97.8, 100.12, 97.6]), bars([99.2] * 21))
    assert rejected["state"] == "REJECTED_BREAKOUT"
    assert rejected["rejection_reason"] == "failed_hold"


def test_price_and_relative_volume_confirm_setup():
    one = bars([99.8] * 20 + [100.4])
    five = bars([99.0] * 20 + [100.5], latest_volume=160.0, timeframe_minutes=5)
    result = evaluate("CONFIRMING", one, five)
    assert result["state"] in {"APPROVED", "STRONGLY_CONFIRMED"}
    assert result["price_confirmation"] is True
    assert result["volume_confirmation"] is True
    assert result["manual_order_plan"]["execution"] == "MANUAL_ONLY"


def test_high_volume_rejection_is_not_bullish():
    one = bars([99.8] * 20 + [99.7])
    five = bars([99.0] * 20 + [99.7], latest_volume=200.0, timeframe_minutes=5, rejection=True)
    result = evaluate("CONFIRMING", one, five)
    assert result["state"] == "REJECTED_BREAKOUT"
    assert result["rejection_reason"] == "high_volume_rejection"


def test_missed_and_invalidated_hard_states():
    missed = evaluate("APPROVED", bars([100.0] * 20 + [102.0]), bars([100.2] * 21, latest_volume=160.0, timeframe_minutes=5))
    assert missed["state"] == "MISSED"
    assert "maximum_chase_exceeded" in missed["hard_blockers"]
    invalid = evaluate("WATCHING", bars([95.0] * 20 + [94.4]), bars([95.0] * 21, timeframe_minutes=5))
    assert invalid["state"] == "INVALIDATED"


def test_stale_data_blocks_new_approval():
    old = NOW - timedelta(minutes=10)
    one = bars([100.4] * 21)
    five = bars([100.5] * 21, latest_volume=200.0, timeframe_minutes=5)
    one[-1]["date"] = old
    five[-1]["date"] = old
    result = evaluate("CONFIRMING", one, five)
    assert result["state"] == "DATA_STALE"
    assert "data_stale" in result["hard_blockers"]


def test_llm_schema_and_hard_gates():
    evaluation = evaluate("CONFIRMING", bars([99.8] * 20 + [100.4]), bars([99.0] * 20 + [100.5], latest_volume=160.0, timeframe_minutes=5))
    packet = build_advisory_packet(baseline={"ticker": "INTC"}, evaluation=evaluation, historical_profile={"evidence_strength": "LOW"}, similar_cases=[])
    review = review_advisory_packet(packet, lambda _packet: {"decision": "APPROVE", "confidence": 0.8, "reason_summary": "Grounded review"})
    assert review["decision"] == "APPROVE"
    blocked_packet = {**packet, "confirmation_evidence": {**evaluation, "hard_blockers": ["setup_invalidated"]}}
    blocked = review_advisory_packet(blocked_packet, lambda _packet: {"decision": "APPROVE", "confidence": 1.0})
    assert blocked["decision"] == "REJECT"


def test_llm_failure_keeps_deterministic_result_available():
    evaluation = evaluate("CONFIRMING", bars([99.8] * 20 + [100.4]), bars([99.0] * 20 + [100.5], latest_volume=160.0, timeframe_minutes=5))
    packet = build_advisory_packet(baseline={"ticker": "INTC"}, evaluation=evaluation, historical_profile={}, similar_cases=[])

    def fail(_packet):
        raise RuntimeError("provider offline")

    review = review_advisory_packet(packet, fail)
    assert review["status"] == "unavailable"
    assert review["decision"] == "WAIT"
    assert evaluation["state"] in {"APPROVED", "STRONGLY_CONFIRMED"}


def test_sparse_and_mature_history_weights_are_sample_sensitive():
    sparse = hierarchical_weights(ticker_samples=3, setup_samples=30, sector_samples=50)
    mature = hierarchical_weights(ticker_samples=60, setup_samples=100, sector_samples=120)
    assert sparse["ticker"] < sparse["global"] + sparse["sector"] + sparse["setup"]
    assert mature["ticker"] > sparse["ticker"]


@pytest.fixture()
def isolated_service(monkeypatch, tmp_path):
    database = tmp_path / "monitor.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module

    monkeypatch.setattr(service_module, "SessionLocal", factory)
    service = LiveMonitorService(config=replace(CONFIG, auto_llm_min_setup_score=0.0), bars_loader=lambda *_args: [])
    return service, factory


def sample_plan():
    return {
        "ticker": "INTC",
        "current_price": 99.0,
        "primary_entry_trigger": {"price": 100.0},
        "invalidation_level": 94.5,
        "suggested_stop": 95.0,
        "atr": 2.0,
        "take_profit_1": 110.0,
        "take_profit_2": 114.0,
        "take_profit_3": 118.0,
        "raw_setup_score": 8.5,
        "broader_structure": "UPTREND",
        "setup_type": "deep_pullback",
        "execution_structure": "base_building",
        "market_regime": "risk_on",
    }


def test_manual_add_persists_and_override_preserves_original(isolated_service):
    service, factory = isolated_service
    added = service.add_monitor("INTC", planner_payload=sample_plan())
    watch_id = added["id"]
    service.edit_levels(watch_id, {"primary_entry_trigger": 100.25})
    detail = service.get_monitor(watch_id)
    assert detail["monitor_active"] is True
    assert detail["active_levels"]["primary_entry_trigger"] == 100.25
    assert detail["planner_levels"]["primary_entry_trigger"] == 100.0
    assert detail["trigger_source"] == "MANUAL"
    restarted = LiveMonitorService(config=CONFIG, bars_loader=lambda *_args: [])
    import app.live_monitor.service as service_module
    service_module.SessionLocal = factory
    assert restarted.list_monitors()[0]["ticker"] == "INTC"


def test_multiple_attempts_are_separate_records(isolated_service):
    service, factory = isolated_service
    watch_id = service.add_monitor("INTC", planner_payload=sample_plan())["id"]
    service.evaluate_watch(watch_id, bars_1m=bars([99.8, 100.12]), bars_5m=bars([99.0] * 21), now=NOW)
    service.evaluate_watch(watch_id, bars_1m=bars([100.12, 97.6]), bars_5m=bars([99.0] * 21), now=NOW)
    service.evaluate_watch(watch_id, bars_1m=bars([99.8, 100.2]), bars_5m=bars([99.0] * 21), now=NOW)
    service.evaluate_watch(watch_id, bars_1m=bars([100.1] * 20 + [100.5]), bars_5m=bars([99.0] * 20 + [100.5], latest_volume=180, timeframe_minutes=5), now=NOW)
    with factory() as db:
        attempts = db.query(ConfirmationAttempt).order_by(ConfirmationAttempt.attempt_number).all()
        assert len(attempts) == 2
        assert attempts[0].outcome == "REJECTED_BREAKOUT"
        assert attempts[1].outcome in {"APPROVED", "STRONGLY_CONFIRMED"}


def test_skipped_recommendation_is_separate_from_execution(isolated_service):
    service, factory = isolated_service
    watch_id = service.add_monitor("INTC", planner_payload=sample_plan())["id"]
    result = service.record_manual_action(watch_id, {"action": "skipped", "notes": "No fill"})
    assert result["status"] == "tracking"
    detail = service.get_monitor(watch_id)
    assert detail["manual_trades"] == []


def test_entered_trade_and_recommendation_outcome_are_separate_records(isolated_service):
    service, factory = isolated_service
    watch_id = service.add_monitor("INTC", planner_payload=sample_plan())["id"]
    result = service.record_manual_action(
        watch_id,
        {"action": "entered", "quantity": 10, "actual_entry": 100.2, "stop_price": 95.0},
    )
    assert result["trade_id"] != result["recommendation_outcome_id"]
    with factory() as db:
        assert db.query(ManualMonitorTrade).count() == 1
        assert db.query(RecommendationOutcome).count() == 1


def test_learning_proposal_requires_user_decision_and_paper_test_is_shadow(isolated_service):
    service, factory = isolated_service
    proposal = service.create_proposal({"scope_type": "ticker", "scope_value": "INTC", "title": "Require retest", "proposed_change": {"retest_weight": 1.2}, "evidence": {"samples": 24}})
    assert proposal["status"] == "PENDING"
    paper = service.decide_proposal(proposal["id"], decision="PAPER_TEST")
    assert paper["status"] == "PAPER_TESTING"
    with factory() as db:
        assert db.query(ShadowRuleEvaluation).count() == 1
        assert db.query(MonitorRuleVersion).count() == 0


def test_user_approved_proposal_creates_new_rule_version(isolated_service):
    service, factory = isolated_service
    proposal = service.create_proposal({"scope_type": "ticker", "scope_value": "INTC", "title": "Retest weight", "proposed_change": {"retest_weight": 1.2}})
    approved = service.decide_proposal(proposal["id"], decision="APPROVE")
    assert approved["status"] == "APPROVED"
    with factory() as db:
        assert db.query(MonitorRuleVersion).count() == 1
        assert db.query(LearningProposal).first().status == "APPROVED"
