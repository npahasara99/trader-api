from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.live_monitor.config import LiveMonitorConfig
from app.live_monitor.learning import derive_bounded_adjustments, evidence_strength
from app.live_monitor.level_sanity import can_auto_apply_chart_correction, evaluate_level_sanity
from app.live_monitor.memory import data_quality_flags, load_historical_context, persist_adjustments, refresh_profile
from app.live_monitor.service import LiveMonitorService
from app.models import (
    ConfirmationAttempt,
    LearnedAdjustment,
    LearningProposal,
    LevelRevision,
    LiveWatch,
    MonitorBarSummary,
    MonitorDailySummary,
    MonitorDecisionSnapshot,
    MonitorSetup,
    RecommendationOutcome,
    ShadowRuleEvaluation,
)


NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)
CONFIG = LiveMonitorConfig(
    chart_review_on_add=False,
    stale_data_seconds=900,
    auto_llm_min_setup_score=0.0,
)


def bars(prices: list[float], minutes: int = 5, *, volumes: list[float] | None = None) -> list[dict]:
    output = []
    for index, price in enumerate(prices):
        output.append({
            "date": NOW - timedelta(minutes=(len(prices) - 1 - index) * minutes),
            "open": price - 0.10,
            "high": price + 0.25,
            "low": price - 0.25,
            "close": price,
            "volume": (volumes or [100.0] * len(prices))[index],
        })
    return output


def reaction_bars(minutes: int = 5) -> list[dict]:
    prices = [100.0, 100.15, 100.75, 100.10, 99.95] * 8
    return bars(prices, minutes)


def sample_plan(primary: float = 101.0) -> dict:
    return {
        "ticker": "INTC",
        "current_price": 100.0,
        "primary_entry_trigger": {"price": primary},
        "near_confirmation": 100.5,
        "strong_confirmation": 102.0,
        "major_trend_repair": 108.0,
        "invalidation_level": 96.0,
        "suggested_stop": 96.5,
        "optional_support_level": 98.5,
        "atr": 2.0,
        "atr_pct": 0.02,
        "rsi": 52.0,
        "take_profit_1": 103.0,
        "take_profit_2": 105.0,
        "take_profit_3": 108.0,
        "raw_setup_score": 8.0,
        "broader_structure": "UPTREND",
        "setup_type": "deep_pullback",
        "execution_structure": "base_building",
        "sector": "Technology",
        "market_regime": "risk_on",
    }


@pytest.fixture()
def memory_service(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    import app.live_monitor.service as service_module

    monkeypatch.setattr(service_module, "SessionLocal", factory)

    def loader(_ticker, timeframe, _lookback, **_kwargs):
        minutes = {
            "one_minute": 1, "five_minute": 5, "thirty_minute": 30,
            "hourly": 60, "daily": 390,
        }.get(timeframe, 5)
        return reaction_bars(minutes)

    service = LiveMonitorService(
        config=replace(CONFIG, chart_snapshot_dir=str(tmp_path / "charts")),
        bars_loader=loader,
    )
    return service, factory, tmp_path


def test_cold_start_uses_broader_priors_without_ticker_adjustment(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan())
    with factory() as db:
        setup = db.get(MonitorSetup, detail["current_setup_id"])
        context = load_historical_context(db, setup, service.config)
        assert context["ticker_profile"]["evidence_strength"] == "INSUFFICIENT"
        assert context["hierarchical_weights"]["ticker"] == 0.0
        assert db.query(LearnedAdjustment).count() == 0


def _seed_mature_confirmation_history(factory, setup_id: str, watch_id: str):
    with factory() as db:
        for index in range(36):
            retest = index < 18
            attempt_id = str(uuid.uuid4())
            attempt = ConfirmationAttempt(
                id=attempt_id, watch_id=watch_id, setup_id=setup_id, ticker="INTC",
                attempt_number=index + 1,
                started_at=NOW - timedelta(days=index % 10, minutes=10),
                ended_at=NOW - timedelta(days=index % 10),
                trigger_price=101.0, peak_price=104.0 if retest else 101.3,
                lowest_retest_price=100.4 if retest else 99.5,
                rvol_1m=1.4 if retest else 0.9, rvol_5m=1.5 if retest else 0.8,
                price_confirmation=True, volume_confirmation=retest,
                retest_result="HELD" if retest else "NOT_TESTED",
                confirmation_method="BREAK_RETEST" if retest else "FIRST_TOUCH",
                outcome="APPROVED" if retest else "REJECTED_BREAKOUT",
                evidence_json=json.dumps({"market_session": "REGULAR", "price_confirmation": True}),
            )
            db.add(attempt)
            db.add(RecommendationOutcome(
                id=str(uuid.uuid4()), watch_id=watch_id, setup_id=setup_id,
                attempt_id=attempt_id, ticker="INTC", user_action="SKIPPED",
                outcome="TP1_REACHED" if retest else "STOP_OR_INVALIDATION_REACHED",
                r_multiple=1.1 if retest else -0.4,
                created_at=attempt.started_at, resolved_at=attempt.ended_at,
            ))
        db.commit()


def test_mature_history_creates_bounded_retest_preference(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan())
    _seed_mature_confirmation_history(factory, detail["current_setup_id"], detail["id"])
    with factory() as db:
        setup = db.get(MonitorSetup, detail["current_setup_id"])
        watch = db.get(LiveWatch, detail["id"])
        profile = refresh_profile(db, scope_type="ticker", scope_value="INTC", config=service.config)
        assert profile["evidence_strength"] in {"EMERGING", "MODERATE", "STRONG"}
        context = load_historical_context(db, setup, service.config)
        result = persist_adjustments(
            db, watch=watch, setup=setup, context=context,
            current_features={"tp1_distance_atr": 1.0}, config=service.config,
        )
        db.commit()
        preference = next(item for item in result["adjustments"] if item["adjustment_type"] == "CONFIRMATION_PREFERENCE")
        assert preference["preferred_confirmation"] == "BREAK_RETEST"
        assert abs(preference["adjustment_value"]) <= service.config.max_historical_score_adjustment


def test_history_cannot_override_invalidated_current_structure(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan())
    _seed_mature_confirmation_history(factory, detail["current_setup_id"], detail["id"])
    invalid_prices = [95.8] * 24
    result = service.evaluate_watch(
        detail["id"], bars_1m=bars(invalid_prices, 1),
        bars_5m=bars(invalid_prices, 5), now=NOW,
    )
    assert result["state"] == "INVALIDATED"
    assert result["evaluation"]["base_deterministic_state"] == "INVALIDATED"


def test_completed_bars_and_eod_summary_are_saved_without_trade(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan(primary=103.0))
    service.evaluate_watch(
        detail["id"], bars_1m=reaction_bars(1), bars_5m=reaction_bars(5), now=NOW,
    )
    result = service.run_learning_cycle(NOW.astimezone().date())
    with factory() as db:
        assert db.query(MonitorBarSummary).count() > 0
        assert db.query(MonitorDailySummary).count() == 1
        no_trade_outcome = db.query(RecommendationOutcome).filter(
            RecommendationOutcome.user_action == "NO_ACTION",
        ).one()
        assert no_trade_outcome.outcome == "NO_TRIGGER"
        summary = db.query(MonitorDailySummary).one()
        assert summary.actual_trade_executed is False
        assert result["summaries_finalized"] == 1


def test_level_sanity_detects_distant_primary_skipped_target_and_wide_stop():
    levels = {
        "primary_entry_trigger": 108.0,
        "invalidation_level": 96.0,
        "suggested_stop": 90.0,
        "tp1": 118.0,
        "tp2": 120.0,
        "tp3": 122.0,
    }
    result = evaluate_level_sanity(
        current_price=100.0, atr=2.0, levels=levels,
        structure_bars=reaction_bars(30), execution_bars=reaction_bars(5),
        config=CONFIG,
    )
    assert "PRIMARY_TRIGGER_TOO_DISTANT" in result["anomalies"]
    assert "STOP_TOO_WIDE" in result["anomalies"]
    assert "TP1_TOO_DISTANT" in result["anomalies"]
    assert result["stop_invalidation"]["structural_invalidation"] == 96.0


def test_true_wide_invalidation_is_not_artificially_tightened():
    result = evaluate_level_sanity(
        current_price=100.0, atr=1.0,
        levels={"primary_entry_trigger": 101.0, "invalidation_level": 94.0, "suggested_stop": 94.0, "tp1": 104.0},
        structure_bars=bars([94.0, 97.0, 100.0, 99.0, 100.0], 30),
        execution_bars=reaction_bars(5), config=CONFIG,
    )
    assert result["stop_invalidation"]["structural_invalidation"] == 94.0
    assert result["stop_invalidation"]["tradeable_geometry"] is False
    assert "STOP_TOO_WIDE" in result["anomalies"]


def test_high_confidence_anomaly_can_auto_correct_current_setup(memory_service):
    _, factory, tmp_path = memory_service

    def provider(_packet):
        return {
            "chart_assessment": {
                "broader_structure": "repair", "setup_type": "deep_pullback",
                "execution_structure": "base_building", "setup_quality": "conditional",
                "setup_stale": False,
            },
            "levels": {
                "support_zone": {"low": 98.5, "high": 98.5, "reason": "reaction support"},
                "near_confirmation": {"price": 100.5, "reason": "near pivot"},
                "primary_entry_trigger": {"price": 101.0, "reason": "local reaction pivot"},
                "strong_confirmation": {"price": 102.0, "reason": "upper shelf"},
                "major_trend_repair": {"price": 108.0, "reason": "old primary is major repair"},
                "structural_invalidation": {"price": 96.0, "reason": "structural low"},
                "suggested_stop": {"price": 96.5, "reason": "executable stop"},
            },
            "targets": {
                "tp1": {"price": 103.0, "reason": "first resistance"},
                "tp2": {"price": 105.0, "reason": "next resistance"},
                "tp3": {"price": 108.0, "reason": "major resistance"},
                "stretch_target": {"price": None, "reason": "none"},
            },
            "planner_comparison": {
                "agrees_with_primary_trigger": False,
                "planner_trigger_issue": "major repair misclassified as primary",
                "recommended_action": "MODIFY_LEVELS",
            },
            "decision": "MODIFY_LEVELS", "confidence": 0.97,
            "positive_factors": ["local pivot"], "risk_factors": ["repair incomplete"],
            "rationale_tags": ["LOCAL_RESISTANCE_PRIMARY_TRIGGER"],
            "historical_evidence_used": [], "historical_evidence_ignored": ["insufficient sample"],
            "suggested_level_changes": ["reclassify old primary as major repair"],
            "planner_disagreements": ["primary role"],
            "reason_summary": "Use local confirmation and keep the distant level as major repair.",
        }

    def correction_loader(_ticker, timeframe, _lookback, **_kwargs):
        loaded = reaction_bars({
            "one_minute": 1, "five_minute": 5, "thirty_minute": 30,
            "hourly": 60, "daily": 390,
        }.get(timeframe, 5))
        if timeframe in {"thirty_minute", "hourly", "daily"}:
            loaded[5].update({"open": 97.0, "high": 97.4, "low": 96.0, "close": 96.8})
            loaded[12].update({"open": 102.2, "high": 103.0, "low": 101.8, "close": 102.6})
        return loaded

    service = LiveMonitorService(
        config=replace(CONFIG, chart_review_on_add=True, level_auto_correct_confidence=0.90, chart_snapshot_dir=str(tmp_path / "auto")),
        bars_loader=correction_loader,
        chart_review_provider=provider,
    )
    plan = sample_plan(primary=108.0)
    plan["suggested_stop"] = 90.0
    plan["take_profit_1"] = 118.0
    detail = service.add_monitor("INTC", planner_payload=plan)
    assert detail["active_levels"]["primary_entry_trigger"] == pytest.approx(101.0)
    assert detail["planner_levels"]["primary_entry_trigger"] == pytest.approx(108.0)
    assert detail["active_levels"]["suggested_stop"] == pytest.approx(96.5)
    assert detail["active_levels"]["tp1"] == pytest.approx(103.0)
    assert detail["trigger_source"] == "VALIDATED_CHART_LLM"
    with factory() as db:
        assert db.query(LevelRevision).filter(LevelRevision.source == "VALIDATED_CHART_LLM").count() > 0


def test_low_confidence_or_manual_ownership_blocks_auto_correction():
    sanity = {"review_required": True}
    review = {
        "confidence": 0.70, "decision": "MODIFY_LEVELS",
        "validated_levels": {"primary_entry_trigger": 101.0},
        "validation": {"status": "VALID"},
    }
    result = can_auto_apply_chart_correction(
        review=review, manual_overrides={"primary_entry_trigger": 102.0},
        sanity=sanity, config=CONFIG,
    )
    assert result["allowed"] is False
    assert "confidence_below_threshold" in result["blockers"]
    assert "manual_levels_own_active_setup" in result["blockers"]


def test_evidence_labels_use_effective_sample_size():
    assert evidence_strength(3) == "INSUFFICIENT"
    assert evidence_strength(10) == "WEAK"
    assert evidence_strength(22) == "EMERGING"
    assert evidence_strength(40) == "MODERATE"
    assert evidence_strength(80) == "STRONG"


def test_regular_trading_hours_are_not_marked_extended():
    flags = data_quality_flags(
        {"market_session": "RTH", "price_confirmation": True},
        [{"volume": 100.0}] * 5,
    )
    assert "EXTENDED_HOURS" not in flags


def test_extreme_historical_effect_is_clamped():
    profile = {
        "observation_count": 100,
        "evidence_strength": "STRONG",
        "statistics": {
            "weighted_sample_size": 100.0,
            "reliability": 0.95,
            "false_breakout_rate": 1.0,
            "confirmation_method_stats": {},
        },
    }
    adjustments = derive_bounded_adjustments(profile, {}, replace(CONFIG, max_historical_score_adjustment=0.4))
    penalty = next(item for item in adjustments if item["adjustment_type"] == "FALSE_BREAKOUT_PENALTY")
    assert penalty["adjustment_value"] == pytest.approx(-0.4)


def test_distant_primary_can_remain_when_review_keeps_planner():
    review = {
        "confidence": 0.98,
        "decision": "KEEP_PLANNER",
        "validated_levels": {},
        "validation": {"status": "VALID"},
    }
    policy = can_auto_apply_chart_correction(
        review=review, manual_overrides={},
        sanity={"review_required": True}, config=CONFIG,
    )
    assert policy["allowed"] is False
    assert "review_did_not_approve_modification" in policy["blockers"]


def test_decision_snapshot_never_receives_later_outcome_data(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan())
    one = bars([100.0] * 20 + [101.2], 1)
    five = bars([100.0] * 20 + [101.2], 5, volumes=[100.0] * 20 + [220.0])
    five[-1]["bar_complete"] = True
    service.evaluate_watch(detail["id"], bars_1m=one, bars_5m=five, now=NOW)
    with factory() as db:
        snapshot = db.query(MonitorDecisionSnapshot).filter(
            MonitorDecisionSnapshot.snapshot_type != "MONITOR_CREATED",
        ).order_by(MonitorDecisionSnapshot.created_at.desc()).first()
        assert snapshot is not None
        original_payload = snapshot.payload_json
        decision_data = json.loads(original_payload)
        assert datetime.fromisoformat(str(decision_data["market_data_as_of"])) <= NOW
        assert "recommendation_outcome" not in decision_data
        snapshot_id = snapshot.id
        db.add(RecommendationOutcome(
            id=str(uuid.uuid4()), watch_id=detail["id"], setup_id=detail["current_setup_id"],
            ticker="INTC", user_action="SKIPPED", outcome="TP1_REACHED",
            r_multiple=1.0, created_at=NOW, resolved_at=NOW + timedelta(hours=4),
        ))
        db.commit()
    service.run_learning_cycle(NOW.astimezone().date())
    with factory() as db:
        persisted = db.get(MonitorDecisionSnapshot, snapshot_id)
        assert persisted.payload_json == original_payload


def test_manual_level_source_receives_its_own_outcome_attribution(memory_service):
    service, factory, _ = memory_service
    detail = service.add_monitor("INTC", planner_payload=sample_plan(primary=103.0))
    service.edit_levels(detail["id"], {"primary_entry_trigger": 102.5})
    service.evaluate_watch(
        detail["id"], bars_1m=reaction_bars(1), bars_5m=reaction_bars(5), now=NOW,
    )
    service.run_learning_cycle(NOW.astimezone().date())
    with factory() as db:
        manual = db.query(LevelRevision).filter(
            LevelRevision.setup_id == detail["current_setup_id"],
            LevelRevision.level_name == "primary_entry_trigger",
            LevelRevision.source == "MANUAL",
        ).order_by(LevelRevision.created_at.desc()).first()
        assert manual is not None
        associated = json.loads(manual.outcome_json)["associated_outcome"]
        assert associated["summary_id"]
        assert associated["outcome_type"] == "RECOMMENDATION_OUTCOME"


def test_paper_rule_records_shadow_result_without_changing_production(memory_service):
    service, factory, _ = memory_service
    plan = sample_plan()
    plan["suggested_stop"] = 99.5
    plan["take_profit_1"] = 108.0
    detail = service.add_monitor("INTC", planner_payload=plan)
    proposal = service.create_proposal({
        "scope_type": "ticker", "scope_value": "INTC",
        "title": "Require break and retest",
        "proposed_change": {"preferred_confirmation": "BREAK_RETEST"},
    })
    service.decide_proposal(proposal["id"], decision="PAPER_TEST")
    one = bars([100.0] * 20 + [101.2], 1)
    five = bars([100.0] * 20 + [101.2], 5, volumes=[100.0] * 20 + [220.0])
    five[-1]["bar_complete"] = True
    five[-1].update({"open": 100.9, "high": 101.25, "low": 100.8, "close": 101.2})
    result = service.evaluate_watch(detail["id"], bars_1m=one, bars_5m=five, now=NOW)
    assert result["state"] in {"APPROVED", "STRONGLY_CONFIRMED"}
    with factory() as db:
        shadow = db.query(ShadowRuleEvaluation).filter(
            ShadowRuleEvaluation.setup_id == detail["current_setup_id"],
        ).one()
        assert shadow.production_decision == result["state"]
        assert shadow.shadow_decision == "WAIT_FOR_RETEST"
        assert json.loads(shadow.evidence_json)["production_unchanged"] is True
        db.add(RecommendationOutcome(
            id=str(uuid.uuid4()), watch_id=detail["id"], setup_id=detail["current_setup_id"],
            ticker="INTC", user_action="SKIPPED", outcome="TP1_REACHED",
            r_multiple=1.0, created_at=NOW, resolved_at=NOW + timedelta(hours=3),
        ))
        db.commit()
    service.run_learning_cycle(NOW.astimezone().date())
    with factory() as db:
        resolved = db.query(ShadowRuleEvaluation).filter(
            ShadowRuleEvaluation.setup_id == detail["current_setup_id"],
        ).one()
        assert resolved.production_outcome == "TP1_REACHED"
        assert resolved.shadow_hypothetical_outcome == "NO_SHADOW_ENTRY"
        assert resolved.resolved_at is not None
        assert db.get(LearningProposal, proposal["id"]).status == "PAPER_TESTING"


def test_failed_validated_level_source_only_adds_bounded_penalty():
    profile = {
        "observation_count": 40,
        "evidence_strength": "MODERATE",
        "statistics": {
            "weighted_sample_size": 35.0,
            "reliability": 0.60,
            "false_breakout_rate": 0.35,
            "confirmation_method_stats": {},
            "level_source_stats": {
                "VALIDATED_CHART_LLM": {
                    "sample_size": 15,
                    "weighted_sample_size": 12.0,
                    "success_rate": 0.20,
                    "expectancy_r": -0.5,
                },
            },
        },
    }
    adjustments = derive_bounded_adjustments(
        profile, {"level_source": "VALIDATED_CHART_LLM"},
        replace(CONFIG, max_historical_score_adjustment=0.4),
    )
    source = next(item for item in adjustments if item["adjustment_type"] == "LEVEL_SOURCE_CONFIDENCE")
    assert -0.4 <= source["adjustment_value"] < 0
