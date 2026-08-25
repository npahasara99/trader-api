"""Persistent historical memory for the advisory live monitor."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from typing import Any
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    BehaviorProfileVersion,
    ConfirmationAttempt,
    LearnedAdjustment,
    LearningJobRun,
    LearningObservation,
    LearningProposal,
    LLMAdvisoryReview,
    LLMDecisionPostmortem,
    LevelRevision,
    LiveWatch,
    ManualMonitorTrade,
    MonitorBarSummary,
    MonitorDailySummary,
    MonitorEvent,
    MonitorSetup,
    RecommendationOutcome,
    ShadowRuleEvaluation,
    StockBehaviorProfile,
)

from .chart_levels import number
from .config import LiveMonitorConfig
from .learning import (
    FORMULA_VERSION,
    adjustment_breakdown,
    aggregate_observations,
    derive_bounded_adjustments,
    hierarchical_weights,
)


NEW_YORK = ZoneInfo("America/New_York")


def _id() -> str:
    return str(uuid.uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _loads(value: str | None, fallback: Any = None) -> Any:
    if not value:
        return {} if fallback is None else fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {} if fallback is None else fallback


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def trading_day_bounds(trading_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(trading_date, time.min, tzinfo=NEW_YORK)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def data_quality_flags(evaluation: dict[str, Any], bars: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if evaluation.get("data_stale"):
        flags.append("STALE_DATA")
    if not bars or all(number(bar.get("volume")) is None for bar in bars):
        flags.append("MISSING_VOLUME")
    if str(evaluation.get("market_session") or "").upper() not in {"REGULAR", "RTH", "OPEN", "MARKET_OPEN"}:
        flags.append("EXTENDED_HOURS")
    if len(bars) < 5:
        flags.append("PARTIAL_HISTORY")
    if not evaluation.get("price_confirmation"):
        flags.append("NO_EXECUTION_CONFIRMATION")
    return flags


def persist_completed_bars(
    db: Session,
    *,
    watch: LiveWatch,
    setup: MonitorSetup,
    timeframe: str,
    bars: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> int:
    """Store newly completed bars only; repeated polls are ignored."""
    if not bars:
        return 0
    current_time = _as_datetime(evaluation.get("evaluated_at")) or datetime.now(timezone.utc)
    duration = timedelta(minutes=1 if timeframe == "1m" else 5)
    candidates = bars[:-1] if (_as_datetime(bars[-1].get("date")) or current_time) + duration > current_time else bars
    flags = data_quality_flags(evaluation, candidates)
    inserted = 0
    for bar in candidates[-4:]:
        timestamp = _as_datetime(bar.get("date"))
        values = {name: number(bar.get(name)) for name in ("open", "high", "low", "close")}
        if timestamp is None or any(value is None for value in values.values()):
            continue
        exists = db.query(MonitorBarSummary.id).filter(
            MonitorBarSummary.setup_id == setup.id,
            MonitorBarSummary.timeframe == timeframe,
            MonitorBarSummary.bar_timestamp == timestamp,
        ).first()
        if exists:
            continue
        db.add(MonitorBarSummary(
            id=_id(), watch_id=watch.id, setup_id=setup.id,
            market_snapshot_id=setup.market_snapshot_id, ticker=watch.ticker,
            timeframe=timeframe, bar_timestamp=timestamp,
            open_price=float(values["open"]), high_price=float(values["high"]),
            low_price=float(values["low"]), close_price=float(values["close"]),
            volume=number(bar.get("volume")),
            indicators_json=_dumps({
                "rvol_1m": evaluation.get("rvol_1m"),
                "rvol_5m": evaluation.get("rvol_5m"),
                "close_location": evaluation.get("close_location_value"),
                "upper_wick_ratio": evaluation.get("upper_wick_ratio"),
            }),
            data_quality_flags_json=_dumps(flags),
        ))
        inserted += 1
    return inserted


def _attempt_observation(
    attempt: ConfirmationAttempt,
    setup: MonitorSetup,
    recommendation: RecommendationOutcome | None,
) -> dict[str, Any]:
    evidence = _loads(attempt.evidence_json)
    levels = evidence.get("levels") or _loads(setup.active_levels_json)
    atr = number(levels.get("atr")) or number((_loads(setup.planner_baseline_json)).get("atr"))
    mfe_atr = None
    mae_atr = None
    if atr and number(attempt.trigger_price) is not None:
        if number(attempt.peak_price) is not None:
            mfe_atr = (float(attempt.peak_price) - float(attempt.trigger_price)) / atr
        if number(attempt.lowest_retest_price) is not None:
            mae_atr = (float(attempt.lowest_retest_price) - float(attempt.trigger_price)) / atr
    # Attempt rows are already completed event summaries, so an absent raw-bar
    # payload must not be mistaken for missing historical data. Apply only the
    # quality penalties that can be established from the persisted evidence.
    quality: list[str] = []
    if evidence.get("data_stale") or evidence.get("snapshot_status") == "STALE":
        quality.append("STALE_DATA")
    if attempt.rvol_1m is None and attempt.rvol_5m is None:
        quality.append("MISSING_VOLUME")
    if str(evidence.get("session_bucket") or evidence.get("market_session") or "").upper() not in {"", "REGULAR", "RTH"}:
        quality.append("EXTENDED_HOURS")
    if not attempt.price_confirmation and not attempt.volume_confirmation:
        quality.append("NO_EXECUTION_CONFIRMATION")
    return {
        "attempt_id": attempt.id,
        "ticker": attempt.ticker,
        "occurred_at": attempt.ended_at or attempt.started_at,
        "outcome": (recommendation.outcome if recommendation else None) or attempt.outcome,
        "r_multiple": recommendation.r_multiple if recommendation else None,
        "mfe_atr": mfe_atr,
        "mae_atr": mae_atr,
        "rvol_1m": attempt.rvol_1m,
        "rvol_5m": attempt.rvol_5m,
        "entry_distance_pct": recommendation.entry_distance_from_trigger_pct if recommendation else None,
        "confirmation_method": attempt.confirmation_method,
        "attempt_number": attempt.attempt_number,
        "setup_type": setup.setup_type,
        "setup_family": setup.setup_family,
        "broader_structure": setup.broader_structure,
        "execution_structure": setup.execution_structure,
        "sector": setup.sector,
        "market_regime": setup.market_regime,
        "level_source": setup.trigger_source,
        "session_bucket": evidence.get("session_bucket") or evidence.get("market_session"),
        "atr_pct": evidence.get("atr_pct"),
        "rsi": evidence.get("rsi"),
        "distance_from_support_atr": evidence.get("distance_from_support_atr"),
        "primary_trigger_distance_atr": evidence.get("distance_to_trigger_atr"),
        "tp1_reached": "TP1" in str((recommendation.outcome if recommendation else None) or "").upper(),
        "tp2_reached": "TP2" in str((recommendation.outcome if recommendation else None) or "").upper(),
        "tp3_reached": "TP3" in str((recommendation.outcome if recommendation else None) or "").upper(),
        "data_quality_flags": quality,
    }


def _daily_observation(row: MonitorDailySummary) -> dict[str, Any]:
    outcome = _loads(row.outcome_json)
    decisions = _loads(row.decisions_json)
    indicators = _loads(row.indicators_json)
    return {
        "ticker": row.ticker,
        "occurred_at": row.finalized_at,
        "trading_date": row.trading_date,
        "outcome": outcome.get("recommendation_outcome") or row.highest_state_reached or "NO_TRIGGER",
        "r_multiple": row.recommendation_r_multiple,
        "mfe_atr": row.mfe_atr,
        "mae_atr": row.mae_atr,
        "rvol_1m": indicators.get("rvol_1m"),
        "rvol_5m": indicators.get("rvol_5m"),
        "confirmation_method": decisions.get("confirmation_method") or "NO_TRIGGER",
        "attempt_number": row.number_of_trigger_attempts or 0,
        "setup_type": row.setup_type,
        "setup_family": row.setup_family,
        "broader_structure": row.broader_structure,
        "execution_structure": row.execution_structure,
        "sector": row.sector,
        "market_regime": row.market_regime,
        "level_source": decisions.get("trigger_source"),
        "session_bucket": decisions.get("session_bucket"),
        "tp1_reached": outcome.get("tp1_reached"),
        "tp2_reached": outcome.get("tp2_reached"),
        "tp3_reached": outcome.get("tp3_reached"),
        "runner_state": decisions.get("runner_state"),
        "breakout_rejected": outcome.get("breakout_rejected"),
        "breakout_confirmed": outcome.get("breakout_confirmed"),
        "runner_extension_atr": outcome.get("runner_extension_atr"),
        "data_quality_flags": _loads(row.data_quality_flags_json, []),
    }


def collect_scope_observations(db: Session, scope_type: str, scope_value: str) -> list[dict[str, Any]]:
    setup_query = db.query(MonitorSetup)
    normalized = str(scope_type).lower()
    if normalized == "ticker":
        setup_query = setup_query.filter(MonitorSetup.ticker == scope_value)
    elif normalized == "setup_type":
        setup_query = setup_query.filter(MonitorSetup.setup_type == scope_value)
    elif normalized == "setup_family":
        setup_query = setup_query.filter(MonitorSetup.setup_family == scope_value)
    elif normalized == "sector":
        setup_query = setup_query.filter(MonitorSetup.sector == scope_value)
    elif normalized == "market_regime":
        setup_query = setup_query.filter(MonitorSetup.market_regime == scope_value)
    setups = setup_query.all()
    if not setups:
        return []
    setup_by_id = {row.id: row for row in setups}
    setup_ids = list(setup_by_id)
    recommendations = db.query(RecommendationOutcome).filter(RecommendationOutcome.setup_id.in_(setup_ids)).all()
    recommendation_by_attempt = {row.attempt_id: row for row in recommendations if row.attempt_id}
    attempts = db.query(ConfirmationAttempt).filter(
        ConfirmationAttempt.setup_id.in_(setup_ids),
        ConfirmationAttempt.ended_at.is_not(None),
    ).all()
    observations = [
        _attempt_observation(row, setup_by_id[row.setup_id], recommendation_by_attempt.get(row.id))
        for row in attempts
    ]
    attempted_setup_ids = {row.setup_id for row in attempts}
    daily_rows = db.query(MonitorDailySummary).filter(MonitorDailySummary.setup_id.in_(setup_ids)).all()
    observations.extend(_daily_observation(row) for row in daily_rows if row.setup_id not in attempted_setup_ids)
    return observations


def refresh_profile(
    db: Session,
    *,
    scope_type: str,
    scope_value: str,
    config: LiveMonitorConfig,
    force_version: bool = False,
) -> dict[str, Any]:
    observations = collect_scope_observations(db, scope_type, scope_value)
    prior = {
        "ticker": config.ticker_prior_strength,
        "setup_type": config.setup_prior_strength,
        "setup_family": config.setup_prior_strength,
        "sector": config.sector_prior_strength,
        "market_regime": config.regime_prior_strength,
    }.get(scope_type, config.sector_prior_strength)
    statistics = aggregate_observations(
        observations,
        half_life_days=config.learning_recency_half_life_days,
        prior_strength=prior,
        evidence_thresholds=config.evidence_thresholds,
    )
    current = db.query(StockBehaviorProfile).filter(
        StockBehaviorProfile.scope_type == scope_type,
        StockBehaviorProfile.scope_value == scope_value,
    ).one_or_none()
    if current is None:
        current = StockBehaviorProfile(
            id=_id(), scope_type=scope_type, scope_value=scope_value,
            statistics_json="{}",
        )
        db.add(current)
        db.flush()
    prior_statistics = _loads(current.statistics_json)
    current.observation_count = len(observations)
    current.evidence_strength = statistics["evidence_strength"]
    current.statistics_json = _dumps(statistics)
    current.updated_at = datetime.now(timezone.utc)

    latest_version = db.query(BehaviorProfileVersion).filter(
        BehaviorProfileVersion.scope_type == scope_type,
        BehaviorProfileVersion.scope_value == scope_value,
    ).order_by(BehaviorProfileVersion.version.desc()).first()
    changed = prior_statistics != statistics
    if latest_version is None or force_version or changed:
        version = 1 if latest_version is None else latest_version.version + 1
        latest_version = BehaviorProfileVersion(
            id=_id(), profile_id=current.id, scope_type=scope_type, scope_value=scope_value,
            version=version, observation_count=len(observations),
            weighted_observation_count=float(statistics["weighted_sample_size"]),
            evidence_strength=statistics["evidence_strength"], reliability=float(statistics["reliability"]),
            statistics_json=_dumps(statistics), formula_version=FORMULA_VERSION,
            source_cutoff_at=max((_as_datetime(row.get("occurred_at")) for row in observations), default=None),
        )
        db.add(latest_version)
        db.flush()
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "observation_count": len(observations),
        "evidence_strength": statistics["evidence_strength"],
        "statistics": statistics,
        "profile_version_id": latest_version.id if latest_version else None,
        "profile_version": latest_version.version if latest_version else 0,
        "profile_last_updated_at": current.updated_at,
    }


def load_historical_context(db: Session, setup: MonitorSetup, config: LiveMonitorConfig) -> dict[str, Any]:
    setup_scope = (
        ("setup_family", setup.setup_family)
        if setup.setup_family
        else ("setup_type", setup.setup_type)
    )
    scopes = {
        "global": ("global", "all"),
        "ticker": ("ticker", setup.ticker),
        "setup": setup_scope,
        "legacy_setup_type": ("setup_type", setup.setup_type) if setup.setup_family else ("setup_type", None),
        "sector": ("sector", setup.sector),
        "regime": ("market_regime", setup.market_regime),
    }
    profiles: dict[str, Any] = {}
    for name, (scope_type, scope_value) in scopes.items():
        if not scope_value:
            continue
        profiles[name] = refresh_profile(
            db, scope_type=scope_type, scope_value=str(scope_value), config=config,
        )
    weights = hierarchical_weights(
        ticker_samples=float((profiles.get("ticker") or {}).get("statistics", {}).get("weighted_sample_size") or 0),
        setup_samples=float((profiles.get("setup") or {}).get("statistics", {}).get("weighted_sample_size") or 0),
        sector_samples=float((profiles.get("sector") or {}).get("statistics", {}).get("weighted_sample_size") or 0),
        regime_samples=float((profiles.get("regime") or {}).get("statistics", {}).get("weighted_sample_size") or 0),
        ticker_prior=config.ticker_prior_strength,
        setup_prior=config.setup_prior_strength,
        sector_prior=config.sector_prior_strength,
        regime_prior=config.regime_prior_strength,
    )
    ticker_profile = profiles.get("ticker") or {
        "scope_type": "ticker", "scope_value": setup.ticker,
        "observation_count": 0, "evidence_strength": "INSUFFICIENT", "statistics": {},
    }
    ticker_profile["hierarchical_weights"] = weights
    return {"ticker_profile": ticker_profile, "broader_profiles": profiles, "hierarchical_weights": weights}


def persist_adjustments(
    db: Session,
    *,
    watch: LiveWatch,
    setup: MonitorSetup,
    context: dict[str, Any],
    current_features: dict[str, Any],
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    profile = context.get("ticker_profile") or {}
    adjustments = derive_bounded_adjustments(profile, current_features, config)
    existing_types = {
        row[0]
        for row in db.query(LearnedAdjustment.adjustment_type).filter(LearnedAdjustment.setup_id == setup.id).all()
    }
    for item in adjustments:
        if item["adjustment_type"] in existing_types:
            continue
        db.add(LearnedAdjustment(
            id=_id(), watch_id=watch.id, setup_id=setup.id,
            market_snapshot_id=setup.market_snapshot_id, ticker=setup.ticker,
            adjustment_type=item["adjustment_type"], base_value=item.get("base_value"),
            learned_value=item.get("learned_value"), adjustment_value=float(item.get("adjustment_value") or 0.0),
            adjustment_strength=float(item.get("adjustment_strength") or 0.0),
            evidence_strength=item.get("evidence_strength") or "INSUFFICIENT",
            sample_size=int(item.get("sample_size") or 0),
            weighted_sample_size=float(item.get("weighted_sample_size") or 0.0),
            reason=str(item.get("reason") or "Historical evidence adjustment"),
            supporting_stats_json=_dumps(item.get("supporting_stats") or {}),
            bounds_json=_dumps(item.get("bounds") or {}),
            profile_version_id=profile.get("profile_version_id"),
        ))
    breakdown = adjustment_breakdown(setup.setup_quality_score, adjustments, config.max_historical_score_adjustment)
    return {"adjustments": adjustments, "recommendation_breakdown": breakdown}


def learned_adjustment_payloads(db: Session, setup_id: str) -> list[dict[str, Any]]:
    rows = db.query(LearnedAdjustment).filter(LearnedAdjustment.setup_id == setup_id).order_by(LearnedAdjustment.created_at).all()
    return [{
        "id": row.id, "adjustment_type": row.adjustment_type, "base_value": row.base_value,
        "learned_value": row.learned_value, "adjustment_value": row.adjustment_value,
        "adjustment_strength": row.adjustment_strength, "evidence_strength": row.evidence_strength,
        "sample_size": row.sample_size, "weighted_sample_size": row.weighted_sample_size,
        "reason": row.reason, "supporting_stats": _loads(row.supporting_stats_json),
        "bounds": _loads(row.bounds_json), "profile_version_id": row.profile_version_id,
        "created_at": row.created_at,
    } for row in rows]


def finalize_daily_summary(
    db: Session,
    *,
    watch: LiveWatch,
    setup: MonitorSetup,
    trading_date: date,
) -> MonitorDailySummary | None:
    existing = db.query(MonitorDailySummary).filter(
        MonitorDailySummary.setup_id == setup.id,
        MonitorDailySummary.trading_date == trading_date,
    ).one_or_none()
    if existing:
        return None
    start, end = trading_day_bounds(trading_date)
    bars = db.query(MonitorBarSummary).filter(
        MonitorBarSummary.setup_id == setup.id,
        MonitorBarSummary.bar_timestamp >= start,
        MonitorBarSummary.bar_timestamp < end,
    ).order_by(MonitorBarSummary.bar_timestamp).all()
    events = db.query(MonitorEvent).filter(
        MonitorEvent.setup_id == setup.id,
        MonitorEvent.created_at >= start,
        MonitorEvent.created_at < end,
    ).order_by(MonitorEvent.created_at).all()
    attempts = db.query(ConfirmationAttempt).filter(
        ConfirmationAttempt.setup_id == setup.id,
        ConfirmationAttempt.started_at >= start,
        ConfirmationAttempt.started_at < end,
    ).all()
    if not bars and not events and not attempts:
        return None
    baseline = _loads(setup.planner_baseline_json)
    levels = _loads(setup.active_levels_json)
    atr = number(levels.get("atr")) or number(baseline.get("atr"))
    open_price = bars[0].open_price if bars else setup.plan_reference_price
    close_price = bars[-1].close_price if bars else watch.current_price
    high_price = max((bar.high_price for bar in bars), default=close_price)
    low_price = min((bar.low_price for bar in bars), default=close_price)
    trigger = number(levels.get("primary_entry_trigger"))
    mfe_atr = None if not atr or trigger is None or high_price is None else (high_price - trigger) / atr
    mae_atr = None if not atr or trigger is None or low_price is None else (low_price - trigger) / atr
    reviews = db.query(LLMAdvisoryReview).filter(
        LLMAdvisoryReview.setup_id == setup.id,
        LLMAdvisoryReview.created_at >= start,
        LLMAdvisoryReview.created_at < end,
    ).order_by(LLMAdvisoryReview.created_at).all()
    recommendations = db.query(RecommendationOutcome).filter(
        RecommendationOutcome.setup_id == setup.id,
        RecommendationOutcome.created_at >= start,
        RecommendationOutcome.created_at < end,
    ).order_by(RecommendationOutcome.created_at).all()
    trades = db.query(ManualMonitorTrade).filter(
        ManualMonitorTrade.setup_id == setup.id,
        ManualMonitorTrade.created_at >= start,
        ManualMonitorTrade.created_at < end,
    ).all()
    latest_evaluation = _loads(watch.latest_evaluation_json)
    quality = data_quality_flags(latest_evaluation, [{"volume": bar.volume} for bar in bars])
    if _loads(setup.manual_overrides_json):
        quality.append("MANUAL_LEVEL_OVERRIDE")
    state_priority = {
        "WATCHING": 0, "NEAR_TRIGGER": 1, "ARMED": 2, "CONFIRMING": 3,
        "REJECTED_BREAKOUT": 4, "APPROVED": 5, "STRONGLY_CONFIRMED": 6,
        "MISSED": 7, "INVALIDATED": 8,
    }
    reached_states = [event.to_state for event in events if event.to_state]
    highest_state = max(reached_states or [watch.state], key=lambda item: state_priority.get(str(item), -1))
    tp1, tp2, tp3 = (number(levels.get(name)) for name in ("tp1", "tp2", "tp3"))
    invalidation = number(levels.get("invalidation_level"))
    recommendation = recommendations[-1] if recommendations else None
    if recommendation is None:
        recommendation = RecommendationOutcome(
            id=_id(), watch_id=watch.id, setup_id=setup.id, ticker=setup.ticker,
            user_action="NO_ACTION",
            outcome="NO_TRIGGER" if not attempts else str(highest_state),
            details_json=_dumps({
                "generated_by": "daily_learning_cycle",
                "actual_trade_executed": False,
                "selection_bias_control": True,
            }),
            created_at=end - timedelta(microseconds=1), resolved_at=end,
        )
        db.add(recommendation)
        db.flush()
    actual_r = next((row.r_multiple for row in reversed(trades) if row.r_multiple is not None), None)
    runner_extension_atr = (
        None
        if tp1 is None or high_price is None or not atr
        else round(max(float(high_price) - float(tp1), 0.0) / float(atr), 4)
    )
    row = MonitorDailySummary(
        id=_id(), trading_date=trading_date, watch_id=watch.id, setup_id=setup.id,
        market_snapshot_id=setup.market_snapshot_id, ticker=setup.ticker,
        open_price=open_price, high_price=high_price, low_price=low_price, close_price=close_price,
        starting_monitor_price=bars[0].close_price if bars else setup.plan_reference_price,
        ending_monitor_price=close_price, broader_structure=setup.broader_structure,
        setup_type=setup.setup_type, execution_structure=setup.execution_structure,
        setup_family=setup.setup_family,
        market_regime=setup.market_regime, sector=setup.sector,
        levels_json=_dumps({
            "planner": _loads(setup.planner_levels_json),
            "chart_llm": _loads(setup.llm_proposed_levels_json),
            "validated": _loads(setup.validated_chart_levels_json),
            "manual": _loads(setup.manual_overrides_json),
            "final_active": levels,
        }),
        indicators_json=_dumps({
            key: baseline.get(key)
            for key in ("atr", "atr_pct", "rsi", "ema20", "ema50", "ema100", "ema200", "vwap")
        } | {"rvol_1m": latest_evaluation.get("rvol_1m"), "rvol_5m": latest_evaluation.get("rvol_5m")}),
        context_json=_dumps({
            "market_regime": setup.market_regime, "spy_context": baseline.get("spy_context"),
            "qqq_context": baseline.get("qqq_context"), "sector_context": baseline.get("sector_context"),
            "relative_strength": baseline.get("relative_strength"),
        }),
        decisions_json=_dumps({
            "deterministic_decision": highest_state,
            "llm_decision": reviews[-1].decision if reviews else None,
            "llm_confidence": reviews[-1].confidence if reviews else None,
            "user_decision": recommendation.user_action if recommendation else None,
            "confirmation_method": attempts[-1].confirmation_method if attempts else None,
            "trigger_source": setup.trigger_source,
            "rule_version": setup.rule_version,
            "runner_state": latest_evaluation.get("runner_state"),
            "runner_trailing_methods": baseline.get("runner_trailing_methods") or [],
        }),
        outcome_json=_dumps({
            "trigger_reached": bool(attempts),
            "confirmation_passed": any(row.outcome in {"APPROVED", "STRONGLY_CONFIRMED"} for row in attempts),
            "tp1_reached": bool(tp1 is not None and high_price is not None and high_price >= tp1),
            "tp2_reached": bool(tp2 is not None and high_price is not None and high_price >= tp2),
            "tp3_reached": bool(tp3 is not None and high_price is not None and high_price >= tp3),
            "breakout_rejected": bool(latest_evaluation.get("breakout_rejected")),
            "breakout_confirmed": bool(latest_evaluation.get("breakout_confirmed")),
            "runner_extension_atr": runner_extension_atr,
            "invalidation_reached": bool(invalidation is not None and low_price is not None and low_price <= invalidation),
            "recommendation_outcome": recommendation.outcome if recommendation else "NO_USER_ACTION",
            "outcome_type": "ACTUAL_TRADE" if trades else "RECOMMENDATION_OUTCOME",
        }),
        data_quality_flags_json=_dumps(sorted(set(quality))),
        number_of_trigger_attempts=len(attempts),
        number_of_rejections=sum("REJECT" in str(row.outcome or "") for row in attempts),
        highest_state_reached=highest_state,
        mfe_atr=None if mfe_atr is None else round(mfe_atr, 4),
        mae_atr=None if mae_atr is None else round(mae_atr, 4),
        recommendation_r_multiple=recommendation.r_multiple if recommendation else None,
        actual_trade_executed=bool(trades), actual_trade_r_multiple=actual_r,
    )
    db.add(row)
    db.flush()
    _associate_pricing_and_shadow_outcomes(db, row, attempts)
    return row


def _associate_pricing_and_shadow_outcomes(
    db: Session,
    summary: MonitorDailySummary,
    attempts: list[ConfirmationAttempt],
) -> None:
    """Resolve append-only decision lineage without mixing actual and shadow PnL."""
    outcome = _loads(summary.outcome_json)
    association = {
        "summary_id": summary.id,
        "outcome_type": outcome.get("outcome_type"),
        "recommendation_outcome": outcome.get("recommendation_outcome"),
        "trigger_reached": outcome.get("trigger_reached"),
        "confirmation_passed": outcome.get("confirmation_passed"),
        "tp1_reached": outcome.get("tp1_reached"),
        "tp2_reached": outcome.get("tp2_reached"),
        "tp3_reached": outcome.get("tp3_reached"),
        "invalidation_reached": outcome.get("invalidation_reached"),
        "mfe_atr": summary.mfe_atr,
        "mae_atr": summary.mae_atr,
        "recommendation_r_multiple": summary.recommendation_r_multiple,
    }
    revisions = db.query(LevelRevision).filter(LevelRevision.setup_id == summary.setup_id).all()
    for revision in revisions:
        payload = _loads(revision.outcome_json)
        payload["associated_outcome"] = association
        revision.outcome_json = _dumps(payload)

    shadow_rows = db.query(ShadowRuleEvaluation).filter(
        ShadowRuleEvaluation.setup_id == summary.setup_id,
        ShadowRuleEvaluation.resolved_at.is_(None),
    ).all()
    production_label = (
        str(outcome.get("recommendation_outcome"))
        if outcome.get("recommendation_outcome") not in {None, "NO_USER_ACTION"}
        else "TP1_REACHED"
        if outcome.get("tp1_reached")
        else "INVALIDATED"
        if outcome.get("invalidation_reached")
        else str(summary.highest_state_reached or "NO_TRIGGER")
    )
    for shadow in shadow_rows:
        shadow_time = _as_datetime(shadow.created_at) or datetime.min.replace(tzinfo=timezone.utc)
        later_retest = any(
            attempt.retest_result == "HELD"
            and (_as_datetime(attempt.ended_at or attempt.started_at) or shadow_time) >= shadow_time
            for attempt in attempts
        )
        if shadow.shadow_decision == shadow.production_decision:
            hypothetical = f"SAME_AS_PRODUCTION:{production_label}"
        elif shadow.shadow_decision == "WAIT_FOR_RETEST":
            hypothetical = (
                f"RETEST_CONFIRMED:{production_label}"
                if later_retest
                else "NO_SHADOW_ENTRY"
            )
        else:
            hypothetical = f"UNRESOLVED_RULE_SEMANTICS:{shadow.shadow_decision}"
        evidence = _loads(shadow.evidence_json)
        evidence["resolution"] = {
            "summary_id": summary.id,
            "production_outcome": production_label,
            "shadow_hypothetical_outcome": hypothetical,
            "actual_trade_outcome_kept_separate": True,
        }
        shadow.production_outcome = production_label
        shadow.shadow_hypothetical_outcome = hypothetical
        shadow.evidence_json = _dumps(evidence)
        shadow.resolved_at = summary.finalized_at


def _create_postmortems(db: Session, summary: MonitorDailySummary) -> int:
    reviews = db.query(LLMAdvisoryReview).filter(LLMAdvisoryReview.setup_id == summary.setup_id).all()
    outcome = _loads(summary.outcome_json)
    success = bool(outcome.get("tp1_reached") or (summary.recommendation_r_multiple or 0) > 0)
    created = 0
    for review in reviews:
        exists = db.query(LLMDecisionPostmortem.id).filter(LLMDecisionPostmortem.llm_review_id == review.id).first()
        if exists:
            continue
        aligned = (review.decision == "APPROVE" and success) or (review.decision in {"WAIT", "REJECT"} and not success)
        tags = ["LLM_CORRECTION_HELPFUL" if aligned else "LLM_CORRECTION_HARMFUL"]
        tags.append("TP1_REALISTIC" if outcome.get("tp1_reached") else "TP1_TOO_HIGH")
        lessons = {
            "decision_aligned_with_outcome": aligned,
            "entry_level_appropriate": outcome.get("trigger_reached"),
            "confirmation_sufficient": outcome.get("confirmation_passed"),
            "historical_evidence_was_helpful": aligned,
        }
        review.actual_outcome_json = _dumps({"summary_id": summary.id, **outcome})
        db.add(LLMDecisionPostmortem(
            id=_id(), llm_review_id=review.id, watch_id=review.watch_id, setup_id=review.setup_id,
            ticker=review.ticker, outcome_type=str(outcome.get("outcome_type") or "RECOMMENDATION_OUTCOME"),
            original_decision=review.decision, outcome_json=_dumps(outcome),
            rationale_tags_json=_dumps(tags), lessons_json=_dumps(lessons),
            model=review.model, prompt_version=review.prompt_version,
        ))
        created += 1
    return created


def _generate_observation_and_proposal(
    db: Session,
    *,
    profile: dict[str, Any],
    config: LiveMonitorConfig,
) -> tuple[int, int]:
    if profile["evidence_strength"] not in {"MODERATE", "STRONG"}:
        return 0, 0
    methods = (profile.get("statistics") or {}).get("confirmation_method_stats") or {}
    retest = next((value for key, value in methods.items() if "RETEST" in key.upper()), None)
    first = next((value for key, value in methods.items() if "FIRST" in key.upper() or "5M_CLOSE" in key.upper()), None)
    if not retest or not first or retest.get("expectancy_r") is None or first.get("expectancy_r") is None:
        return 0, 0
    effect = float(retest["expectancy_r"]) - float(first["expectancy_r"])
    if effect < config.proposal_min_effect_r:
        return 0, 0
    scope_type = profile["scope_type"]
    scope_value = profile["scope_value"]
    sample_size = int(profile["observation_count"])
    latest = db.query(LearningObservation).filter(
        LearningObservation.scope_type == scope_type,
        LearningObservation.scope_value == scope_value,
        LearningObservation.observation_type == "retest_vs_first_touch",
    ).order_by(LearningObservation.created_at.desc()).first()
    if latest and sample_size - latest.sample_size < config.learning_min_observations:
        return 0, 0
    observation = LearningObservation(
        id=_id(), scope_type=scope_type, scope_value=scope_value,
        observation_type="retest_vs_first_touch",
        summary=f"{scope_value}: break/retest expectancy exceeds first-touch expectancy by {effect:.2f}R.",
        sample_size=sample_size, evidence_strength=profile["evidence_strength"],
        evidence_json=_dumps({"effect_size_r": effect, "retest": retest, "first_touch": first}),
    )
    db.add(observation)
    db.flush()
    pending = db.query(LearningProposal.id).filter(
        LearningProposal.scope_type == scope_type,
        LearningProposal.scope_value == scope_value,
        LearningProposal.status.in_(["PENDING", "PAPER_TESTING"]),
    ).first()
    if pending:
        return 1, 0
    db.add(LearningProposal(
        id=_id(), observation_id=observation.id, scope_type=scope_type, scope_value=scope_value,
        status="PENDING", title=f"Prefer validated retest confirmation for {scope_value}",
        proposed_change_json=_dumps({"confirmation_preference": "BREAK_RETEST"}),
        evidence_json=observation.evidence_json,
    ))
    return 1, 1


def run_daily_learning_cycle(db: Session, trading_date: date, config: LiveMonitorConfig) -> dict[str, Any]:
    run = LearningJobRun(
        id=_id(), trading_date=trading_date, status="RUNNING", details_json="{}",
    )
    db.add(run)
    db.flush()
    summaries: list[MonitorDailySummary] = []
    setups = db.query(MonitorSetup).all()
    watches = {row.id: row for row in db.query(LiveWatch).all()}
    for setup in setups:
        watch = watches.get(setup.watch_id)
        if watch is None:
            continue
        summary = finalize_daily_summary(db, watch=watch, setup=setup, trading_date=trading_date)
        if summary:
            summaries.append(summary)
    postmortems = sum(_create_postmortems(db, summary) for summary in summaries)
    scope_values: set[tuple[str, str]] = {("global", "all")}
    for setup in setups:
        scope_values.add(("ticker", setup.ticker))
        if setup.setup_type:
            scope_values.add(("setup_type", setup.setup_type))
        if setup.setup_family:
            scope_values.add(("setup_family", setup.setup_family))
        if setup.sector:
            scope_values.add(("sector", setup.sector))
        if setup.market_regime:
            scope_values.add(("market_regime", setup.market_regime))
    profiles = [
        refresh_profile(db, scope_type=scope_type, scope_value=scope_value, config=config, force_version=True)
        for scope_type, scope_value in sorted(scope_values)
    ]
    observations_created = 0
    proposals_created = 0
    for profile in profiles:
        created, proposed = _generate_observation_and_proposal(db, profile=profile, config=config)
        observations_created += created
        proposals_created += proposed
    retention_cutoff = datetime.now(timezone.utc) - timedelta(days=config.bar_retention_days)
    bars_pruned = db.query(MonitorBarSummary).filter(
        MonitorBarSummary.bar_timestamp < retention_cutoff,
    ).delete(synchronize_session=False)
    run.status = "COMPLETED"
    run.summaries_finalized = len(summaries)
    run.profiles_updated = len(profiles)
    run.observations_created = observations_created
    run.details_json = _dumps({
        "postmortems_created": postmortems,
        "proposals_created": proposals_created,
        "completed_bars_pruned": bars_pruned,
    })
    run.completed_at = datetime.now(timezone.utc)
    return {
        "job_id": run.id,
        "trading_date": trading_date,
        "summaries_finalized": len(summaries),
        "profiles_updated": len(profiles),
        "observations_created": observations_created,
        "proposals_created": proposals_created,
        "postmortems_created": postmortems,
        "completed_bars_pruned": bars_pruned,
    }


def past_postmortems(db: Session, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = db.query(LLMDecisionPostmortem).filter(
        LLMDecisionPostmortem.ticker == ticker,
    ).order_by(LLMDecisionPostmortem.created_at.desc()).limit(limit).all()
    return [{
        "outcome_type": row.outcome_type,
        "original_decision": row.original_decision,
        "rationale_tags": _loads(row.rationale_tags_json, []),
        "lessons": _loads(row.lessons_json),
        "created_at": row.created_at,
    } for row in rows]


__all__ = [
    "collect_scope_observations",
    "data_quality_flags",
    "finalize_daily_summary",
    "learned_adjustment_payloads",
    "load_historical_context",
    "past_postmortems",
    "persist_adjustments",
    "persist_completed_bars",
    "refresh_profile",
    "run_daily_learning_cycle",
    "trading_day_bounds",
]
