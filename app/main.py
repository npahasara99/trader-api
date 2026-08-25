
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta, date
from collections import Counter
from .logic import bucket_news, classify_assumption
from .config import DEFAULT_PLANNING_CONFIG
from .actionability import build_actionability_soon
from .execution_view import build_chart_execution_view
from .llm_reasoning import classify_final_action, reconcile_actions
from .monitoring import build_wait_monitoring_plan
from .ranking import build_ranking_profile
from .scanner import build_pre_scan_profile, sector_benchmark_symbol_for_meta
from .suitability import build_swing_trade_suitability
from .supabase_reporting import persist_scan_workflow_to_supabase, persist_sp100_workflow_to_supabase
from .opportunity_ranking import build_portfolio_snapshot, rank_daily_opportunities
from .candidate_discovery import (
    build_multilane_candidate_order,
    classify_best_setup_quality,
    classify_search_exhaustiveness_with_coverage,
    run_adaptive_batches,
    sector_counts,
    setup_family_counts,
    validate_sp500_universe,
)
from .market_session import classify_market_session
from .universe import get_sp500_universe
from .what_to_watch import build_what_to_watch
from .watchlist import build_watchlist_profile
import json
import os
import time

from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sqlalchemy.exc import IntegrityError

from .db import Base, SessionLocal, engine, get_db
from .models import DailyBarCacheStatus, ManagedPosition, SwingDecision, DailyBar, TradingViewSignalEvent
from .bot.enums import PositionStatus
from .settings import settings
from .logic import (
    build_swing_plan,
    get_upcoming_earnings_calendar,
    get_last_price,
    evaluate_plan_row,
    get_sp100_universe,
    detect_market_regime,
    estimate_trade_probabilities,
    SP100_CLASSIFICATION,
    compute_earnings_signal,
    get_last_price_or_recent_close,
)
from .market_data import (
    ensure_cached_daily_closes,
    backfill_universe_daily_bars,
    build_bulk_cached_daily_loaders,
    fetch_finnhub_daily_bars_with_meta,
    get_bars as get_timeframe_bars,
    last_completed_market_date,
    repair_daily_bar_cache,
    resolve_expected_market_date,
)
from .bot.api import router as bot_router
from .live_monitor.api import router as live_monitor_router
from .live_monitor.service import get_live_monitor_service
from .tradingview_webhook import NormalizedTradingViewEvent, create_tradingview_router


DEFAULT_BAR_LOOKBACK_DAYS = 320


def _ensure_runtime_columns() -> None:
    required_cols = {
        "news_score": "INTEGER",
        "news_json": "TEXT",
        "earnings_score": "INTEGER",
        "earnings_context_json": "TEXT",
    }
    monitor_cols = {
        "setup_family": "VARCHAR(80)",
        "llm_proposed_levels_json": "TEXT",
        "validated_chart_levels_json": "TEXT",
        "level_sources_json": "TEXT",
        "chart_analysis_status": "VARCHAR(40) NOT NULL DEFAULT 'NOT_RUN'",
        "latest_chart_review_id": "VARCHAR(80)",
        "plan_stale_reason": "TEXT",
        "proposed_setup_json": "TEXT",
        "final_active_plan_id": "VARCHAR(80)",
        "final_active_plan_json": "TEXT",
        "final_plan_validation_json": "TEXT",
        "plan_integrity_status": "VARCHAR(20) NOT NULL DEFAULT 'INVALID'",
        "reconciliation_status": "VARCHAR(40) NOT NULL DEFAULT 'PLANNER_ACCEPTED'",
        "market_snapshot_id": "VARCHAR(80)",
        "plan_reference_price": "DOUBLE PRECISION",
        "plan_created_at": "TIMESTAMP",
        "market_data_timestamp": "TIMESTAMP",
        "plan_stale": "BOOLEAN NOT NULL DEFAULT FALSE",
        "plan_stale_reasons_json": "TEXT",
        "previous_setup_id": "VARCHAR(80)",
        "replacement_reason": "TEXT",
    }
    monitor_summary_cols = {"setup_family": "VARCHAR(80)"}
    watch_cols = {
        "market_snapshot_id": "VARCHAR(80)",
        "last_backend_evaluation_at": "TIMESTAMP",
        "last_market_data_update_at": "TIMESTAMP",
    }
    snapshot_cols = {"image_data_base64": "TEXT", "market_snapshot_id": "VARCHAR(80)"}
    review_cols = {"market_snapshot_id": "VARCHAR(80)"}
    shadow_cols = {
        "production_outcome": "VARCHAR(80)",
        "shadow_hypothetical_outcome": "VARCHAR(80)",
        "resolved_at": "TIMESTAMP",
    }
    try:
        with engine.begin() as conn:
            dialect = conn.dialect.name
            if dialect == "sqlite":
                existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(swing_decisions)")).fetchall()
                }
                for col, col_type in required_cols.items():
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE swing_decisions ADD COLUMN {col} {col_type}"))
                monitor_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(monitor_setups)")).fetchall()
                }
                for col, col_type in monitor_cols.items():
                    if col not in monitor_existing:
                        conn.execute(text(f"ALTER TABLE monitor_setups ADD COLUMN {col} {col_type}"))
                summary_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(monitor_daily_summaries)")).fetchall()
                }
                for col, col_type in monitor_summary_cols.items():
                    if col not in summary_existing:
                        conn.execute(text(f"ALTER TABLE monitor_daily_summaries ADD COLUMN {col} {col_type}"))
                watch_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(live_watches)")).fetchall()
                }
                for col, col_type in watch_cols.items():
                    if col not in watch_existing:
                        conn.execute(text(f"ALTER TABLE live_watches ADD COLUMN {col} {col_type}"))
                snapshot_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(chart_snapshots)")).fetchall()
                }
                for col, col_type in snapshot_cols.items():
                    if col not in snapshot_existing:
                        conn.execute(text(f"ALTER TABLE chart_snapshots ADD COLUMN {col} {col_type}"))
                review_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(chart_structure_reviews)")).fetchall()
                }
                for col, col_type in review_cols.items():
                    if col not in review_existing:
                        conn.execute(text(f"ALTER TABLE chart_structure_reviews ADD COLUMN {col} {col_type}"))
                shadow_existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(shadow_rule_evaluations)")).fetchall()
                }
                for col, col_type in shadow_cols.items():
                    if col not in shadow_existing:
                        conn.execute(text(f"ALTER TABLE shadow_rule_evaluations ADD COLUMN {col} {col_type}"))
                return

            for col, col_type in required_cols.items():
                conn.execute(text(f"ALTER TABLE swing_decisions ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in monitor_cols.items():
                conn.execute(text(f"ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in monitor_summary_cols.items():
                conn.execute(text(f"ALTER TABLE monitor_daily_summaries ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in watch_cols.items():
                conn.execute(text(f"ALTER TABLE live_watches ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in snapshot_cols.items():
                conn.execute(text(f"ALTER TABLE chart_snapshots ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in review_cols.items():
                conn.execute(text(f"ALTER TABLE chart_structure_reviews ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            for col, col_type in shadow_cols.items():
                conn.execute(text(f"ALTER TABLE shadow_rule_evaluations ADD COLUMN IF NOT EXISTS {col} {col_type}"))
            if dialect == "postgresql":
                conn.execute(
                    text(
                        "ALTER TABLE swing_decisions "
                        "ALTER COLUMN mode TYPE VARCHAR(80)"
                    )
                )
    except Exception as exc:
        # Do not block startup if migration cannot be applied here.
        print(f"Runtime compatibility migration warning: {type(exc).__name__}: {exc}")


# Create tables + best-effort additive columns
Base.metadata.create_all(bind=engine)
_ensure_runtime_columns()

@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    if settings.LIVE_MONITOR_ENABLED:
        get_live_monitor_service().start()
    try:
        yield
    finally:
        get_live_monitor_service().stop_service()


app = FastAPI(
    title="Trader Backend (Stocks Only)",
    version="0.1.3",
    servers=[
        {"url": "https://trader-api-production-7875.up.railway.app", "description": "Production"}
    ],
    lifespan=app_lifespan,
)


@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok"}


def require_bearer_token(authorization: Optional[str] = Header(default=None)):
    expected = os.getenv("API_BEARER_TOKEN")
    # If you haven't set a token, don't block (useful for local dev).
    if not expected:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


app.include_router(bot_router, dependencies=[Depends(require_bearer_token)])
app.include_router(live_monitor_router, dependencies=[Depends(require_bearer_token)])


def _persist_tradingview_monitoring_event(event: NormalizedTradingViewEvent) -> None:
    """Persist or de-duplicate a signal; this path cannot submit broker orders."""
    with SessionLocal() as db:
        existing = (
            db.query(TradingViewSignalEvent)
            .filter(TradingViewSignalEvent.event_id == event.event_id)
            .one_or_none()
        )
        if existing is not None:
            return
        db.add(
            TradingViewSignalEvent(
                event_id=event.event_id,
                ticker=event.ticker,
                timeframe=event.timeframe,
                event_type=event.event_type.value,
                price=event.price,
                occurred_at=event.occurred_at,
                received_at=event.received_at,
                indicators_json=json.dumps(event.indicators, default=str),
                payload_json=json.dumps(event.payload, default=str),
                processed=False,
                processing_status="pending_replan",
                re_evaluation_requested=True,
                execution_requested=False,
            )
        )
        try:
            db.commit()
        except IntegrityError:
            # TradingView may retry the same alert; deterministic event ids make
            # duplicate delivery safe and idempotent.
            db.rollback()
            duplicate = (
                db.query(TradingViewSignalEvent.id)
                .filter(TradingViewSignalEvent.event_id == event.event_id)
                .one_or_none()
            )
            if duplicate is None:
                raise


# TradingView cannot attach the normal bearer token, so this endpoint uses its
# own dedicated secret. Accepted events only queue structured re-evaluation.
app.include_router(
    create_tradingview_router(
        expected_secret=settings.TRADINGVIEW_WEBHOOK_SECRET,
        event_sink=_persist_tradingview_monitoring_event,
    )
)


# --- Requests/Responses ---
class NewsItem(BaseModel):
    headline: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    datetime: Optional[str] = None
    url: Optional[str] = None


class ScanRequest(BaseModel):
    universe: List[str]
    top_n: int = 8


class ScanResponse(BaseModel):
    tickers: List[str]


class QuoteRowOut(BaseModel):
    ticker: str
    live_price: Optional[float] = None
    live_price_asof: Optional[datetime] = None
    available: bool = False
    status: str = "unavailable"
    price_source: str = "unavailable"


class QuoteBatchResponse(BaseModel):
    as_of: datetime
    quote_count: int
    available_count: int
    unavailable_count: int
    rows: List[QuoteRowOut] = Field(default_factory=list)


class Sp100ScanRequest(BaseModel):
    top_n: int = 100
    sector: Optional[str] = None
    industry: Optional[str] = None


class PlanRequest(BaseModel):
    tickers: List[str]
    mode: str = "manual"  # manual/scan
    llm_used: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_style: Optional[str] = None


class PlanRowOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    last: Optional[float] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_date: Optional[datetime] = None

    strategy_action: Optional[str] = None
    strategy_reason: Optional[str] = None

    news: Optional[List[NewsItem]] = None
    news_score: int = 0
    earnings_score: int = 0
    earnings_context: Optional[dict] = None

    signal_score: int = 0
    market_regime: Optional[str] = None
    prob_tp: Optional[float] = None
    prob_sl: Optional[float] = None
    prob_open: Optional[float] = None
    expected_return: Optional[float] = None
    confidence: Optional[float] = None
    buy_threshold: Optional[int] = None
    avoid_threshold: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    hold_days: Optional[int] = None
    risk_tuning_reason: Optional[str] = None
    current_price: Optional[float] = None
    trend_state: Optional[str] = None
    structure_state: Optional[str] = None
    enhanced_trend_state: Optional[str] = None
    ema_structure: Optional[dict] = None
    universe_suitability: Optional[dict] = None
    universe_eligible: Optional[bool] = None
    universe_rejection_reasons: List[str] = Field(default_factory=list)
    average_daily_volume: Optional[float] = None
    liquidity_score: Optional[float] = None
    range_position_1m: Optional[float] = None
    range_position_3m: Optional[float] = None
    range_position_12m: Optional[float] = None
    local_range_position: Optional[float] = None
    distance_to_1m_high_pct: Optional[float] = None
    distance_to_1m_low_pct: Optional[float] = None
    distance_to_3m_high_pct: Optional[float] = None
    distance_to_3m_low_pct: Optional[float] = None
    distance_to_12m_high_pct: Optional[float] = None
    distance_to_12m_low_pct: Optional[float] = None
    distance_from_ema20_pct: Optional[float] = None
    distance_from_sma50_pct: Optional[float] = None
    distance_from_sma100_pct: Optional[float] = None
    distance_from_sma200_pct: Optional[float] = None
    recent_expansion_state: Optional[str] = None
    recent_compression_state: Optional[str] = None
    breakout_extension_state: Optional[str] = None
    historical_range_context: Optional[str] = None
    price_location_context: Optional[str] = None
    price_location_score: Optional[float] = None
    price_location_category: Optional[str] = None
    price_location_reasons: List[str] = Field(default_factory=list)
    consecutive_green_sessions: Optional[int] = None
    broader_structure: Optional[str] = None
    setup_type: Optional[str] = None
    setup_family: Optional[str] = None
    setup_family_score: Optional[float] = None
    setup_family_scores: Optional[dict] = None
    setup_family_components: Optional[dict] = None
    setup_family_weights: Optional[dict] = None
    setup_family_policy: Optional[dict] = None
    execution_structure: Optional[str] = None
    scenario_setup_type: Optional[str] = None
    setup_id: Optional[str] = None
    setup_created_at: Optional[str] = None
    setup_last_validated_at: Optional[str] = None
    setup_status: Optional[str] = None
    setup_invalidated_at: Optional[str] = None
    setup_invalidation_reason: Optional[str] = None
    replaced_setup: Optional[dict] = None
    catalyst_signals: List[str] = Field(default_factory=list)
    news_directional_bias: Optional[str] = None
    catalyst_strength_score: Optional[float] = None
    catalyst_recency_score: Optional[float] = None
    chart_news_alignment: Optional[str] = None
    news_supports_continuation: Optional[bool] = None
    news_supports_rebound: Optional[bool] = None
    news_conflicts_with_chart: Optional[bool] = None
    news_neutral: Optional[bool] = None
    sector_regime: Optional[str] = None
    macro_sensitivity_tag: Optional[str] = None
    macro_alignment_score: Optional[float] = None
    macro_context_label: Optional[str] = None
    setup_scenario: Optional[str] = None
    continuation_vs_reversion_bias: Optional[str] = None
    news_regime_alignment: Optional[str] = None
    tp_aggressiveness: Optional[str] = None
    sl_tolerance: Optional[str] = None
    expected_move_profile: Optional[str] = None
    scenario_confidence: Optional[float] = None
    scenario_rationale: Optional[str] = None
    chart_context: Optional[dict] = None
    timeframe_context: Optional[dict] = None
    preferred_trade_shape: Optional[str] = None
    execution_scenarios: Optional[dict] = None
    enter_now_scenario: Optional[dict] = None
    pullback_scenario: Optional[dict] = None
    breakout_scenario: Optional[dict] = None
    repair_scenario: Optional[dict] = None
    preferred_scenario: Optional[str] = None
    execution_action: Optional[str] = None
    execution_scenario_confidence: Optional[float] = None
    scenario_selection_reason: Optional[str] = None
    pullback_entry_zone: Optional[dict] = None
    breakout_trigger_zone: Optional[dict] = None
    repair_trigger_zone: Optional[dict] = None
    live_scenario_status: Optional[str] = None
    replan_needed: Optional[bool] = None
    setup_context_summary: Optional[str] = None
    location_context_summary: Optional[str] = None
    support_zone_1: Optional[dict] = None
    support_zone_2: Optional[dict] = None
    resistance_zone_1: Optional[dict] = None
    resistance_zone_2: Optional[dict] = None
    support_levels: List[dict] = Field(default_factory=list)
    resistance_levels: List[dict] = Field(default_factory=list)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    major_resistance_cluster: List[dict] = Field(default_factory=list)
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    atr_percent: Optional[float] = None
    volatility_regime: Optional[str] = None
    volatility_suitability_score: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema100: Optional[float] = None
    ema200: Optional[float] = None
    fib_levels: Optional[dict] = None
    moving_averages: Optional[dict] = None
    volume_context: Optional[dict] = None
    relative_strength: Optional[dict] = None
    earnings: Optional[dict] = None
    entry_candidates: List[dict] = Field(default_factory=list)
    preferred_entry: Optional[float] = None
    preferred_entry_type: Optional[str] = None
    entry_quality_score: Optional[float] = None
    entry_distance_from_current_price_pct: Optional[float] = None
    entry_confluence_score: Optional[float] = None
    entry_requires_confirmation: Optional[bool] = None
    confirmation_trigger: Optional[str] = None
    preferred_entry_low: Optional[float] = None
    preferred_entry_high: Optional[float] = None
    confirmation_trigger_price: Optional[float] = None
    near_confirmation: Optional[dict] = None
    primary_entry_trigger: Optional[dict] = None
    strong_confirmation: Optional[dict] = None
    major_trend_repair: Optional[dict] = None
    confirmation_levels: Optional[dict] = None
    confirmation_reason: Optional[str] = None
    confirmation_state: Optional[str] = None
    entry_status: Optional[str] = None
    confirmation_required: Optional[bool] = None
    price_confirmed: Optional[bool] = None
    volume_confirmed: Optional[bool] = None
    confirmation_score: Optional[float] = None
    confirmation_style: Optional[str] = None
    confirmation_requirements: List[str] = Field(default_factory=list)
    stop_loss: Optional[float] = None
    suggested_stop: Optional[float] = None
    invalidation_level: Optional[float] = None
    invalidation_reason: Optional[str] = None
    invalidation_width_pct: Optional[float] = None
    invalidation_width_atr: Optional[float] = None
    executable_stop_technically_valid: Optional[bool] = None
    stop_basis: Optional[str] = None
    stop_distance_pct: Optional[float] = None
    stop_width_pct: Optional[float] = None
    stop_width_atr: Optional[float] = None
    stop_too_tight_flag: Optional[bool] = None
    stop_style: Optional[str] = None
    trade_geometry_status: Optional[str] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    stretch_target: Optional[float] = None
    take_profit_final: Optional[float] = None
    tp1_distance_pct: Optional[float] = None
    tp1_distance_atr: Optional[float] = None
    tp1_atr_distance: Optional[float] = None
    tp2_atr_distance: Optional[float] = None
    tp3_atr_distance: Optional[float] = None
    tp1_reason: Optional[str] = None
    tp2_reason: Optional[str] = None
    tp3_reason: Optional[str] = None
    stretch_target_reason: Optional[str] = None
    tp_basis: Optional[str] = None
    reward_risk: Optional[dict] = None
    tp_too_optimistic_flag: Optional[bool] = None
    hold_window_reachability_score: Optional[float] = None
    target_realism_score: Optional[float] = None
    swing_realism_flag: Optional[str] = None
    risk_width_flag: Optional[str] = None
    target_reachability_flag: Optional[str] = None
    level_geometry_flag: Optional[str] = None
    stop_generation_reason: Optional[str] = None
    tp1_generation_reason: Optional[str] = None
    target_style: Optional[str] = None
    runner_plan: Optional[dict] = None
    runner_eligible: Optional[bool] = None
    tp1_partial_profit_min_pct: Optional[float] = None
    tp1_partial_profit_max_pct: Optional[float] = None
    runner_activation_level: Optional[float] = None
    runner_trailing_methods: List[str] = Field(default_factory=list)
    runner_state: Optional[str] = None
    max_hold_days: Optional[int] = None
    expected_hold_days: Optional[int] = None
    trend_quality_score: Optional[float] = None
    trend_score: Optional[float] = None
    pullback_quality_score: Optional[float] = None
    support_quality_score: Optional[float] = None
    support_confluence_score: Optional[float] = None
    volatility_quality_score: Optional[float] = None
    relative_strength_score: Optional[float] = None
    volume_confirmation_score: Optional[float] = None
    earnings_risk_score: Optional[float] = None
    reward_risk_score: Optional[float] = None
    historical_analogue_score: Optional[float] = None
    llm_quality_score: Optional[float] = None
    context_score: Optional[float] = None
    catalyst_score: Optional[float] = None
    macro_score: Optional[float] = None
    scenario_score: Optional[float] = None
    trend_strength_score: Optional[float] = None
    pullback_volume_quality: Optional[float] = None
    continuation_structure_score: Optional[float] = None
    target_quality_score: Optional[float] = None
    composite_score: Optional[float] = None
    component_scores: Optional[dict] = None
    setup_downgrade_reasons: List[str] = Field(default_factory=list)
    llm_review: Optional[dict] = None
    quant_action: Optional[str] = None
    reconciled_action: Optional[str] = None
    final_action: Optional[str] = None
    action_alignment: Optional[str] = None
    action_reason_bucket: Optional[str] = None
    monitorable_setup: Optional[bool] = None
    avoid_severity_score: Optional[float] = None
    wait_reason: Optional[str] = None
    avoid_reason: Optional[str] = None
    buy_blockers: List[str] = Field(default_factory=list)
    constructive_traits: List[str] = Field(default_factory=list)
    wait_type: Optional[str] = None
    monitor_window_days: Optional[int] = None
    monitor_until_date: Optional[datetime] = None
    stale_after_date: Optional[datetime] = None
    watch_priority: Optional[str] = None
    days_to_trigger_estimate: Optional[float] = None
    support_zone_1_display: Optional[str] = None
    support_zone_2_display: Optional[str] = None
    resistance_zone_1_display: Optional[str] = None
    resistance_zone_2_display: Optional[str] = None
    support_zone_1_midpoint: Optional[float] = None
    support_zone_2_midpoint: Optional[float] = None
    support_zone_1_width_pct: Optional[float] = None
    support_zone_2_width_pct: Optional[float] = None
    support_zone_1_note: Optional[str] = None
    support_zone_2_note: Optional[str] = None
    support_zone_summary: List[str] = Field(default_factory=list)
    resistance_zone_summary: List[str] = Field(default_factory=list)
    upgrade_triggers: List[str] = Field(default_factory=list)
    failure_triggers: List[str] = Field(default_factory=list)
    next_check_focus: List[str] = Field(default_factory=list)
    setup_monitoring_summary: Optional[str] = None
    chart_execution_view: Optional[dict] = None
    what_to_watch: Optional[dict] = None
    swing_trade_suitability: Optional[dict] = None
    actionability_soon: Optional[dict] = None
    watchlist_tier: Optional[str] = None
    watchlist_bucket: Optional[str] = None
    watchlist_summary: Optional[str] = None
    watchlist_reason: Optional[str] = None
    is_primary_watchlist_candidate: Optional[bool] = None
    is_secondary_watchlist_candidate: Optional[bool] = None
    pre_scan_score: Optional[float] = None
    legacy_pre_scan_score: Optional[float] = None
    setup_lane_qualified: Optional[bool] = None
    setup_lane_scores: Optional[dict] = None
    setup_lane_components: Optional[dict] = None
    alternative_setup_families: List[dict] = Field(default_factory=list)
    pre_scan_reason_tags: List[str] = Field(default_factory=list)
    sector_relative_strength: Optional[float] = None
    scanner_rank_score: Optional[float] = None
    immediate_rank_score: Optional[float] = None
    watchlist_rank_score: Optional[float] = None
    ranking_bucket: Optional[str] = None
    scan_shortlisted: Optional[bool] = None
    scan_rejection_reason: Optional[str] = None
    structure_flags: List[str] = Field(default_factory=list)
    breakout_level: Optional[float] = None
    prior_breakout_retest_zone: Optional[dict] = None
    consolidation_range: Optional[dict] = None
    gap_zone: Optional[dict] = None
    recent_swing_highs: List[dict] = Field(default_factory=list)
    recent_swing_lows: List[dict] = Field(default_factory=list)
    daily_trend: Optional[str] = None
    four_hour_trend: Optional[str] = None
    one_hour_trend: Optional[str] = None
    thirty_minute_trend: Optional[str] = None
    multi_timeframe_alignment_score: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    correlation_group: Optional[str] = None
    raw_setup_score: Optional[float] = None
    trade_grade: Optional[str] = None
    actionability_score: Optional[float] = None
    actionability_state: Optional[str] = None
    current_reward_risk: Optional[float] = None
    distance_to_preferred_entry_pct: Optional[float] = None
    waiting_for: List[dict] = Field(default_factory=list)
    portfolio_fit_score: Optional[float] = None
    sector_concentration_penalty: Optional[float] = None
    correlation_penalty: Optional[float] = None
    trade_today_score: Optional[float] = None
    daily_exclusion_reasons: List[str] = Field(default_factory=list)

    llm_action: Optional[str] = None
    llm_rationale: Optional[str] = None


class PlanResponse(BaseModel):
    planned_at: datetime
    market_regime: Optional[str] = None
    regime_score: Optional[float] = None
    buy_threshold: Optional[int] = None
    avoid_threshold: Optional[int] = None
    rows: List[PlanRowOut]


class LogRequest(BaseModel):
    planned_at: datetime
    mode: str = "manual"
    rows: List[PlanRowOut]
    meta: dict = Field(default_factory=dict)


class SwingPlanLogWorkflowRequest(BaseModel):
    ticker: str
    lookback_days: int = 30
    learning_limit: int = 200
    mode: str = "manual"
    llm_provider: Optional[str] = "chatgpt-actions"
    llm_model: Optional[str] = None
    llm_style: Optional[str] = "swing_v1"


class SwingPlanLogWorkflowResponse(BaseModel):
    planned_at: datetime
    ticker: str
    market_regime: Optional[str] = None
    regime_score: Optional[float] = None
    buy_threshold: Optional[int] = None
    avoid_threshold: Optional[int] = None
    learning_samples: int = 0
    learning_prompt_context: Optional[str] = None
    rows_logged: int
    logging_skipped_reason: Optional[str] = None
    row: PlanRowOut


class Sp100WorkflowRequest(BaseModel):
    top_scan: int = 100
    top_plan: int = 10
    pre_scan_shortlist: Optional[int] = None
    lookback_days: int = 180
    min_history_samples: int = 3
    sector: Optional[str] = None
    industry: Optional[str] = None
    max_hold_days: Optional[int] = None
    max_hold_date: Optional[datetime] = None
    mode: str = "sp100_auto"
    llm_provider: Optional[str] = "chatgpt-actions"
    llm_model: Optional[str] = None
    llm_style: Optional[str] = "sp100_ranker_v1"
    compact_response: bool = False


class RankedPlanOut(BaseModel):
    rank: int
    score: float
    signal_score: int
    history_boost: float = 0.0
    history_samples: int = 0
    history_win_rate: Optional[float] = None
    history_avg_return: Optional[float] = None
    row: PlanRowOut


class Sp500DailyOpportunitiesRequest(BaseModel):
    prescan_limit: int = DEFAULT_PLANNING_CONFIG.sp500_prescan_limit
    deep_analysis_limit: int = DEFAULT_PLANNING_CONFIG.sp500_deep_analysis_limit
    best_setups_count: int = DEFAULT_PLANNING_CONFIG.best_setups_count
    best_trades_today_max: int = DEFAULT_PLANNING_CONFIG.best_trades_today_max
    next_to_trigger_count: int = DEFAULT_PLANNING_CONFIG.next_to_trigger_count
    lookback_days: int = 180
    min_history_samples: int = 3
    sector: Optional[str] = None
    industry: Optional[str] = None
    mode: str = Field(default="sp500_daily_opportunities", max_length=80)
    llm_provider: Optional[str] = "chatgpt-actions"
    llm_model: Optional[str] = None
    llm_style: Optional[str] = "sp500_daily_ranker_v1"
    compact_response: bool = False
    adaptive_expansion: bool = True
    deep_analysis_batch_size: int = DEFAULT_PLANNING_CONFIG.sp500_deep_analysis_batch_size
    max_deep_analysis_limit: int = DEFAULT_PLANNING_CONFIG.sp500_max_deep_analysis_limit
    min_deep_candidates_per_sector: int = DEFAULT_PLANNING_CONFIG.sp500_min_deep_candidates_per_sector
    target_actionable_candidates: int = DEFAULT_PLANNING_CONFIG.sp500_target_actionable_candidates


class DailyOpportunityOut(BaseModel):
    rank: int
    ticker: str
    company_name: Optional[str] = None
    sector: str
    industry: str
    correlation_group: str
    setup_type: Optional[str] = None
    setup_family: Optional[str] = None
    setup_family_score: Optional[float] = None
    broader_structure: Optional[str] = None
    execution_structure: Optional[str] = None
    entry_style: Optional[str] = None
    confirmation_style: Optional[str] = None
    stop_style: Optional[str] = None
    target_style: Optional[str] = None
    trend_strength_score: Optional[float] = None
    pullback_quality_score: Optional[float] = None
    continuation_structure_score: Optional[float] = None
    grade: str
    action: str
    planner_action: Optional[str] = None
    raw_setup_score: float
    actionability_score: float
    actionability_raw: float = 0.0
    actionability_penalties: dict = Field(default_factory=dict)
    actionability_positive: List[str] = Field(default_factory=list)
    actionability_negative: List[str] = Field(default_factory=list)
    portfolio_fit_score: float
    trade_today_score: float
    actionability_state: str
    execution_timing: Optional[str] = None
    confirmation_status: str
    current_price: Optional[float] = None
    preferred_entry: Optional[float] = None
    confirmation_trigger: Optional[float] = None
    near_confirmation: Optional[dict] = None
    primary_entry_trigger: Optional[dict] = None
    strong_confirmation: Optional[dict] = None
    major_trend_repair: Optional[dict] = None
    distance_to_primary_trigger_pct: Optional[float] = None
    next_trigger_rank_score: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    stretch_target: Optional[float] = None
    runner_eligible: bool = False
    runner_plan: Optional[dict] = None
    tp1_partial_profit_min_pct: Optional[float] = None
    tp1_partial_profit_max_pct: Optional[float] = None
    runner_trailing_methods: List[str] = Field(default_factory=list)
    runner_state: Optional[str] = None
    risk_reward: Optional[dict] = None
    current_reward_risk: Optional[float] = None
    distance_to_preferred_entry_pct: Optional[float] = None
    waiting_for: List[dict] = Field(default_factory=list)
    exclusion_reasons: List[str] = Field(default_factory=list)
    raw_setup_components: dict = Field(default_factory=dict)
    actionability_components: dict = Field(default_factory=dict)
    sector_concentration_penalty: float = 0.0
    correlation_penalty: float = 0.0
    position_limit_penalty: float = 0.0
    row: Optional[PlanRowOut] = None


class Sp500DailyOpportunitiesResponse(BaseModel):
    planned_at: datetime
    market_regime: str
    universe_name: str = "SP500"
    universe_size: int
    universe_as_of: str
    universe_source: str
    universe_used_fallback: bool = False
    universe_warning: Optional[str] = None
    symbols_loaded: int = 0
    market_session: str = "closed"
    search_exhaustiveness: str = "partial"
    best_setup_quality_state: str = "no_quality_setups"
    scanned_universe_size: int
    pre_scanned_count: int
    pre_scan_shortlist_count: int
    candidates_with_price: int
    eligible_count: int
    selected_count: int
    rows_logged: int
    selection_message: str
    scan_summary: dict
    portfolio_summary: dict
    scoring_configuration: dict
    best_setups: List[DailyOpportunityOut] = Field(default_factory=list)
    best_trades_today: List[DailyOpportunityOut] = Field(default_factory=list)
    next_to_trigger: List[DailyOpportunityOut] = Field(default_factory=list)
    best_by_setup_family: dict[str, DailyOpportunityOut] = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    supabase_persisted: bool = False
    supabase_scan_run_id: Optional[str] = None
    supabase_persistence_error: Optional[str] = None


class Sp100WorkflowResponse(BaseModel):
    planned_at: datetime
    market_regime: str
    regime_score: float
    buy_threshold: int
    avoid_threshold: int
    sector: Optional[str] = None
    industry: Optional[str] = None
    max_hold_days: Optional[int] = None
    requested_max_hold_date: Optional[datetime] = None
    scanned_universe_size: int
    pre_scanned_count: int
    pre_scan_shortlist_count: int
    candidates_with_price: int
    eligible_count: int = 0
    selected_count: int
    rows_logged: int
    selection_message: Optional[str] = None
    planner_crash_count: int = 0
    planner_crashed_tickers: List[str] = Field(default_factory=list)
    planner_crash_reasons: List[str] = Field(default_factory=list)
    selected_tickers: List[str] = Field(default_factory=list)
    best_immediate_tickers: List[str] = Field(default_factory=list)
    best_watchlist_tickers: List[str] = Field(default_factory=list)
    rejected_or_low_priority_tickers: List[str] = Field(default_factory=list)
    supabase_persisted: bool = False
    supabase_scan_run_id: Optional[str] = None
    supabase_persistence_error: Optional[str] = None
    rows: List[RankedPlanOut] = Field(default_factory=list)
    best_immediate_setups: List[RankedPlanOut] = Field(default_factory=list)
    best_watchlist_setups: List[RankedPlanOut] = Field(default_factory=list)
    rejected_or_low_priority: List[RankedPlanOut] = Field(default_factory=list)

class DailyBarsBackfillRequest(BaseModel):
    symbols: Optional[List[str]] = None
    use_sp100: bool = True
    use_sp500: bool = False
    top_n: int = 100
    years: int = 10
    refresh: bool = False
    commit_every: int = 5
    start_index: int = 0
    batch_size: Optional[int] = None
    include_results: bool = True


class DailyBarsBackfillResponse(BaseModel):
    as_of: datetime
    universe_size: int
    requested_total: int
    start_index: int
    end_index: int
    processed_count: int
    remaining: int
    next_start_index: Optional[int] = None
    total: int
    updated: int
    skipped_cached: int
    failed: int
    no_data: int = 0
    results: List[dict]


class DailyBarsStatusRow(BaseModel):
    symbol: str
    count: int = 0
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    provider_symbol: Optional[str] = None
    provider: Optional[str] = None
    freshness_status: Optional[str] = None
    history_sufficient: bool = False
    last_error_code: Optional[str] = None


class DailyBarsStatusResponse(BaseModel):
    as_of: datetime
    requested_symbols: int
    symbols_with_data: int
    total_rows: int
    expected_market_date: Optional[date] = None
    symbols_current: int = 0
    symbols_stale: int = 0
    symbols_missing: int = 0
    symbols_with_sufficient_history: int = 0
    market_data_coverage_pct: float = 0.0
    provider_counts: dict = Field(default_factory=dict)
    last_backfill: Optional[datetime] = None
    rows: List[DailyBarsStatusRow]


class EarningsCalendarRowOut(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    earnings_date: str
    earnings_session: str
    earnings_time: Optional[str] = None
    days_to_earnings: int
    sector: Optional[str] = None
    industry: Optional[str] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    avg_post_earnings_move_pct: Optional[float] = None
    post_earnings_up_rate: Optional[float] = None
    reaction_samples: Optional[int] = None
    avg_surprise_percent: Optional[float] = None
    earnings_risk_flag: Optional[bool] = None


class EarningsCalendarResponse(BaseModel):
    as_of: datetime
    days_ahead: int
    sector: Optional[str] = None
    industry: Optional[str] = None
    sp100_only: bool = False
    event_count: int
    before_open_count: int
    after_close_count: int
    unknown_session_count: int
    rows: List[EarningsCalendarRowOut] = Field(default_factory=list)


class EarningsCalendarDetailResponse(EarningsCalendarRowOut):
    as_of: datetime


def _to_plan_row_out(r) -> PlanRowOut:
    return PlanRowOut(
        ticker=r.ticker,
        last=r.last,
        entry=r.entry,
        stop=r.stop,
        take_profit=r.take_profit,
        max_hold_date=r.max_hold_date,
        strategy_action=r.strategy_action,
        strategy_reason=r.strategy_reason,
        llm_action=r.llm_action,
        llm_rationale=r.llm_rationale,
        news_score=getattr(r, "news_score", 0),
        earnings_score=getattr(r, "earnings_score", 0),
        earnings_context=getattr(r, "earnings_context", None),
        signal_score=getattr(r, "signal_score", 0),
        market_regime=getattr(r, "market_regime", None),
        prob_tp=getattr(r, "prob_tp", None),
        prob_sl=getattr(r, "prob_sl", None),
        prob_open=getattr(r, "prob_open", None),
        expected_return=getattr(r, "expected_return", None),
        confidence=getattr(r, "confidence", None),
        buy_threshold=getattr(r, "buy_threshold", None),
        avoid_threshold=getattr(r, "avoid_threshold", None),
        stop_loss_pct=getattr(r, "stop_loss_pct", None),
        take_profit_pct=getattr(r, "take_profit_pct", None),
        hold_days=getattr(r, "hold_days", None),
        risk_tuning_reason=getattr(r, "risk_tuning_reason", None),
        current_price=getattr(r, "current_price", None),
        trend_state=getattr(r, "trend_state", None),
        structure_state=getattr(r, "structure_state", None),
        enhanced_trend_state=getattr(r, "enhanced_trend_state", None),
        ema_structure=getattr(r, "ema_structure", None),
        universe_suitability=getattr(r, "universe_suitability", None),
        universe_eligible=getattr(r, "universe_eligible", None),
        universe_rejection_reasons=list(getattr(r, "universe_rejection_reasons", []) or []),
        average_daily_volume=getattr(r, "average_daily_volume", None),
        liquidity_score=getattr(r, "liquidity_score", None),
        range_position_1m=getattr(r, "range_position_1m", None),
        range_position_3m=getattr(r, "range_position_3m", None),
        range_position_12m=getattr(r, "range_position_12m", None),
        local_range_position=getattr(r, "local_range_position", None),
        distance_to_1m_high_pct=getattr(r, "distance_to_1m_high_pct", None),
        distance_to_1m_low_pct=getattr(r, "distance_to_1m_low_pct", None),
        distance_to_3m_high_pct=getattr(r, "distance_to_3m_high_pct", None),
        distance_to_3m_low_pct=getattr(r, "distance_to_3m_low_pct", None),
        distance_to_12m_high_pct=getattr(r, "distance_to_12m_high_pct", None),
        distance_to_12m_low_pct=getattr(r, "distance_to_12m_low_pct", None),
        distance_from_ema20_pct=getattr(r, "distance_from_ema20_pct", None),
        distance_from_sma50_pct=getattr(r, "distance_from_sma50_pct", None),
        distance_from_sma100_pct=getattr(r, "distance_from_sma100_pct", None),
        distance_from_sma200_pct=getattr(r, "distance_from_sma200_pct", None),
        recent_expansion_state=getattr(r, "recent_expansion_state", None),
        recent_compression_state=getattr(r, "recent_compression_state", None),
        breakout_extension_state=getattr(r, "breakout_extension_state", None),
        historical_range_context=getattr(r, "historical_range_context", None),
        price_location_context=getattr(r, "price_location_context", None),
        price_location_score=getattr(r, "price_location_score", None),
        price_location_category=getattr(r, "price_location_category", None),
        price_location_reasons=list(getattr(r, "price_location_reasons", []) or []),
        consecutive_green_sessions=getattr(r, "consecutive_green_sessions", None),
        broader_structure=getattr(r, "broader_structure", None),
        setup_type=getattr(r, "setup_type", None),
        setup_family=getattr(r, "setup_family", None),
        setup_family_score=getattr(r, "setup_family_score", None),
        setup_family_scores=getattr(r, "setup_family_scores", None),
        setup_family_components=getattr(r, "setup_family_components", None),
        setup_family_weights=getattr(r, "setup_family_weights", None),
        setup_family_policy=getattr(r, "setup_family_policy", None),
        execution_structure=getattr(r, "execution_structure", None),
        scenario_setup_type=getattr(r, "scenario_setup_type", None),
        setup_id=getattr(r, "setup_id", None),
        setup_created_at=getattr(r, "setup_created_at", None),
        setup_last_validated_at=getattr(r, "setup_last_validated_at", None),
        setup_status=getattr(r, "setup_status", None),
        setup_invalidated_at=getattr(r, "setup_invalidated_at", None),
        setup_invalidation_reason=getattr(r, "setup_invalidation_reason", None),
        replaced_setup=getattr(r, "replaced_setup", None),
        catalyst_signals=list(getattr(r, "catalyst_signals", []) or []),
        news_directional_bias=getattr(r, "news_directional_bias", None),
        catalyst_strength_score=getattr(r, "catalyst_strength_score", None),
        catalyst_recency_score=getattr(r, "catalyst_recency_score", None),
        chart_news_alignment=getattr(r, "chart_news_alignment", None),
        news_supports_continuation=getattr(r, "news_supports_continuation", None),
        news_supports_rebound=getattr(r, "news_supports_rebound", None),
        news_conflicts_with_chart=getattr(r, "news_conflicts_with_chart", None),
        news_neutral=getattr(r, "news_neutral", None),
        sector_regime=getattr(r, "sector_regime", None),
        macro_sensitivity_tag=getattr(r, "macro_sensitivity_tag", None),
        macro_alignment_score=getattr(r, "macro_alignment_score", None),
        macro_context_label=getattr(r, "macro_context_label", None),
        setup_scenario=getattr(r, "setup_scenario", None),
        continuation_vs_reversion_bias=getattr(r, "continuation_vs_reversion_bias", None),
        news_regime_alignment=getattr(r, "news_regime_alignment", None),
        tp_aggressiveness=getattr(r, "tp_aggressiveness", None),
        sl_tolerance=getattr(r, "sl_tolerance", None),
        expected_move_profile=getattr(r, "expected_move_profile", None),
        scenario_confidence=getattr(r, "scenario_confidence", None),
        scenario_rationale=getattr(r, "scenario_rationale", None),
        chart_context=getattr(r, "chart_context", None),
        timeframe_context=getattr(r, "timeframe_context", None),
        preferred_trade_shape=getattr(r, "preferred_trade_shape", None),
        execution_scenarios=getattr(r, "execution_scenarios", None),
        enter_now_scenario=getattr(r, "enter_now_scenario", None),
        pullback_scenario=getattr(r, "pullback_scenario", None),
        breakout_scenario=getattr(r, "breakout_scenario", None),
        repair_scenario=getattr(r, "repair_scenario", None),
        preferred_scenario=getattr(r, "preferred_scenario", None),
        execution_action=getattr(r, "execution_action", None),
        execution_scenario_confidence=getattr(r, "execution_scenario_confidence", None),
        scenario_selection_reason=getattr(r, "scenario_selection_reason", None),
        pullback_entry_zone=getattr(r, "pullback_entry_zone", None),
        breakout_trigger_zone=getattr(r, "breakout_trigger_zone", None),
        repair_trigger_zone=getattr(r, "repair_trigger_zone", None),
        live_scenario_status=getattr(r, "live_scenario_status", None),
        replan_needed=getattr(r, "replan_needed", None),
        setup_context_summary=getattr(r, "setup_context_summary", None),
        location_context_summary=getattr(r, "location_context_summary", None),
        support_zone_1=getattr(r, "support_zone_1", None),
        support_zone_2=getattr(r, "support_zone_2", None),
        resistance_zone_1=getattr(r, "resistance_zone_1", None),
        resistance_zone_2=getattr(r, "resistance_zone_2", None),
        support_levels=list(getattr(r, "support_levels", []) or []),
        resistance_levels=list(getattr(r, "resistance_levels", []) or []),
        nearest_support=getattr(r, "nearest_support", None),
        nearest_resistance=getattr(r, "nearest_resistance", None),
        major_resistance_cluster=list(getattr(r, "major_resistance_cluster", []) or []),
        atr=getattr(r, "atr", None),
        atr_pct=getattr(r, "atr_pct", None),
        atr_percent=getattr(r, "atr_percent", None),
        volatility_regime=getattr(r, "volatility_regime", None),
        volatility_suitability_score=getattr(r, "volatility_suitability_score", None),
        ema20=getattr(r, "ema20", None),
        ema50=getattr(r, "ema50", None),
        ema100=getattr(r, "ema100", None),
        ema200=getattr(r, "ema200", None),
        fib_levels=getattr(r, "fib_levels", None),
        moving_averages=getattr(r, "moving_averages", None),
        volume_context=getattr(r, "volume_context", None),
        relative_strength=getattr(r, "relative_strength", None),
        earnings=getattr(r, "earnings", None),
        entry_candidates=list(getattr(r, "entry_candidates", []) or []),
        preferred_entry=getattr(r, "preferred_entry", None),
        preferred_entry_type=getattr(r, "preferred_entry_type", None),
        entry_quality_score=getattr(r, "entry_quality_score", None),
        entry_distance_from_current_price_pct=getattr(r, "entry_distance_from_current_price_pct", None),
        entry_confluence_score=getattr(r, "entry_confluence_score", None),
        entry_requires_confirmation=getattr(r, "entry_requires_confirmation", None),
        confirmation_trigger=getattr(r, "confirmation_trigger", None),
        preferred_entry_low=getattr(r, "preferred_entry_low", None),
        preferred_entry_high=getattr(r, "preferred_entry_high", None),
        confirmation_trigger_price=getattr(r, "confirmation_trigger_price", None),
        near_confirmation=getattr(r, "near_confirmation", None),
        primary_entry_trigger=getattr(r, "primary_entry_trigger", None),
        strong_confirmation=getattr(r, "strong_confirmation", None),
        major_trend_repair=getattr(r, "major_trend_repair", None),
        confirmation_levels=getattr(r, "confirmation_levels", None),
        confirmation_reason=getattr(r, "confirmation_reason", None),
        confirmation_state=getattr(r, "confirmation_state", None),
        entry_status=getattr(r, "entry_status", None),
        confirmation_required=getattr(r, "confirmation_required", None),
        price_confirmed=getattr(r, "price_confirmed", None),
        volume_confirmed=getattr(r, "volume_confirmed", None),
        confirmation_score=getattr(r, "confirmation_score", None),
        confirmation_style=getattr(r, "confirmation_style", None),
        confirmation_requirements=list(getattr(r, "confirmation_requirements", []) or []),
        stop_loss=getattr(r, "stop_loss", None),
        suggested_stop=getattr(r, "suggested_stop", None),
        invalidation_level=getattr(r, "invalidation_level", None),
        invalidation_reason=getattr(r, "invalidation_reason", None),
        invalidation_width_pct=getattr(r, "invalidation_width_pct", None),
        invalidation_width_atr=getattr(r, "invalidation_width_atr", None),
        executable_stop_technically_valid=getattr(r, "executable_stop_technically_valid", None),
        stop_basis=getattr(r, "stop_basis", None),
        stop_distance_pct=getattr(r, "stop_distance_pct", None),
        stop_width_pct=getattr(r, "stop_width_pct", None),
        stop_width_atr=getattr(r, "stop_width_atr", None),
        stop_too_tight_flag=getattr(r, "stop_too_tight_flag", None),
        stop_style=getattr(r, "stop_style", None),
        trade_geometry_status=getattr(r, "trade_geometry_status", None),
        take_profit_1=getattr(r, "take_profit_1", None),
        take_profit_2=getattr(r, "take_profit_2", None),
        take_profit_3=getattr(r, "take_profit_3", None),
        stretch_target=getattr(r, "stretch_target", None),
        take_profit_final=getattr(r, "take_profit_final", None),
        tp1_distance_pct=getattr(r, "tp1_distance_pct", None),
        tp1_distance_atr=getattr(r, "tp1_distance_atr", None),
        tp1_atr_distance=getattr(r, "tp1_atr_distance", None),
        tp2_atr_distance=getattr(r, "tp2_atr_distance", None),
        tp3_atr_distance=getattr(r, "tp3_atr_distance", None),
        tp1_reason=getattr(r, "tp1_reason", None),
        tp2_reason=getattr(r, "tp2_reason", None),
        tp3_reason=getattr(r, "tp3_reason", None),
        stretch_target_reason=getattr(r, "stretch_target_reason", None),
        tp_basis=getattr(r, "tp_basis", None),
        reward_risk=getattr(r, "reward_risk", None),
        tp_too_optimistic_flag=getattr(r, "tp_too_optimistic_flag", None),
        hold_window_reachability_score=getattr(r, "hold_window_reachability_score", None),
        target_realism_score=getattr(r, "target_realism_score", None),
        swing_realism_flag=getattr(r, "swing_realism_flag", None),
        risk_width_flag=getattr(r, "risk_width_flag", None),
        target_reachability_flag=getattr(r, "target_reachability_flag", None),
        level_geometry_flag=getattr(r, "level_geometry_flag", None),
        stop_generation_reason=getattr(r, "stop_generation_reason", None),
        tp1_generation_reason=getattr(r, "tp1_generation_reason", None),
        target_style=getattr(r, "target_style", None),
        runner_plan=getattr(r, "runner_plan", None),
        runner_eligible=getattr(r, "runner_eligible", None),
        tp1_partial_profit_min_pct=getattr(r, "tp1_partial_profit_min_pct", None),
        tp1_partial_profit_max_pct=getattr(r, "tp1_partial_profit_max_pct", None),
        runner_activation_level=getattr(r, "runner_activation_level", None),
        runner_trailing_methods=list(getattr(r, "runner_trailing_methods", []) or []),
        runner_state=getattr(r, "runner_state", None),
        max_hold_days=getattr(r, "max_hold_days", None),
        expected_hold_days=getattr(r, "expected_hold_days", None),
        trend_quality_score=getattr(r, "trend_quality_score", None),
        trend_score=getattr(r, "trend_score", None),
        pullback_quality_score=getattr(r, "pullback_quality_score", None),
        support_quality_score=getattr(r, "support_quality_score", None),
        support_confluence_score=getattr(r, "support_confluence_score", None),
        volatility_quality_score=getattr(r, "volatility_quality_score", None),
        relative_strength_score=getattr(r, "relative_strength_score", None),
        volume_confirmation_score=getattr(r, "volume_confirmation_score", None),
        earnings_risk_score=getattr(r, "earnings_risk_score", None),
        reward_risk_score=getattr(r, "reward_risk_score", None),
        historical_analogue_score=getattr(r, "historical_analogue_score", None),
        llm_quality_score=getattr(r, "llm_quality_score", None),
        context_score=getattr(r, "context_score", None),
        catalyst_score=getattr(r, "catalyst_score", None),
        macro_score=getattr(r, "macro_score", None),
        scenario_score=getattr(r, "scenario_score", None),
        trend_strength_score=getattr(r, "trend_strength_score", None),
        pullback_volume_quality=getattr(r, "pullback_volume_quality", None),
        continuation_structure_score=getattr(r, "continuation_structure_score", None),
        target_quality_score=getattr(r, "target_quality_score", None),
        composite_score=getattr(r, "composite_score", None),
        component_scores=getattr(r, "component_scores", None),
        setup_downgrade_reasons=list(getattr(r, "setup_downgrade_reasons", []) or []),
        llm_review=getattr(r, "llm_review", None),
        quant_action=getattr(r, "quant_action", None),
        reconciled_action=getattr(r, "reconciled_action", None),
        final_action=getattr(r, "final_action", None),
        action_alignment=getattr(r, "action_alignment", None),
        action_reason_bucket=getattr(r, "action_reason_bucket", None),
        monitorable_setup=getattr(r, "monitorable_setup", None),
        avoid_severity_score=getattr(r, "avoid_severity_score", None),
        wait_reason=getattr(r, "wait_reason", None),
        avoid_reason=getattr(r, "avoid_reason", None),
        buy_blockers=list(getattr(r, "buy_blockers", []) or []),
        constructive_traits=list(getattr(r, "constructive_traits", []) or []),
        wait_type=getattr(r, "wait_type", None),
        monitor_window_days=getattr(r, "monitor_window_days", None),
        monitor_until_date=getattr(r, "monitor_until_date", None),
        stale_after_date=getattr(r, "stale_after_date", None),
        watch_priority=getattr(r, "watch_priority", None),
        days_to_trigger_estimate=getattr(r, "days_to_trigger_estimate", None),
        support_zone_1_display=getattr(r, "support_zone_1_display", None),
        support_zone_2_display=getattr(r, "support_zone_2_display", None),
        resistance_zone_1_display=getattr(r, "resistance_zone_1_display", None),
        resistance_zone_2_display=getattr(r, "resistance_zone_2_display", None),
        support_zone_1_midpoint=getattr(r, "support_zone_1_midpoint", None),
        support_zone_2_midpoint=getattr(r, "support_zone_2_midpoint", None),
        support_zone_1_width_pct=getattr(r, "support_zone_1_width_pct", None),
        support_zone_2_width_pct=getattr(r, "support_zone_2_width_pct", None),
        support_zone_1_note=getattr(r, "support_zone_1_note", None),
        support_zone_2_note=getattr(r, "support_zone_2_note", None),
        support_zone_summary=list(getattr(r, "support_zone_summary", []) or []),
        resistance_zone_summary=list(getattr(r, "resistance_zone_summary", []) or []),
        upgrade_triggers=list(getattr(r, "upgrade_triggers", []) or []),
        failure_triggers=list(getattr(r, "failure_triggers", []) or []),
        next_check_focus=list(getattr(r, "next_check_focus", []) or []),
        setup_monitoring_summary=getattr(r, "setup_monitoring_summary", None),
        chart_execution_view=getattr(r, "chart_execution_view", None),
        what_to_watch=getattr(r, "what_to_watch", None),
        swing_trade_suitability=getattr(r, "swing_trade_suitability", None),
        actionability_soon=getattr(r, "actionability_soon", None),
        watchlist_tier=getattr(r, "watchlist_tier", None),
        watchlist_bucket=getattr(r, "watchlist_bucket", None),
        watchlist_summary=getattr(r, "watchlist_summary", None),
        watchlist_reason=getattr(r, "watchlist_reason", None),
        is_primary_watchlist_candidate=getattr(r, "is_primary_watchlist_candidate", None),
        is_secondary_watchlist_candidate=getattr(r, "is_secondary_watchlist_candidate", None),
        pre_scan_score=getattr(r, "pre_scan_score", None),
        legacy_pre_scan_score=getattr(r, "legacy_pre_scan_score", None),
        setup_lane_qualified=getattr(r, "setup_lane_qualified", None),
        setup_lane_scores=getattr(r, "setup_lane_scores", None),
        setup_lane_components=getattr(r, "setup_lane_components", None),
        alternative_setup_families=list(getattr(r, "alternative_setup_families", []) or []),
        pre_scan_reason_tags=list(getattr(r, "pre_scan_reason_tags", []) or []),
        sector_relative_strength=getattr(r, "sector_relative_strength", None),
        scanner_rank_score=getattr(r, "scanner_rank_score", None),
        immediate_rank_score=getattr(r, "immediate_rank_score", None),
        watchlist_rank_score=getattr(r, "watchlist_rank_score", None),
        ranking_bucket=getattr(r, "ranking_bucket", None),
        scan_shortlisted=getattr(r, "scan_shortlisted", None),
        scan_rejection_reason=getattr(r, "scan_rejection_reason", None),
        structure_flags=list(getattr(r, "structure_flags", []) or []),
        breakout_level=getattr(r, "breakout_level", None),
        prior_breakout_retest_zone=getattr(r, "prior_breakout_retest_zone", None),
        consolidation_range=getattr(r, "consolidation_range", None),
        gap_zone=getattr(r, "gap_zone", None),
        recent_swing_highs=list(getattr(r, "recent_swing_highs", []) or []),
        recent_swing_lows=list(getattr(r, "recent_swing_lows", []) or []),
        daily_trend=getattr(r, "daily_trend", None),
        four_hour_trend=getattr(r, "four_hour_trend", None),
        one_hour_trend=getattr(r, "one_hour_trend", None),
        thirty_minute_trend=getattr(r, "thirty_minute_trend", None),
        multi_timeframe_alignment_score=getattr(r, "multi_timeframe_alignment_score", None),
        sector=getattr(r, "sector", None),
        industry=getattr(r, "industry", None),
        correlation_group=getattr(r, "correlation_group", None),
        raw_setup_score=getattr(r, "raw_setup_score", None),
        trade_grade=getattr(r, "trade_grade", None),
        actionability_score=getattr(r, "actionability_score", None),
        actionability_state=getattr(r, "actionability_state", None),
        current_reward_risk=getattr(r, "current_reward_risk", None),
        distance_to_preferred_entry_pct=getattr(r, "distance_to_preferred_entry_pct", None),
        waiting_for=list(getattr(r, "waiting_for", []) or []),
        portfolio_fit_score=getattr(r, "portfolio_fit_score", None),
        sector_concentration_penalty=getattr(r, "sector_concentration_penalty", None),
        correlation_penalty=getattr(r, "correlation_penalty", None),
        trade_today_score=getattr(r, "trade_today_score", None),
        daily_exclusion_reasons=list(getattr(r, "daily_exclusion_reasons", []) or []),
        news=[NewsItem(**n) for n in (getattr(r, "news", None) or [])],
    )


def _queue_rows_for_logging(db: Session, *, planned_at: datetime, mode: str, rows: List[PlanRowOut], meta: dict) -> int:
    rows_logged = 0
    for r in rows:
        if r.entry is None or r.stop is None or r.take_profit is None:
            continue

        entry_val = float(r.entry)
        stop_val = float(r.stop)
        tp_val = float(r.take_profit)

        news_items = []
        for n in (r.news or []):
            if isinstance(n, dict):
                news_items.append(n)
            else:
                news_items.append(n.model_dump())

        primary_trigger = (r.primary_entry_trigger or {}).get("price") if r.primary_entry_trigger else None
        lifecycle_tags = [
            value
            for value in (
                f"setup_id:{r.setup_id}" if r.setup_id else None,
                f"setup_created_at:{r.setup_created_at}" if r.setup_created_at else None,
                f"setup_status:{r.setup_status}" if r.setup_status else None,
                f"primary_trigger:{primary_trigger}" if primary_trigger is not None else None,
            )
            if value
        ]

        db.add(
            SwingDecision(
                ticker=r.ticker,
                planned_at=planned_at,
                mode=mode,
                entry=entry_val,
                stop=stop_val,
                take_profit=tp_val,
                max_hold_date=r.max_hold_date,
                strategy_action=r.strategy_action,
                strategy_reason=r.strategy_reason,
                llm_used=bool(meta.get("llm_used", False)),
                llm_provider=meta.get("llm_provider"),
                llm_model=meta.get("llm_model"),
                llm_style=meta.get("llm_style"),
                llm_action=r.llm_action,
                llm_rationale=r.llm_rationale,
                news_score=int(r.news_score) if r.news_score is not None else None,
                earnings_score=int(r.earnings_score) if r.earnings_score is not None else None,
                earnings_context_json=(json.dumps(r.earnings_context) if r.earnings_context is not None else None),
                news_json=json.dumps(news_items),
                tags_json=json.dumps(lifecycle_tags) if lifecycle_tags else None,
            )
        )
        rows_logged += 1

    return rows_logged

def _normalize_symbols(symbols: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for s in symbols:
        sym = (s or "").strip().upper()
        if not sym or sym in seen:
            continue
        out.append(sym)
        seen.add(sym)
    return out


def _resolve_universe(
    symbols: Optional[List[str]],
    *,
    use_sp100: bool,
    use_sp500: bool = False,
    top_n: int,
) -> List[str]:
    if symbols:
        return _normalize_symbols(symbols)
    if use_sp500:
        n = max(1, min(int(top_n), 600))
        universe, _, _ = get_sp500_universe(n)
        return universe
    if use_sp100:
        n = max(1, min(int(top_n), 100))
        return get_sp100_universe(n)
    return []


def _build_daily_closes_loader(db: Session):
    memo: dict[tuple[str, date, date], dict[date, float]] = {}

    def _loader(symbol: str, frm: date, to: date) -> dict[date, float]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return {}

        key = (sym, frm, to)
        if key in memo:
            return memo[key]

        closes = ensure_cached_daily_closes(db, sym, frm, to, auto_fetch=True, commit=False)
        memo[key] = closes
        return closes

    return _loader


def _build_daily_bars_loader(db: Session):
    memo: dict[str, list[dict]] = {}

    def _loader(symbol: str) -> list[dict]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        if sym in memo:
            return memo[sym]

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=DEFAULT_BAR_LOOKBACK_DAYS)
        ensure_cached_daily_closes(db, sym, start, end, auto_fetch=True, commit=False)
        rows = (
            db.query(DailyBar)
            .filter(DailyBar.symbol == sym)
            .filter(DailyBar.bar_date >= start)
            .filter(DailyBar.bar_date <= end)
            .order_by(DailyBar.bar_date.asc())
            .all()
        )
        bars: list[dict] = []
        for row in rows:
            bars.append(
                {
                    "symbol": row.symbol,
                    "bar_date": row.bar_date,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "adjusted_close": row.adjusted_close,
                    "source": row.source,
                }
            )
        memo[sym] = bars
        return bars

    return _loader


def _rank_pre_scan_universe(
    symbols: List[str],
    *,
    daily_closes_loader,
    daily_bars_loader,
    ticker_metadata_by_ticker: dict[str, dict] | None = None,
    include_earnings: bool = True,
    prefer_bar_close: bool = False,
    allow_price_fallback: bool = True,
) -> List[dict]:
    """Cheap swing pre-scan used to shortlist names before full planning."""

    ranked: List[dict] = []
    benchmark_symbols = {"SPY", "QQQ"}
    sector_symbols: set[str] = set()
    for sym in symbols:
        metadata = (ticker_metadata_by_ticker or {}).get(sym) or SP100_CLASSIFICATION.get(sym)
        sector_symbol = sector_benchmark_symbol_for_meta(metadata)
        if sector_symbol:
            sector_symbols.add(sector_symbol)

    benchmark_bars = {}
    for symbol in sorted(benchmark_symbols | sector_symbols):
        try:
            benchmark_bars[symbol] = daily_bars_loader(symbol)
        except Exception as exc:
            print(f"Pre-scan benchmark load failed for {symbol}: {exc}")
            benchmark_bars[symbol] = []

    for sym in symbols:
        try:
            bars = daily_bars_loader(sym)
            last = None
            if prefer_bar_close and bars:
                last = bars[-1].get("close")
            if last is None and allow_price_fallback:
                last = get_last_price_or_recent_close(sym, daily_closes_loader=daily_closes_loader)
            earnings_context = None
            if include_earnings:
                _, earnings_context = compute_earnings_signal(
                    sym,
                    last,
                    daily_closes_loader=daily_closes_loader,
                )
            metadata = (ticker_metadata_by_ticker or {}).get(sym) or SP100_CLASSIFICATION.get(sym)
            profile = build_pre_scan_profile(
                ticker=sym,
                current_price=last,
                bars=bars,
                benchmark_bars=benchmark_bars,
                sector_benchmark_symbol=sector_benchmark_symbol_for_meta(metadata),
                earnings_context=earnings_context,
                config=DEFAULT_PLANNING_CONFIG,
            )
            ranked.append(profile)
        except Exception as exc:
            print(f"Pre-scan crashed for {sym}: {exc}")
            ranked.append(
                {
                    "ticker": sym,
                    "pre_scan_score": -9999.0,
                    "pre_scan_reason_tags": ["prescan_crashed"],
                    "scan_rejection_reason": "prescan_crashed",
                    "prescan_error": f"{type(exc).__name__}: {exc}",
                    "sector_relative_strength": None,
                }
            )

    ranked.sort(key=lambda item: float(item.get("pre_scan_score", 0.0)), reverse=True)
    return ranked


def _daily_bars_status_rows(db: Session, symbols: List[str]) -> List[DailyBarsStatusRow]:
    if not symbols:
        return []

    agg = (
        db.query(
            DailyBar.symbol.label("symbol"),
            func.min(DailyBar.bar_date).label("min_date"),
            func.max(DailyBar.bar_date).label("max_date"),
            func.count(DailyBar.symbol).label("count"),
        )
        .filter(DailyBar.symbol.in_(symbols))
        .group_by(DailyBar.symbol)
        .all()
    )

    by_symbol = {str(r.symbol): r for r in agg}
    status_rows = (
        db.query(DailyBarCacheStatus)
        .filter(DailyBarCacheStatus.canonical_symbol.in_(symbols))
        .all()
    )
    status_by_symbol = {row.canonical_symbol: row for row in status_rows}
    expected_date = resolve_expected_market_date(db)
    out: List[DailyBarsStatusRow] = []
    for sym in symbols:
        row = by_symbol.get(sym)
        status = status_by_symbol.get(sym)
        if row is None:
            out.append(
                DailyBarsStatusRow(
                    symbol=sym,
                    count=0,
                    min_date=None,
                    max_date=None,
                    provider_symbol=getattr(status, "provider_symbol", None),
                    provider=getattr(status, "provider", None),
                    freshness_status="CACHE_MISSING",
                    history_sufficient=False,
                    last_error_code=getattr(status, "last_error_code", None),
                )
            )
            continue

        count = int(row.count or 0)
        freshness = (
            "CACHE_STALE"
            if row.max_date and row.max_date < expected_date
            else "INSUFFICIENT_HISTORY"
            if count < DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars
            else "CURRENT"
        )
        out.append(
            DailyBarsStatusRow(
                symbol=sym,
                count=count,
                min_date=row.min_date,
                max_date=row.max_date,
                provider_symbol=getattr(status, "provider_symbol", None),
                provider=getattr(status, "provider", None),
                freshness_status=freshness,
                history_sufficient=count >= DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars,
                last_error_code=getattr(status, "last_error_code", None),
            )
        )
    return out


def _history_stats_by_ticker(db: Session, lookback_days: int) -> dict[str, dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.last_eval_return.isnot(None))
        .all()
    )

    raw: dict[str, dict] = {}
    for d in rows:
        if d.last_eval_return is None:
            continue
        r = float(d.last_eval_return)
        t = d.ticker
        s = raw.setdefault(t, {"samples": 0, "wins": 0, "ret_sum": 0.0, "abs_ret_sum": 0.0})
        s["samples"] += 1
        s["ret_sum"] += r
        s["abs_ret_sum"] += abs(r)
        if r > 0:
            s["wins"] += 1

    out: dict[str, dict] = {}
    for t, s in raw.items():
        n = s["samples"]
        out[t] = {
            "samples": n,
            "avg_return": (s["ret_sum"] / max(n, 1)),
            "avg_abs_return": (s["abs_ret_sum"] / max(n, 1)),
            "win_rate": (s["wins"] / max(n, 1)),
        }
    return out


def _rolling_performance_snapshot(db: Session, lookback_days: int = 180) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.last_eval_return.isnot(None))
        .all()
    )

    overall_samples = 0
    overall_wins = 0
    overall_sum = 0.0
    overall_abs_sum = 0.0

    buy_samples = 0
    buy_wins = 0
    buy_sum = 0.0

    for d in rows:
        if d.last_eval_return is None:
            continue
        ret = float(d.last_eval_return)
        overall_samples += 1
        overall_sum += ret
        overall_abs_sum += abs(ret)
        if ret > 0:
            overall_wins += 1

        action = (d.llm_action or d.strategy_action or "").strip().upper()
        if action.startswith("BUY"):
            buy_samples += 1
            buy_sum += ret
            if ret > 0:
                buy_wins += 1

    return {
        "overall_samples": overall_samples,
        "overall_avg_return": (overall_sum / max(overall_samples, 1)) if overall_samples else 0.0,
        "overall_abs_return": (overall_abs_sum / max(overall_samples, 1)) if overall_samples else 0.0,
        "overall_win_rate": (overall_wins / max(overall_samples, 1)) if overall_samples else 0.0,
        "buy_samples": buy_samples,
        "buy_avg_return": (buy_sum / max(buy_samples, 1)) if buy_samples else 0.0,
        "buy_win_rate": (buy_wins / max(buy_samples, 1)) if buy_samples else 0.0,
    }


def _compute_dynamic_thresholds(regime: str, perf: dict) -> dict:
    base_buy = 4
    base_avoid = -4
    if regime == "risk_on":
        base_buy = 3
        base_avoid = -5
    elif regime == "risk_off":
        base_buy = 6
        base_avoid = -3

    buy_adj = 0
    avoid_adj = 0

    if perf.get("buy_samples", 0) >= 12:
        buy_win = float(perf.get("buy_win_rate", 0.0))
        buy_avg = float(perf.get("buy_avg_return", 0.0))

        if buy_win < 0.45 or buy_avg < 0.0:
            buy_adj += 1
            avoid_adj += 1
        elif buy_win > 0.58 and buy_avg > 0.01:
            buy_adj -= 1
            avoid_adj -= 1

    overall_avg = float(perf.get("overall_avg_return", 0.0))
    if perf.get("overall_samples", 0) >= 20:
        if overall_avg < -0.004:
            buy_adj += 1
        elif overall_avg > 0.008:
            buy_adj -= 1

    buy_threshold = max(3, min(7, base_buy + buy_adj))
    avoid_threshold = max(-7, min(-2, base_avoid + avoid_adj))

    return {
        "buy_threshold": int(buy_threshold),
        "avoid_threshold": int(avoid_threshold),
    }


def _normalize_max_hold_days(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(1, min(int(value), 60))


def _resolve_requested_hold_window(
    *,
    planned_at: datetime,
    max_hold_days: Optional[int],
    max_hold_date: Optional[datetime],
) -> tuple[Optional[int], Optional[datetime]]:
    normalized_days = _normalize_max_hold_days(max_hold_days)
    normalized_date = max_hold_date

    if normalized_date is not None:
        delta_days = (normalized_date.date() - planned_at.date()).days
        date_days = max(1, min(delta_days, 60))
        if normalized_days is None or date_days < normalized_days:
            normalized_days = date_days

    return normalized_days, normalized_date


def _holding_window_message(*, max_hold_days: int, eligible_count: int, candidate_count: int, regime: str) -> str:
    if eligible_count <= 0:
        return (
            f"No setups fit a max hold of {max_hold_days} days in the current {regime} regime. "
            "The workflow kept discipline and returned no forced picks."
        )
    if eligible_count < candidate_count:
        return (
            f"Filtered for a max hold of {max_hold_days} days. "
            f"{eligible_count} of {candidate_count} priced candidates fit the holding window."
        )
    return f"All ranked candidates fit the requested max hold of {max_hold_days} days."


def _compute_adaptive_trade_levels(
    *,
    entry: float,
    ticker: str,
    regime: str,
    planned_at: datetime,
    ticker_stats: dict,
    perf: dict,
) -> dict:
    base_stop_pct = 0.03
    base_tp_pct = 0.06
    base_hold_days = 20

    samples = int(ticker_stats.get("samples", 0)) if ticker_stats else 0
    ticker_win = float(ticker_stats.get("win_rate", perf.get("overall_win_rate", 0.5))) if ticker_stats else float(perf.get("overall_win_rate", 0.5))
    ticker_avg = float(ticker_stats.get("avg_return", perf.get("overall_avg_return", 0.0))) if ticker_stats else float(perf.get("overall_avg_return", 0.0))
    ticker_abs = float(ticker_stats.get("avg_abs_return", perf.get("overall_abs_return", 0.025))) if ticker_stats else float(perf.get("overall_abs_return", 0.025))

    confidence = min(1.0, samples / 12.0)

    stop_pct = max(0.018, min(0.06, 0.012 + 0.6 * max(ticker_abs, 0.005)))
    tp_pct = base_tp_pct
    hold_days = base_hold_days

    edge = (ticker_win - 0.5) * 2.2 + ticker_avg * 24.0
    edge *= (0.55 + 0.45 * confidence)

    if edge > 0:
        tp_pct += min(0.03, 0.012 * edge)
        stop_pct += min(0.01, 0.005 * edge)
        hold_days += int(round(min(8.0, 3.0 * edge)))
    else:
        tp_pct -= min(0.02, 0.012 * abs(edge))
        stop_pct -= min(0.01, 0.004 * abs(edge))
        hold_days -= int(round(min(8.0, 3.0 * abs(edge))))

    if regime == "risk_on":
        tp_pct += 0.006
        stop_pct += 0.002
        hold_days += 2
    elif regime == "risk_off":
        tp_pct -= 0.010
        stop_pct -= 0.004
        hold_days -= 4

    stop_pct = max(0.015, min(0.065, stop_pct))
    tp_pct = max(0.03, min(0.12, tp_pct))
    hold_days = max(7, min(35, hold_days))

    stop = entry * (1.0 - stop_pct)
    take_profit = entry * (1.0 + tp_pct)
    max_hold_date = planned_at + timedelta(days=hold_days)

    reason = (
        f"adaptive-risk ticker={ticker}; samples={samples}; win={ticker_win:.2f}; "
        f"avg_ret={ticker_avg:.3f}; regime={regime}; sl%={stop_pct:.3f}; "
        f"tp%={tp_pct:.3f}; hold={hold_days}"
    )

    return {
        "stop": float(stop),
        "take_profit": float(take_profit),
        "max_hold_date": max_hold_date,
        "stop_loss_pct": float(round(stop_pct, 6)),
        "take_profit_pct": float(round(tp_pct, 6)),
        "hold_days": int(hold_days),
        "risk_tuning_reason": reason,
    }


def _apply_adaptive_risk_controls(
    row,
    *,
    planned_at: datetime,
    regime: str,
    ticker_stats: dict,
    perf: dict,
) -> None:
    if row.entry is None:
        return

    levels = _compute_adaptive_trade_levels(
        entry=float(row.entry),
        ticker=row.ticker,
        regime=regime,
        planned_at=planned_at,
        ticker_stats=ticker_stats,
        perf=perf,
    )

    row.stop = levels["stop"]
    row.take_profit = levels["take_profit"]
    row.max_hold_date = levels["max_hold_date"]
    row.stop_loss_pct = levels["stop_loss_pct"]
    row.take_profit_pct = levels["take_profit_pct"]
    row.hold_days = levels["hold_days"]
    row.risk_tuning_reason = levels["risk_tuning_reason"]

    prior = row.strategy_reason or ""
    row.strategy_reason = (prior + " | " + levels["risk_tuning_reason"]).strip(" |")


def _row_fits_hold_window(row, max_hold_days: Optional[int]) -> bool:
    if max_hold_days is None:
        return True
    hold_days = getattr(row, "hold_days", None)
    if hold_days is None:
        return False
    return int(hold_days) <= int(max_hold_days)


def _apply_prob_and_action(
    row,
    *,
    regime: str,
    buy_threshold: int,
    avoid_threshold: int,
    history_win_rate: float | None,
    history_samples: int,
) -> dict:
    if row.entry is None or row.stop is None or row.take_profit is None:
        return {
            "p_tp": None,
            "p_sl": None,
            "p_open": None,
            "expected_return": None,
            "confidence": None,
            "quant_action": "WAIT",
            "llm_action": "WAIT",
            "reconciled_action": "WAIT",
            "action": "WAIT",
        }

    signal_score = int(getattr(row, "signal_score", getattr(row, "news_score", 0) + getattr(row, "earnings_score", 0)))
    probs = estimate_trade_probabilities(
        signal_score=signal_score,
        entry=float(row.entry),
        stop=float(row.stop),
        take_profit=float(row.take_profit),
        regime=regime,
        history_win_rate=history_win_rate,
        history_samples=history_samples,
    )

    review = getattr(row, "llm_review", None) or {}
    review_action = str(review.get("llm_action") or getattr(row, "llm_action", "") or "").upper().strip()
    # Final action uses probability-aware severity scoring so "not BUY" does not
    # automatically collapse into AVOID. This keeps BUY selective while allowing
    # constructive-but-unconfirmed setups to remain WAIT/watchlist candidates.
    classification = classify_final_action(
        payload={
            "trend_state": getattr(row, "trend_state", None),
            "structure_state": getattr(row, "structure_state", None),
            "market_regime": regime,
            "buy_threshold": buy_threshold,
            "entry_quality_score": getattr(row, "entry_quality_score", None),
            "entry_requires_confirmation": getattr(row, "entry_requires_confirmation", None),
            "confirmation_trigger": getattr(row, "confirmation_trigger", None),
            "confirmation_state": getattr(row, "confirmation_state", None),
            "entry_status": getattr(row, "entry_status", None),
            "executable_stop_technically_valid": getattr(row, "executable_stop_technically_valid", None),
            "universe_eligible": getattr(row, "universe_eligible", None),
            "support_quality_score": getattr(row, "support_quality_score", None),
            "relative_strength_score": getattr(row, "relative_strength_score", None),
            "volume_confirmation_score": getattr(row, "volume_confirmation_score", None),
            "reward_risk": getattr(row, "reward_risk", None),
            "earnings": getattr(row, "earnings", None),
            "volume_context": getattr(row, "volume_context", None),
            "price_location_context": getattr(row, "price_location_context", None),
            "setup_scenario": getattr(row, "setup_scenario", None),
            "chart_news_alignment": getattr(row, "chart_news_alignment", None),
            "macro_alignment_score": getattr(row, "macro_alignment_score", None),
            "scenario_confidence": getattr(row, "scenario_confidence", None),
            "composite_score": getattr(row, "composite_score", None),
            "expected_return": probs["expected_return"],
            "prob_tp": probs["p_tp"],
            "prob_sl": probs["p_sl"],
        },
        config=DEFAULT_PLANNING_CONFIG,
    )

    quant_action = str(classification["quant_action"])
    reconciled = reconcile_actions(
        quant_action=quant_action,
        llm_action=(review_action or quant_action),
        monitorable_setup=bool(classification["monitorable_setup"]),
        avoid_severity_score=float(classification["avoid_severity_score"]),
        constructive_traits=list(classification["constructive_traits"]),
        trend_state=str(getattr(row, "trend_state", None) or ""),
        relative_strength_score=float(getattr(row, "relative_strength_score", 5.0) or 5.0),
        config=DEFAULT_PLANNING_CONFIG,
    )
    llm_action = review_action or quant_action
    action = str(reconciled["reconciled_action"])
    strategy_action = action

    row.signal_score = signal_score
    row.market_regime = regime
    row.prob_tp = probs["p_tp"]
    row.prob_sl = probs["p_sl"]
    row.prob_open = probs["p_open"]
    row.expected_return = probs["expected_return"]
    row.confidence = probs["confidence"]
    row.buy_threshold = buy_threshold
    row.avoid_threshold = avoid_threshold
    row.strategy_action = strategy_action
    row.quant_action = quant_action
    row.llm_action = llm_action
    row.reconciled_action = action
    row.final_action = action
    scenario_action = str(getattr(row, "execution_action", None) or "MONITOR")
    if action == "AVOID":
        row.execution_action = "AVOID"
    elif action == "BUY" and scenario_action == "BUY_NOW":
        row.execution_action = "BUY_NOW"
    elif scenario_action == "BUY_NOW":
        row.execution_action = "MONITOR"
    else:
        row.execution_action = scenario_action
    row.action_alignment = str(reconciled["action_alignment"])
    row.action_reason_bucket = classification["action_reason_bucket"]
    row.monitorable_setup = bool(classification["monitorable_setup"])
    row.avoid_severity_score = float(classification["avoid_severity_score"])
    row.wait_reason = classification["wait_reason"]
    row.avoid_reason = classification["avoid_reason"]
    row.buy_blockers = list(classification["buy_blockers"])
    row.constructive_traits = list(classification["constructive_traits"])
    monitoring_plan = build_wait_monitoring_plan(row, config=DEFAULT_PLANNING_CONFIG)
    if monitoring_plan:
        row.wait_type = monitoring_plan["wait_type"]
        row.monitor_window_days = monitoring_plan["monitor_window_days"]
        row.monitor_until_date = monitoring_plan["monitor_until_date"]
        row.stale_after_date = monitoring_plan["stale_after_date"]
        row.watch_priority = monitoring_plan["watch_priority"]
        row.days_to_trigger_estimate = monitoring_plan["days_to_trigger_estimate"]
        row.support_zone_1_display = monitoring_plan["support_zone_1_display"]
        row.support_zone_2_display = monitoring_plan["support_zone_2_display"]
        row.resistance_zone_1_display = monitoring_plan["resistance_zone_1_display"]
        row.resistance_zone_2_display = monitoring_plan["resistance_zone_2_display"]
        row.support_zone_1_midpoint = monitoring_plan["support_zone_1_midpoint"]
        row.support_zone_2_midpoint = monitoring_plan["support_zone_2_midpoint"]
        row.support_zone_1_width_pct = monitoring_plan["support_zone_1_width_pct"]
        row.support_zone_2_width_pct = monitoring_plan["support_zone_2_width_pct"]
        row.support_zone_1_note = monitoring_plan["support_zone_1_note"]
        row.support_zone_2_note = monitoring_plan["support_zone_2_note"]
        row.support_zone_summary = list(monitoring_plan["support_zone_summary"])
        row.resistance_zone_summary = list(monitoring_plan["resistance_zone_summary"])
        row.upgrade_triggers = list(monitoring_plan["upgrade_triggers"])
        row.failure_triggers = list(monitoring_plan["failure_triggers"])
        row.next_check_focus = list(monitoring_plan["next_check_focus"])
        row.setup_monitoring_summary = monitoring_plan["setup_monitoring_summary"]
    row.chart_execution_view = build_chart_execution_view(row, config=DEFAULT_PLANNING_CONFIG)
    row.what_to_watch = build_what_to_watch(row, config=DEFAULT_PLANNING_CONFIG)
    row.swing_trade_suitability = build_swing_trade_suitability(row, config=DEFAULT_PLANNING_CONFIG)
    watchlist_profile = build_watchlist_profile(row, config=DEFAULT_PLANNING_CONFIG)
    row.watchlist_tier = watchlist_profile["watchlist_tier"]
    row.watchlist_bucket = watchlist_profile["watchlist_bucket"]
    row.watchlist_summary = watchlist_profile["watchlist_summary"]
    row.watchlist_reason = watchlist_profile["watchlist_reason"]
    row.is_primary_watchlist_candidate = watchlist_profile["is_primary_watchlist_candidate"]
    row.is_secondary_watchlist_candidate = watchlist_profile["is_secondary_watchlist_candidate"]
    row.actionability_soon = build_actionability_soon(row, config=DEFAULT_PLANNING_CONFIG)
    ranking_profile = build_ranking_profile(row, config=DEFAULT_PLANNING_CONFIG)
    row.immediate_rank_score = ranking_profile["immediate_rank_score"]
    row.watchlist_rank_score = ranking_profile["watchlist_rank_score"]
    row.scanner_rank_score = ranking_profile["scanner_rank_score"]
    row.ranking_bucket = ranking_profile["ranking_bucket"]
    rationale_bits = list(review.get("rationale") or [])
    rationale_bits.append(
        f"regime={regime}; signal={signal_score}; p_tp={probs['p_tp']:.2f}; "
        f"p_sl={probs['p_sl']:.2f}; exp_ret={probs['expected_return']:.3f}; "
        f"confidence={probs['confidence']:.2f}; history_samples={history_samples}; "
        f"quant_action={quant_action}; llm_action={llm_action}; final_action={action}; "
        f"severity={classification['avoid_severity_score']:.2f}; alignment={reconciled['action_alignment']}"
    )
    row.llm_rationale = " | ".join(rationale_bits)

    return {
        "p_tp": probs["p_tp"],
        "p_sl": probs["p_sl"],
        "p_open": probs["p_open"],
        "expected_return": probs["expected_return"],
        "confidence": probs["confidence"],
        "quant_action": quant_action,
        "llm_action": llm_action,
        "reconciled_action": action,
        "action": action,
        "classification": classification,
        "action_alignment": reconciled["action_alignment"],
    }


@app.get("/debug/model")
def debug_model(_=Depends(require_bearer_token)):
    cols = list(SwingDecision.__table__.columns.keys())
    return {"columns": cols}


@app.get("/debug/finnhub")
def debug_finnhub(_=Depends(require_bearer_token)):
    today = datetime.now(timezone.utc).date()
    bars, fetch_status = fetch_finnhub_daily_bars_with_meta("AAPL", today - timedelta(days=14), today)
    last = get_last_price("AAPL")
    return {
        "finnhub_key_configured": bool(os.getenv("FINNHUB_API_KEY")),
        "candle_fetch_status": fetch_status,
        "candle_bars": len(bars),
        "aapl_last_price": last,
    }


@app.get("/market/quotes", response_model=QuoteBatchResponse)
def get_market_quotes(
    tickers: str = Query(..., description="Comma-separated ticker list"),
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    normalized_tickers = _normalize_symbols(str(tickers or "").split(","))
    if not normalized_tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")

    as_of = datetime.now(timezone.utc)
    daily_closes_loader = _build_daily_closes_loader(db)
    rows: list[QuoteRowOut] = []
    for ticker in normalized_tickers[:100]:
        live_price = None
        live_price_asof = None
        price_source = "unavailable"
        status = "unavailable"
        try:
            live_price = get_last_price(ticker)
        except Exception:
            live_price = None
        if live_price is not None:
            live_price_asof = as_of
            price_source = "live_quote"
            status = "ok"
        else:
            end = as_of.date()
            start = end - timedelta(days=14)
            try:
                close_map = daily_closes_loader(ticker, start, end)
            except Exception:
                close_map = {}
            if close_map:
                latest_day = max(close_map.keys())
                live_price = float(close_map[latest_day])
                live_price_asof = datetime.combine(latest_day, datetime.min.time(), tzinfo=timezone.utc)
                price_source = "recent_close_fallback"
                status = "stale_close"
        rows.append(
            QuoteRowOut(
                ticker=ticker,
                live_price=live_price,
                live_price_asof=live_price_asof,
                available=live_price is not None,
                status=status,
                price_source=price_source,
            )
        )

    available_count = sum(1 for row in rows if row.available)
    return QuoteBatchResponse(
        as_of=as_of,
        quote_count=len(rows),
        available_count=available_count,
        unavailable_count=len(rows) - available_count,
        rows=rows,
    )


@app.get("/scan/sp100", response_model=ScanResponse)
def scan_sp100(
    top_n: int = 100,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    universe = get_sp100_universe(None, sector=sector, industry=industry)
    ranked = _rank_pre_scan_universe(
        universe,
        daily_closes_loader=_build_daily_closes_loader(db),
        daily_bars_loader=_build_daily_bars_loader(db),
    )
    n = max(1, min(int(top_n), len(ranked))) if ranked else 0
    return {"tickers": [row["ticker"] for row in ranked[:n]]}


@app.get("/calendar/earnings", response_model=EarningsCalendarResponse)
def get_earnings_calendar(
    days_ahead: int = Query(default=30, ge=1, le=90),
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    sp100_only: bool = False,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    rows = get_upcoming_earnings_calendar(
        days_ahead=days_ahead,
        sector=sector,
        industry=industry,
        sp100_only=sp100_only,
    )
    normalized_rows: list[dict] = []
    for row in rows:
        event_days = row.get("days_to_earnings")
        normalized_rows.append(
            {
                **row,
                "avg_post_earnings_move_pct": None,
                "post_earnings_up_rate": None,
                "reaction_samples": None,
                "avg_surprise_percent": None,
                "earnings_risk_flag": bool(event_days is not None and int(event_days) <= 10),
            }
        )

    before_open_count = sum(1 for row in normalized_rows if row.get("earnings_session") == "before_open")
    after_close_count = sum(1 for row in normalized_rows if row.get("earnings_session") == "after_close")
    unknown_session_count = sum(1 for row in normalized_rows if row.get("earnings_session") == "unknown")
    return EarningsCalendarResponse(
        as_of=datetime.now(timezone.utc),
        days_ahead=days_ahead,
        sector=sector,
        industry=industry,
        sp100_only=sp100_only,
        event_count=len(normalized_rows),
        before_open_count=before_open_count,
        after_close_count=after_close_count,
        unknown_session_count=unknown_session_count,
        rows=[EarningsCalendarRowOut(**row) for row in normalized_rows],
    )


@app.get("/calendar/earnings/{ticker}", response_model=EarningsCalendarDetailResponse)
def get_earnings_calendar_detail(
    ticker: str,
    days_ahead: int = Query(default=30, ge=1, le=90),
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    matching_rows = get_upcoming_earnings_calendar(
        days_ahead=days_ahead,
        tickers=[normalized_ticker],
    )
    if not matching_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No upcoming earnings event found for {normalized_ticker} in the next {days_ahead} days.",
        )

    event_row = matching_rows[0]
    daily_closes_loader = _build_daily_closes_loader(db)
    earnings_context = {}
    try:
        last_price = get_last_price_or_recent_close(
            normalized_ticker,
            daily_closes_loader=daily_closes_loader,
        )
        _, earnings_context = compute_earnings_signal(
            normalized_ticker,
            last_price,
            daily_closes_loader=daily_closes_loader,
        )
    except Exception:
        earnings_context = {}

    event_days = event_row.get("days_to_earnings")
    return EarningsCalendarDetailResponse(
        as_of=datetime.now(timezone.utc),
        **event_row,
        avg_post_earnings_move_pct=earnings_context.get("avg_post_earnings_move_pct"),
        post_earnings_up_rate=earnings_context.get("post_earnings_up_rate"),
        reaction_samples=earnings_context.get("reaction_samples"),
        avg_surprise_percent=earnings_context.get("avg_surprise_percent"),
        earnings_risk_flag=bool(event_days is not None and int(event_days) <= 10),
    )


@app.post("/scan/sp100", response_model=ScanResponse)
def scan_sp100_post(req: Sp100ScanRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    """POST variant of SP100 scan for Action clients that behave better with bodies."""

    universe = get_sp100_universe(None, sector=req.sector, industry=req.industry)
    ranked = _rank_pre_scan_universe(
        universe,
        daily_closes_loader=_build_daily_closes_loader(db),
        daily_bars_loader=_build_daily_bars_loader(db),
    )
    n = max(1, min(int(req.top_n), len(ranked))) if ranked else 0
    return {"tickers": [row["ticker"] for row in ranked[:n]]}


@app.post("/scan/swing", response_model=ScanResponse)
def scan_swing(req: ScanRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    ranked = _rank_pre_scan_universe(
        _normalize_symbols(req.universe),
        daily_closes_loader=_build_daily_closes_loader(db),
        daily_bars_loader=_build_daily_bars_loader(db),
    )
    n = max(1, min(int(req.top_n), len(ranked))) if ranked else 0
    return {"tickers": [row["ticker"] for row in ranked[:n]]}


@app.post("/plan/swing", response_model=PlanResponse)
def plan_swing(req: PlanRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)
    tickers = _normalize_symbols(req.tickers)

    daily_closes_loader = _build_daily_closes_loader(db)
    daily_bars_loader = _build_daily_bars_loader(db)
    regime_snapshot = detect_market_regime(tickers, daily_closes_loader=daily_closes_loader)
    perf = _rolling_performance_snapshot(db, lookback_days=180)
    thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)
    ticker_hist = _history_stats_by_ticker(db, lookback_days=180)
    ranked_prescan = _rank_pre_scan_universe(
        tickers,
        daily_closes_loader=daily_closes_loader,
        daily_bars_loader=daily_bars_loader,
    )
    pre_scan_by_ticker = {
        item["ticker"]: {
            **item,
            "scan_shortlisted": True,
            "scan_rejection_reason": None,
        }
        for item in ranked_prescan
    }

    try:
        rows = build_swing_plan(
            tickers,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            daily_closes_loader=daily_closes_loader,
            daily_bars_loader=daily_bars_loader,
            timeframe_bars_loader=get_timeframe_bars,
            history_stats_by_ticker=ticker_hist,
            pre_scan_by_ticker=pre_scan_by_ticker,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_style=req.llm_style,
        )
    except Exception as e:
        out = [
            PlanRowOut(
                ticker=t,
                last=None,
                entry=None,
                stop=None,
                take_profit=None,
                max_hold_date=datetime.now(timezone.utc),
                strategy_action="WAIT",
                strategy_reason=f"Planner crashed: {e}",
                news=[],
                news_score=0,
                earnings_score=0,
                earnings_context=None,
                signal_score=0,
                market_regime=regime_snapshot["regime"],
                buy_threshold=thresholds["buy_threshold"],
                avoid_threshold=thresholds["avoid_threshold"],
                pre_scan_score=(pre_scan_by_ticker.get(t, {}) or {}).get("pre_scan_score"),
                pre_scan_reason_tags=list((pre_scan_by_ticker.get(t, {}) or {}).get("pre_scan_reason_tags") or []),
                sector_relative_strength=(pre_scan_by_ticker.get(t, {}) or {}).get("sector_relative_strength"),
                scan_shortlisted=True,
                scan_rejection_reason="planner_crashed",
                llm_action="WAIT",
                llm_rationale="no-data",
            )
            for t in tickers
        ]
        return {
            "planned_at": planned_at,
            "market_regime": regime_snapshot["regime"],
            "regime_score": regime_snapshot["score"],
            "buy_threshold": thresholds["buy_threshold"],
            "avoid_threshold": thresholds["avoid_threshold"],
            "rows": out,
        }

    for r in rows:
        h = ticker_hist.get(r.ticker, {})
        _apply_prob_and_action(
            r,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            history_win_rate=(float(h["win_rate"]) if "win_rate" in h else None),
            history_samples=int(h.get("samples", 0)),
        )

    out = [_to_plan_row_out(r) for r in rows]
    return {
        "planned_at": planned_at,
        "market_regime": regime_snapshot["regime"],
        "regime_score": regime_snapshot["score"],
        "buy_threshold": thresholds["buy_threshold"],
        "avoid_threshold": thresholds["avoid_threshold"],
        "rows": out,
    }


def _apply_daily_opportunity_fields(candidate: dict) -> None:
    """Expose daily ranking fields on the underlying planner row for logging/debugging."""

    row = candidate["row"]
    row.sector = candidate["sector"]
    row.industry = candidate["industry"]
    row.correlation_group = candidate["correlation_group"]
    row.raw_setup_score = candidate["raw_setup_score"]
    row.trade_grade = candidate["grade"]
    row.actionability_score = candidate["actionability_score"]
    row.actionability_state = candidate["actionability_state"]
    row.current_reward_risk = candidate["current_reward_risk"]
    row.distance_to_preferred_entry_pct = candidate["distance_to_preferred_entry_pct"]
    row.waiting_for = list(candidate["waiting_for"] or [])
    row.portfolio_fit_score = candidate["portfolio_fit_score"]
    row.sector_concentration_penalty = candidate["sector_concentration_penalty"]
    row.correlation_penalty = candidate["correlation_penalty"]
    row.trade_today_score = candidate["trade_today_score"]
    row.daily_exclusion_reasons = list(candidate["exclusion_reasons"] or [])


def _daily_opportunity_out(candidate: dict, *, compact: bool) -> DailyOpportunityOut:
    _apply_daily_opportunity_fields(candidate)
    return DailyOpportunityOut(
        rank=int(candidate["rank"]),
        ticker=candidate["ticker"],
        company_name=candidate.get("company_name"),
        sector=candidate["sector"],
        industry=candidate["industry"],
        correlation_group=candidate["correlation_group"],
        setup_type=candidate.get("setup_type"),
        setup_family=candidate.get("setup_family"),
        setup_family_score=candidate.get("setup_family_score"),
        broader_structure=candidate.get("broader_structure"),
        execution_structure=candidate.get("execution_structure"),
        entry_style=candidate.get("entry_style"),
        confirmation_style=candidate.get("confirmation_style"),
        stop_style=candidate.get("stop_style"),
        target_style=candidate.get("target_style"),
        trend_strength_score=candidate.get("trend_strength_score"),
        pullback_quality_score=candidate.get("pullback_quality_score"),
        continuation_structure_score=candidate.get("continuation_structure_score"),
        grade=candidate["grade"],
        action=candidate["action"],
        planner_action=candidate.get("planner_action"),
        raw_setup_score=candidate["raw_setup_score"],
        actionability_raw=candidate.get("actionability_raw", candidate["actionability_score"]),
        actionability_penalties=dict(candidate.get("actionability_penalties") or {}),
        actionability_positive=list(candidate.get("actionability_positive") or []),
        actionability_negative=list(candidate.get("actionability_negative") or []),
        actionability_score=candidate["actionability_score"],
        portfolio_fit_score=candidate["portfolio_fit_score"],
        trade_today_score=candidate["trade_today_score"],
        actionability_state=candidate["actionability_state"],
        execution_timing=candidate.get("execution_timing"),
        confirmation_status=candidate["confirmation_status"],
        current_price=candidate.get("current_price"),
        preferred_entry=candidate.get("preferred_entry"),
        confirmation_trigger=candidate.get("confirmation_trigger"),
        near_confirmation=candidate.get("near_confirmation"),
        primary_entry_trigger=candidate.get("primary_entry_trigger"),
        strong_confirmation=candidate.get("strong_confirmation"),
        major_trend_repair=candidate.get("major_trend_repair"),
        distance_to_primary_trigger_pct=candidate.get("distance_to_primary_trigger_pct"),
        next_trigger_rank_score=candidate.get("next_trigger_rank_score"),
        stop_loss=candidate.get("stop_loss"),
        take_profit_1=candidate.get("take_profit_1"),
        take_profit_2=candidate.get("take_profit_2"),
        take_profit_3=candidate.get("take_profit_3"),
        stretch_target=candidate.get("stretch_target"),
        runner_eligible=bool(candidate.get("runner_eligible")),
        runner_plan=candidate.get("runner_plan"),
        tp1_partial_profit_min_pct=candidate.get("tp1_partial_profit_min_pct"),
        tp1_partial_profit_max_pct=candidate.get("tp1_partial_profit_max_pct"),
        runner_trailing_methods=list(candidate.get("runner_trailing_methods") or []),
        runner_state=candidate.get("runner_state"),
        risk_reward=candidate.get("risk_reward"),
        current_reward_risk=candidate.get("current_reward_risk"),
        distance_to_preferred_entry_pct=candidate.get("distance_to_preferred_entry_pct"),
        waiting_for=list(candidate.get("waiting_for") or []),
        exclusion_reasons=list(candidate.get("exclusion_reasons") or []),
        raw_setup_components=dict(candidate.get("raw_setup_components") or {}),
        actionability_components=dict(candidate.get("actionability_components") or {}),
        sector_concentration_penalty=float(candidate.get("sector_concentration_penalty") or 0.0),
        correlation_penalty=float(candidate.get("correlation_penalty") or 0.0),
        position_limit_penalty=float(candidate.get("position_limit_penalty") or 0.0),
        row=None if compact else _to_plan_row_out(candidate["row"]),
    )


@app.post("/workflow/sp500/daily-opportunities", response_model=Sp500DailyOpportunitiesResponse)
def workflow_sp500_daily_opportunities(
    req: Sp500DailyOpportunitiesRequest,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    """Run the staged S&P 500 scan and return three distinct daily leaderboards."""

    workflow_started = time.monotonic()
    planned_at = datetime.now(timezone.utc)
    market_session = classify_market_session(planned_at)
    prescan_limit = max(10, min(int(req.prescan_limit), 200))
    max_deep_limit = max(
        1,
        min(int(req.max_deep_analysis_limit), prescan_limit, DEFAULT_PLANNING_CONFIG.sp500_max_deep_analysis_limit),
    )
    deep_limit = max(1, min(int(req.deep_analysis_limit), max_deep_limit))
    deep_batch_size = max(1, min(int(req.deep_analysis_batch_size), max_deep_limit))
    target_actionable = max(1, min(int(req.target_actionable_candidates), 5))
    min_per_sector = max(0, min(int(req.min_deep_candidates_per_sector), 10))
    best_setups_count = max(1, min(int(req.best_setups_count), deep_limit, 25))
    best_trades_max = max(0, min(int(req.best_trades_today_max), 5))
    next_to_trigger_count = max(1, min(int(req.next_to_trigger_count), 10))
    lookback_days = max(30, min(int(req.lookback_days), 720))
    min_history_samples = max(1, min(int(req.min_history_samples), 20))

    base_universe, universe_snapshot, metadata_by_ticker = get_sp500_universe(
        sector=req.sector,
        industry=req.industry,
    )
    universe_validation = validate_sp500_universe(
        universe_size=len(base_universe),
        sector_filter=req.sector,
        industry_filter=req.industry,
        minimum_broad_size=DEFAULT_PLANNING_CONFIG.sp500_universe_minimum_broad_size,
    )
    sector_benchmarks = {
        benchmark
        for ticker in base_universe
        if (benchmark := sector_benchmark_symbol_for_meta(metadata_by_ticker.get(ticker)))
    }
    prescan_support_symbols = [
        *base_universe,
        *DEFAULT_PLANNING_CONFIG.benchmark_symbols,
        *sorted(sector_benchmarks),
    ]
    benchmark_symbols = sorted({
        *DEFAULT_PLANNING_CONFIG.benchmark_symbols,
        *sector_benchmarks,
    })
    benchmark_repair: dict = {}
    market_data_repair: dict = {}
    if DEFAULT_PLANNING_CONFIG.sp500_market_data_auto_repair:
        try:
            # Repair benchmarks first so their latest completed session resolves
            # exchange holidays for the broader constituent refresh.
            benchmark_repair = repair_daily_bar_cache(
                db,
                benchmark_symbols,
                history_days=DEFAULT_PLANNING_CONFIG.sp500_market_data_history_days,
                min_history_bars=DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars,
                max_workers=DEFAULT_PLANNING_CONFIG.sp500_market_data_max_workers,
                commit_every=DEFAULT_PLANNING_CONFIG.sp500_market_data_commit_every,
                incremental_overlap_days=DEFAULT_PLANNING_CONFIG.sp500_market_data_incremental_overlap_days,
            )
            expected_market_date = resolve_expected_market_date(
                db,
                benchmark_symbols=tuple(DEFAULT_PLANNING_CONFIG.benchmark_symbols),
                at=planned_at,
            )
            market_data_repair = repair_daily_bar_cache(
                db,
                base_universe,
                history_days=DEFAULT_PLANNING_CONFIG.sp500_market_data_history_days,
                min_history_bars=DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars,
                expected_date=expected_market_date,
                max_workers=DEFAULT_PLANNING_CONFIG.sp500_market_data_max_workers,
                commit_every=DEFAULT_PLANNING_CONFIG.sp500_market_data_commit_every,
                incremental_overlap_days=DEFAULT_PLANNING_CONFIG.sp500_market_data_incremental_overlap_days,
            )
            print(
                "SP500 daily-bar repair complete: "
                f"universe={len(base_universe)} attempted={market_data_repair.get('fetch_attempted', 0)} "
                f"success={market_data_repair.get('fetch_success', 0)} "
                f"failed={market_data_repair.get('fetch_failed', 0)} "
                f"current={market_data_repair.get('current', 0)} "
                f"seconds={market_data_repair.get('duration_seconds', 0)}"
            )
        except Exception as exc:
            db.rollback()
            expected_market_date = last_completed_market_date(planned_at)
            market_data_repair = {
                "fetch_attempted": 0,
                "fetch_success": 0,
                "fetch_failed": len(base_universe),
                "failure_reasons": [{
                    "ticker": None,
                    "reason": "FETCH_FAILED",
                    "details": f"repair_service:{type(exc).__name__}: {exc}",
                }],
            }
            print(f"SP500 daily-bar repair warning: {type(exc).__name__}: {exc}")
    else:
        expected_market_date = resolve_expected_market_date(
            db,
            benchmark_symbols=tuple(DEFAULT_PLANNING_CONFIG.benchmark_symbols),
            at=planned_at,
        )
    prescan_closes_loader, prescan_bars_loader, prescan_cache_coverage = build_bulk_cached_daily_loaders(
        db,
        prescan_support_symbols,
        lookback_days=max(
            DEFAULT_BAR_LOOKBACK_DAYS,
            DEFAULT_PLANNING_CONFIG.sp500_market_data_history_days,
        ),
        max_age_days=DEFAULT_PLANNING_CONFIG.sp500_prescan_cache_max_age_days,
        min_history_bars=DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars,
        expected_market_date=expected_market_date,
    )
    prescan_cache_coverage["constituent_symbols"] = len(base_universe)
    prescan_cache_coverage["constituents_current"] = sum(
        1 for ticker in base_universe if prescan_bars_loader(ticker)
    )
    prescan_cache_coverage["constituents_with_sufficient_history"] = sum(
        1
        for ticker in base_universe
        if len(prescan_bars_loader(ticker)) >= DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars
    )
    prescan_cache_coverage["market_data_coverage_pct"] = round(
        prescan_cache_coverage["constituents_current"] / max(len(base_universe), 1),
        4,
    )
    prescan_cache_coverage["history_coverage_pct"] = round(
        prescan_cache_coverage["constituents_with_sufficient_history"] / max(len(base_universe), 1),
        4,
    )
    prescan_cache_coverage["effective_search_coverage_pct"] = min(
        prescan_cache_coverage["market_data_coverage_pct"],
        prescan_cache_coverage["history_coverage_pct"],
    )
    prescan_cache_coverage["repair"] = {
        key: value
        for key, value in market_data_repair.items()
        if key not in {"results"}
    }
    prescan_cache_coverage["benchmark_repair"] = {
        key: value
        for key, value in benchmark_repair.items()
        if key not in {"results"}
    }
    prescan_cache_coverage["missing_symbols"] = list(prescan_cache_coverage.get("missing_symbols") or [])[:25]
    prescan_cache_coverage["stale_symbols"] = list(prescan_cache_coverage.get("stale_symbols") or [])[:25]
    broad_scope = not bool((req.sector or "").strip() or (req.industry or "").strip())
    market_data_minimum = max(
        1,
        int(len(base_universe) * DEFAULT_PLANNING_CONFIG.sp500_market_data_min_coverage_pct + 0.9999),
    )
    market_data_validation = {
        "status": "valid",
        "valid": True,
        "expected_minimum": market_data_minimum if broad_scope else None,
        "warning": None,
    }
    if (
        prescan_cache_coverage["constituents_current"] < market_data_minimum
        or prescan_cache_coverage["constituents_with_sufficient_history"] < market_data_minimum
    ):
        market_data_validation = {
            "status": "SCAN_DATA_INCOMPLETE",
            "valid": False,
            "expected_minimum": market_data_minimum if broad_scope else None,
            "coverage_pct": prescan_cache_coverage["market_data_coverage_pct"],
            "warning": (
                "SCAN INCOMPLETE - market data is available for only "
                f"{prescan_cache_coverage['constituents_current']} of {len(base_universe)} requested constituents. "
                f"Only {prescan_cache_coverage['constituents_with_sufficient_history']} have sufficient technical history. "
                "Scanner conclusions are not representative."
            ),
        }
    market_data_is_complete = bool(market_data_validation.get("valid"))
    ranked_prescan = _rank_pre_scan_universe(
        base_universe,
        daily_closes_loader=prescan_closes_loader,
        daily_bars_loader=prescan_bars_loader,
        ticker_metadata_by_ticker=metadata_by_ticker,
        include_earnings=False,
        prefer_bar_close=True,
        allow_price_fallback=False,
    )
    prescan_seconds = round(time.monotonic() - workflow_started, 3)
    print(
        "SP500 workflow pre-scan complete: "
        f"universe={len(base_universe)} current_cache={prescan_cache_coverage['constituents_current']} "
        f"eligible={sum(1 for item in ranked_prescan if not item.get('scan_rejection_reason'))} "
        f"seconds={prescan_seconds}"
    )
    prescan_failures = [item for item in ranked_prescan if item.get("prescan_error")]
    prescan_rejected = [
        item for item in ranked_prescan
        if item.get("scan_rejection_reason") and not item.get("prescan_error")
    ]
    prescan_passed = [item for item in ranked_prescan if not item.get("scan_rejection_reason")]
    sector_aware_order = build_multilane_candidate_order(
        prescan_passed,
        metadata_by_ticker=metadata_by_ticker,
        initial_limit=deep_limit,
        min_per_sector=min_per_sector,
        minimum_by_family=DEFAULT_PLANNING_CONFIG.setup_lane_min_candidates,
        minimum_family_score=DEFAULT_PLANNING_CONFIG.setup_lane_min_score,
    )
    prescan_ranked = sector_aware_order[:prescan_limit]
    shortlist = prescan_ranked[:deep_limit]
    deep_universe = [item["ticker"] for item in shortlist]
    pre_scan_by_ticker = {
        item["ticker"]: {**item, "scan_shortlisted": True}
        for item in shortlist
    }

    # Deep analysis may request richer/timeframe data after broad daily bars
    # have already been repaired and loaded from the persistent cache.
    daily_closes_loader = _build_daily_closes_loader(db)
    daily_bars_loader = _build_daily_bars_loader(db)
    deep_analysis_started = time.monotonic()

    try:
        if not deep_universe:
            raise ValueError("No current cached constituents were available for regime breadth")
        regime_snapshot = detect_market_regime(deep_universe[:20], daily_closes_loader=daily_closes_loader)
    except Exception as exc:
        regime_snapshot = {
            "regime": "neutral",
            "score": 0.0,
            "spy_price": None,
            "spy_ma20": None,
            "spy_ma50": None,
            "breadth_ratio": None,
            "breadth_samples": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        perf = _rolling_performance_snapshot(db, lookback_days=lookback_days)
    except Exception:
        perf = {
            "overall_samples": 0,
            "overall_avg_return": 0.0,
            "overall_abs_return": 0.0,
            "overall_win_rate": 0.0,
            "buy_samples": 0,
            "buy_avg_return": 0.0,
            "buy_win_rate": 0.0,
        }
    thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)
    try:
        history_stats = _history_stats_by_ticker(db, lookback_days=lookback_days)
    except Exception:
        history_stats = {}

    previous_setup_error = None
    previous_setup_by_ticker: dict[str, dict] = {}
    try:
        prior_decisions = (
            db.query(SwingDecision)
            .filter(SwingDecision.ticker.in_([item["ticker"] for item in prescan_ranked]))
            .order_by(SwingDecision.planned_at.desc())
            .all()
        )
        for decision in prior_decisions:
            if decision.ticker in previous_setup_by_ticker:
                continue
            lifecycle_values: dict[str, str] = {}
            try:
                for tag in json.loads(decision.tags_json or "[]"):
                    key, separator, value = str(tag).partition(":")
                    if separator:
                        lifecycle_values[key] = value
            except (TypeError, ValueError, json.JSONDecodeError):
                lifecycle_values = {}
            prior_trigger = lifecycle_values.get("primary_trigger")
            previous_setup_by_ticker[decision.ticker] = {
                "setup_id": lifecycle_values.get("setup_id") or f"legacy-decision-{decision.id}",
                "setup_created_at": lifecycle_values.get("setup_created_at") or decision.planned_at.isoformat(),
                "invalidation_level": float(decision.stop),
                # Legacy rows fall back to entry as context only. New rows store
                # the actual primary trigger in tags_json.
                "primary_entry_trigger": {
                    "price": float(prior_trigger) if prior_trigger is not None else float(decision.entry)
                },
            }
    except Exception as exc:
        previous_setup_error = f"{type(exc).__name__}: {exc}"

    portfolio_error = None
    try:
        active_statuses = {
            PositionStatus.OPEN.value,
            PositionStatus.EXTERNAL.value,
            PositionStatus.RECONCILIATION_REQUIRED.value,
        }
        position_rows = db.query(ManagedPosition).filter(ManagedPosition.status.in_(active_statuses)).all()
        position_payloads = [
            {
                "ticker": position.ticker,
                "status": position.status,
                "quantity": position.quantity,
                "average_entry_price": position.average_entry_price,
            }
            for position in position_rows
        ]
    except Exception as exc:
        position_payloads = []
        portfolio_error = f"{type(exc).__name__}: {exc}"

    portfolio = build_portfolio_snapshot(
        position_payloads,
        metadata_by_ticker=metadata_by_ticker,
        max_positions=int(settings.MAX_OPEN_POSITIONS),
        trading_budget=float(settings.TRADING_BUDGET),
    )

    def analyze_candidate_batch(batch_items: list[dict]):
        batch_tickers = [item["ticker"] for item in batch_items]
        batch_pre_scan = {
            item["ticker"]: {**item, "scan_shortlisted": True}
            for item in batch_items
        }
        batch_rows = build_swing_plan(
            batch_tickers,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            daily_closes_loader=daily_closes_loader,
            daily_bars_loader=daily_bars_loader,
            timeframe_bars_loader=get_timeframe_bars,
            history_stats_by_ticker=history_stats,
            pre_scan_by_ticker=batch_pre_scan,
            ticker_metadata_by_ticker=metadata_by_ticker,
            previous_setup_by_ticker=previous_setup_by_ticker,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_style=req.llm_style,
        )
        for row in batch_rows:
            if row.entry is None or row.stop is None or row.take_profit is None:
                continue
            stats = history_stats.get(row.ticker) or {}
            _apply_prob_and_action(
                row,
                regime=regime_snapshot["regime"],
                buy_threshold=thresholds["buy_threshold"],
                avoid_threshold=thresholds["avoid_threshold"],
                history_win_rate=(
                    float(stats["win_rate"])
                    if stats.get("samples", 0) >= min_history_samples
                    else None
                ),
                history_samples=int(stats.get("samples", 0)),
            )
        return batch_rows

    def count_strict_actionable(candidate_rows) -> int:
        interim = rank_daily_opportunities(
            candidate_rows,
            metadata_by_ticker=metadata_by_ticker,
            market_regime=regime_snapshot["regime"],
            portfolio=portfolio,
            best_setups_count=max(best_setups_count, target_actionable),
            best_trades_max=target_actionable,
            next_to_trigger_count=next_to_trigger_count,
        )
        return len(interim["best_trades_today"])

    rows, adaptive_history = run_adaptive_batches(
        prescan_ranked,
        initial_limit=deep_limit,
        batch_size=deep_batch_size,
        maximum_limit=max_deep_limit,
        target_actionable=target_actionable,
        adaptive=bool(req.adaptive_expansion),
        analyze_batch=analyze_candidate_batch,
        count_actionable=count_strict_actionable,
    ) if prescan_ranked else ([], [])
    deep_analysis_seconds = round(time.monotonic() - deep_analysis_started, 3)
    print(
        "SP500 workflow deep analysis complete: "
        f"initial={len(deep_universe)} analyzed={len(rows)} batches={len(adaptive_history)} "
        f"seconds={deep_analysis_seconds}"
    )

    candidates_with_price = sum(
        1 for row in rows if row.entry is not None and row.stop is not None and row.take_profit is not None
    )
    ranking = rank_daily_opportunities(
        rows,
        metadata_by_ticker=metadata_by_ticker,
        market_regime=regime_snapshot["regime"],
        portfolio=portfolio,
        best_setups_count=best_setups_count,
        best_trades_max=best_trades_max,
        next_to_trigger_count=next_to_trigger_count,
    )
    for bucket_name in ("all_candidates", "best_setups", "best_trades_today", "next_to_trigger"):
        for candidate in ranking[bucket_name]:
            if candidate["actionability_state"] == "actionable":
                candidate["execution_timing"] = (
                    "actionable_now" if market_session == "regular" else "ready_for_next_session"
                )
            else:
                candidate["execution_timing"] = candidate["actionability_state"]
    for candidate in ranking["all_candidates"]:
        _apply_daily_opportunity_fields(candidate)

    best_setups = [_daily_opportunity_out(item, compact=req.compact_response) for item in ranking["best_setups"]]
    best_trades_today = [_daily_opportunity_out(item, compact=req.compact_response) for item in ranking["best_trades_today"]]
    next_to_trigger = [_daily_opportunity_out(item, compact=req.compact_response) for item in ranking["next_to_trigger"]]
    best_by_setup_family = {
        family: _daily_opportunity_out(item, compact=req.compact_response)
        for family, item in ranking["best_by_setup_family"].items()
    }

    ranked_rows = [
        RankedPlanOut(
            rank=item.rank,
            score=item.raw_setup_score,
            signal_score=int(getattr(ranking["best_setups"][item.rank - 1]["row"], "signal_score", 0) or 0),
            row=_to_plan_row_out(ranking["best_setups"][item.rank - 1]["row"]),
        )
        for item in best_setups
    ]
    meta = {
        "llm_used": True,
        "llm_provider": req.llm_provider,
        "llm_model": req.llm_model,
        "llm_style": req.llm_style,
    }
    rows_logged = 0
    if market_data_is_complete:
        try:
            rows_logged = _queue_rows_for_logging(
                db,
                planned_at=planned_at,
                mode=req.mode,
                rows=[item.row for item in ranked_rows],
                meta=meta,
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"SP500 workflow logging failed: {exc}")

    failed_rows = ranking["failures"]
    failure_reasons = [
        *[
            {"ticker": item.get("ticker"), "reason": "prescan_crashed", "details": item.get("prescan_error")}
            for item in prescan_failures
        ],
        *failed_rows,
    ]
    rejection_reason_aliases = {
        "price_below_minimum": "price_too_low",
        "average_daily_volume_below_minimum": "low_liquidity",
        "planner_crashed": "data_failure",
        "prescan_crashed": "data_failure",
    }
    rejection_reason_counts = Counter(
        rejection_reason_aliases.get(reason.strip(), reason.strip())
        for item in prescan_rejected
        for reason in str(item.get("scan_rejection_reason") or "missing_required_data").split(",")
        if reason.strip()
    )
    actionable_count = sum(
        1 for item in ranking["all_candidates"] if item["actionability_state"] == "actionable"
    )
    a_grade_count = sum(
        1 for item in ranking["all_candidates"] if item["grade"] in {"A-", "A", "A+"}
    )
    search_exhaustiveness = classify_search_exhaustiveness_with_coverage(
        analyzed=len(rows),
        viable=len(prescan_ranked),
        initial_limit=deep_limit,
        maximum_limit=max_deep_limit,
        data_coverage_pct=float(prescan_cache_coverage.get("effective_search_coverage_pct") or 0.0),
        minimum_data_coverage_pct=DEFAULT_PLANNING_CONFIG.sp500_market_data_min_coverage_pct,
    )
    best_setup_quality_state = classify_best_setup_quality(ranking["all_candidates"])
    market_data_tickers = [ticker for ticker in base_universe if prescan_bars_loader(ticker)]
    valid_setup_tickers = [item["ticker"] for item in ranking["all_candidates"]]
    best_setup_tickers = [item["ticker"] for item in ranking["best_setups"]]
    sector_stage_counts = {
        "universe_sector_counts": sector_counts(base_universe, metadata_by_ticker),
        "market_data_sector_counts": sector_counts(market_data_tickers, metadata_by_ticker),
        "suitability_sector_counts": sector_counts(prescan_passed, metadata_by_ticker),
        "prescan_sector_counts": sector_counts(prescan_ranked, metadata_by_ticker),
        "initial_shortlist_sector_counts": sector_counts(shortlist, metadata_by_ticker),
        "deep_analysis_sector_counts": sector_counts(rows, metadata_by_ticker),
        "valid_setup_sector_counts": sector_counts(valid_setup_tickers, metadata_by_ticker),
        "best_setup_sector_counts": sector_counts(best_setup_tickers, metadata_by_ticker),
    }
    history_sufficient_tickers = [
        ticker
        for ticker in base_universe
        if len(prescan_bars_loader(ticker)) >= DEFAULT_PLANNING_CONFIG.sp500_market_data_min_history_bars
    ]
    failed_market_data_tickers = {
        str(item.get("ticker") or "")
        for item in (market_data_repair.get("failure_reasons") or [])
        if item.get("ticker")
    }
    universe_sector_counts = sector_counts(base_universe, metadata_by_ticker)
    current_sector_counts = sector_counts(market_data_tickers, metadata_by_ticker)
    history_sector_counts = sector_counts(history_sufficient_tickers, metadata_by_ticker)
    failed_sector_counts = sector_counts(failed_market_data_tickers, metadata_by_ticker)
    market_data_sector_coverage = {
        sector: {
            "universe": total,
            "current": current_sector_counts.get(sector, 0),
            "history_sufficient": history_sector_counts.get(sector, 0),
            "failed": failed_sector_counts.get(sector, 0),
            "coverage_pct": round(current_sector_counts.get(sector, 0) / max(total, 1), 4),
        }
        for sector, total in universe_sector_counts.items()
    }
    setup_family_stage_counts = {
        "prescan_primary_family_counts": setup_family_counts(prescan_passed),
        "prescan_lane_qualified_counts": {
            family: sum(
                1
                for item in prescan_passed
                if float((item.get("setup_lane_scores") or {}).get(family) or 0.0)
                >= DEFAULT_PLANNING_CONFIG.setup_lane_min_score
            )
            for family in DEFAULT_PLANNING_CONFIG.setup_lane_min_candidates
        },
        "initial_shortlist_counts": setup_family_counts(shortlist),
        "deep_analysis_counts": setup_family_counts(rows),
        "valid_setup_counts": setup_family_counts(ranking["all_candidates"]),
        "best_setup_counts": setup_family_counts(ranking["best_setups"]),
        "actionable_counts": setup_family_counts(
            item for item in ranking["all_candidates"] if item["actionability_state"] == "actionable"
        ),
    }
    strategy_dominance_warnings: list[str] = []
    for stage, counts in setup_family_stage_counts.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        family, family_count = max(counts.items(), key=lambda item: item[1])
        share = family_count / total
        if share >= DEFAULT_PLANNING_CONFIG.setup_lane_dominance_threshold:
            strategy_dominance_warnings.append(
                f"{stage}: {family} represents {share:.0%} of classified candidates"
            )
    failed_symbol_count = len({item.get("ticker") for item in failure_reasons if item.get("ticker")})
    candidate_funnel = {
        "universe_loaded": len(base_universe),
        "market_data_success": len(market_data_tickers),
        "technical_history_sufficient": len(history_sufficient_tickers),
        "prescan_evaluated": len(history_sufficient_tickers),
        "basic_suitability_passed": len(prescan_passed),
        "prescan_passed": len(prescan_ranked),
        "initial_shortlisted": len(shortlist),
        "deep_analyzed": len(rows),
        "valid_setups": len(ranking["all_candidates"]),
        "a_grade_setups": a_grade_count,
        "actionable_setups": actionable_count,
        "market_data_failed": int(market_data_repair.get("fetch_failed") or 0),
        "failed_symbols": failed_symbol_count,
    }
    data_incomplete = search_exhaustiveness == "data_incomplete"
    if data_incomplete:
        selection_message = (
            "SCAN INCOMPLETE - MARKET DATA AVAILABLE FOR ONLY "
            f"{len(market_data_tickers)} OF {len(base_universe)} CONSTITUENTS. "
            f"SUFFICIENT TECHNICAL HISTORY EXISTS FOR {len(history_sufficient_tickers)}. "
            "Trade-quality conclusions are suppressed until data coverage recovers."
        )
    elif not shortlist:
        selection_message = (
            "No constituents passed the current technical prescreen after broad market-data validation."
        )
    elif best_trades_today:
        selection_message = (
            f"{len(best_trades_today)} high-quality trade"
            f"{'s' if len(best_trades_today) != 1 else ''} confirmed today."
        )
    else:
        selection_message = (
            "NO HIGH-QUALITY TRADE CURRENTLY CONFIRMED. "
            f"Deep analysis covered {len(rows)} of {len(prescan_ranked)} viable prescan candidates "
            f"({search_exhaustiveness})."
        )
    scan_summary = {
        "universe": "SP500",
        "universe_size": len(base_universe),
        "symbols_loaded": len(base_universe),
        "market_data_success": len(market_data_tickers),
        "suitability_passed": len(prescan_passed),
        "prescan_passed": len(prescan_ranked),
        "shortlisted": len(shortlist),
        "initial_deep_analysis_size": len(shortlist),
        "expanded_deep_analysis_size": max(len(rows) - len(shortlist), 0),
        "deep_analyzed": len(rows),
        "actionable_count": actionable_count,
        "a_grade_setup_count": a_grade_count,
        "search_exhaustiveness": search_exhaustiveness,
        "best_setup_quality_state": best_setup_quality_state,
        "market_session": market_session,
        "market_regime": regime_snapshot["regime"],
        "cached_constituents_current": prescan_cache_coverage["constituents_current"],
        "cached_constituents_with_sufficient_history": prescan_cache_coverage[
            "constituents_with_sufficient_history"
        ],
        "market_data_coverage_pct": prescan_cache_coverage["market_data_coverage_pct"],
        "history_sufficient": len(history_sufficient_tickers),
        "prescan_evaluated": len(history_sufficient_tickers),
        "target_actionable_trades_per_day": DEFAULT_PLANNING_CONFIG.target_actionable_trades_per_day,
        "min_required_trades_per_day": DEFAULT_PLANNING_CONFIG.min_required_trades_per_day,
        "results_representative": not data_incomplete,
    }
    portfolio_summary = {
        "max_positions": portfolio.max_positions,
        "open_positions": portfolio.open_positions,
        "available_position_slots": portfolio.available_position_slots,
        "max_new_trades_today": min(best_trades_max, DEFAULT_PLANNING_CONFIG.max_new_trades_per_day),
        "trading_budget": portfolio.trading_budget,
        "capital_in_use": portfolio.capital_in_use,
        "available_capital": portfolio.available_capital,
        "sector_exposures": portfolio.sector_exposures,
        "correlation_exposures": portfolio.correlation_exposures,
    }
    scoring_configuration = {
        "minimums": {
            "grade": DEFAULT_PLANNING_CONFIG.min_actionable_grade,
            "raw_setup_score": DEFAULT_PLANNING_CONFIG.min_raw_setup_score,
            "actionability_score": DEFAULT_PLANNING_CONFIG.min_actionability_score,
            "portfolio_fit_score": DEFAULT_PLANNING_CONFIG.min_portfolio_fit_score,
        },
        "raw_setup_weights": DEFAULT_PLANNING_CONFIG.raw_setup_weights,
        "actionability_weights": DEFAULT_PLANNING_CONFIG.daily_actionability_weights,
        "trade_today_weights": DEFAULT_PLANNING_CONFIG.trade_today_weights,
        "setup_family_score_weights": DEFAULT_PLANNING_CONFIG.setup_family_score_weights,
        "setup_lane_min_candidates": DEFAULT_PLANNING_CONFIG.setup_lane_min_candidates,
        "setup_lane_min_score": DEFAULT_PLANNING_CONFIG.setup_lane_min_score,
        "portfolio_limits": {
            "max_per_sector": DEFAULT_PLANNING_CONFIG.max_open_positions_per_sector,
            "max_per_correlation_group": DEFAULT_PLANNING_CONFIG.max_open_positions_per_correlation_group,
        },
    }
    diagnostics = {
        "universe_validation": universe_validation,
        "market_data_validation": market_data_validation,
        "candidate_funnel": candidate_funnel,
        "sector_stage_counts": sector_stage_counts,
        "market_data_sector_coverage": market_data_sector_coverage,
        "setup_family_stage_counts": setup_family_stage_counts,
        "strategy_dominance_warnings": strategy_dominance_warnings,
        "adaptive_expansion": {
            "enabled": bool(req.adaptive_expansion),
            "target_actionable": target_actionable,
            "initial_limit": deep_limit,
            "batch_size": deep_batch_size,
            "maximum_limit": max_deep_limit,
            "batches": adaptive_history,
            "remaining_viable_candidates": max(len(prescan_ranked) - len(rows), 0),
            "search_exhaustiveness": search_exhaustiveness,
        },
        "successful_tickers": [item["ticker"] for item in ranking["all_candidates"]],
        "failed_tickers": [item.get("ticker") for item in failure_reasons],
        "failure_reasons": failure_reasons,
        "prescan_rejection_counts": dict(rejection_reason_counts),
        "prescan_rejected_count": len(prescan_rejected),
        "market_regime_details": regime_snapshot,
        "portfolio_read_error": portfolio_error,
        "previous_setup_read_error": previous_setup_error,
        "daily_bar_cache_coverage": prescan_cache_coverage,
        "market_data_repair": {
            key: value
            for key, value in market_data_repair.items()
            if key not in {"results"}
        },
        "performance": {
            "scan_duration_seconds": round(time.monotonic() - workflow_started, 3),
            "market_data_requests": int(market_data_repair.get("fetch_attempted") or 0),
            "market_data_request_count_available": True,
            "cache_hits": int(market_data_repair.get("cache_hits") or 0),
            "symbols_failed": int(market_data_repair.get("fetch_failed") or 0) + failed_symbol_count,
        },
        "stage_seconds": {
            "prescan": prescan_seconds,
            "deep_analysis": deep_analysis_seconds,
            "total_before_reporting_persistence": round(time.monotonic() - workflow_started, 3),
        },
    }
    response = Sp500DailyOpportunitiesResponse(
        planned_at=planned_at,
        market_regime=regime_snapshot["regime"],
        universe_size=len(base_universe),
        symbols_loaded=len(base_universe),
        universe_as_of=universe_snapshot.as_of,
        universe_source=universe_snapshot.source,
        universe_used_fallback=universe_snapshot.used_fallback,
        universe_warning=universe_snapshot.warning or universe_validation.get("warning"),
        market_session=market_session,
        search_exhaustiveness=search_exhaustiveness,
        best_setup_quality_state=best_setup_quality_state,
        scanned_universe_size=len(base_universe),
        pre_scanned_count=len(prescan_ranked),
        pre_scan_shortlist_count=len(shortlist),
        candidates_with_price=candidates_with_price,
        eligible_count=len(ranking["all_candidates"]),
        selected_count=0 if data_incomplete else len(best_setups),
        rows_logged=rows_logged,
        selection_message=selection_message,
        scan_summary=scan_summary,
        portfolio_summary=portfolio_summary,
        scoring_configuration=scoring_configuration,
        best_setups=[] if data_incomplete else best_setups,
        best_trades_today=[] if data_incomplete else best_trades_today,
        next_to_trigger=[] if data_incomplete else next_to_trigger,
        best_by_setup_family={} if data_incomplete else best_by_setup_family,
        diagnostics=diagnostics,
    )
    supabase_status = persist_scan_workflow_to_supabase(
        workflow_type="sp500_daily_opportunities",
        workflow_request=req,
        workflow_response=response,
        selected_rows=[] if data_incomplete else ranked_rows,
    )
    response.supabase_persisted = bool((supabase_status or {}).get("persisted"))
    response.supabase_scan_run_id = (supabase_status or {}).get("scan_run_id")
    response.supabase_persistence_error = (supabase_status or {}).get("error")
    print(
        "SP500 workflow complete: "
        f"selected={response.selected_count} rows_logged={response.rows_logged} "
        f"seconds={round(time.monotonic() - workflow_started, 3)}"
    )
    return response


@app.post("/workflow/sp100/top10-log", response_model=Sp100WorkflowResponse)
def workflow_sp100_top10_log(req: Sp100WorkflowRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)
    top_scan = max(10, min(int(req.top_scan), 100))
    top_plan = max(1, min(int(req.top_plan), 20))
    pre_scan_shortlist = max(top_plan, min(int(req.pre_scan_shortlist or DEFAULT_PLANNING_CONFIG.pre_scan_shortlist_size), 60))
    lookback_days = max(30, min(int(req.lookback_days), 720))
    min_history_samples = max(1, min(int(req.min_history_samples), 20))
    max_hold_days, requested_max_hold_date = _resolve_requested_hold_window(
        planned_at=planned_at,
        max_hold_days=req.max_hold_days,
        max_hold_date=req.max_hold_date,
    )

    base_universe = get_sp100_universe(None, sector=req.sector, industry=req.industry)
    daily_closes_loader = _build_daily_closes_loader(db)
    daily_bars_loader = _build_daily_bars_loader(db)
    if not base_universe:
        return Sp100WorkflowResponse(
            planned_at=planned_at,
            market_regime="neutral",
            regime_score=0.0,
            buy_threshold=4,
            avoid_threshold=-4,
            sector=req.sector,
            industry=req.industry,
            max_hold_days=max_hold_days,
            requested_max_hold_date=requested_max_hold_date,
            scanned_universe_size=0,
            pre_scanned_count=0,
            pre_scan_shortlist_count=0,
            candidates_with_price=0,
            eligible_count=0,
            selected_count=0,
            rows_logged=0,
            selection_message="No SP100 stocks matched the requested sector/industry filter.",
            planner_crash_count=0,
            planner_crashed_tickers=[],
            planner_crash_reasons=[],
            selected_tickers=[],
            best_immediate_tickers=[],
            best_watchlist_tickers=[],
            rejected_or_low_priority_tickers=[],
            rows=[],
        )

    ranked_prescan = _rank_pre_scan_universe(
        base_universe,
        daily_closes_loader=daily_closes_loader,
        daily_bars_loader=daily_bars_loader,
    )
    eligible_prescan = [item for item in ranked_prescan if not item.get("scan_rejection_reason")]
    pre_scanned = eligible_prescan[:top_scan]
    shortlist = pre_scanned[: min(pre_scan_shortlist, len(pre_scanned))]
    universe = [item["ticker"] for item in shortlist]
    pre_scan_by_ticker = {
        item["ticker"]: {
            **item,
            "scan_shortlisted": not bool(item.get("scan_rejection_reason")),
            "scan_rejection_reason": item.get("scan_rejection_reason"),
        }
        for item in shortlist
    }

    try:
        regime_snapshot = detect_market_regime(universe[:20], daily_closes_loader=daily_closes_loader)
    except Exception as exc:
        print(f"Market regime detection failed during SP100 workflow: {exc}")
        regime_snapshot = {
            "as_of": planned_at,
            "regime": "neutral",
            "score": 0.0,
            "spy_price": None,
            "spy_ma20": None,
            "spy_ma50": None,
            "breadth_ratio": None,
            "breadth_samples": 0,
        }

    try:
        perf = _rolling_performance_snapshot(db, lookback_days=lookback_days)
    except Exception as exc:
        print(f"Rolling performance snapshot failed during SP100 workflow: {exc}")
        perf = {
            "overall_samples": 0,
            "overall_avg_return": 0.0,
            "overall_abs_return": 0.0,
            "overall_win_rate": 0.0,
            "buy_samples": 0,
            "buy_avg_return": 0.0,
            "buy_win_rate": 0.0,
        }

    thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)

    try:
        history_stats = _history_stats_by_ticker(db, lookback_days=lookback_days)
    except Exception as exc:
        print(f"History stats lookup failed during SP100 workflow: {exc}")
        history_stats = {}

    rows = build_swing_plan(
        universe,
        regime=regime_snapshot["regime"],
        buy_threshold=thresholds["buy_threshold"],
        avoid_threshold=thresholds["avoid_threshold"],
        daily_closes_loader=daily_closes_loader,
        daily_bars_loader=daily_bars_loader,
        timeframe_bars_loader=get_timeframe_bars,
        history_stats_by_ticker=history_stats,
        pre_scan_by_ticker=pre_scan_by_ticker,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        llm_style=req.llm_style,
    )

    planner_crashed_rows = [
        r for r in rows
        if str(getattr(r, "scan_rejection_reason", None) or "") == "planner_crashed"
    ]
    planner_crashed_tickers = [r.ticker for r in planner_crashed_rows]
    planner_crash_reasons = [
        f"{r.ticker}: {getattr(r, 'strategy_reason', 'planner crashed')}"
        for r in planner_crashed_rows[:12]
    ]

    ranked: list[dict] = []
    skipped_rows: list = []
    priced_candidates = 0
    eligible_count = 0
    for r in rows:
        if r.entry is None or r.stop is None or r.take_profit is None:
            skipped_rows.append(r)
            continue
        priced_candidates += 1

        h = history_stats.get(r.ticker)
        history_samples = 0
        history_win_rate = None
        history_avg_return = None
        history_boost = 0.0

        if h:
            history_samples = int(h.get("samples", 0))
            history_win_rate = float(h.get("win_rate"))
            history_avg_return = float(h.get("avg_return"))
            if history_samples >= min_history_samples:
                confidence = min(1.0, history_samples / 8.0)
                hist_raw = (history_avg_return * 100.0) * 0.35 + (history_win_rate - 0.5) * 4.0
                history_boost = max(-3.0, min(3.0, hist_raw * confidence))

        if not _row_fits_hold_window(r, max_hold_days):
            continue
        eligible_count += 1

        decision = _apply_prob_and_action(
            r,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            history_win_rate=history_win_rate,
            history_samples=history_samples,
        )

        signal_score = int(getattr(r, "signal_score", 0))
        score = float(getattr(r, "scanner_rank_score", 0.0) or 0.0) + float(history_boost)

        ranked.append(
            {
                "score": score,
                "signal_score": signal_score,
                "history_boost": history_boost,
                "history_samples": history_samples,
                "history_win_rate": history_win_rate,
                "history_avg_return": history_avg_return,
                "row": r,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    selected = ranked[:top_plan]
    selection_message = (
        _holding_window_message(
            max_hold_days=max_hold_days,
            eligible_count=eligible_count,
            candidate_count=priced_candidates,
            regime=regime_snapshot["regime"],
        )
        if max_hold_days is not None
        else f"Pre-scanned {len(pre_scanned)} names and fully planned {len(shortlist)} shortlisted candidates."
    )
    if not ranked and planner_crashed_rows:
        preview = ", ".join(planner_crashed_tickers[:6])
        suffix = "..." if len(planner_crashed_tickers) > 6 else ""
        reason_preview = " | ".join(planner_crash_reasons[:3])
        selection_message = (
            f"No ranked setups were produced because the structured planner crashed for "
            f"{len(planner_crashed_rows)} tickers. Affected names: {preview}{suffix}. "
            f"Crash details: {reason_preview}"
        )
    elif planner_crashed_rows:
        reason_preview = " | ".join(planner_crash_reasons[:2])
        selection_message = (
            f"{selection_message} Structured planner crashes were isolated for "
            f"{len(planner_crashed_rows)} tickers. Crash details: {reason_preview}"
        )

    out_rows: list[RankedPlanOut] = []
    for idx, item in enumerate(selected, start=1):
        row_out = _to_plan_row_out(item["row"])
        out_rows.append(
            RankedPlanOut(
                rank=idx,
                score=float(round(item["score"], 4)),
                signal_score=int(item["signal_score"]),
                history_boost=float(round(item["history_boost"], 4)),
                history_samples=int(item["history_samples"]),
                history_win_rate=item["history_win_rate"],
                history_avg_return=item["history_avg_return"],
                row=row_out,
            )
        )

    immediate_items = [item for item in ranked if getattr(item["row"], "ranking_bucket", None) == "best_immediate_setups"]
    watchlist_items = [item for item in ranked if getattr(item["row"], "ranking_bucket", None) == "best_watchlist_setups"]
    rejected_items = [item for item in ranked if getattr(item["row"], "ranking_bucket", None) == "rejected_or_low_priority"]
    if not rejected_items and skipped_rows:
        rejected_items = [
            {
                "score": -999.0,
                "signal_score": int(getattr(r, "signal_score", 0) or 0),
                "history_boost": 0.0,
                "history_samples": 0,
                "history_win_rate": None,
                "history_avg_return": None,
                "row": r,
            }
            for r in skipped_rows[:top_plan]
        ]

    def _ranked_rows_for(items: list[dict], limit: int) -> list[RankedPlanOut]:
        out: list[RankedPlanOut] = []
        for idx, item in enumerate(items[:limit], start=1):
            out.append(
                RankedPlanOut(
                    rank=idx,
                    score=float(round(item["score"], 4)),
                    signal_score=int(item["signal_score"]),
                    history_boost=float(round(item["history_boost"], 4)),
                    history_samples=int(item["history_samples"]),
                    history_win_rate=item["history_win_rate"],
                    history_avg_return=item["history_avg_return"],
                    row=_to_plan_row_out(item["row"]),
                )
            )
        return out

    best_immediate_setups = _ranked_rows_for(immediate_items, top_plan)
    best_watchlist_setups = _ranked_rows_for(watchlist_items, top_plan)
    rejected_or_low_priority = _ranked_rows_for(rejected_items, top_plan)
    selected_tickers = [item["row"].ticker for item in selected]
    best_immediate_tickers = [item["row"].ticker for item in immediate_items[:top_plan]]
    best_watchlist_tickers = [item["row"].ticker for item in watchlist_items[:top_plan]]
    rejected_or_low_priority_tickers = [item["row"].ticker for item in rejected_items[:top_plan]]

    meta = {
        "llm_used": True,
        "llm_provider": req.llm_provider,
        "llm_model": req.llm_model,
        "llm_style": req.llm_style,
    }

    rows_logged = 0
    try:
        rows_logged = _queue_rows_for_logging(
            db,
            planned_at=planned_at,
            mode=req.mode,
            rows=[x.row for x in out_rows],
            meta=meta,
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"SP100 workflow logging failed: {e}")

    response = Sp100WorkflowResponse(
        planned_at=planned_at,
        market_regime=regime_snapshot["regime"],
        regime_score=float(regime_snapshot["score"]),
        buy_threshold=thresholds["buy_threshold"],
        avoid_threshold=thresholds["avoid_threshold"],
        sector=req.sector,
        industry=req.industry,
        max_hold_days=max_hold_days,
        requested_max_hold_date=requested_max_hold_date,
        scanned_universe_size=len(base_universe),
        pre_scanned_count=len(pre_scanned),
        pre_scan_shortlist_count=len(shortlist),
        candidates_with_price=priced_candidates,
        eligible_count=eligible_count,
        selected_count=len(out_rows),
        rows_logged=rows_logged,
        selection_message=selection_message,
        planner_crash_count=len(planner_crashed_rows),
        planner_crashed_tickers=planner_crashed_tickers,
        planner_crash_reasons=planner_crash_reasons,
        selected_tickers=selected_tickers,
        best_immediate_tickers=best_immediate_tickers,
        best_watchlist_tickers=best_watchlist_tickers,
        rejected_or_low_priority_tickers=rejected_or_low_priority_tickers,
        rows=[] if req.compact_response else out_rows,
        best_immediate_setups=[] if req.compact_response else best_immediate_setups,
        best_watchlist_setups=[] if req.compact_response else best_watchlist_setups,
        rejected_or_low_priority=[] if req.compact_response else rejected_or_low_priority,
    )
    supabase_status = persist_sp100_workflow_to_supabase(
        workflow_request=req,
        workflow_response=response,
        selected_rows=out_rows,
    )
    response.supabase_persisted = bool((supabase_status or {}).get("persisted"))
    response.supabase_scan_run_id = (supabase_status or {}).get("scan_run_id")
    response.supabase_persistence_error = (supabase_status or {}).get("error")
    return response


@app.post("/workflow/swing-plan-log", response_model=SwingPlanLogWorkflowResponse)
def workflow_swing_plan_log(
    req: SwingPlanLogWorkflowRequest,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    ticker = (req.ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    learning = learning_patterns(
        lookback_days=max(7, min(int(req.lookback_days), 720)),
        limit=max(20, min(int(req.learning_limit), 500)),
        db=db,
        _=None,
    )

    plan = plan_swing(
        PlanRequest(
            tickers=[ticker],
            mode=req.mode,
            llm_used=True,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_style=req.llm_style,
        ),
        db=db,
        _=None,
    )

    rows = list(plan.get("rows", []))
    if not rows:
        raise HTTPException(status_code=500, detail=f"Planner returned no row for ticker={ticker}")

    row = rows[0]
    rows_logged = 0
    logging_skipped_reason = None

    if row.entry is None or row.stop is None or row.take_profit is None:
        logging_skipped_reason = "Plan has incomplete price levels, so nothing was logged."
    else:
        try:
            rows_logged = _queue_rows_for_logging(
                db,
                planned_at=plan["planned_at"],
                mode=req.mode,
                rows=[row],
                meta={
                    "llm_used": True,
                    "llm_provider": req.llm_provider,
                    "llm_model": req.llm_model,
                    "llm_style": req.llm_style,
                },
            )
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Swing workflow logging failed: {e}")

    return SwingPlanLogWorkflowResponse(
        planned_at=plan["planned_at"],
        ticker=ticker,
        market_regime=plan.get("market_regime"),
        regime_score=plan.get("regime_score"),
        buy_threshold=plan.get("buy_threshold"),
        avoid_threshold=plan.get("avoid_threshold"),
        learning_samples=int(learning.get("samples", 0)),
        learning_prompt_context=learning.get("prompt_context"),
        rows_logged=rows_logged,
        logging_skipped_reason=logging_skipped_reason,
        row=row,
    )



@app.post("/data/daily-bars/backfill", response_model=DailyBarsBackfillResponse)
def daily_bars_backfill(req: DailyBarsBackfillRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    universe = _resolve_universe(
        req.symbols,
        use_sp100=req.use_sp100,
        use_sp500=req.use_sp500,
        top_n=req.top_n,
    )
    if not universe:
        raise HTTPException(
            status_code=400,
            detail="No symbols provided. Pass symbols or set use_sp100=true/use_sp500=true.",
        )

    years = max(1, min(int(req.years), 15))
    commit_every = max(1, min(int(req.commit_every), 50))
    start_index = max(0, int(req.start_index))

    if start_index >= len(universe):
        raise HTTPException(status_code=400, detail=f"start_index={start_index} is out of range for universe_size={len(universe)}")

    batch_size = req.batch_size
    if batch_size is None:
        selected = universe[start_index:]
    else:
        size = max(1, min(int(batch_size), 100))
        selected = universe[start_index : start_index + size]

    if not selected:
        raise HTTPException(status_code=400, detail="No symbols selected for this batch.")

    end_index = start_index + len(selected)
    remaining = max(0, len(universe) - end_index)
    next_start_index = end_index if remaining > 0 else None

    try:
        result = backfill_universe_daily_bars(
            db,
            selected,
            years=years,
            refresh=bool(req.refresh),
            commit_every=commit_every,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Daily bars backfill failed: {e}")

    results_payload = list(result.get("results", [])) if req.include_results else []

    return DailyBarsBackfillResponse(
        as_of=datetime.now(timezone.utc),
        universe_size=len(universe),
        requested_total=len(universe),
        start_index=start_index,
        end_index=end_index,
        processed_count=len(selected),
        remaining=remaining,
        next_start_index=next_start_index,
        total=int(result.get("total", 0)),
        updated=int(result.get("updated", 0)),
        skipped_cached=int(result.get("skipped_cached", 0)),
        failed=int(result.get("failed", 0)),
        no_data=int(result.get("no_data", 0)),
        results=results_payload,
    )


@app.get("/data/daily-bars/status", response_model=DailyBarsStatusResponse)
def daily_bars_status(
    symbols: Optional[List[str]] = Query(default=None),
    use_sp100: bool = True,
    use_sp500: bool = False,
    top_n: int = 100,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    universe = _resolve_universe(
        symbols,
        use_sp100=use_sp100,
        use_sp500=use_sp500,
        top_n=top_n,
    )
    if not universe:
        raise HTTPException(
            status_code=400,
            detail="No symbols provided. Pass symbols or set use_sp100=true/use_sp500=true.",
        )

    rows = _daily_bars_status_rows(db, universe)
    symbols_with_data = sum(1 for r in rows if int(r.count) > 0)
    total_rows = sum(int(r.count) for r in rows)
    status_counts = Counter(str(row.freshness_status or "CACHE_MISSING") for row in rows)
    provider_counts = Counter(str(row.provider) for row in rows if row.provider)
    last_backfill = db.query(func.max(DailyBarCacheStatus.last_attempt_at)).scalar()

    return DailyBarsStatusResponse(
        as_of=datetime.now(timezone.utc),
        requested_symbols=len(universe),
        symbols_with_data=symbols_with_data,
        total_rows=total_rows,
        expected_market_date=resolve_expected_market_date(db),
        symbols_current=int(status_counts.get("CURRENT", 0)),
        symbols_stale=int(status_counts.get("CACHE_STALE", 0)),
        symbols_missing=int(status_counts.get("CACHE_MISSING", 0)),
        symbols_with_sufficient_history=sum(bool(row.history_sufficient) for row in rows),
        market_data_coverage_pct=round(int(status_counts.get("CURRENT", 0)) / max(len(universe), 1), 4),
        provider_counts=dict(provider_counts),
        last_backfill=last_backfill,
        rows=rows,
    )

@app.post("/history/log")
def log_history(req: LogRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    try:
        rows_logged = _queue_rows_for_logging(
            db,
            planned_at=req.planned_at,
            mode=req.mode,
            rows=req.rows,
            meta=req.meta,
        )
        db.commit()
        return {"ok": True, "rows_logged": rows_logged}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Logging failed: {e}")


@app.get("/history/evaluate")
def evaluate_history(limit: int = 200, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    q = (
        db.query(SwingDecision)
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )
    results = []
    for d in q:
        last = get_last_price(d.ticker)
        if last is None:
            continue

        outcome, ret = evaluate_plan_row(d.entry, d.stop, d.take_profit, last, d.max_hold_date)
        d.last_eval_ts = datetime.now(timezone.utc)
        d.last_eval_price = float(last)
        d.last_eval_outcome = outcome
        d.last_eval_return = float(ret)
        results.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "entry": d.entry,
                "stop": d.stop,
                "tp": d.take_profit,
                "max_hold_date": d.max_hold_date,
                "last_price": last,
                "outcome": outcome,
                "return_since_entry": ret,
                "strategy_action": d.strategy_action,
                "llm_action": d.llm_action,
                "news_score": getattr(d, "news_score", None),
                "earnings_score": getattr(d, "earnings_score", None),
            }
        )
    db.commit()
    return {"rows": results, "evaluated": len(results)}


@app.get("/analysis/earnings-score")
def earnings_score_analysis(
    lookback_days: int = 180,
    limit: int = 500,
    refresh_prices: bool = True,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    q = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.earnings_score.isnot(None))
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )

    samples = []
    for d in q:
        if d.entry is None or d.stop is None or d.take_profit is None or d.earnings_score is None:
            continue

        outcome = d.last_eval_outcome
        ret = d.last_eval_return
        last_price = d.last_eval_price

        if refresh_prices:
            live_last = get_last_price(d.ticker)
            if live_last is not None:
                outcome, ret = evaluate_plan_row(d.entry, d.stop, d.take_profit, live_last, d.max_hold_date)
                d.last_eval_ts = now
                d.last_eval_price = float(live_last)
                d.last_eval_outcome = outcome
                d.last_eval_return = float(ret)
                last_price = float(live_last)

        if ret is None:
            continue

        score = int(d.earnings_score)
        if score <= -4:
            bucket = "negative"
        elif score >= 4:
            bucket = "positive"
        else:
            bucket = "neutral"

        samples.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "earnings_score": score,
                "bucket": bucket,
                "outcome": outcome,
                "return_since_entry": float(ret),
                "last_price": last_price,
            }
        )

    db.commit()

    def rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {
                "samples": 0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "tp_rate": 0.0,
                "sl_rate": 0.0,
                "expired_rate": 0.0,
                "open_rate": 0.0,
            }

        avg_return = sum(r["return_since_entry"] for r in rows) / n
        wins = sum(1 for r in rows if r["return_since_entry"] > 0)
        tp = sum(1 for r in rows if r["outcome"] == "TP hit")
        sl = sum(1 for r in rows if r["outcome"] == "SL hit")
        expired = sum(1 for r in rows if r["outcome"] == "Expired")
        open_ = sum(1 for r in rows if r["outcome"] == "Open / In range")

        return {
            "samples": n,
            "avg_return": avg_return,
            "win_rate": rate(wins, n),
            "tp_rate": rate(tp, n),
            "sl_rate": rate(sl, n),
            "expired_rate": rate(expired, n),
            "open_rate": rate(open_, n),
        }

    by_bucket = {
        "negative": summarize([r for r in samples if r["bucket"] == "negative"]),
        "neutral": summarize([r for r in samples if r["bucket"] == "neutral"]),
        "positive": summarize([r for r in samples if r["bucket"] == "positive"]),
    }

    def pearson(rows: list[dict]) -> float | None:
        n = len(rows)
        if n < 2:
            return None
        xs = [float(r["earnings_score"]) for r in rows]
        ys = [float(r["return_since_entry"]) for r in rows]
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 1e-12 or vy <= 1e-12:
            return None
        return cov / ((vx ** 0.5) * (vy ** 0.5))

    return {
        "as_of": now,
        "lookback_days": lookback_days,
        "samples": len(samples),
        "refresh_prices": refresh_prices,
        "overall": summarize(samples),
        "score_return_correlation": pearson(samples),
        "by_bucket": by_bucket,
        "rows_preview": samples[:25],
    }


@app.get("/learning/patterns")
def learning_patterns(
    lookback_days: int = 120,
    limit: int = 500,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    q = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )

    rows = []
    for d in q:
        # only learn from rows with valid levels
        if d.entry is None or d.stop is None or d.take_profit is None:
            continue

        last = get_last_price(d.ticker)
        if last is None:
            continue

        assumed_executed, label, ret = classify_assumption(
            llm_action=d.llm_action,
            entry=float(d.entry),
            stop=float(d.stop),
            take_profit=float(d.take_profit),
            last_price=float(last),
            max_hold_date=d.max_hold_date,
            now=now,
        )

        rows.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "max_hold_date": d.max_hold_date,
                "llm_action": d.llm_action,
                "news_score": getattr(d, "news_score", None),
                "news_bucket": bucket_news(getattr(d, "news_score", None)),
                "entry": float(d.entry),
                "stop": float(d.stop),
                "take_profit": float(d.take_profit),
                "last_price": float(last),
                "assumed_executed": assumed_executed,
                "label": label,
                "return_since_entry": float(ret),
            }
        )

    # --- aggregate ---
    total = len(rows)
    by_label = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    def rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    buy_total = sum(1 for r in rows if r["assumed_executed"])
    buy_success = sum(1 for r in rows if r["label"] in ("buy_success_tp", "buy_expired_win"))
    buy_fail = sum(1 for r in rows if r["label"] in ("buy_fail_sl", "buy_expired_loss"))

    wait_total = total - buy_total
    wait_good_avoid = sum(1 for r in rows if r["label"] == "wait_good_avoid")
    wait_missed = sum(1 for r in rows if r["label"] in ("wait_missed_tp", "wait_missed_tp_expired"))

    # by news bucket (BUY success rate, WAIT missed rate)
    buckets = ["negative", "neutral", "positive", "unknown"]
    by_bucket = {}
    for b in buckets:
        br = [x for x in rows if x["news_bucket"] == b]
        b_buy = [x for x in br if x["assumed_executed"]]
        b_wait = [x for x in br if not x["assumed_executed"]]
        by_bucket[b] = {
            "samples": len(br),
            "buy_samples": len(b_buy),
            "buy_success_rate": rate(sum(1 for x in b_buy if x["label"] in ("buy_success_tp", "buy_expired_win")), len(b_buy)),
            "wait_samples": len(b_wait),
            "wait_missed_rate": rate(sum(1 for x in b_wait if x["label"] in ("wait_missed_tp", "wait_missed_tp_expired")), len(b_wait)),
            "wait_good_avoid_rate": rate(sum(1 for x in b_wait if x["label"] == "wait_good_avoid"), len(b_wait)),
        }

    # by ticker quick stats
    by_ticker = {}
    for r in rows:
        t = r["ticker"]
        d = by_ticker.setdefault(t, {"samples": 0, "buy": 0, "buy_success": 0, "wait": 0, "wait_missed": 0, "avg_ret": 0.0})
        d["samples"] += 1
        d["avg_ret"] += r["return_since_entry"]
        if r["assumed_executed"]:
            d["buy"] += 1
            if r["label"] in ("buy_success_tp", "buy_expired_win"):
                d["buy_success"] += 1
        else:
            d["wait"] += 1
            if r["label"] in ("wait_missed_tp", "wait_missed_tp_expired"):
                d["wait_missed"] += 1

    for t, d in by_ticker.items():
        d["avg_ret"] = d["avg_ret"] / max(d["samples"], 1)
        d["buy_success_rate"] = rate(d["buy_success"], d["buy"])
        d["wait_missed_rate"] = rate(d["wait_missed"], d["wait"])

    # --- prompt context (what you inject into next plans) ---
    prompt_context = (
        "Learning snapshot (assumptions: BUY executed; WAIT not executed).\n"
        f"Lookback: {lookback_days}d, samples: {total}\n"
        f"BUY success rate: {buy_success}/{buy_total} = {rate(buy_success,buy_total):.0%}; "
        f"BUY fail rate: {buy_fail}/{buy_total} = {rate(buy_fail,buy_total):.0%}\n"
        f"WAIT missed rate: {wait_missed}/{wait_total} = {rate(wait_missed,wait_total):.0%}; "
        f"WAIT good-avoid rate: {wait_good_avoid}/{wait_total} = {rate(wait_good_avoid,wait_total):.0%}\n"
        "News buckets impact:\n"
        + "\n".join(
            [
                f"- {b}: buy_success_rate={by_bucket[b]['buy_success_rate']:.0%} "
                f"(n={by_bucket[b]['buy_samples']}), "
                f"wait_missed_rate={by_bucket[b]['wait_missed_rate']:.0%} "
                f"(n={by_bucket[b]['wait_samples']})"
                for b in buckets
            ]
        )
    )

    return {
        "as_of": now,
        "lookback_days": lookback_days,
        "samples": total,
        "by_label": by_label,
        "rates": {
            "buy_success_rate": rate(buy_success, buy_total),
            "buy_fail_rate": rate(buy_fail, buy_total),
            "wait_missed_rate": rate(wait_missed, wait_total),
            "wait_good_avoid_rate": rate(wait_good_avoid, wait_total),
        },
        "by_bucket": by_bucket,
        "by_ticker": by_ticker,
        "prompt_context": prompt_context,
    }



