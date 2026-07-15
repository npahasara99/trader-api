
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta, date
from .logic import bucket_news, classify_assumption
from .config import DEFAULT_PLANNING_CONFIG
from .actionability import build_actionability_soon
from .execution_view import build_chart_execution_view
from .llm_reasoning import classify_final_action, reconcile_actions
from .monitoring import build_wait_monitoring_plan
from .ranking import build_ranking_profile
from .scanner import build_pre_scan_profile, sector_benchmark_symbol_for_meta
from .suitability import build_swing_trade_suitability
from .supabase_reporting import persist_sp100_workflow_to_supabase
from .what_to_watch import build_what_to_watch
from .watchlist import build_watchlist_profile
import json
import os

from sqlalchemy.orm import Session
from sqlalchemy import text, func

from .db import Base, engine, get_db
from .models import SwingDecision, DailyBar
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
    fetch_finnhub_daily_bars_with_meta,
)
from .bot.api import router as bot_router


DEFAULT_BAR_LOOKBACK_DAYS = 320


def _ensure_runtime_columns() -> None:
    required_cols = {
        "news_score": "INTEGER",
        "news_json": "TEXT",
        "earnings_score": "INTEGER",
        "earnings_context_json": "TEXT",
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
                return

            for col, col_type in required_cols.items():
                conn.execute(text(f"ALTER TABLE swing_decisions ADD COLUMN IF NOT EXISTS {col} {col_type}"))
    except Exception:
        # Do not block startup if migration cannot be applied here.
        pass


# Create tables + best-effort additive columns
Base.metadata.create_all(bind=engine)
_ensure_runtime_columns()

app = FastAPI(
    title="Trader Backend (Stocks Only)",
    version="0.1.3",
    servers=[
        {"url": "https://trader-api-production-7875.up.railway.app", "description": "Production"}
    ],
)


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
    setup_type: Optional[str] = None
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
    setup_context_summary: Optional[str] = None
    location_context_summary: Optional[str] = None
    support_zone_1: Optional[dict] = None
    support_zone_2: Optional[dict] = None
    resistance_zone_1: Optional[dict] = None
    resistance_zone_2: Optional[dict] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
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
    stop_loss: Optional[float] = None
    stop_basis: Optional[str] = None
    stop_distance_pct: Optional[float] = None
    stop_width_pct: Optional[float] = None
    stop_width_atr: Optional[float] = None
    stop_too_tight_flag: Optional[bool] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_final: Optional[float] = None
    tp1_distance_pct: Optional[float] = None
    tp1_distance_atr: Optional[float] = None
    tp_basis: Optional[str] = None
    reward_risk: Optional[dict] = None
    tp_too_optimistic_flag: Optional[bool] = None
    hold_window_reachability_score: Optional[float] = None
    swing_realism_flag: Optional[str] = None
    risk_width_flag: Optional[str] = None
    target_reachability_flag: Optional[str] = None
    level_geometry_flag: Optional[str] = None
    stop_generation_reason: Optional[str] = None
    tp1_generation_reason: Optional[str] = None
    max_hold_days: Optional[int] = None
    trend_quality_score: Optional[float] = None
    pullback_quality_score: Optional[float] = None
    support_quality_score: Optional[float] = None
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
    composite_score: Optional[float] = None
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


class DailyBarsStatusResponse(BaseModel):
    as_of: datetime
    requested_symbols: int
    symbols_with_data: int
    total_rows: int
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
        setup_type=getattr(r, "setup_type", None),
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
        setup_context_summary=getattr(r, "setup_context_summary", None),
        location_context_summary=getattr(r, "location_context_summary", None),
        support_zone_1=getattr(r, "support_zone_1", None),
        support_zone_2=getattr(r, "support_zone_2", None),
        resistance_zone_1=getattr(r, "resistance_zone_1", None),
        resistance_zone_2=getattr(r, "resistance_zone_2", None),
        atr=getattr(r, "atr", None),
        atr_pct=getattr(r, "atr_pct", None),
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
        stop_loss=getattr(r, "stop_loss", None),
        stop_basis=getattr(r, "stop_basis", None),
        stop_distance_pct=getattr(r, "stop_distance_pct", None),
        stop_width_pct=getattr(r, "stop_width_pct", None),
        stop_width_atr=getattr(r, "stop_width_atr", None),
        stop_too_tight_flag=getattr(r, "stop_too_tight_flag", None),
        take_profit_1=getattr(r, "take_profit_1", None),
        take_profit_2=getattr(r, "take_profit_2", None),
        take_profit_final=getattr(r, "take_profit_final", None),
        tp1_distance_pct=getattr(r, "tp1_distance_pct", None),
        tp1_distance_atr=getattr(r, "tp1_distance_atr", None),
        tp_basis=getattr(r, "tp_basis", None),
        reward_risk=getattr(r, "reward_risk", None),
        tp_too_optimistic_flag=getattr(r, "tp_too_optimistic_flag", None),
        hold_window_reachability_score=getattr(r, "hold_window_reachability_score", None),
        swing_realism_flag=getattr(r, "swing_realism_flag", None),
        risk_width_flag=getattr(r, "risk_width_flag", None),
        target_reachability_flag=getattr(r, "target_reachability_flag", None),
        level_geometry_flag=getattr(r, "level_geometry_flag", None),
        stop_generation_reason=getattr(r, "stop_generation_reason", None),
        tp1_generation_reason=getattr(r, "tp1_generation_reason", None),
        max_hold_days=getattr(r, "max_hold_days", None),
        trend_quality_score=getattr(r, "trend_quality_score", None),
        pullback_quality_score=getattr(r, "pullback_quality_score", None),
        support_quality_score=getattr(r, "support_quality_score", None),
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
        composite_score=getattr(r, "composite_score", None),
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


def _resolve_universe(symbols: Optional[List[str]], *, use_sp100: bool, top_n: int) -> List[str]:
    if symbols:
        return _normalize_symbols(symbols)
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
) -> List[dict]:
    """Cheap swing pre-scan used to shortlist names before full planning."""

    ranked: List[dict] = []
    benchmark_symbols = {"SPY", "QQQ"}
    sector_symbols: set[str] = set()
    for sym in symbols:
        sector_symbol = sector_benchmark_symbol_for_meta(SP100_CLASSIFICATION.get(sym))
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
            last = get_last_price_or_recent_close(sym, daily_closes_loader=daily_closes_loader)
            earnings_score, earnings_context = compute_earnings_signal(
                sym,
                last,
                daily_closes_loader=daily_closes_loader,
            )
            _ = earnings_score
            bars = daily_bars_loader(sym)
            profile = build_pre_scan_profile(
                ticker=sym,
                current_price=last,
                bars=bars,
                benchmark_bars=benchmark_bars,
                sector_benchmark_symbol=sector_benchmark_symbol_for_meta(SP100_CLASSIFICATION.get(sym)),
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
    out: List[DailyBarsStatusRow] = []
    for sym in symbols:
        row = by_symbol.get(sym)
        if row is None:
            out.append(DailyBarsStatusRow(symbol=sym, count=0, min_date=None, max_date=None))
            continue

        out.append(
            DailyBarsStatusRow(
                symbol=sym,
                count=int(row.count or 0),
                min_date=row.min_date,
                max_date=row.max_date,
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
            "market_regime": regime,
            "buy_threshold": buy_threshold,
            "entry_quality_score": getattr(row, "entry_quality_score", None),
            "entry_requires_confirmation": getattr(row, "entry_requires_confirmation", None),
            "confirmation_trigger": getattr(row, "confirmation_trigger", None),
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
    pre_scanned = ranked_prescan[:top_scan]
    shortlist = pre_scanned[: min(pre_scan_shortlist, len(pre_scanned))]
    universe = [item["ticker"] for item in shortlist]
    pre_scan_by_ticker = {
        item["ticker"]: {
            **item,
            "scan_shortlisted": str(item.get("scan_rejection_reason") or "") != "prescan_crashed",
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
    universe = _resolve_universe(req.symbols, use_sp100=req.use_sp100, top_n=req.top_n)
    if not universe:
        raise HTTPException(status_code=400, detail="No symbols provided. Pass symbols or set use_sp100=true.")

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
    top_n: int = 100,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    universe = _resolve_universe(symbols, use_sp100=use_sp100, top_n=top_n)
    if not universe:
        raise HTTPException(status_code=400, detail="No symbols provided. Pass symbols or set use_sp100=true.")

    rows = _daily_bars_status_rows(db, universe)
    symbols_with_data = sum(1 for r in rows if int(r.count) > 0)
    total_rows = sum(int(r.count) for r in rows)

    return DailyBarsStatusResponse(
        as_of=datetime.now(timezone.utc),
        requested_symbols=len(universe),
        symbols_with_data=symbols_with_data,
        total_rows=total_rows,
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



