from sqlalchemy import String, Float, DateTime, Date, Integer, Text, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone, date
from .db import Base


def utcnow():
    return datetime.now(timezone.utc)


class SwingDecision(Base):
    __tablename__ = "swing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(20), index=True)
    planned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=utcnow)
    # Workflow identifiers such as ``sp500_daily_opportunities`` exceed the
    # original 20-character manual/scan-era limit.
    mode: Mapped[str] = mapped_column(String(80), default="manual")

    entry: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)

    max_hold_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    strategy_action: Mapped[str | None] = mapped_column(String(40), nullable=True)
    strategy_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_provider: Mapped[str | None] = mapped_column(String(30), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(60), nullable=True)
    llm_style: Mapped[str | None] = mapped_column(String(40), nullable=True)

    llm_action: Mapped[str | None] = mapped_column(String(40), nullable=True)
    llm_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    news_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    earnings_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    earnings_context_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    news_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Evaluation fields (latest evaluation snapshot)
    last_eval_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_eval_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_eval_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)  # TP hit / SL hit / Open / Expired
    last_eval_return: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Learning tags
    decision_accuracy: Mapped[str | None] = mapped_column(String(20), nullable=True)  # accurate/inaccurate/unknown
    success_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of tags

    MODEL_VERSION = "2026-03-08-adaptive-risk-v2"


class DailyBar(Base):
    __tablename__ = "daily_bars"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    bar_date: Mapped[date] = mapped_column(Date, primary_key=True)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(20), default="finnhub")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class DailyBarCacheStatus(Base):
    """Latest ingestion state for one canonical daily-bar symbol."""

    __tablename__ = "daily_bar_cache_status"

    canonical_symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    provider_symbol: Mapped[str | None] = mapped_column(String(30), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    last_bar_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    data_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    freshness_status: Mapped[str] = mapped_column(String(40), default="CACHE_MISSING", index=True)
    history_sufficient: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    last_error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class BotConfiguration(Base):
    __tablename__ = "bot_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_key: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    config_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_by: Mapped[str | None] = mapped_column(String(80), nullable=True)


class BotRun(Base):
    __tablename__ = "bot_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    state: Mapped[str] = mapped_column(String(40), index=True)
    trading_mode: Mapped[str] = mapped_column(String(40), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class BotEvent(Base):
    __tablename__ = "bot_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_run_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    outcome: Mapped[str] = mapped_column(String(40), index=True)
    message: Mapped[str] = mapped_column(Text)
    level: Mapped[str] = mapped_column(String(20), default="info", index=True)
    ticker: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    candidate_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    proposal_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BotHealthSnapshot(Base):
    __tablename__ = "bot_health_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_run_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    broker_state: Mapped[str] = mapped_column(String(40), index=True)
    bot_state: Mapped[str] = mapped_column(String(40), index=True)
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class BrokerAccount(Base):
    __tablename__ = "broker_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id_masked: Mapped[str] = mapped_column(String(40), index=True)
    broker_name: Mapped[str] = mapped_column(String(40), index=True)
    account_type: Mapped[str] = mapped_column(String(20), index=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class TradeCandidate(Base):
    __tablename__ = "trade_candidates"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_trade_candidates_candidate_id"),
        Index("ix_trade_candidates_ticker_generated_at", "ticker", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    basket: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    trigger_state: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(40), nullable=True)
    setup_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    setup_scenario: Mapped[str | None] = mapped_column(String(80), nullable=True)
    actionability_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    suitability_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ranking_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    preferred_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    maximum_holding_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejection_code: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class CandidateScoreComponent(Base):
    __tablename__ = "candidate_score_components"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[str] = mapped_column(String(80), index=True)
    component_name: Mapped[str] = mapped_column(String(80), index=True)
    component_value: Mapped[float] = mapped_column(Float)
    component_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TradeProposal(Base):
    __tablename__ = "trade_proposals"
    __table_args__ = (
        UniqueConstraint("proposal_id", name="uq_trade_proposals_proposal_id"),
        UniqueConstraint("idempotency_key", name="uq_trade_proposals_idempotency_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(String(80), index=True)
    candidate_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    side: Mapped[str] = mapped_column(String(12), default="BUY")
    order_type: Mapped[str] = mapped_column(String(20), default="LIMIT")
    idempotency_key: Mapped[str] = mapped_column(String(120), index=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_price_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_price_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    planned_risk_dollars: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_max_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_risk_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    rejection_codes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    preview_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BrokerOrder(Base):
    __tablename__ = "broker_orders"
    __table_args__ = (
        UniqueConstraint("broker_order_id", name="uq_broker_orders_broker_order_id"),
        Index("ix_broker_orders_ticker_status", "ticker", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    broker_order_id: Mapped[str] = mapped_column(String(80), index=True)
    broker_permanent_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    proposal_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    parent_order_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    child_role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(12))
    order_type: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    broker_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BrokerOrderEvent(Base):
    __tablename__ = "broker_order_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    broker_order_id: Mapped[str] = mapped_column(String(80), index=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BrokerExecution(Base):
    __tablename__ = "executions"
    __table_args__ = (UniqueConstraint("execution_id", name="uq_executions_execution_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(120), index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    position_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(12))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class BrokerCommission(Base):
    __tablename__ = "commissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(120), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(12), default="USD")
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ManagedPosition(Base):
    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("position_id", name="uq_positions_position_id"),
        Index("ix_positions_ticker_status", "ticker", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    proposal_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    candidate_id: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    average_entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    position_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class PositionEvent(Base):
    __tablename__ = "position_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(80), index=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    event_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TradeReview(Base):
    __tablename__ = "trade_reviews"
    __table_args__ = (UniqueConstraint("position_id", name="uq_trade_reviews_position_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(80), index=True)
    success_category: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    failure_category: Mapped[str | None] = mapped_column(String(80), index=True, nullable=True)
    realised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    deterministic_review_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TradeMemoryStatistic(Base):
    __tablename__ = "trade_memory_statistics"
    __table_args__ = (
        UniqueConstraint("memory_key", name="uq_trade_memory_statistics_memory_key"),
        Index("ix_trade_memory_statistics_scope", "scope_type", "scope_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_key: Mapped[str] = mapped_column(String(160), index=True)
    scope_type: Mapped[str] = mapped_column(String(60), index=True)
    scope_value: Mapped[str] = mapped_column(String(120), index=True)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    bounded_adjustment: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class DailyPerformance(Base):
    __tablename__ = "daily_performance"
    __table_args__ = (UniqueConstraint("performance_date", name="uq_daily_performance_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    performance_date: Mapped[date] = mapped_column(Date, index=True)
    realised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    closed_positions: Mapped[int] = mapped_column(Integer, default=0)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ReconciliationRun(Base):
    __tablename__ = "reconciliation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reconciliation_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason: Mapped[str] = mapped_column(Text)
    triggered_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TradingViewSignalEvent(Base):
    """Monitoring-only TradingView signal queued for structured re-evaluation."""

    __tablename__ = "tradingview_signal_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_tradingview_signal_events_event_id"),
        Index("ix_tradingview_signal_events_monitor", "ticker", "processed", "received_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    price: Mapped[float] = mapped_column(Float)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    indicators_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    processing_status: Mapped[str] = mapped_column(String(40), default="pending_replan", index=True)
    re_evaluation_requested: Mapped[bool] = mapped_column(Boolean, default=True)
    execution_requested: Mapped[bool] = mapped_column(Boolean, default=False)


class LiveWatch(Base):
    """Mutable pointer to the latest setup for a manually monitored symbol."""

    __tablename__ = "live_watches"
    __table_args__ = (
        UniqueConstraint("ticker", name="uq_live_watches_ticker"),
        Index("ix_live_watches_active_state", "monitor_active", "state"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(40), default="manual")
    monitor_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    state: Mapped[str] = mapped_column(String(40), default="WATCHING", index=True)
    current_setup_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    market_data_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    session_label: Mapped[str | None] = mapped_column(String(24), nullable=True)
    last_event: Mapped[str | None] = mapped_column(Text, nullable=True)
    latest_evaluation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_backend_evaluation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_market_data_update_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MonitorSetup(Base):
    """Versioned planner baseline plus a non-destructive active level overlay."""

    __tablename__ = "monitor_setups"
    __table_args__ = (
        UniqueConstraint("watch_id", "version", name="uq_monitor_setups_watch_version"),
        Index("ix_monitor_setups_watch_status", "watch_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(40), default="active", index=True)
    valid_setup: Mapped[bool] = mapped_column(Boolean, default=True)
    setup_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    broader_structure: Mapped[str | None] = mapped_column(String(80), nullable=True)
    setup_type: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    setup_family: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    execution_structure: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    industry: Mapped[str | None] = mapped_column(String(160), nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    planner_baseline_json: Mapped[str] = mapped_column(Text)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    plan_reference_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    plan_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    market_data_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    plan_stale: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    plan_stale_reasons_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    previous_setup_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    replacement_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    planner_levels_json: Mapped[str] = mapped_column(Text)
    active_levels_json: Mapped[str] = mapped_column(Text)
    manual_overrides_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_proposed_levels_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    validated_chart_levels_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    level_sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    chart_analysis_status: Mapped[str] = mapped_column(String(40), default="NOT_RUN", index=True)
    latest_chart_review_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    plan_stale_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposed_setup_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_active_plan_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    final_active_plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_plan_validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan_integrity_status: Mapped[str] = mapped_column(String(20), default="INVALID", index=True)
    reconciliation_status: Mapped[str] = mapped_column(String(40), default="PLANNER_ACCEPTED", index=True)
    trigger_source: Mapped[str] = mapped_column(String(20), default="PLANNER")
    max_chase_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidation_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    replaced_by_setup_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    rule_version: Mapped[str] = mapped_column(String(60), default="live-monitor-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ConfirmationAttempt(Base):
    __tablename__ = "confirmation_attempts"
    __table_args__ = (
        UniqueConstraint("setup_id", "attempt_number", name="uq_confirmation_attempt_number"),
        Index("ix_confirmation_attempts_setup_started", "setup_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trigger_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    lowest_retest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    rvol_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    rvol_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_confirmation: Mapped[bool] = mapped_column(Boolean, default=False)
    volume_confirmation: Mapped[bool] = mapped_column(Boolean, default=False)
    retest_result: Mapped[str | None] = mapped_column(String(60), nullable=True)
    confirmation_method: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    outcome: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class MonitorEvent(Base):
    """Append-only decision journal event."""

    __tablename__ = "monitor_events"
    __table_args__ = (Index("ix_monitor_events_watch_created", "watch_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    from_state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    to_state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MarketSnapshot(Base):
    """Immutable market-data vintage used by one planner setup and its charts."""

    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_market_snapshots_ticker_created", "ticker", "created_at"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    quote_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reference_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_source: Mapped[str | None] = mapped_column(String(160), nullable=True)
    daily_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hourly_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    thirty_min_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    five_min_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    one_min_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consistency_status: Mapped[str] = mapped_column(String(40), default="CONSISTENT", index=True)
    cache_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)


class MonitorDecisionSnapshot(Base):
    __tablename__ = "monitor_decision_snapshots"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    snapshot_type: Mapped[str] = mapped_column(String(60), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LLMAdvisoryReview(Base):
    __tablename__ = "llm_advisory_reviews"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(60), default="live-advisor-v1")
    decision: Mapped[str] = mapped_column(String(20), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(40), default="available")
    reason_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_snapshot_json: Mapped[str] = mapped_column(Text)
    output_json: Mapped[str] = mapped_column(Text)
    hard_blockers_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_user_action: Mapped[str | None] = mapped_column(String(40), nullable=True)
    actual_outcome_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LLMDecisionPostmortem(Base):
    """Outcome-time review linked to the immutable original LLM decision."""

    __tablename__ = "llm_decision_postmortems"
    __table_args__ = (UniqueConstraint("llm_review_id", name="uq_llm_postmortem_review"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    llm_review_id: Mapped[str] = mapped_column(String(80), index=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    outcome_type: Mapped[str] = mapped_column(String(40), index=True)
    original_decision: Mapped[str] = mapped_column(String(20), index=True)
    outcome_json: Mapped[str] = mapped_column(Text)
    rationale_tags_json: Mapped[str] = mapped_column(Text)
    lessons_json: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ManualMonitorTrade(Base):
    """User-reported trade record. This table is never a broker submission queue."""

    __tablename__ = "manual_monitor_trades"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    planned_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    targets_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    entered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realised_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RecommendationOutcome(Base):
    __tablename__ = "recommendation_outcomes"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    user_action: Mapped[str] = mapped_column(String(40), index=True)
    outcome: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    entry_distance_from_trigger_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StockBehaviorProfile(Base):
    __tablename__ = "stock_behavior_profiles"
    __table_args__ = (UniqueConstraint("scope_type", "scope_value", name="uq_stock_behavior_scope"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(40), index=True)
    scope_value: Mapped[str] = mapped_column(String(160), index=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=0)
    evidence_strength: Mapped[str] = mapped_column(String(30), default="INSUFFICIENT")
    statistics_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MonitorBarSummary(Base):
    """A completed monitor bar retained for evidence without storing every quote poll."""

    __tablename__ = "monitor_bar_summaries"
    __table_args__ = (
        UniqueConstraint("setup_id", "timeframe", "bar_timestamp", name="uq_monitor_bar_summary"),
        Index("ix_monitor_bar_summaries_ticker_time", "ticker", "bar_timestamp"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(20), index=True)
    bar_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open_price: Mapped[float] = mapped_column(Float)
    high_price: Mapped[float] = mapped_column(Float)
    low_price: Mapped[float] = mapped_column(Float)
    close_price: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicators_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_quality_flags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MonitorDailySummary(Base):
    """Immutable finalized monitor-day evidence, including setups that never traded."""

    __tablename__ = "monitor_daily_summaries"
    __table_args__ = (
        UniqueConstraint("setup_id", "trading_date", name="uq_monitor_daily_setup_date"),
        Index("ix_monitor_daily_ticker_date", "ticker", "trading_date"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    trading_date: Mapped[date] = mapped_column(Date, index=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    open_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    starting_monitor_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ending_monitor_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    broader_structure: Mapped[str | None] = mapped_column(String(80), nullable=True)
    setup_type: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    setup_family: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    execution_structure: Mapped[str | None] = mapped_column(String(80), nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    levels_json: Mapped[str] = mapped_column(Text)
    indicators_json: Mapped[str] = mapped_column(Text)
    context_json: Mapped[str] = mapped_column(Text)
    decisions_json: Mapped[str] = mapped_column(Text)
    outcome_json: Mapped[str] = mapped_column(Text)
    data_quality_flags_json: Mapped[str] = mapped_column(Text)
    number_of_trigger_attempts: Mapped[int] = mapped_column(Integer, default=0)
    number_of_rejections: Mapped[int] = mapped_column(Integer, default=0)
    highest_state_reached: Mapped[str | None] = mapped_column(String(40), nullable=True)
    mfe_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_trade_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    actual_trade_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    finalized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class BehaviorProfileVersion(Base):
    """Append-only aggregate version so each decision can identify its historical prior."""

    __tablename__ = "behavior_profile_versions"
    __table_args__ = (
        UniqueConstraint("scope_type", "scope_value", "version", name="uq_behavior_profile_version"),
        Index("ix_behavior_profile_scope_created", "scope_type", "scope_value", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    profile_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    scope_type: Mapped[str] = mapped_column(String(40), index=True)
    scope_value: Mapped[str] = mapped_column(String(160), index=True)
    version: Mapped[int] = mapped_column(Integer)
    observation_count: Mapped[int] = mapped_column(Integer, default=0)
    weighted_observation_count: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_strength: Mapped[str] = mapped_column(String(30), default="INSUFFICIENT")
    reliability: Mapped[float] = mapped_column(Float, default=0.0)
    statistics_json: Mapped[str] = mapped_column(Text)
    formula_version: Mapped[str] = mapped_column(String(60), default="historical-memory-v1")
    source_cutoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LearnedAdjustment(Base):
    """Append-only, bounded interpretation change applied to one current setup."""

    __tablename__ = "learned_adjustments"
    __table_args__ = (Index("ix_learned_adjustments_setup_created", "setup_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    adjustment_type: Mapped[str] = mapped_column(String(80), index=True)
    base_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    learned_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    adjustment_value: Mapped[float] = mapped_column(Float, default=0.0)
    adjustment_strength: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_strength: Mapped[str] = mapped_column(String(30), default="INSUFFICIENT")
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    weighted_sample_size: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text)
    supporting_stats_json: Mapped[str] = mapped_column(Text)
    bounds_json: Mapped[str] = mapped_column(Text)
    profile_version_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LevelRevision(Base):
    """Append-only pricing lineage from planner through validation to the active level."""

    __tablename__ = "level_revisions"
    __table_args__ = (Index("ix_level_revisions_setup_level", "setup_id", "level_name", "created_at"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    chart_review_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    level_name: Mapped[str] = mapped_column(String(60), index=True)
    level_role: Mapped[str] = mapped_column(String(60), index=True)
    planner_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_proposed_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    manual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_active_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    validation_result: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    anomaly_flags_json: Mapped[str] = mapped_column(Text)
    outcome_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LearningJobRun(Base):
    __tablename__ = "learning_job_runs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    trading_date: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    summaries_finalized: Mapped[int] = mapped_column(Integer, default=0)
    profiles_updated: Mapped[int] = mapped_column(Integer, default=0)
    observations_created: Mapped[int] = mapped_column(Integer, default=0)
    details_json: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class LearningObservation(Base):
    __tablename__ = "learning_observations"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(40), index=True)
    scope_value: Mapped[str] = mapped_column(String(160), index=True)
    observation_type: Mapped[str] = mapped_column(String(80), index=True)
    summary: Mapped[str] = mapped_column(Text)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    evidence_strength: Mapped[str] = mapped_column(String(30), default="INSUFFICIENT")
    evidence_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LearningProposal(Base):
    __tablename__ = "learning_proposals"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    observation_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    scope_type: Mapped[str] = mapped_column(String(40), index=True)
    scope_value: Mapped[str] = mapped_column(String(160), index=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING", index=True)
    title: Mapped[str] = mapped_column(Text)
    proposed_change_json: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class MonitorRuleVersion(Base):
    __tablename__ = "monitor_rule_versions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    proposal_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    rules_json: Mapped[str] = mapped_column(Text)
    approved_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ShadowRuleEvaluation(Base):
    __tablename__ = "shadow_rule_evaluations"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String(80), index=True)
    watch_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    setup_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    production_decision: Mapped[str] = mapped_column(String(40))
    shadow_decision: Mapped[str] = mapped_column(String(40))
    production_outcome: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    shadow_hypothetical_outcome: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    evidence_json: Mapped[str] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ChartSnapshot(Base):
    """Immutable decision-time chart image and exact time-bounded metadata."""

    __tablename__ = "chart_snapshots"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_chart_snapshots_content_hash"),
        Index("ix_chart_snapshots_setup_event", "setup_id", "event_type", "generated_at"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    decision_event_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(24), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    image_path: Mapped[str] = mapped_column(Text)
    image_data_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(80), index=True)
    data_source: Mapped[str | None] = mapped_column(String(80), nullable=True)
    data_last_bar_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_time_boundary: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    metadata_json: Mapped[str] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    retain_permanently: Mapped[bool] = mapped_column(Boolean, default=True)


class ChartStructureReview(Base):
    __tablename__ = "chart_structure_reviews"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    market_snapshot_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    review_type: Mapped[str] = mapped_column(String(60), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(80), index=True)
    chart_snapshot_ids_json: Mapped[str] = mapped_column(Text)
    deterministic_input_json: Mapped[str] = mapped_column(Text)
    planner_levels_json: Mapped[str] = mapped_column(Text)
    llm_output_json: Mapped[str] = mapped_column(Text)
    llm_proposed_levels_json: Mapped[str] = mapped_column(Text)
    validated_levels_json: Mapped[str] = mapped_column(Text)
    validation_json: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_consistency_status: Mapped[str] = mapped_column(String(40), default="CONSISTENT")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ChartLevelDecision(Base):
    __tablename__ = "chart_level_decisions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    watch_id: Mapped[str] = mapped_column(String(80), index=True)
    setup_id: Mapped[str] = mapped_column(String(80), index=True)
    chart_review_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    decision: Mapped[str] = mapped_column(String(40), index=True)
    previous_active_levels_json: Mapped[str] = mapped_column(Text)
    selected_levels_json: Mapped[str] = mapped_column(Text)
    level_sources_json: Mapped[str] = mapped_column(Text)
    decided_by: Mapped[str] = mapped_column(String(80), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
