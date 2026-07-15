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
    mode: Mapped[str] = mapped_column(String(20), default="manual")  # manual/scan

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
