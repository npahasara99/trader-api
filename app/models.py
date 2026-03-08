from sqlalchemy import String, Float, DateTime, Date, Integer, Text, Boolean
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
