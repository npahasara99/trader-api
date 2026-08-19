"""Planner adapter that creates a stable monitor baseline once per setup."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.market_data import get_bars
from app.models import DailyBar
from app.planner import generate_structured_plan

from .config import LiveMonitorConfig
from .engine import derive_max_chase


def _price(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("price") or value.get("upper") or value.get("lower")
    try:
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _daily_bars_from_db(db: Session, ticker: str, limit: int = 360) -> list[dict]:
    rows = (
        db.query(DailyBar)
        .filter(DailyBar.symbol == ticker)
        .order_by(DailyBar.bar_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "symbol": row.symbol,
            "date": row.bar_date,
            "bar_date": row.bar_date,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "adjusted_close": row.adjusted_close,
            "source": row.source,
        }
        for row in reversed(rows)
    ]


def build_monitor_baseline(
    db: Session,
    ticker: str,
    *,
    supplied_plan: dict | None = None,
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    """Return planner payload and normalized live levels.

    Scanner-originated additions should supply their existing planner row. A
    manual symbol runs the planner once. Missing data produces a persistent
    no-valid-setup baseline rather than fabricated levels.
    """
    symbol = str(ticker or "").strip().upper()
    if supplied_plan:
        plan = dict(supplied_plan)
    else:
        bars = _daily_bars_from_db(db, symbol)
        if not bars:
            bars = get_bars(symbol, "daily", 360, cache_ttl_seconds=300)
        if not bars:
            plan = {
                "ticker": symbol,
                "planned_at": datetime.now(timezone.utc),
                "valid_setup": False,
                "setup_status": "NO_VALID_SWING_SETUP",
                "strategy_reason": "Daily market data is unavailable; reanalysis is required.",
            }
        else:
            current = float(bars[-1]["close"])
            timeframes = {
                "hourly": get_bars(symbol, "hourly", 60, cache_ttl_seconds=300),
                "thirty_minute": get_bars(symbol, "thirty_minute", 30, cache_ttl_seconds=300),
            }
            try:
                plan = generate_structured_plan(
                    ticker=symbol,
                    current_price=current,
                    bars=bars,
                    timeframe_bars=timeframes,
                    news_items=[],
                    news_score=0,
                    earnings_score=0,
                    earnings_context={},
                    market_regime="neutral",
                    buy_threshold=4,
                    avoid_threshold=-4,
                    history_stats={},
                )
            except Exception as exc:
                plan = {
                    "ticker": symbol,
                    "current_price": current,
                    "planned_at": datetime.now(timezone.utc),
                    "valid_setup": False,
                    "setup_status": "NO_VALID_SWING_SETUP",
                    "strategy_reason": f"Planner could not create a valid monitor baseline: {type(exc).__name__}: {exc}",
                }

    primary = _price(plan.get("primary_entry_trigger") or plan.get("confirmation_trigger") or plan.get("breakout_level"))
    invalidation = _price(plan.get("invalidation_level") or plan.get("stop_loss"))
    atr = _price(plan.get("atr"))
    levels = {
        "near_confirmation": _price(plan.get("near_confirmation")),
        "primary_entry_trigger": primary,
        "strong_confirmation": _price(plan.get("strong_confirmation")),
        "major_trend_repair": _price(plan.get("major_trend_repair")),
        "invalidation_level": invalidation,
        "suggested_stop": _price(plan.get("suggested_stop") or plan.get("stop_loss")),
        "optional_support_level": _price(plan.get("nearest_support") or plan.get("support_zone_1")),
        "preferred_entry_zone": plan.get("preferred_entry_zone") or plan.get("pullback_entry_zone"),
        "atr": atr,
        "tp1": _price(plan.get("take_profit_1")),
        "tp2": _price(plan.get("take_profit_2")),
        "tp3": _price(plan.get("take_profit_3")),
        "stretch_target": _price(plan.get("stretch_target") or plan.get("take_profit_final")),
        "expected_hold_window": {
            "expected_hold_days": plan.get("expected_hold_days"),
            "max_hold_days": plan.get("max_hold_days"),
            "max_hold_date": plan.get("max_hold_date"),
        },
    }
    levels["max_chase_price"] = derive_max_chase(primary, atr, config)
    valid = bool(plan.get("valid_setup", True) and primary and invalidation and primary > invalidation)
    plan["valid_setup"] = valid
    return {"plan": plan, "levels": levels, "valid_setup": valid}

