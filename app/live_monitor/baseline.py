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


def assess_source_plan_freshness(plan: dict, snapshot: dict, config: LiveMonitorConfig) -> dict[str, Any]:
    """Decide whether scanner geometry is safe to promote into a live setup."""
    current = _price(snapshot.get("reference_price"))
    reference = _price(plan.get("plan_reference_price") or plan.get("current_price"))
    atr = _price(plan.get("atr") or snapshot.get("atr")) or (current or 1.0) * 0.02
    planned_at = _as_datetime(plan.get("plan_created_at") or plan.get("planned_at"))
    now = _as_datetime(snapshot.get("created_at")) or datetime.now(timezone.utc)
    drift = None if current is None or reference is None else current - reference
    drift_pct = None if drift is None or reference is None else drift / reference
    drift_atr = None if drift is None else drift / max(atr, 1e-9)
    reasons: list[str] = []
    if current is None:
        reasons.append("MARKET_DATA_UNAVAILABLE")
    if drift_pct is not None and abs(drift_pct) > config.plan_price_drift_pct:
        reasons.append("PRICE_DRIFT")
    if drift_atr is not None and abs(drift_atr) > config.plan_price_drift_atr:
        reasons.append("ATR_DRIFT")
    if planned_at is not None and (now - planned_at).total_seconds() > config.source_plan_max_age_minutes * 60:
        reasons.append("PLAN_TOO_OLD")
    support = _price(plan.get("nearest_support") or plan.get("support_zone_1") or plan.get("optional_support_level"))
    if current is not None and support is not None and current < support - atr * config.support_failure_atr:
        reasons.append("SUPPORT_FAILED")
    if snapshot.get("consistency_status") == "MARKET_DATA_MISMATCH":
        reasons.append("DATA_MISMATCH")
    return {
        "fresh": not reasons,
        "reasons": reasons,
        "source_plan_reference_price": reference,
        "current_reference_price": current,
        "price_drift_pct": None if drift_pct is None else round(drift_pct, 6),
        "price_drift_atr": None if drift_atr is None else round(drift_atr, 4),
        "source_plan_created_at": planned_at.isoformat() if planned_at else None,
    }


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
    market_snapshot: dict | None = None,
    allow_source_reuse: bool = True,
) -> dict[str, Any]:
    """Return planner payload and normalized live levels.

    Scanner-originated additions should supply their existing planner row. A
    manual symbol runs the planner once. Missing data produces a persistent
    no-valid-setup baseline rather than fabricated levels.
    """
    symbol = str(ticker or "").strip().upper()
    snapshot = dict(market_snapshot or {})
    snapshot_bars = snapshot.get("bars") or {}
    source_validation = assess_source_plan_freshness(supplied_plan, snapshot, config) if supplied_plan and snapshot else {
        "fresh": False,
        "reasons": ["NO_CANONICAL_SNAPSHOT"] if supplied_plan else [],
    }
    reuse_supplied = bool(supplied_plan and allow_source_reuse and source_validation.get("fresh"))
    if reuse_supplied:
        plan = dict(supplied_plan or {})
        plan_source = "validated_scanner_context"
    else:
        bars = list(snapshot_bars.get("daily") or [])
        if not bars and not snapshot:
            bars = _daily_bars_from_db(db, symbol)
        if not bars and not snapshot:
            bars = get_bars(symbol, "daily", 360, cache_ttl_seconds=0)
        if not bars:
            plan = {
                "ticker": symbol,
                "planned_at": datetime.now(timezone.utc),
                "valid_setup": False,
                "setup_status": "NO_VALID_SWING_SETUP",
                "strategy_reason": "Daily market data is unavailable; reanalysis is required.",
            }
            plan_source = "fresh_snapshot_unavailable"
        else:
            current = _price(snapshot.get("reference_price")) or float(bars[-1]["close"])
            timeframes = {
                "hourly": list(snapshot_bars.get("hourly") or []),
                "thirty_minute": list(snapshot_bars.get("thirty_minute") or []),
            }
            try:
                context = supplied_plan or {}
                plan = generate_structured_plan(
                    ticker=symbol,
                    current_price=current,
                    bars=bars,
                    timeframe_bars=timeframes,
                    news_items=context.get("news_items") or context.get("news") or [],
                    news_score=context.get("news_score") or 0,
                    earnings_score=context.get("earnings_score") or 0,
                    earnings_context=context.get("earnings_context") or {},
                    market_regime=context.get("market_regime") or "neutral",
                    buy_threshold=4,
                    avoid_threshold=-4,
                    history_stats=context.get("history_stats") or {},
                )
                plan_source = "fresh_canonical_replan"
            except Exception as exc:
                plan = {
                    "ticker": symbol,
                    "current_price": current,
                    "planned_at": datetime.now(timezone.utc),
                    "valid_setup": False,
                    "setup_status": "NO_VALID_SWING_SETUP",
                    "strategy_reason": f"Planner could not create a valid monitor baseline: {type(exc).__name__}: {exc}",
                }
                plan_source = "fresh_canonical_replan_failed"

    snapshot_id = snapshot.get("market_snapshot_id")
    snapshot_created = snapshot.get("created_at") or datetime.now(timezone.utc).isoformat()
    snapshot_price = _price(snapshot.get("reference_price")) or _price(plan.get("current_price"))
    plan["source_plan_context"] = dict(supplied_plan or {}) if supplied_plan and not reuse_supplied else None
    plan["source_plan_validation"] = source_validation
    plan["monitor_plan_source"] = plan_source
    plan["market_snapshot_id"] = snapshot_id
    plan["plan_reference_price"] = snapshot_price
    plan["current_price"] = snapshot_price
    plan["plan_created_at"] = snapshot_created
    plan["market_data_timestamp"] = snapshot.get("quote_timestamp")
    plan["market_data_source"] = snapshot.get("data_source")

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
    support = _price(levels.get("optional_support_level"))
    if snapshot_price is not None and support is not None and support > snapshot_price:
        levels["historical_support_lost"] = support
        levels["optional_support_level"] = None
        plan.setdefault("level_semantic_warnings", []).append("OLD_SUPPORT_LOST")
    levels["max_chase_price"] = derive_max_chase(primary, atr, config)
    valid = bool(plan.get("valid_setup", True) and primary and invalidation and primary > invalidation)
    plan["valid_setup"] = valid
    return {
        "plan": plan,
        "levels": levels,
        "valid_setup": valid,
        "market_snapshot": snapshot,
        "source_plan_validation": source_validation,
        "source_plan_reused": reuse_supplied,
    }


__all__ = ["assess_source_plan_freshness", "build_monitor_baseline"]
