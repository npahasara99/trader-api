"""Pure, deterministic live confirmation and trade-geometry evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from .config import LiveMonitorConfig
from .enums import MonitorState


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _timestamp(bar: dict) -> datetime | None:
    value = bar.get("date") or bar.get("timestamp") or bar.get("datetime")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def market_session(at: datetime | None = None) -> str:
    current = (at or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    if current.weekday() >= 5:
        return "CLOSED"
    minutes = current.hour * 60 + current.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "PREMARKET"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "RTH"
    if 16 * 60 <= minutes < 20 * 60:
        return "AFTER_HOURS"
    return "CLOSED"


def relative_volume(bars: list[dict], lookback: int) -> float | None:
    if len(bars) < 2:
        return None
    current = _number(bars[-1].get("volume"))
    history = [
        value
        for value in (_number(bar.get("volume")) for bar in bars[-(lookback + 1):-1])
        if value is not None and value > 0
    ]
    if current is None or not history:
        return None
    baseline = median(history)
    return None if baseline <= 0 else round(current / baseline, 4)


def candle_quality(bar: dict) -> dict[str, float | bool | None]:
    opened = _number(bar.get("open"))
    high = _number(bar.get("high"))
    low = _number(bar.get("low"))
    close = _number(bar.get("close"))
    if None in {opened, high, low, close} or high <= low:
        return {"upper_wick_ratio": None, "close_location_value": None, "bullish_body": False}
    candle_range = high - low
    upper_wick = high - max(opened, close)
    return {
        "upper_wick_ratio": round(max(0.0, upper_wick) / candle_range, 4),
        "close_location_value": round((close - low) / candle_range, 4),
        "bullish_body": close >= opened,
    }


def derive_max_chase(trigger: float | None, atr: float | None, config: LiveMonitorConfig) -> float | None:
    if trigger is None or trigger <= 0:
        return None
    pct_buffer = trigger * config.max_chase_pct
    atr_buffer = (atr or 0.0) * config.max_chase_atr_fraction
    return round(trigger + max(pct_buffer, atr_buffer), 6)


def build_manual_order_plan(*, current_price: float, levels: dict, config: LiveMonitorConfig) -> dict[str, Any]:
    trigger = _number(levels.get("primary_entry_trigger"))
    max_chase = _number(levels.get("max_chase_price")) or derive_max_chase(trigger, _number(levels.get("atr")), config)
    stop = _number(levels.get("suggested_stop"))
    invalidation = _number(levels.get("invalidation_level"))
    targets = [_number(levels.get(name)) for name in ("tp1", "tp2", "tp3", "stretch_target")]
    targets = [target for target in targets if target is not None and target > current_price]
    risk = current_price - stop if stop is not None else None
    reward_risks = [round((target - current_price) / risk, 4) for target in targets if risk and risk > 0]
    entry_low = max(trigger or current_price, current_price)
    entry_high = max_chase if max_chase and max_chase >= entry_low else entry_low
    return {
        "execution": "MANUAL_ONLY",
        "suggested_manual_order_type": "BUY LIMIT / STOP-LIMIT",
        "current_entry_candidate": round(current_price, 6),
        "recommended_entry_zone": {"lower": round(entry_low, 6), "upper": round(entry_high, 6)},
        "max_chase_price": max_chase,
        "suggested_stop": stop,
        "structural_invalidation": invalidation,
        "tp1": _number(levels.get("tp1")),
        "tp2": _number(levels.get("tp2")),
        "tp3": _number(levels.get("tp3")),
        "stretch_target": _number(levels.get("stretch_target")),
        "reward_risk_candidates": reward_risks,
        "expected_hold_window": levels.get("expected_hold_window"),
    }


def evaluate_monitor(
    *,
    previous_state: str,
    levels: dict,
    bars_1m: list[dict],
    bars_5m: list[dict],
    setup_valid: bool,
    now: datetime | None,
    config: LiveMonitorConfig,
    prior_attempt_count: int = 0,
) -> dict[str, Any]:
    """Evaluate one symbol without side effects.

    A tick crossing can arm a setup, but approval requires a fresh 5-minute
    close, constructive relative volume, acceptable candle quality, and live R:R.
    """
    evaluated_at = now or datetime.now(timezone.utc)
    latest_1m = bars_1m[-1] if bars_1m else {}
    closed_5m = [
        bar
        for bar in bars_5m
        if bar.get("bar_complete") is True
        or _timestamp(bar) is None
        or _timestamp(bar) <= evaluated_at - timedelta(minutes=5)
    ]
    latest_5m = closed_5m[-1] if closed_5m else {}
    current = _number(latest_1m.get("close")) or _number(latest_5m.get("close"))
    data_at = _timestamp(latest_1m) or _timestamp(latest_5m)
    session = market_session(data_at or evaluated_at)
    trigger = _number(levels.get("primary_entry_trigger"))
    invalidation = _number(levels.get("invalidation_level"))
    stop = _number(levels.get("suggested_stop"))
    tp1 = _number(levels.get("tp1"))
    atr = _number(levels.get("atr"))
    max_chase = _number(levels.get("max_chase_price")) or derive_max_chase(trigger, atr, config)
    rvol_1m = relative_volume(bars_1m, config.volume_lookback_bars)
    rvol_5m = relative_volume(closed_5m, config.volume_lookback_bars)
    quality = candle_quality(latest_5m)
    stale_seconds = None if data_at is None else max(0.0, (evaluated_at - data_at.astimezone(timezone.utc)).total_seconds())
    stale = current is None or data_at is None or stale_seconds > config.stale_data_seconds

    distance_pct = None if current is None or trigger is None else round((current - trigger) / trigger, 6)
    one_minute_close = bool(current is not None and trigger is not None and current >= trigger)
    five_minute_close = bool(_number(latest_5m.get("close")) is not None and trigger is not None and _number(latest_5m.get("close")) >= trigger)
    constructive_candle = bool(
        quality["upper_wick_ratio"] is not None
        and quality["upper_wick_ratio"] <= config.max_upper_wick_ratio
        and quality["close_location_value"] is not None
        and quality["close_location_value"] >= config.min_close_location
    )
    volume_confirmation = bool(rvol_5m is not None and rvol_5m >= config.constructive_rvol)
    high_volume_rejection = bool(
        rvol_5m is not None
        and rvol_5m >= config.constructive_rvol
        and trigger is not None
        and _number(latest_5m.get("high")) is not None
        and _number(latest_5m.get("high")) >= trigger
        and not five_minute_close
        and (quality["upper_wick_ratio"] or 0.0) > config.max_upper_wick_ratio
    )
    risk = current - stop if current is not None and stop is not None else None
    current_rr = None if not risk or risk <= 0 or tp1 is None else round((tp1 - current) / risk, 4)
    rr_valid = bool(current_rr is not None and current_rr >= config.minimum_current_rr)
    chased = bool(current is not None and max_chase is not None and current > max_chase)
    invalidated = bool(not setup_valid or (current is not None and invalidation is not None and current <= invalidation))
    retest_tolerance = (atr or 0.0) * config.retest_tolerance_atr_fraction
    recent_after_cross = bars_1m[-6:] if bars_1m else []
    crossed_recently = bool(trigger is not None and any((_number(bar.get("high")) or -math.inf) >= trigger for bar in recent_after_cross))
    retest_held = bool(
        crossed_recently
        and trigger is not None
        and current is not None
        and current >= trigger
        and min((_number(bar.get("low")) or current) for bar in recent_after_cross) >= trigger - retest_tolerance
    )

    components = {
        "setup_valid": {"passed": not invalidated, "weight": 2.0},
        "trigger_crossed": {"passed": crossed_recently, "weight": 1.0},
        "one_minute_close": {"passed": one_minute_close, "weight": 0.75},
        "five_minute_close": {"passed": five_minute_close, "weight": 2.0},
        "relative_volume": {"passed": volume_confirmation, "weight": 1.75, "value": rvol_5m},
        "candle_quality": {"passed": constructive_candle, "weight": 1.0},
        "retest_success": {"passed": retest_held, "weight": 0.75},
        "current_rr": {"passed": rr_valid, "weight": 1.0, "value": current_rr},
        "not_chased": {"passed": not chased, "weight": 1.0},
        "fresh_data": {"passed": not stale, "weight": 2.0},
    }
    possible = sum(float(item["weight"]) for item in components.values())
    earned = sum(float(item["weight"]) for item in components.values() if item["passed"])
    score = round(10.0 * earned / possible, 3) if possible else 0.0

    previous = str(previous_state or MonitorState.WATCHING)
    rejection_reason = None
    if stale:
        state = MonitorState.DATA_STALE
    elif invalidated:
        state = MonitorState.INVALIDATED
    elif chased and previous in {
        MonitorState.ARMED,
        MonitorState.CONFIRMING,
        MonitorState.APPROVED,
        MonitorState.STRONGLY_CONFIRMED,
    }:
        state = MonitorState.MISSED
    elif high_volume_rejection:
        state = MonitorState.REJECTED_BREAKOUT
        rejection_reason = "high_volume_rejection"
    elif crossed_recently and current is not None and trigger is not None and current < trigger:
        state = MonitorState.REJECTED_BREAKOUT
        rejection_reason = "failed_hold"
    elif five_minute_close and volume_confirmation and constructive_candle and rr_valid and not chased:
        state = MonitorState.STRONGLY_CONFIRMED if retest_held and rvol_5m >= config.strong_rvol else MonitorState.APPROVED
    elif crossed_recently:
        state = MonitorState.CONFIRMING if previous in {MonitorState.ARMED, MonitorState.CONFIRMING} else MonitorState.ARMED
    elif distance_pct is not None and -config.near_trigger_distance_pct <= distance_pct < 0:
        state = MonitorState.NEAR_TRIGGER
    else:
        state = MonitorState.WATCHING

    manual_plan = build_manual_order_plan(current_price=current, levels={**levels, "max_chase_price": max_chase}, config=config) if current else None
    return {
        "state": state.value,
        "evaluated_at": evaluated_at,
        "market_data_as_of": data_at,
        "market_session": session,
        "current_price": current,
        "primary_entry_trigger": trigger,
        "distance_to_trigger_pct": distance_pct,
        "rvol_1m": rvol_1m,
        "rvol_5m": rvol_5m,
        "volume_baseline_method": "median_previous_bars",
        "volume_baseline_bars": config.volume_lookback_bars,
        "price_confirmation": five_minute_close,
        "volume_confirmation": volume_confirmation,
        "breakout_candle_quality": "constructive" if constructive_candle else "rejection" if high_volume_rejection else "unconfirmed",
        **quality,
        "retest_result": "held" if retest_held else "not_confirmed",
        "live_confirmation_score": score,
        "confirmation_components": components,
        "setup_valid": not invalidated,
        "data_stale": stale,
        "data_age_seconds": stale_seconds,
        "max_chase_price": max_chase,
        "current_rr_tp1": current_rr,
        "target_reachability": "acceptable" if rr_valid and tp1 and current < tp1 else "unacceptable",
        "rejection_reason": rejection_reason,
        "attempt_number": prior_attempt_count + 1 if state in {MonitorState.ARMED, MonitorState.CONFIRMING} and previous not in {MonitorState.ARMED, MonitorState.CONFIRMING} else prior_attempt_count,
        "manual_order_plan": manual_plan,
        "hard_blockers": [
            reason
            for condition, reason in (
                (stale, "data_stale"),
                (invalidated, "setup_invalidated"),
                (chased, "maximum_chase_exceeded"),
                (not rr_valid, "current_reward_risk_unacceptable"),
            )
            if condition
        ],
    }
