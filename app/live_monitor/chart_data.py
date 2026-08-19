"""Canonical, decision-time-bounded OHLCV bundles for monitor charts and reviews."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import math
from typing import Any, Callable
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models import ConfirmationAttempt, DailyBar


BarsLoader = Callable[[str, str, int | None], list[dict]]


TIMEFRAME_SPECS: dict[str, tuple[str, int]] = {
    "daily": ("daily", 420),
    "hourly": ("hourly", 45),
    "structure": ("thirty_minute", 30),
    "execution": ("five_minute", 7),
}


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _normalize_bars(
    bars: list[dict],
    *,
    timeframe: str,
    decision_time_boundary: datetime,
    max_bars: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in bars:
        at = _timestamp(raw.get("date") or raw.get("timestamp") or raw.get("datetime") or raw.get("time"))
        close = _number(raw.get("close"))
        if at is None or close is None or at > decision_time_boundary:
            continue
        opened = _number(raw.get("open")) or close
        high = _number(raw.get("high")) or max(opened, close)
        low = _number(raw.get("low")) or min(opened, close)
        normalized.append(
            {
                "time": at.astimezone(timezone.utc).isoformat(),
                "timestamp": int(at.timestamp()),
                "open": opened,
                "high": high,
                "low": low,
                "close": close,
                "volume": _number(raw.get("volume")) or 0.0,
                "source": str(raw.get("source") or "canonical_provider"),
                "timeframe": timeframe,
            }
        )
    normalized.sort(key=lambda item: item["timestamp"])
    return normalized[-max_bars:]


def _ema(values: list[float], period: int) -> list[float | None]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    current = values[0]
    output: list[float | None] = []
    for index, value in enumerate(values):
        current = value if index == 0 else alpha * value + (1.0 - alpha) * current
        output.append(round(current, 6) if index >= min(period - 1, len(values) - 1) else None)
    return output


def _indicator_series(bars: list[dict], *, intraday: bool) -> dict[str, list[dict[str, Any]]]:
    closes = [float(bar["close"]) for bar in bars]
    output: dict[str, list[dict[str, Any]]] = {}
    for period in (20, 50, 100, 200):
        values = _ema(closes, period)
        output[f"ema{period}"] = [
            {"time": bar["timestamp"], "value": value}
            for bar, value in zip(bars, values)
            if value is not None
        ]
    if intraday:
        cumulative_volume = 0.0
        cumulative_price_volume = 0.0
        current_day: str | None = None
        vwap: list[dict[str, Any]] = []
        for bar in bars:
            day = str(bar["time"])[:10]
            if day != current_day:
                current_day = day
                cumulative_volume = 0.0
                cumulative_price_volume = 0.0
            typical = (float(bar["high"]) + float(bar["low"]) + float(bar["close"])) / 3.0
            volume = float(bar.get("volume") or 0.0)
            cumulative_volume += volume
            cumulative_price_volume += typical * volume
            if cumulative_volume > 0:
                vwap.append({"time": bar["timestamp"], "value": round(cumulative_price_volume / cumulative_volume, 6)})
        output["vwap"] = vwap
    return output


def _derive_four_hour_bars(hourly_bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate complete four-bar market-session blocks from canonical hourly bars."""
    by_session: dict[str, list[dict[str, Any]]] = {}
    eastern = ZoneInfo("America/New_York")
    for bar in hourly_bars:
        at = datetime.fromtimestamp(int(bar["timestamp"]), tz=timezone.utc).astimezone(eastern)
        minutes = at.hour * 60 + at.minute
        if not (9 * 60 + 30 <= minutes < 16 * 60):
            continue
        by_session.setdefault(at.date().isoformat(), []).append(bar)
    output: list[dict[str, Any]] = []
    for session_bars in by_session.values():
        session_bars.sort(key=lambda item: item["timestamp"])
        for offset in range(0, len(session_bars), 4):
            chunk = session_bars[offset:offset + 4]
            if len(chunk) < 4:
                continue
            output.append(
                {
                    "time": chunk[0]["time"],
                    "timestamp": chunk[0]["timestamp"],
                    "open": chunk[0]["open"],
                    "high": max(float(bar["high"]) for bar in chunk),
                    "low": min(float(bar["low"]) for bar in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(float(bar.get("volume") or 0.0) for bar in chunk),
                    "source": f"derived:{chunk[0].get('source') or 'hourly'}",
                    "timeframe": "four_hour",
                }
            )
    return output


def _daily_db_bars(db: Session, ticker: str, limit: int = 420) -> list[dict]:
    rows = (
        db.query(DailyBar)
        .filter(DailyBar.symbol == ticker.upper())
        .order_by(DailyBar.bar_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "date": row.bar_date,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "source": row.source or "daily_bars",
        }
        for row in reversed(rows)
    ]


def semantic_levels(levels: dict[str, Any], sources: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    colors = {
        "near_confirmation": "#6ea8fe",
        "primary_entry_trigger": "#21d4a8",
        "strong_confirmation": "#4cc9f0",
        "major_trend_repair": "#b084ff",
        "optional_support_level": "#f0b35a",
        "suggested_stop": "#ff8a65",
        "invalidation_level": "#ff5f6d",
        "tp1": "#75d69c",
        "tp2": "#49bf84",
        "tp3": "#2ea76e",
        "stretch_target": "#188e5a",
    }
    output: list[dict[str, Any]] = []
    for name, color in colors.items():
        value = levels.get(name)
        if isinstance(value, dict):
            value = value.get("price") or value.get("upper") or value.get("lower")
        parsed = _number(value)
        if parsed is not None:
            output.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "price": parsed,
                    "color": color,
                    "source": (sources or {}).get(name) or "PLANNER",
                }
            )
    return output


def _attempt_markers(attempts: list[ConfirmationAttempt] | list[dict]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for attempt in attempts:
        getter = attempt.get if isinstance(attempt, dict) else lambda name, default=None: getattr(attempt, name, default)
        started = _timestamp(getter("started_at"))
        if started is None:
            continue
        outcome = str(getter("outcome") or "ATTEMPT")
        markers.append(
            {
                "time": int(started.timestamp()),
                "position": "aboveBar" if "REJECT" in outcome else "belowBar",
                "color": "#ff6b78" if "REJECT" in outcome else "#63d7b0",
                "shape": "arrowDown" if "REJECT" in outcome else "arrowUp",
                "text": f"#{getter('attempt_number', '')} {outcome}",
            }
        )
    return markers


def build_chart_bundle(
    db: Session,
    *,
    ticker: str,
    levels: dict[str, Any],
    level_sources: dict[str, Any] | None,
    bars_loader: BarsLoader,
    decision_time_boundary: datetime | None = None,
    max_bars: int = 180,
    attempts: list[ConfirmationAttempt] | list[dict] | None = None,
) -> dict[str, Any]:
    """Build one source-of-truth bundle for UI, renderer, validator, and LLM."""
    boundary = decision_time_boundary or datetime.now(timezone.utc)
    if boundary.tzinfo is None:
        boundary = boundary.replace(tzinfo=timezone.utc)
    raw_by_key: dict[str, list[dict]] = {}
    daily = _daily_db_bars(db, ticker, limit=max(max_bars, 260))
    raw_by_key["daily"] = daily or bars_loader(ticker, "daily", 420)
    for key, (timeframe, lookback) in TIMEFRAME_SPECS.items():
        if key == "daily":
            continue
        raw_by_key[key] = bars_loader(ticker, timeframe, lookback)

    timeframes: dict[str, dict[str, Any]] = {}
    all_sources: set[str] = set()
    last_bar_at: datetime | None = None
    for key, (timeframe, _lookback) in TIMEFRAME_SPECS.items():
        bars = _normalize_bars(
            raw_by_key.get(key) or [],
            timeframe=timeframe,
            decision_time_boundary=boundary,
            max_bars=max_bars,
        )
        for bar in bars:
            all_sources.add(str(bar.get("source") or "canonical_provider"))
        if bars:
            candidate = _timestamp(bars[-1]["time"])
            if candidate and (last_bar_at is None or candidate > last_bar_at):
                last_bar_at = candidate
        timeframes[key] = {
            "timeframe": timeframe,
            "bars": bars,
            "indicators": _indicator_series(bars, intraday=key != "daily"),
            "bar_count": len(bars),
            "last_bar_timestamp": bars[-1]["time"] if bars else None,
            "status": "AVAILABLE" if bars else "UNAVAILABLE",
        }

    four_hour_bars = _derive_four_hour_bars((timeframes.get("hourly") or {}).get("bars") or [])
    timeframes["four_hour"] = {
        "timeframe": "four_hour",
        "bars": four_hour_bars,
        "indicators": _indicator_series(four_hour_bars, intraday=True),
        "bar_count": len(four_hour_bars),
        "last_bar_timestamp": four_hour_bars[-1]["time"] if four_hour_bars else None,
        "status": "AVAILABLE" if four_hour_bars else "UNAVAILABLE",
        "derived_from": "hourly",
    }

    freshness = None if last_bar_at is None else max(0.0, (boundary - last_bar_at).total_seconds())
    return {
        "ticker": ticker.upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_time_boundary": boundary.isoformat(),
        "data_source": ",".join(sorted(all_sources)) if all_sources else "unavailable",
        "data_timestamp": boundary.isoformat(),
        "last_bar_timestamp": last_bar_at.isoformat() if last_bar_at else None,
        "data_freshness_seconds": freshness,
        "latest_chart_close": next(
            (
                tf["bars"][-1]["close"]
                for key in ("execution", "structure", "hourly", "four_hour", "daily")
                if (tf := timeframes[key])["bars"]
            ),
            None,
        ),
        "levels": semantic_levels(levels, level_sources),
        "attempt_markers": _attempt_markers(attempts or []),
        "timeframes": timeframes,
    }


__all__ = ["TIMEFRAME_SPECS", "build_chart_bundle", "semantic_levels"]
