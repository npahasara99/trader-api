"""Canonical market state shared by monitor planning, charts, and review."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
import math
from typing import Any, Callable
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models import MarketSnapshot


BarsLoader = Callable[[str, str, int | None], list[dict]]

SNAPSHOT_TIMEFRAMES: dict[str, tuple[str, int]] = {
    "daily": ("daily", 420),
    "hourly": ("hourly", 45),
    "thirty_minute": ("thirty_minute", 30),
    "five_minute": ("five_minute", 7),
    "one_minute": ("one_minute", 1),
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
        return parsed if math.isfinite(parsed) and parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _bar_timestamp(bar: dict) -> datetime | None:
    return _timestamp(bar.get("date") or bar.get("timestamp") or bar.get("datetime") or bar.get("time") or bar.get("bar_date"))


def _clean_bars(bars: list[dict], *, boundary: datetime) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in bars:
        at = _bar_timestamp(raw)
        close = _number(raw.get("close"))
        if at is None or close is None or at > boundary:
            continue
        row = dict(raw)
        row["date"] = at.astimezone(timezone.utc).isoformat()
        row["close"] = close
        row["source"] = str(raw.get("source") or "canonical_provider")
        output.append(row)
    output.sort(key=lambda row: str(row["date"]))
    return output


def _latest(bars: list[dict]) -> tuple[float | None, datetime | None, str | None]:
    if not bars:
        return None, None, None
    bar = bars[-1]
    return _number(bar.get("close")), _bar_timestamp(bar), str(bar.get("source") or "canonical_provider")


def _daily_atr(bars: list[dict], period: int = 14) -> float | None:
    ranges: list[float] = []
    prior_close: float | None = None
    for bar in bars[-(period + 1):]:
        high = _number(bar.get("high"))
        low = _number(bar.get("low"))
        close = _number(bar.get("close"))
        if high is None or low is None or close is None:
            continue
        ranges.append(max(high - low, abs(high - prior_close) if prior_close else 0.0, abs(low - prior_close) if prior_close else 0.0))
        prior_close = close
    return sum(ranges[-period:]) / len(ranges[-period:]) if ranges else None


def _market_session(at: datetime) -> str:
    eastern = at.astimezone(ZoneInfo("America/New_York"))
    if eastern.weekday() >= 5:
        return "CLOSED"
    minutes = eastern.hour * 60 + eastern.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "PREMARKET"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "RTH"
    if 16 * 60 <= minutes < 20 * 60:
        return "AFTER_HOURS"
    return "CLOSED"


def build_market_snapshot(
    ticker: str,
    *,
    bars_loader: BarsLoader,
    force_refresh: bool,
    created_at: datetime | None = None,
    consistency_max_pct: float = 0.0125,
    consistency_atr_fraction: float = 0.50,
) -> dict[str, Any]:
    """Fetch one time-bounded market state for every downstream consumer."""
    symbol = str(ticker or "").strip().upper()
    boundary = created_at or datetime.now(timezone.utc)
    if boundary.tzinfo is None:
        boundary = boundary.replace(tzinfo=timezone.utc)
    bars_by_timeframe: dict[str, list[dict[str, Any]]] = {}
    last_bars: dict[str, datetime | None] = {}
    sources: set[str] = set()
    for key, (provider_timeframe, lookback_days) in SNAPSHOT_TIMEFRAMES.items():
        bars = _clean_bars(bars_loader(symbol, provider_timeframe, lookback_days) or [], boundary=boundary)
        bars_by_timeframe[key] = bars
        _close, at, source = _latest(bars)
        last_bars[key] = at
        if source:
            sources.add(source)

    reference_price: float | None = None
    quote_timestamp: datetime | None = None
    reference_timeframe: str | None = None
    for key in ("one_minute", "five_minute", "thirty_minute", "hourly", "daily"):
        close, at, _source = _latest(bars_by_timeframe[key])
        if close is not None:
            reference_price, quote_timestamp, reference_timeframe = close, at, key
            break

    atr = _daily_atr(bars_by_timeframe["daily"])
    comparison: dict[str, float] = {}
    for key in ("one_minute", "five_minute", "thirty_minute"):
        close, _at, _source = _latest(bars_by_timeframe[key])
        if close is not None:
            comparison[key] = close
    mismatch_pct = None
    tolerance_pct = max(float(consistency_max_pct), ((atr or 0.0) * consistency_atr_fraction / reference_price) if reference_price else 0.0)
    if len(comparison) >= 2:
        center = sum(comparison.values()) / len(comparison)
        mismatch_pct = (max(comparison.values()) - min(comparison.values())) / max(center, 1e-9)
    consistency_status = "MARKET_DATA_MISMATCH" if mismatch_pct is not None and mismatch_pct > tolerance_pct else "CONSISTENT"
    if reference_price is None:
        consistency_status = "MARKET_DATA_UNAVAILABLE"

    session = _market_session(boundary)
    data_age_seconds = None if quote_timestamp is None else max(0.0, (boundary - quote_timestamp).total_seconds())
    allowed_age_seconds = 180.0
    if session == "AFTER_HOURS":
        allowed_age_seconds = 8 * 60 * 60
    elif session == "PREMARKET":
        allowed_age_seconds = 18 * 60 * 60
    elif session == "CLOSED":
        allowed_age_seconds = 72 * 60 * 60
    freshness_status = (
        "MARKET_DATA_UNAVAILABLE"
        if data_age_seconds is None
        else "MARKET_DATA_STALE"
        if data_age_seconds > allowed_age_seconds
        else "FRESH"
    )

    snapshot_id = str(uuid.uuid4())
    return {
        "market_snapshot_id": snapshot_id,
        "ticker": symbol,
        "created_at": boundary.isoformat(),
        "quote_timestamp": quote_timestamp.isoformat() if quote_timestamp else None,
        "reference_price": reference_price,
        "reference_timeframe": reference_timeframe,
        "data_source": ",".join(sorted(sources)) if sources else "unavailable",
        "daily_last_bar_at": last_bars["daily"].isoformat() if last_bars["daily"] else None,
        "hourly_last_bar_at": last_bars["hourly"].isoformat() if last_bars["hourly"] else None,
        "thirty_min_last_bar_at": last_bars["thirty_minute"].isoformat() if last_bars["thirty_minute"] else None,
        "five_min_last_bar_at": last_bars["five_minute"].isoformat() if last_bars["five_minute"] else None,
        "one_min_last_bar_at": last_bars["one_minute"].isoformat() if last_bars["one_minute"] else None,
        "atr": atr,
        "consistency_status": consistency_status,
        "consistency_prices": comparison,
        "consistency_difference_pct": None if mismatch_pct is None else round(mismatch_pct, 6),
        "consistency_tolerance_pct": round(tolerance_pct, 6),
        "market_session": session,
        "data_age_seconds": data_age_seconds,
        "allowed_data_age_seconds": allowed_age_seconds,
        "freshness_status": freshness_status,
        "cache_status": "BYPASS_REQUESTED" if force_refresh else "TTL_ALLOWED",
        "bars": bars_by_timeframe,
    }


def persist_market_snapshot(db: Session, snapshot: dict[str, Any]) -> MarketSnapshot:
    row = MarketSnapshot(
        id=str(snapshot["market_snapshot_id"]),
        ticker=str(snapshot["ticker"]),
        created_at=_timestamp(snapshot.get("created_at")) or datetime.now(timezone.utc),
        quote_timestamp=_timestamp(snapshot.get("quote_timestamp")),
        reference_price=_number(snapshot.get("reference_price")),
        data_source=str(snapshot.get("data_source") or "unavailable"),
        daily_last_bar_at=_timestamp(snapshot.get("daily_last_bar_at")),
        hourly_last_bar_at=_timestamp(snapshot.get("hourly_last_bar_at")),
        thirty_min_last_bar_at=_timestamp(snapshot.get("thirty_min_last_bar_at")),
        five_min_last_bar_at=_timestamp(snapshot.get("five_min_last_bar_at")),
        one_min_last_bar_at=_timestamp(snapshot.get("one_min_last_bar_at")),
        consistency_status=str(snapshot.get("consistency_status") or "INSUFFICIENT_DATA"),
        cache_status=str(snapshot.get("cache_status") or "UNKNOWN"),
        payload_json=json.dumps(snapshot, default=str, separators=(",", ":")),
    )
    db.add(row)
    db.flush()
    return row


def market_snapshot_payload(row: MarketSnapshot | None) -> dict[str, Any]:
    if row is None:
        return {}
    try:
        payload = json.loads(row.payload_json or "{}")
    except (TypeError, ValueError):
        payload = {}
    payload.setdefault("market_snapshot_id", row.id)
    payload.setdefault("reference_price", row.reference_price)
    payload.setdefault("quote_timestamp", row.quote_timestamp.isoformat() if row.quote_timestamp else None)
    payload.setdefault("data_source", row.data_source)
    return payload


__all__ = ["SNAPSHOT_TIMEFRAMES", "build_market_snapshot", "market_snapshot_payload", "persist_market_snapshot"]
