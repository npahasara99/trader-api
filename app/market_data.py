from datetime import datetime, timezone, timedelta, date
import time
from sqlalchemy.orm import Session
from sqlalchemy import text, func

from .models import DailyBar
from .logic import finnhub_get


def _fetch_finnhub_candles_payload(symbol: str, frm: date, to: date, *, max_attempts: int = 3) -> tuple[dict | None, str]:
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": int(datetime(frm.year, frm.month, frm.day, tzinfo=timezone.utc).timestamp()),
        "to": int(datetime(to.year, to.month, to.day, tzinfo=timezone.utc).timestamp()),
    }

    last_status = "fetch_failed"
    for attempt in range(max_attempts):
        data = finnhub_get("/stock/candle", params)
        if isinstance(data, dict):
            status = str(data.get("s") or "").lower().strip()
            if status == "ok":
                return data, "ok"
            if status:
                last_status = f"api_status:{status}"
            else:
                last_status = "api_status:unknown"

        if attempt < (max_attempts - 1):
            time.sleep(0.35 * (attempt + 1))

    return None, last_status


def fetch_finnhub_daily_bars_with_meta(symbol: str, frm: date, to: date) -> tuple[list[dict], str]:
    data, fetch_status = _fetch_finnhub_candles_payload(symbol, frm, to)
    if not isinstance(data, dict):
        return [], fetch_status

    if data.get("s") != "ok":
        status = str(data.get("s") or "unknown").lower().strip()
        return [], f"api_status:{status}"

    ts = data.get("t") or []
    o = data.get("o") or []
    h = data.get("h") or []
    l = data.get("l") or []
    c = data.get("c") or []
    v = data.get("v") or []

    out: list[dict] = []
    for i, t in enumerate(ts):
        try:
            d = datetime.fromtimestamp(int(t), tz=timezone.utc).date()
            close_val = float(c[i])
            out.append(
                {
                    "symbol": symbol,
                    "bar_date": d,
                    "open": (float(o[i]) if i < len(o) and o[i] is not None else None),
                    "high": (float(h[i]) if i < len(h) and h[i] is not None else None),
                    "low": (float(l[i]) if i < len(l) and l[i] is not None else None),
                    "close": close_val,
                    "volume": (float(v[i]) if i < len(v) and v[i] is not None else None),
                    "adjusted_close": close_val,
                    "source": "finnhub",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        except Exception:
            continue

    if not out:
        return [], "empty_payload"
    return out, "ok"


def fetch_finnhub_daily_bars(symbol: str, frm: date, to: date) -> list[dict]:
    bars, _ = fetch_finnhub_daily_bars_with_meta(symbol, frm, to)
    return bars


def upsert_daily_bars(db: Session, bars: list[dict]) -> int:
    if not bars:
        return 0

    dialect = db.bind.dialect.name if db.bind is not None else ""

    if dialect in ("postgresql", "sqlite"):
        sql = text(
            """
            INSERT INTO daily_bars
            (symbol, bar_date, open, high, low, close, volume, adjusted_close, source, updated_at)
            VALUES
            (:symbol, :bar_date, :open, :high, :low, :close, :volume, :adjusted_close, :source, :updated_at)
            ON CONFLICT(symbol, bar_date)
            DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                adjusted_close = excluded.adjusted_close,
                source = excluded.source,
                updated_at = excluded.updated_at
            """
        )
        db.execute(sql, bars)
        return len(bars)

    count = 0
    for b in bars:
        existing = db.get(DailyBar, (b["symbol"], b["bar_date"]))
        if existing:
            existing.open = b["open"]
            existing.high = b["high"]
            existing.low = b["low"]
            existing.close = b["close"]
            existing.volume = b["volume"]
            existing.adjusted_close = b["adjusted_close"]
            existing.source = b["source"]
            existing.updated_at = b["updated_at"]
        else:
            db.add(DailyBar(**b))
        count += 1
    return count


def get_cached_daily_closes(db: Session, symbol: str, frm: date, to: date) -> dict[date, float]:
    rows = (
        db.query(DailyBar)
        .filter(DailyBar.symbol == symbol)
        .filter(DailyBar.bar_date >= frm)
        .filter(DailyBar.bar_date <= to)
        .order_by(DailyBar.bar_date.asc())
        .all()
    )

    out: dict[date, float] = {}
    for r in rows:
        if r.close is None:
            continue
        out[r.bar_date] = float(r.close)
    return out


def ensure_cached_daily_closes(
    db: Session,
    symbol: str,
    frm: date,
    to: date,
    *,
    auto_fetch: bool = True,
    commit: bool = False,
) -> dict[date, float]:
    cached = get_cached_daily_closes(db, symbol, frm, to)
    if not auto_fetch:
        return cached

    need_fetch = False
    if not cached:
        need_fetch = True
    else:
        keys = sorted(cached.keys())
        if not keys:
            need_fetch = True
        else:
            start_slack = frm + timedelta(days=5)
            end_slack = to - timedelta(days=5)
            if keys[0] > start_slack or keys[-1] < end_slack:
                need_fetch = True

    if need_fetch:
        bars = fetch_finnhub_daily_bars(symbol, frm, to)
        if bars:
            upsert_daily_bars(db, bars)
            if commit:
                db.commit()
            else:
                db.flush()
        cached = get_cached_daily_closes(db, symbol, frm, to)

    return cached


def _coverage_stats(db: Session, symbol: str) -> dict:
    row = (
        db.query(
            func.min(DailyBar.bar_date).label("min_date"),
            func.max(DailyBar.bar_date).label("max_date"),
            func.count().label("cnt"),
        )
        .filter(DailyBar.symbol == symbol)
        .first()
    )
    if not row:
        return {"min_date": None, "max_date": None, "count": 0}
    return {
        "min_date": row.min_date,
        "max_date": row.max_date,
        "count": int(row.cnt or 0),
    }


def backfill_symbol_daily_bars(
    db: Session,
    symbol: str,
    *,
    years: int = 10,
    refresh: bool = False,
) -> dict:
    years = max(1, min(15, int(years)))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365 * years + 40)

    if not refresh:
        coverage = _coverage_stats(db, symbol)
        min_date = coverage.get("min_date")
        max_date = coverage.get("max_date")
        count = int(coverage.get("count", 0))
        if min_date and max_date and count > 1800:
            if min_date <= (start + timedelta(days=15)) and max_date >= (end - timedelta(days=5)):
                return {
                    "symbol": symbol,
                    "status": "skipped_cached",
                    "inserted": 0,
                    "count": count,
                    "fetch_status": "cached",
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                }

    bars, fetch_status = fetch_finnhub_daily_bars_with_meta(symbol, start, end)
    if not bars:
        coverage = _coverage_stats(db, symbol)
        count = int(coverage.get("count", 0))
        status = "kept_existing" if count > 0 else "no_data"
        return {
            "symbol": symbol,
            "status": status,
            "inserted": 0,
            "count": count,
            "fetch_status": fetch_status,
            "min_date": str(coverage.get("min_date")) if coverage.get("min_date") else None,
            "max_date": str(coverage.get("max_date")) if coverage.get("max_date") else None,
        }

    inserted = upsert_daily_bars(db, bars)

    coverage = _coverage_stats(db, symbol)
    return {
        "symbol": symbol,
        "status": "updated",
        "inserted": inserted,
        "count": int(coverage.get("count", 0)),
        "fetch_status": fetch_status,
        "min_date": str(coverage.get("min_date")) if coverage.get("min_date") else None,
        "max_date": str(coverage.get("max_date")) if coverage.get("max_date") else None,
    }


def backfill_universe_daily_bars(
    db: Session,
    symbols: list[str],
    *,
    years: int = 10,
    refresh: bool = False,
    commit_every: int = 5,
) -> dict:
    commit_every = max(1, min(50, int(commit_every)))

    results: list[dict] = []
    updated = 0
    failed = 0
    skipped = 0
    no_data = 0

    for i, sym in enumerate(symbols, start=1):
        try:
            r = backfill_symbol_daily_bars(db, sym, years=years, refresh=refresh)
            results.append(r)
            status = str(r.get("status") or "")

            if status == "updated":
                updated += 1
            elif status in ("skipped_cached", "kept_existing"):
                skipped += 1
            elif status == "no_data":
                no_data += 1
                failed += 1
            else:
                failed += 1

            if i % commit_every == 0:
                db.commit()
        except Exception as e:
            failed += 1
            db.rollback()
            results.append({"symbol": sym, "status": "error", "error": str(e)})

    db.commit()

    return {
        "total": len(symbols),
        "updated": updated,
        "skipped_cached": skipped,
        "no_data": no_data,
        "failed": failed,
        "results": results,
    }
