from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta, date, time as datetime_time
import csv
import io
import math
import re
import threading
import time
import requests
from typing import Callable, Protocol
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy import text, func

from .config import DEFAULT_PLANNING_CONFIG
from .models import DailyBar, DailyBarCacheStatus
from .logic import FINNHUB_API_KEY, FINNHUB_BASE


YAHOO_CHART_BASES = [
    "https://query1.finance.yahoo.com/v8/finance/chart",
    "https://query2.finance.yahoo.com/v8/finance/chart",
]
YAHOO_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"

_LAST_YAHOO_REQUEST_TS = 0.0
_YAHOO_PACE_LOCK = threading.Lock()
_TIMEFRAME_CACHE: dict[tuple[str, str, int], tuple[float, list[dict]]] = {}
_TIMEFRAME_CACHE_LOCK = threading.Lock()

TIMEFRAME_ALIASES = {
    "1m": "one_minute",
    "one_minute": "one_minute",
    "5m": "five_minute",
    "five_minute": "five_minute",
    "d": "daily",
    "1d": "daily",
    "daily": "daily",
    "h": "hourly",
    "1h": "hourly",
    "60m": "hourly",
    "hourly": "hourly",
    "30": "thirty_minute",
    "30m": "thirty_minute",
    "thirty_minute": "thirty_minute",
}

YAHOO_INTERVALS = {
    "one_minute": "1m",
    "five_minute": "5m",
    "daily": "1d",
    "hourly": "60m",
    "thirty_minute": "30m",
}

DEFAULT_TIMEFRAME_LOOKBACK_DAYS = {
    "one_minute": 1,
    "five_minute": 5,
    "daily": 320,
    "hourly": 90,
    "thirty_minute": 30,
}


class StructuredMarketDataProvider(Protocol):
    """Provider-neutral interface consumed by chart-context planning."""

    def get_bars(self, ticker: str, timeframe: str, lookback_days: int | None = None) -> list[dict]: ...


def last_completed_market_date(at: datetime | None = None) -> date:
    """Return the latest completed US regular-session date.

    A post-close grace period avoids treating today's candle as complete while
    providers are still publishing it. Exchange holidays are resolved from a
    current benchmark cache by ``resolve_expected_market_date`` when available.
    """

    current_et = (at or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    candidate = current_et.date()
    if current_et.weekday() < 5 and current_et.time() < datetime_time(16, 15):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def resolve_expected_market_date(
    db: Session,
    *,
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ"),
    at: datetime | None = None,
) -> date:
    """Use fresh benchmark bars to account for weekends and exchange holidays."""

    calendar_date = last_completed_market_date(at)
    normalized = [canonicalize_symbol(symbol) for symbol in benchmark_symbols]
    benchmark_date = (
        db.query(func.max(DailyBar.bar_date))
        .filter(DailyBar.symbol.in_(normalized))
        .filter(DailyBar.bar_date <= calendar_date)
        .scalar()
    )
    if benchmark_date and (calendar_date - benchmark_date).days <= 4:
        return benchmark_date
    return calendar_date


def build_bulk_cached_daily_loaders(
    db: Session,
    symbols: list[str],
    *,
    lookback_days: int = 320,
    max_age_days: int | None = None,
    min_history_bars: int = 60,
    expected_market_date: date | None = None,
) -> tuple[
    Callable[[str, date, date], dict[date, float]],
    Callable[[str], list[dict]],
    dict,
]:
    """Load a large universe from ``daily_bars`` without provider calls.

    This read primitive performs one database query after the orchestration
    layer has repaired missing/stale symbols, then exposes the same loader
    interfaces used by the scanner.
    """

    normalized = sorted(
        {
            str(symbol or "").strip().upper()
            for symbol in symbols
            if str(symbol or "").strip()
        }
    )
    end = expected_market_date or last_completed_market_date()
    start = end - timedelta(days=max(30, int(lookback_days)))
    # ``max_age_days`` remains in the signature for compatibility. Daily-bar
    # freshness is tied to the last completed US session, not elapsed wall time.
    freshness_cutoff = None if max_age_days is None else end
    bars_by_symbol: dict[str, list[dict]] = {symbol: [] for symbol in normalized}

    if normalized:
        rows = (
            db.query(
                DailyBar.symbol,
                DailyBar.bar_date,
                DailyBar.open,
                DailyBar.high,
                DailyBar.low,
                DailyBar.close,
                DailyBar.volume,
                DailyBar.adjusted_close,
                DailyBar.source,
            )
            .filter(DailyBar.symbol.in_(normalized))
            .filter(DailyBar.bar_date >= start)
            .filter(DailyBar.bar_date <= end)
            .order_by(DailyBar.symbol.asc(), DailyBar.bar_date.asc())
            .all()
        )
        for row in rows:
            symbol = str(row.symbol).upper()
            bars_by_symbol.setdefault(symbol, []).append(
                {
                    "symbol": symbol,
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

    latest_by_symbol = {
        symbol: bars[-1]["bar_date"]
        for symbol, bars in bars_by_symbol.items()
        if bars
    }
    stale_symbols = {
        symbol
        for symbol, latest in latest_by_symbol.items()
        if freshness_cutoff is not None and latest < freshness_cutoff
    }

    def _bars_loader(symbol: str) -> list[dict]:
        normalized_symbol = str(symbol or "").strip().upper()
        if normalized_symbol in stale_symbols:
            return []
        return list(bars_by_symbol.get(normalized_symbol, []))

    def _closes_loader(symbol: str, frm: date, to: date) -> dict[date, float]:
        return {
            bar["bar_date"]: float(bar["close"])
            for bar in _bars_loader(symbol)
            if frm <= bar["bar_date"] <= to and bar.get("close") is not None
        }

    sufficient_history_symbols = {
        symbol
        for symbol, bars in bars_by_symbol.items()
        if len(bars) >= max(1, int(min_history_bars)) and symbol not in stale_symbols
    }
    current_symbols = set(latest_by_symbol) - stale_symbols
    cache_as_of = max(latest_by_symbol.values()) if latest_by_symbol else None
    coverage = {
        "requested_symbols": len(normalized),
        "symbols_with_data": len(latest_by_symbol),
        "symbols_current": len(current_symbols),
        "symbols_with_sufficient_history": len(sufficient_history_symbols),
        "cache_as_of": cache_as_of.isoformat() if cache_as_of else None,
        "expected_market_date": end.isoformat(),
        "freshness_cutoff": freshness_cutoff.isoformat() if freshness_cutoff else None,
        "market_data_coverage_pct": round(len(current_symbols) / max(len(normalized), 1), 4),
        "history_coverage_pct": round(len(sufficient_history_symbols) / max(len(normalized), 1), 4),
        "oldest_current_cache": (
            min(latest_by_symbol[symbol] for symbol in current_symbols).isoformat()
            if current_symbols else None
        ),
        "missing_symbols": [symbol for symbol in normalized if symbol not in latest_by_symbol],
        "stale_symbols": sorted(stale_symbols),
    }
    return _closes_loader, _bars_loader, coverage


def canonicalize_symbol(symbol: str) -> str:
    """Return the stable cache key used across universe and planner code."""

    return str(symbol or "").strip().upper().replace(".", "-")


def provider_symbol_for(symbol: str, provider: str) -> str:
    """Map a canonical symbol to provider syntax without special-case tickers."""

    canonical = canonicalize_symbol(symbol)
    if not canonical:
        raise ValueError("Ticker is empty")
    provider_name = str(provider or "").strip().lower()
    if provider_name == "yahoo":
        return canonical
    if provider_name in {"finnhub", "stooq"} and re.fullmatch(r"[A-Z]+-[A-Z]", canonical):
        return canonical.rsplit("-", 1)[0] + "." + canonical.rsplit("-", 1)[1]
    return canonical


def _normalize_symbol_for_yahoo(symbol: str) -> str:
    # Yahoo uses '-' for class shares (e.g. BRK-B instead of BRK.B)
    return provider_symbol_for(symbol, "yahoo")


def _stooq_symbol_candidates(symbol: str) -> list[str]:
    s = symbol.upper()
    cands: list[str] = []
    for cand in (s, s.replace(".", "-"), s.replace("-", "."), s.replace(".", ""), s.replace("-", "")):
        cand = cand.strip().lower()
        if cand and cand not in cands:
            cands.append(cand)
    return cands


def _yahoo_pace(min_interval_sec: float = 0.9) -> None:
    global _LAST_YAHOO_REQUEST_TS
    with _YAHOO_PACE_LOCK:
        now = time.time()
        delta = now - _LAST_YAHOO_REQUEST_TS
        if delta < min_interval_sec:
            time.sleep(min_interval_sec - delta)
        _LAST_YAHOO_REQUEST_TS = time.time()


def normalize_timeframe(timeframe: str) -> str:
    normalized = TIMEFRAME_ALIASES.get(str(timeframe or "").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return normalized


def _fetch_yahoo_interval_payload(
    symbol: str,
    *,
    timeframe: str,
    lookback_days: int,
    max_attempts: int = 2,
) -> tuple[dict | None, str]:
    normalized = normalize_timeframe(timeframe)
    interval = YAHOO_INTERVALS[normalized]
    now = datetime.now(timezone.utc)
    params = {
        "period1": int((now - timedelta(days=max(1, int(lookback_days)))).timestamp()),
        "period2": int(now.timestamp()) + 60,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    yahoo_symbol = _normalize_symbol_for_yahoo(symbol)
    last_status = "yahoo_fetch_failed"
    for attempt in range(max_attempts):
        for base in YAHOO_CHART_BASES:
            try:
                _yahoo_pace()
                response = requests.get(
                    f"{base}/{yahoo_symbol}",
                    params=params,
                    headers=YAHOO_REQUEST_HEADERS,
                    timeout=12,
                )
                payload = response.json() if response.content else None
                if response.status_code == 200 and isinstance(payload, dict):
                    chart = payload.get("chart") or {}
                    if chart.get("result"):
                        return payload, "ok"
                    error = chart.get("error") or {}
                    last_status = str(error.get("description") or "yahoo_empty_result")[:120]
                else:
                    last_status = f"http_{response.status_code}"
            except (requests.RequestException, ValueError):
                last_status = "request_exception"
        if attempt < max_attempts - 1:
            time.sleep(0.3 * (attempt + 1))
    return None, last_status


def _normalized_bars_from_yahoo_payload(symbol: str, timeframe: str, data: dict) -> list[dict]:
    chart = data.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return []
    result = results[0] or {}
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    output: list[dict] = []
    for index, timestamp in enumerate(timestamps):
        try:
            close = (quote.get("close") or [])[index]
            if close is None:
                continue

            def value(name: str):
                values = quote.get(name) or []
                return float(values[index]) if index < len(values) and values[index] is not None else None

            output.append(
                {
                    "symbol": symbol.upper(),
                    "date": datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
                    "open": value("open"),
                    "high": value("high"),
                    "low": value("low"),
                    "close": float(close),
                    "volume": value("volume"),
                    "timeframe": normalize_timeframe(timeframe),
                    "source": "yahoo",
                }
            )
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    return output


def get_bars(
    ticker: str,
    timeframe: str,
    lookback_days: int | None = None,
    *,
    cache_ttl_seconds: int = 900,
) -> list[dict]:
    """Return provider-normalized bars without exposing vendor payload shapes.

    Failures intentionally return an empty list so the chart-context engine can
    report a missing timeframe without crashing the structured swing plan.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return []
    normalized = normalize_timeframe(timeframe)
    requested_lookback = int(lookback_days or DEFAULT_TIMEFRAME_LOOKBACK_DAYS[normalized])
    cache_key = (symbol, normalized, requested_lookback)
    now = time.time()
    with _TIMEFRAME_CACHE_LOCK:
        cached = _TIMEFRAME_CACHE.get(cache_key)
        if cached and now - cached[0] <= max(0, cache_ttl_seconds):
            return [dict(bar) for bar in cached[1]]

    if normalized == "daily":
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=requested_lookback)
        bars, _ = fetch_finnhub_daily_bars_with_meta(symbol, start, end)
    else:
        payload, status = _fetch_yahoo_interval_payload(
            symbol,
            timeframe=normalized,
            lookback_days=requested_lookback,
        )
        bars = _normalized_bars_from_yahoo_payload(symbol, normalized, payload) if payload and status == "ok" else []

    with _TIMEFRAME_CACHE_LOCK:
        _TIMEFRAME_CACHE[cache_key] = (now, [dict(bar) for bar in bars])
    return bars


class DefaultStructuredMarketDataProvider:
    """Default cached provider used by planner adapters and future replacements."""

    def get_bars(self, ticker: str, timeframe: str, lookback_days: int | None = None) -> list[dict]:
        return get_bars(ticker, timeframe, lookback_days)


def _fetch_finnhub_candles_payload(symbol: str, frm: date, to: date, *, max_attempts: int = 3) -> tuple[dict | None, str]:
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": int(datetime(frm.year, frm.month, frm.day, tzinfo=timezone.utc).timestamp()),
        "to": int(datetime(to.year, to.month, to.day, tzinfo=timezone.utc).timestamp()),
        "token": FINNHUB_API_KEY,
    }

    if not FINNHUB_API_KEY:
        return None, "missing_api_key"

    last_status = "fetch_failed"
    for attempt in range(max_attempts):
        try:
            r = requests.get(f"{FINNHUB_BASE}/stock/candle", params=params, timeout=12)
            status_code = int(r.status_code)

            payload = None
            try:
                payload = r.json()
            except Exception:
                payload = None

            if status_code == 200 and isinstance(payload, dict):
                status = str(payload.get("s") or "").lower().strip()
                if status == "ok":
                    return payload, "ok"
                if status:
                    last_status = f"api_status:{status}"
                else:
                    err = payload.get("error") if isinstance(payload, dict) else None
                    last_status = f"api_error:{str(err)[:100]}" if err else "api_status:unknown"
            else:
                if isinstance(payload, dict) and payload.get("error"):
                    last_status = f"http_{status_code}:{str(payload.get('error'))[:80]}"
                else:
                    last_status = f"http_{status_code}"

        except requests.Timeout:
            last_status = "request_timeout"
        except requests.RequestException:
            last_status = "request_exception"

        if attempt < (max_attempts - 1):
            time.sleep(0.35 * (attempt + 1))

    return None, last_status


def _fetch_yahoo_candles_payload(
    symbol: str,
    frm: date,
    to: date,
    *,
    max_attempts: int = 4,
    min_interval_sec: float = 0.9,
) -> tuple[dict | None, str]:
    yahoo_symbol = _normalize_symbol_for_yahoo(symbol)
    period1 = int(datetime(frm.year, frm.month, frm.day, tzinfo=timezone.utc).timestamp())
    # Yahoo period2 is exclusive; add one day to include `to`.
    period2 = int(datetime(to.year, to.month, to.day, tzinfo=timezone.utc).timestamp()) + 86400

    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    last_status = "yahoo_fetch_failed"
    for attempt in range(max_attempts):
        for base in YAHOO_CHART_BASES:
            try:
                _yahoo_pace(min_interval_sec)
                r = requests.get(f"{base}/{yahoo_symbol}", params=params, headers=YAHOO_REQUEST_HEADERS, timeout=12)
                status_code = int(r.status_code)

                payload = None
                try:
                    payload = r.json()
                except Exception:
                    payload = None

                if status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        delay = max(1.5, float(retry_after)) if retry_after is not None else (1.5 + attempt)
                    except Exception:
                        delay = 1.5 + attempt
                    time.sleep(delay)
                    last_status = "yahoo_http_429"
                    continue

                if status_code == 200 and isinstance(payload, dict):
                    chart = payload.get("chart") or {}
                    err = chart.get("error")
                    if err:
                        msg = str((err or {}).get("description") or (err or {}).get("code") or "unknown")
                        last_status = f"yahoo_error:{msg[:80]}"
                    else:
                        results = chart.get("result") or []
                        if results:
                            return payload, "ok"
                        last_status = "yahoo_empty_result"
                else:
                    last_status = f"yahoo_http_{status_code}"

            except requests.Timeout:
                last_status = "yahoo_timeout"
            except requests.RequestException:
                last_status = "yahoo_request_exception"

        if attempt < (max_attempts - 1):
            time.sleep(0.8 * (attempt + 1))

    return None, last_status


def _fetch_stooq_daily_csv(symbol: str, frm: date, to: date, *, max_attempts: int = 2) -> tuple[list[dict], str]:
    last_status = "stooq_fetch_failed"

    for candidate in _stooq_symbol_candidates(symbol):
        ticker = f"{candidate}.us"
        for attempt in range(max_attempts):
            try:
                r = requests.get(
                    STOOQ_DAILY_URL,
                    params={"s": ticker, "i": "d"},
                    headers={"User-Agent": YAHOO_REQUEST_HEADERS["User-Agent"]},
                    timeout=12,
                )
                status_code = int(r.status_code)
                if status_code != 200:
                    last_status = f"stooq_http_{status_code}"
                    if attempt < (max_attempts - 1):
                        time.sleep(0.6 * (attempt + 1))
                    continue

                text_body = (r.text or "").strip()
                if not text_body or text_body.lower().startswith("no data"):
                    last_status = "stooq_no_data"
                    break

                reader = csv.DictReader(io.StringIO(text_body))
                rows = list(reader)
                if not rows:
                    last_status = "stooq_empty_csv"
                    break

                bars: list[dict] = []
                for row in rows:
                    try:
                        d = datetime.strptime(str(row.get("Date", "")), "%Y-%m-%d").date()
                        if d < frm or d > to:
                            continue

                        close_raw = row.get("Close")
                        if close_raw in (None, "", "null"):
                            continue

                        close_val = float(close_raw)
                        open_val = row.get("Open")
                        high_val = row.get("High")
                        low_val = row.get("Low")
                        vol_val = row.get("Volume")

                        bars.append(
                            {
                                "symbol": symbol,
                                "bar_date": d,
                                "open": (float(open_val) if open_val not in (None, "", "null") else None),
                                "high": (float(high_val) if high_val not in (None, "", "null") else None),
                                "low": (float(low_val) if low_val not in (None, "", "null") else None),
                                "close": close_val,
                                "volume": (float(vol_val) if vol_val not in (None, "", "null") else None),
                                "adjusted_close": close_val,
                                "source": "stooq",
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                    except Exception:
                        continue

                if bars:
                    return bars, f"ok:{ticker}"
                last_status = "stooq_no_inrange_rows"

            except requests.Timeout:
                last_status = "stooq_timeout"
            except requests.RequestException:
                last_status = "stooq_request_exception"

            if attempt < (max_attempts - 1):
                time.sleep(0.6 * (attempt + 1))

    return [], last_status


def _bars_from_finnhub_payload(symbol: str, data: dict) -> list[dict]:
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

    return out


def _bars_from_yahoo_payload(symbol: str, data: dict) -> list[dict]:
    chart = data.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return []

    r0 = results[0] or {}
    ts = r0.get("timestamp") or []
    indicators = r0.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    adj = (indicators.get("adjclose") or [{}])[0] or {}

    o = quote.get("open") or []
    h = quote.get("high") or []
    l = quote.get("low") or []
    c = quote.get("close") or []
    v = quote.get("volume") or []
    ac = adj.get("adjclose") or []

    out: list[dict] = []
    for i, t in enumerate(ts):
        try:
            if i >= len(c) or c[i] is None:
                continue
            d = datetime.fromtimestamp(int(t), tz=timezone.utc).date()
            close_val = float(c[i])
            adj_val = float(ac[i]) if i < len(ac) and ac[i] is not None else close_val
            out.append(
                {
                    "symbol": symbol,
                    "bar_date": d,
                    "open": (float(o[i]) if i < len(o) and o[i] is not None else None),
                    "high": (float(h[i]) if i < len(h) and h[i] is not None else None),
                    "low": (float(l[i]) if i < len(l) and l[i] is not None else None),
                    "close": close_val,
                    "volume": (float(v[i]) if i < len(v) and v[i] is not None else None),
                    "adjusted_close": adj_val,
                    "source": "yahoo",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        except Exception:
            continue

    return out


def fetch_finnhub_daily_bars_with_meta(symbol: str, frm: date, to: date) -> tuple[list[dict], str]:
    data, fetch_status = _fetch_finnhub_candles_payload(symbol, frm, to)
    if isinstance(data, dict) and str(data.get("s") or "").lower().strip() == "ok":
        bars = _bars_from_finnhub_payload(symbol, data)
        if bars:
            return bars, "ok"
        fetch_status = "empty_payload_finnhub"

    y_data, y_status = _fetch_yahoo_candles_payload(symbol, frm, to)
    if isinstance(y_data, dict) and y_status == "ok":
        y_bars = _bars_from_yahoo_payload(symbol, y_data)
        if y_bars:
            return y_bars, f"fallback_yahoo_after:{fetch_status}"
        y_status = "yahoo_empty_rows"

    s_bars, s_status = _fetch_stooq_daily_csv(symbol, frm, to)
    if s_bars:
        return s_bars, f"fallback_stooq_after:{fetch_status}|yahoo:{y_status}|stooq:{s_status}"

    return [], f"{fetch_status}|yahoo:{y_status}|stooq:{s_status}"


def fetch_finnhub_daily_bars(symbol: str, frm: date, to: date) -> list[dict]:
    bars, _ = fetch_finnhub_daily_bars_with_meta(symbol, frm, to)
    return bars


def fetch_daily_bars_for_cache(
    symbol: str,
    frm: date,
    to: date,
) -> tuple[list[dict], str, str | None, str | None]:
    """Fetch broad-universe bars with a rate-aware provider fallback chain.

    Yahoo is the primary bulk source because it supports the complete US
    equity universe without requiring one paid candle entitlement per symbol.
    Existing Finnhub and Stooq adapters remain deterministic fallbacks.
    """

    canonical = canonicalize_symbol(symbol)
    if not canonical:
        return [], "ticker_mapping_error", None, None

    yahoo_symbol = provider_symbol_for(canonical, "yahoo")
    payload, yahoo_status = _fetch_yahoo_candles_payload(
        yahoo_symbol,
        frm,
        to,
        max_attempts=3,
        min_interval_sec=DEFAULT_PLANNING_CONFIG.sp500_market_data_yahoo_interval_seconds,
    )
    if isinstance(payload, dict) and yahoo_status == "ok":
        bars = _bars_from_yahoo_payload(canonical, payload)
        if bars:
            return bars, "ok", "yahoo", yahoo_symbol

    finnhub_symbol = provider_symbol_for(canonical, "finnhub")
    payload, finnhub_status = _fetch_finnhub_candles_payload(finnhub_symbol, frm, to, max_attempts=2)
    if isinstance(payload, dict) and str(payload.get("s") or "").lower().strip() == "ok":
        bars = _bars_from_finnhub_payload(canonical, payload)
        if bars:
            return bars, "ok", "finnhub", finnhub_symbol

    stooq_symbol = provider_symbol_for(canonical, "stooq")
    bars, stooq_status = _fetch_stooq_daily_csv(stooq_symbol, frm, to, max_attempts=2)
    if bars:
        for bar in bars:
            bar["symbol"] = canonical
        return bars, "ok", "stooq", stooq_symbol

    return (
        [],
        f"yahoo:{yahoo_status}|finnhub:{finnhub_status}|stooq:{stooq_status}",
        None,
        yahoo_symbol,
    )


def normalize_fetch_failure(status: str) -> str:
    """Collapse provider-specific text into stable scanner diagnostic codes."""

    value = str(status or "").lower()
    if "mapping" in value:
        return "TICKER_MAPPING_ERROR"
    if "429" in value or "rate" in value:
        return "PROVIDER_RATE_LIMIT"
    if "timeout" in value:
        return "TIMEOUT"
    if "404" in value or "not found" in value or "invalid symbol" in value:
        return "INVALID_SYMBOL"
    if "delisted" in value:
        return "DELISTED"
    if "malformed" in value or "decode" in value:
        return "MALFORMED_RESPONSE"
    if "empty" in value or "no_data" in value or "no data" in value:
        return "NO_DATA"
    return "FETCH_FAILED"


def validate_daily_bars(
    bars: list[dict],
    *,
    canonical_symbol: str,
) -> tuple[list[dict], dict]:
    """Normalize fetched bars and reject materially malformed OHLCV payloads."""

    canonical = canonicalize_symbol(canonical_symbol)
    by_date: dict[date, dict] = {}
    malformed = 0
    duplicate_dates = 0
    volume_present = 0
    for raw in bars or []:
        try:
            bar_date = raw.get("bar_date")
            if isinstance(bar_date, str):
                bar_date = date.fromisoformat(bar_date[:10])
            close = float(raw.get("close"))
            if not isinstance(bar_date, date) or not math.isfinite(close) or close <= 0:
                raise ValueError("invalid date or close")
            normalized = dict(raw)
            normalized.update({"symbol": canonical, "bar_date": bar_date, "close": close})
            for key in ("open", "high", "low", "adjusted_close", "volume"):
                value = normalized.get(key)
                normalized[key] = float(value) if value is not None else None
                if normalized[key] is not None and not math.isfinite(normalized[key]):
                    raise ValueError(f"invalid {key}")
            if normalized.get("volume") is not None and normalized["volume"] < 0:
                raise ValueError("negative volume")
            high = normalized.get("high")
            low = normalized.get("low")
            open_price = normalized.get("open")
            if high is not None and low is not None:
                observed = [value for value in (open_price, close) if value is not None]
                if high < low or (observed and (high < max(observed) or low > min(observed))):
                    raise ValueError("invalid OHLC relationship")
            if normalized.get("volume") is not None and normalized["volume"] > 0:
                volume_present += 1
            normalized["source"] = str(normalized.get("source") or "unknown")[:20]
            normalized["updated_at"] = normalized.get("updated_at") or datetime.now(timezone.utc)
            if bar_date in by_date:
                duplicate_dates += 1
            by_date[bar_date] = normalized
        except (TypeError, ValueError, OverflowError):
            malformed += 1

    normalized_bars = [by_date[key] for key in sorted(by_date)]
    warnings: list[str] = []
    if duplicate_dates:
        warnings.append(f"duplicate_dates:{duplicate_dates}")
    if len(normalized_bars) >= 20:
        max_gap = max(
            (right["bar_date"] - left["bar_date"]).days
            for left, right in zip(normalized_bars, normalized_bars[1:])
        )
        if max_gap > 14:
            warnings.append(f"large_calendar_gap_days:{max_gap}")
        if volume_present / len(normalized_bars) < 0.10:
            warnings.append("volume_mostly_missing")
    malformed_ratio = malformed / max(len(bars or []), 1)
    invalid = not normalized_bars or malformed_ratio > 0.05
    return normalized_bars, {
        "valid": not invalid,
        "input_rows": len(bars or []),
        "valid_rows": len(normalized_bars),
        "malformed_rows": malformed,
        "malformed_ratio": round(malformed_ratio, 4),
        "duplicate_dates": duplicate_dates,
        "warnings": warnings,
    }


def _coverage_stats_by_symbols(db: Session, symbols: list[str]) -> dict[str, dict]:
    normalized = sorted({canonicalize_symbol(symbol) for symbol in symbols if canonicalize_symbol(symbol)})
    if not normalized:
        return {}
    rows = (
        db.query(
            DailyBar.symbol,
            func.min(DailyBar.bar_date).label("min_date"),
            func.max(DailyBar.bar_date).label("max_date"),
            func.count().label("cnt"),
        )
        .filter(DailyBar.symbol.in_(normalized))
        .group_by(DailyBar.symbol)
        .all()
    )
    return {
        str(row.symbol).upper(): {
            "min_date": row.min_date,
            "max_date": row.max_date,
            "count": int(row.cnt or 0),
        }
        for row in rows
    }


def _cache_state(stats: dict, *, expected_date: date, min_history_bars: int) -> str:
    count = int(stats.get("count") or 0)
    last_bar = stats.get("max_date")
    if count <= 0 or last_bar is None:
        return "CACHE_MISSING"
    if last_bar < expected_date:
        return "CACHE_STALE"
    if count < int(min_history_bars):
        return "INSUFFICIENT_HISTORY"
    return "CURRENT"


def _record_cache_status(
    db: Session,
    *,
    symbol: str,
    provider_symbol: str | None,
    provider: str | None,
    stats: dict,
    state: str,
    min_history_bars: int,
    attempted_at: datetime,
    success: bool,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> None:
    existing = db.get(DailyBarCacheStatus, symbol)
    row = existing or DailyBarCacheStatus(canonical_symbol=symbol)
    row.provider_symbol = provider_symbol or row.provider_symbol
    row.provider = provider or row.provider
    row.last_bar_date = stats.get("max_date")
    row.row_count = int(stats.get("count") or 0)
    row.data_source = provider or row.data_source
    row.freshness_status = state
    row.history_sufficient = row.row_count >= int(min_history_bars)
    row.last_updated_at = datetime.now(timezone.utc)
    row.last_attempt_at = attempted_at
    if success:
        row.last_success_at = attempted_at
    row.last_error_code = error_code
    row.last_error_detail = (str(error_detail)[:2000] if error_detail else None)
    if existing is None:
        db.add(row)


def repair_daily_bar_cache(
    db: Session,
    symbols: list[str],
    *,
    history_days: int = 460,
    min_history_bars: int = 200,
    expected_date: date | None = None,
    max_workers: int = 4,
    commit_every: int = 20,
    incremental_overlap_days: int = 5,
    refresh: bool = False,
    fetcher: Callable[[str, date, date], tuple[list[dict], str, str | None, str | None]] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Incrementally repair missing/stale daily bars with per-symbol isolation."""

    started = time.monotonic()
    normalized = sorted({canonicalize_symbol(symbol) for symbol in symbols if canonicalize_symbol(symbol)})
    expected = expected_date or resolve_expected_market_date(db)
    history_days = max(320, int(history_days))
    min_history_bars = max(60, int(min_history_bars))
    history_start = expected - timedelta(days=history_days)
    max_workers = max(1, min(8, int(max_workers)))
    commit_every = max(1, min(100, int(commit_every)))
    overlap = max(0, min(14, int(incremental_overlap_days)))
    fetch_fn = fetcher or fetch_daily_bars_for_cache

    initial_stats = _coverage_stats_by_symbols(db, normalized)
    initial_states = {
        symbol: _cache_state(initial_stats.get(symbol, {}), expected_date=expected, min_history_bars=min_history_bars)
        for symbol in normalized
    }
    initial_counts = Counter(initial_states.values())
    tasks: dict[str, tuple[date, date]] = {}
    for symbol, state in initial_states.items():
        if not refresh and state == "CURRENT":
            continue
        stats = initial_stats.get(symbol, {})
        last_bar = stats.get("max_date")
        if state == "CACHE_STALE" and int(stats.get("count") or 0) >= min_history_bars and last_bar:
            fetch_start = max(history_start, last_bar - timedelta(days=overlap))
        else:
            fetch_start = history_start
        tasks[symbol] = (fetch_start, expected)

    results_by_symbol: dict[str, dict] = {
        symbol: {
            "symbol": symbol,
            "status": "skipped_cached",
            "reason_code": "CURRENT",
            "inserted": 0,
            "count": int(initial_stats.get(symbol, {}).get("count") or 0),
            "min_date": str(initial_stats.get(symbol, {}).get("min_date") or "") or None,
            "max_date": str(initial_stats.get(symbol, {}).get("max_date") or "") or None,
        }
        for symbol in normalized
        if symbol not in tasks
    }
    provider_counts: Counter[str] = Counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="daily-bar-repair") as executor:
        futures = {
            executor.submit(fetch_fn, symbol, fetch_range[0], fetch_range[1]): symbol
            for symbol, fetch_range in tasks.items()
        }
        for future in as_completed(futures):
            symbol = futures[future]
            attempted_at = datetime.now(timezone.utc)
            try:
                fetched_bars, fetch_status, provider, provider_symbol = future.result()
            except Exception as exc:
                fetched_bars, fetch_status, provider, provider_symbol = [], f"{type(exc).__name__}: {exc}", None, None

            normalized_bars, validation = validate_daily_bars(fetched_bars, canonical_symbol=symbol)
            if not normalized_bars:
                reason_code = normalize_fetch_failure(fetch_status)
                stats = initial_stats.get(symbol, {})
                state = _cache_state(stats, expected_date=expected, min_history_bars=min_history_bars)
                try:
                    with db.begin_nested():
                        _record_cache_status(
                            db,
                            symbol=symbol,
                            provider_symbol=provider_symbol,
                            provider=provider,
                            stats=stats,
                            state=state,
                            min_history_bars=min_history_bars,
                            attempted_at=attempted_at,
                            success=False,
                            error_code=reason_code,
                            error_detail=fetch_status,
                        )
                except Exception:
                    pass
                results_by_symbol[symbol] = {
                    "symbol": symbol,
                    "status": "no_data",
                    "reason_code": reason_code,
                    "fetch_status": fetch_status,
                    "inserted": 0,
                    "count": int(stats.get("count") or 0),
                    "provider_symbol": provider_symbol,
                }
            elif not validation["valid"]:
                results_by_symbol[symbol] = {
                    "symbol": symbol,
                    "status": "error",
                    "reason_code": "MALFORMED_RESPONSE",
                    "fetch_status": fetch_status,
                    "inserted": 0,
                    "validation": validation,
                    "provider_symbol": provider_symbol,
                }
            else:
                try:
                    with db.begin_nested():
                        inserted = upsert_daily_bars(db, normalized_bars)
                        db.flush()
                        stats = _coverage_stats(db, symbol)
                        state = _cache_state(stats, expected_date=expected, min_history_bars=min_history_bars)
                        _record_cache_status(
                            db,
                            symbol=symbol,
                            provider_symbol=provider_symbol,
                            provider=provider,
                            stats=stats,
                            state=state,
                            min_history_bars=min_history_bars,
                            attempted_at=attempted_at,
                            success=True,
                        )
                    if provider:
                        provider_counts[provider] += 1
                    results_by_symbol[symbol] = {
                        "symbol": symbol,
                        "status": "updated",
                        "reason_code": state,
                        "fetch_status": fetch_status,
                        "inserted": inserted,
                        "count": int(stats.get("count") or 0),
                        "min_date": str(stats.get("min_date")) if stats.get("min_date") else None,
                        "max_date": str(stats.get("max_date")) if stats.get("max_date") else None,
                        "provider": provider,
                        "provider_symbol": provider_symbol,
                        "validation": validation,
                    }
                except Exception as exc:
                    results_by_symbol[symbol] = {
                        "symbol": symbol,
                        "status": "error",
                        "reason_code": "FETCH_FAILED",
                        "fetch_status": f"database_write:{type(exc).__name__}: {exc}",
                        "inserted": 0,
                        "provider": provider,
                        "provider_symbol": provider_symbol,
                    }
            completed += 1
            if progress_callback is not None:
                processed = list(results_by_symbol.values())
                progress_callback(
                    {
                        "completed": completed,
                        "total": len(tasks),
                        "updated": sum(item.get("status") == "updated" for item in processed),
                        "failed": sum(item.get("status") in {"no_data", "error"} for item in processed),
                    }
                )
            if completed % commit_every == 0:
                db.commit()
                processed = list(results_by_symbol.values())
                print(
                    "Daily-bar repair progress: "
                    f"completed={completed}/{len(tasks)} "
                    f"updated={sum(item.get('status') == 'updated' for item in processed)} "
                    f"failed={sum(item.get('status') in {'no_data', 'error'} for item in processed)}"
                )
    db.commit()

    final_stats = _coverage_stats_by_symbols(db, normalized)
    final_states = {
        symbol: _cache_state(final_stats.get(symbol, {}), expected_date=expected, min_history_bars=min_history_bars)
        for symbol in normalized
    }
    final_counts = Counter(final_states.values())
    results = [results_by_symbol[symbol] for symbol in normalized]
    failures = [
        {
            "ticker": item["symbol"],
            "reason": item.get("reason_code") or "FETCH_FAILED",
            "details": item.get("fetch_status"),
        }
        for item in results
        if item.get("status") in {"no_data", "error"}
    ]
    data_quality_warnings = [
        {"ticker": item["symbol"], "warnings": item["validation"]["warnings"]}
        for item in results
        if (item.get("validation") or {}).get("warnings")
    ]
    fetch_success = sum(item.get("status") == "updated" for item in results)
    current = int(final_counts.get("CURRENT", 0))
    return {
        "universe_count": len(normalized),
        "expected_market_date": expected.isoformat(),
        "history_days_requested": history_days,
        "min_history_bars": min_history_bars,
        "cache_hits": int(initial_counts.get("CURRENT", 0)) if not refresh else 0,
        "cache_missing": int(initial_counts.get("CACHE_MISSING", 0)),
        "cache_stale": int(initial_counts.get("CACHE_STALE", 0)),
        "initial_history_insufficient": int(initial_counts.get("INSUFFICIENT_HISTORY", 0)),
        "fetch_attempted": len(tasks),
        "fetch_success": int(fetch_success),
        "fetch_failed": len(failures),
        "current": current,
        "history_insufficient": int(final_counts.get("INSUFFICIENT_HISTORY", 0)),
        "missing": int(final_counts.get("CACHE_MISSING", 0)),
        "stale": int(final_counts.get("CACHE_STALE", 0)),
        "prescan_eligible": current,
        "market_data_coverage_pct": round(current / max(len(normalized), 1), 4),
        "provider_counts": dict(provider_counts),
        "provider_order": ["yahoo", "finnhub", "stooq"],
        "max_workers": max_workers,
        "failure_reasons": failures,
        "data_quality_warnings": data_quality_warnings,
        "results": results,
        "duration_seconds": round(time.monotonic() - started, 3),
        "last_backfill": datetime.now(timezone.utc).isoformat(),
    }


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
        expected_trading_days = max(180, int(years * 252 * 0.78))
        if min_date and max_date and count >= expected_trading_days:
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
    years = max(1, min(15, int(years)))
    history_days = 365 * years + 40
    min_history_bars = max(180, int(years * 252 * 0.78))
    report = repair_daily_bar_cache(
        db,
        symbols,
        history_days=history_days,
        min_history_bars=min_history_bars,
        max_workers=DEFAULT_PLANNING_CONFIG.sp500_market_data_max_workers,
        commit_every=commit_every,
        incremental_overlap_days=DEFAULT_PLANNING_CONFIG.sp500_market_data_incremental_overlap_days,
        refresh=refresh,
    )
    results = list(report.get("results") or [])
    no_data = sum(item.get("status") == "no_data" for item in results)
    return {
        "total": len(symbols),
        "updated": int(report.get("fetch_success") or 0),
        "skipped_cached": int(report.get("cache_hits") or 0),
        "no_data": no_data,
        "failed": int(report.get("fetch_failed") or 0),
        "results": results,
        "coverage": {key: value for key, value in report.items() if key not in {"results", "failure_reasons"}},
    }
