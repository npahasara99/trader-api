from datetime import datetime, timedelta, timezone
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.candidate_discovery import classify_search_exhaustiveness_with_coverage
from app.db import Base
from app.market_data import (
    last_completed_market_date,
    provider_symbol_for,
    repair_daily_bar_cache,
    validate_daily_bars,
)
from app.models import DailyBar
import app.market_data_jobs as market_data_jobs


def _session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _bars(symbol: str, end_date, count: int = 60, *, source: str = "test") -> list[dict]:
    return [
        {
            "symbol": symbol,
            "bar_date": end_date - timedelta(days=count - index - 1),
            "open": 99.5 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": 1_000_000.0,
            "adjusted_close": 100.0 + index,
            "source": source,
            "updated_at": datetime.now(timezone.utc),
        }
        for index in range(count)
    ]


def test_one_cached_symbol_repairs_remaining_502_and_second_run_is_cache_only():
    db = _session()
    expected = datetime(2026, 8, 21, tzinfo=timezone.utc).date()
    symbols = [f"S{index:03d}" for index in range(503)]
    db.add_all(DailyBar(**bar) for bar in _bars(symbols[0], expected))
    db.commit()
    attempted: list[str] = []
    lock = threading.Lock()

    def fetcher(symbol, _start, end):
        with lock:
            attempted.append(symbol)
        return _bars(symbol, end), "ok", "test", symbol

    first = repair_daily_bar_cache(
        db,
        symbols,
        history_days=320,
        min_history_bars=60,
        expected_date=expected,
        max_workers=4,
        fetcher=fetcher,
    )

    assert first["universe_count"] == 503
    assert first["cache_hits"] == 1
    assert first["fetch_attempted"] == 502
    assert first["fetch_success"] == 502
    assert first["current"] == 503
    assert len(attempted) == 502

    second = repair_daily_bar_cache(
        db,
        symbols,
        history_days=320,
        min_history_bars=60,
        expected_date=expected,
        fetcher=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected provider call")),
    )
    assert second["cache_hits"] == 503
    assert second["fetch_attempted"] == 0
    assert second["current"] == 503


def test_stale_symbol_fetches_only_incremental_range_with_overlap():
    db = _session()
    expected = datetime(2026, 8, 21, tzinfo=timezone.utc).date()
    cached_end = expected - timedelta(days=3)
    db.add_all(DailyBar(**bar) for bar in _bars("AMD", cached_end))
    db.commit()
    requested = {}

    def fetcher(symbol, start, end):
        requested.update({"symbol": symbol, "start": start, "end": end})
        return _bars(symbol, end, count=4), "ok", "test", symbol

    report = repair_daily_bar_cache(
        db,
        ["AMD"],
        min_history_bars=60,
        expected_date=expected,
        incremental_overlap_days=5,
        fetcher=fetcher,
    )

    assert requested["start"] == cached_end - timedelta(days=5)
    assert requested["end"] == expected
    assert report["fetch_attempted"] == 1
    assert report["current"] == 1


def test_provider_failure_is_isolated_and_normalized():
    db = _session()
    expected = datetime(2026, 8, 21, tzinfo=timezone.utc).date()

    def fetcher(symbol, _start, end):
        if symbol == "FAIL":
            return [], "yahoo_http_429", None, symbol
        return _bars(symbol, end), "ok", "test", symbol

    report = repair_daily_bar_cache(
        db,
        ["GOOD", "FAIL"],
        min_history_bars=60,
        expected_date=expected,
        fetcher=fetcher,
    )

    assert report["fetch_success"] == 1
    assert report["fetch_failed"] == 1
    assert report["current"] == 1
    assert report["failure_reasons"] == [
        {"ticker": "FAIL", "reason": "PROVIDER_RATE_LIMIT", "details": "yahoo_http_429"}
    ]


def test_malformed_ohlc_is_rejected_before_cache_write():
    bad = _bars("BAD", datetime(2026, 8, 21, tzinfo=timezone.utc).date(), count=4)
    bad[0]["high"] = bad[0]["low"] - 1.0
    normalized, validation = validate_daily_bars(bad, canonical_symbol="BAD")

    assert len(normalized) == 3
    assert validation["valid"] is False
    assert validation["malformed_rows"] == 1


def test_share_class_provider_mapping_is_generic():
    assert provider_symbol_for("BRK.B", "yahoo") == "BRK-B"
    assert provider_symbol_for("BRK-B", "finnhub") == "BRK.B"
    assert provider_symbol_for("BF-B", "stooq") == "BF.B"
    assert provider_symbol_for("AMD", "finnhub") == "AMD"


def test_market_freshness_uses_last_completed_session():
    saturday = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
    monday_before_open_et = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    assert last_completed_market_date(saturday).isoformat() == "2026-08-21"
    assert last_completed_market_date(monday_before_open_et).isoformat() == "2026-08-21"


def test_low_data_coverage_cannot_be_exhaustive():
    status = classify_search_exhaustiveness_with_coverage(
        analyzed=1,
        viable=1,
        initial_limit=30,
        maximum_limit=75,
        data_coverage_pct=1 / 503,
        minimum_data_coverage_pct=0.90,
    )
    assert status == "data_incomplete"

    complete_status = classify_search_exhaustiveness_with_coverage(
        analyzed=10,
        viable=10,
        initial_limit=4,
        maximum_limit=10,
        data_coverage_pct=0.98,
        minimum_data_coverage_pct=0.90,
    )
    assert complete_status == "exhaustive"


def test_background_repair_job_reports_progress_and_completion(monkeypatch):
    class FakeSessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    calls = []

    def fake_repair(_db, symbols, **kwargs):
        calls.append(list(symbols))
        callback = kwargs.get("progress_callback")
        if callback:
            callback({"completed": 2, "total": 2, "updated": 2, "failed": 0})
        return {
            "fetch_attempted": len(symbols),
            "fetch_success": len(symbols),
            "fetch_failed": 0,
            "current": len(symbols),
            "results": [],
        }

    monkeypatch.setattr(market_data_jobs, "SessionLocal", lambda: FakeSessionContext())
    monkeypatch.setattr(market_data_jobs, "repair_daily_bar_cache", fake_repair)
    monkeypatch.setattr(
        market_data_jobs,
        "resolve_expected_market_date",
        lambda *_args, **_kwargs: datetime(2026, 8, 21, tzinfo=timezone.utc).date(),
    )

    market_data_jobs._run_job("job-test", ["A", "B"], ["SPY"])
    status = market_data_jobs.get_sp500_cache_job_status()

    assert calls == [["SPY"], ["A", "B"]]
    assert status["state"] == "completed"
    assert status["completed_symbols"] == 2
    assert status["fetch_success"] == 2
    assert status["fetch_failed"] == 0


def test_background_repair_job_surfaces_failure(monkeypatch):
    class FakeSessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    def failed_repair(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(market_data_jobs, "SessionLocal", lambda: FakeSessionContext())
    monkeypatch.setattr(market_data_jobs, "repair_daily_bar_cache", failed_repair)

    market_data_jobs._run_job("job-failed", ["A"], ["SPY"])
    status = market_data_jobs.get_sp500_cache_job_status()

    assert status["state"] == "failed"
    assert status["phase"] == "failed"
    assert status["last_error"] == "RuntimeError: provider unavailable"
