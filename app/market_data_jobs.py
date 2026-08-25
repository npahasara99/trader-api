"""Single-process background orchestration for broad daily-bar cache repair."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import uuid

from .config import DEFAULT_PLANNING_CONFIG
from .db import SessionLocal
from .market_data import repair_daily_bar_cache, resolve_expected_market_date


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sp500-cache-job")
_LOCK = threading.Lock()
_FUTURE: Future | None = None
_STATE: dict = {
    "state": "idle",
    "job_id": None,
    "phase": None,
    "requested_symbols": 0,
    "completed_symbols": 0,
    "updated": 0,
    "failed": 0,
    "fetch_attempted": 0,
    "fetch_success": 0,
    "fetch_failed": 0,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "report": None,
}


def _snapshot() -> dict:
    with _LOCK:
        return dict(_STATE)


def get_sp500_cache_job_status() -> dict:
    """Return a JSON-serializable snapshot of the process-local repair job."""

    return _snapshot()


def _set_state(**updates) -> None:
    with _LOCK:
        _STATE.update(updates)


def _run_job(job_id: str, constituents: list[str], benchmarks: list[str]) -> None:
    config = DEFAULT_PLANNING_CONFIG
    _set_state(state="running", phase="benchmarks")
    print(
        "SP500 background daily-bar repair started: "
        f"job_id={job_id} constituents={len(constituents)} benchmarks={len(benchmarks)}"
    )
    try:
        with SessionLocal() as db:
            benchmark_report = repair_daily_bar_cache(
                db,
                benchmarks,
                history_days=config.sp500_market_data_history_days,
                min_history_bars=config.sp500_market_data_min_history_bars,
                max_workers=config.sp500_market_data_max_workers,
                commit_every=config.sp500_market_data_commit_every,
                incremental_overlap_days=config.sp500_market_data_incremental_overlap_days,
            )
            expected_date = resolve_expected_market_date(
                db,
                benchmark_symbols=tuple(config.benchmark_symbols),
            )

            def progress(payload: dict) -> None:
                _set_state(
                    state="running",
                    phase="constituents",
                    completed_symbols=int(payload.get("completed") or 0),
                    updated=int(payload.get("updated") or 0),
                    failed=int(payload.get("failed") or 0),
                    fetch_success=int(payload.get("updated") or 0),
                    fetch_failed=int(payload.get("failed") or 0),
                )

            _set_state(phase="constituents")
            constituent_report = repair_daily_bar_cache(
                db,
                constituents,
                history_days=config.sp500_market_data_history_days,
                min_history_bars=config.sp500_market_data_min_history_bars,
                expected_date=expected_date,
                max_workers=config.sp500_market_data_max_workers,
                commit_every=config.sp500_market_data_commit_every,
                incremental_overlap_days=config.sp500_market_data_incremental_overlap_days,
                progress_callback=progress,
            )
        compact_report = {
            key: value
            for key, value in constituent_report.items()
            if key not in {"results"}
        }
        compact_report["benchmark_repair"] = {
            key: value
            for key, value in benchmark_report.items()
            if key not in {"results"}
        }
        _set_state(
            state="completed",
            phase="completed",
            completed_symbols=len(constituents),
            updated=int(constituent_report.get("fetch_success") or 0),
            failed=int(constituent_report.get("fetch_failed") or 0),
            fetch_attempted=int(constituent_report.get("fetch_attempted") or 0),
            fetch_success=int(constituent_report.get("fetch_success") or 0),
            fetch_failed=int(constituent_report.get("fetch_failed") or 0),
            finished_at=datetime.now(timezone.utc).isoformat(),
            report=compact_report,
        )
        print(
            "SP500 background daily-bar repair completed: "
            f"job_id={job_id} attempted={constituent_report.get('fetch_attempted', 0)} "
            f"success={constituent_report.get('fetch_success', 0)} "
            f"failed={constituent_report.get('fetch_failed', 0)}"
        )
    except Exception as exc:
        _set_state(
            state="failed",
            phase="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            last_error=f"{type(exc).__name__}: {exc}",
        )
        print(
            "SP500 background daily-bar repair failed: "
            f"job_id={job_id} error={type(exc).__name__}: {exc}"
        )


def schedule_sp500_cache_repair(constituents: list[str], benchmarks: list[str]) -> dict:
    """Start one repair job, or return the already-running job unchanged."""

    global _FUTURE
    normalized_constituents = sorted({str(symbol).strip().upper() for symbol in constituents if str(symbol).strip()})
    normalized_benchmarks = sorted({str(symbol).strip().upper() for symbol in benchmarks if str(symbol).strip()})
    with _LOCK:
        if _FUTURE is not None and not _FUTURE.done():
            return dict(_STATE)
        job_id = str(uuid.uuid4())
        _STATE.update(
            {
                "state": "queued",
                "job_id": job_id,
                "phase": "queued",
                "requested_symbols": len(normalized_constituents),
                "completed_symbols": 0,
                "updated": 0,
                "failed": 0,
                "fetch_attempted": len(normalized_constituents),
                "fetch_success": 0,
                "fetch_failed": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "last_error": None,
                "report": None,
            }
        )
        _FUTURE = _EXECUTOR.submit(_run_job, job_id, normalized_constituents, normalized_benchmarks)
        return dict(_STATE)


__all__ = ["get_sp500_cache_job_status", "schedule_sp500_cache_repair"]
