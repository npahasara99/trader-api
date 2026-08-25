"""Constituent providers isolated from scanner and planner orchestration."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import time

import requests

from .config import DEFAULT_PLANNING_CONFIG, PlanningConfig


FALLBACK_PATH = Path(__file__).resolve().parent / "data" / "sp500_constituents.csv"
FALLBACK_AS_OF = "2026-08-17"


@dataclass(frozen=True)
class UniverseSnapshot:
    name: str
    constituents: tuple[dict, ...]
    as_of: str
    source: str
    fetched_at: datetime
    used_fallback: bool = False
    warning: str | None = None

    @property
    def tickers(self) -> list[str]:
        return [str(row["ticker"]) for row in self.constituents]

    @property
    def metadata_by_ticker(self) -> dict[str, dict]:
        return {str(row["ticker"]): dict(row) for row in self.constituents}


class _ConstituentTableParser(HTMLParser):
    """Parse Wikipedia's constituents table without an optional HTML dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "constituents":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag in {"th", "td"} and self.in_cell:
            value = " ".join("".join(self.current_cell).split())
            self.current_row.append(value)
            self.in_cell = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []
        elif tag == "table":
            self.in_table = False


def _normalize_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def _normalize_constituents(rows: list[dict]) -> tuple[dict, ...]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        source_ticker = str(row.get("ticker") or "").strip().upper()
        ticker = _normalize_ticker(source_ticker)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(
            {
                "ticker": ticker,
                "canonical_symbol": ticker,
                "provider_symbols": {
                    "yahoo": ticker,
                    "finnhub": (
                        ticker.rsplit("-", 1)[0] + "." + ticker.rsplit("-", 1)[1]
                        if re.fullmatch(r"[A-Z]+-[A-Z]", ticker) else source_ticker
                    ),
                    "stooq": (
                        ticker.rsplit("-", 1)[0] + "." + ticker.rsplit("-", 1)[1]
                        if re.fullmatch(r"[A-Z]+-[A-Z]", ticker) else source_ticker
                    ),
                },
                "company_name": str(row.get("company_name") or "").strip() or None,
                "sector": str(row.get("sector") or "").strip() or None,
                "industry": str(row.get("industry") or "").strip() or None,
            }
        )
    if len(normalized) < 450:
        raise ValueError(f"Constituent source returned only {len(normalized)} unique symbols")
    return tuple(normalized)


def _fetch_wikipedia(config: PlanningConfig) -> UniverseSnapshot:
    response = requests.get(
        config.sp500_universe_url,
        timeout=20,
        headers={"User-Agent": "trader-api/1.0 (S&P 500 universe refresh)"},
    )
    response.raise_for_status()
    parser = _ConstituentTableParser()
    parser.feed(response.text)
    if not parser.rows:
        raise ValueError("S&P 500 constituents table was not found")

    header = parser.rows[0]
    indexes = {name: header.index(name) for name in ("Symbol", "Security", "GICS Sector", "GICS Sub-Industry")}
    rows: list[dict] = []
    for values in parser.rows[1:]:
        if len(values) <= max(indexes.values()):
            continue
        rows.append(
            {
                "ticker": values[indexes["Symbol"]],
                "company_name": values[indexes["Security"]],
                "sector": values[indexes["GICS Sector"]],
                "industry": values[indexes["GICS Sub-Industry"]],
            }
        )
    now = datetime.now(timezone.utc)
    return UniverseSnapshot(
        name="SP500",
        constituents=_normalize_constituents(rows),
        as_of=now.date().isoformat(),
        source=config.sp500_universe_url,
        fetched_at=now,
    )


def _load_fallback(*, warning: str | None = None) -> UniverseSnapshot:
    with FALLBACK_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return UniverseSnapshot(
        name="SP500",
        constituents=_normalize_constituents(rows),
        as_of=FALLBACK_AS_OF,
        source=f"repository_fallback:{FALLBACK_PATH.name}",
        fetched_at=datetime.now(timezone.utc),
        used_fallback=True,
        warning=warning,
    )


_cached_snapshot: UniverseSnapshot | None = None
_cached_at_monotonic = 0.0


def get_sp500_snapshot(
    *,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
    force_refresh: bool = False,
) -> UniverseSnapshot:
    """Return a cached current universe, falling back safely when the source fails."""

    global _cached_snapshot, _cached_at_monotonic
    cache_valid = (
        _cached_snapshot is not None
        and not force_refresh
        and time.monotonic() - _cached_at_monotonic < config.sp500_universe_cache_seconds
    )
    if cache_valid:
        return _cached_snapshot
    try:
        snapshot = _fetch_wikipedia(config)
    except Exception as exc:
        snapshot = _load_fallback(warning=f"{type(exc).__name__}: {exc}")
    _cached_snapshot = snapshot
    _cached_at_monotonic = time.monotonic()
    return snapshot


def filter_constituents(
    snapshot: UniverseSnapshot,
    *,
    sector: str | None = None,
    industry: str | None = None,
) -> list[dict]:
    """Filter constituent metadata with case-insensitive substring matching."""

    sector_filter = " ".join((sector or "").lower().replace("_", " ").split())
    industry_filter = " ".join((industry or "").lower().replace("_", " ").split())
    rows: list[dict] = []
    for row in snapshot.constituents:
        row_sector = str(row.get("sector") or "").lower()
        row_industry = str(row.get("industry") or "").lower()
        if sector_filter and sector_filter not in row_sector and sector_filter not in row_industry:
            continue
        if industry_filter and industry_filter not in row_industry and industry_filter not in row_sector:
            continue
        rows.append(dict(row))
    return rows


def get_sp500_universe(
    top_n: int | None = None,
    *,
    sector: str | None = None,
    industry: str | None = None,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> tuple[list[str], UniverseSnapshot, dict[str, dict]]:
    snapshot = get_sp500_snapshot(config=config)
    rows = filter_constituents(snapshot, sector=sector, industry=industry)
    if top_n is not None:
        rows = rows[: max(1, min(int(top_n), len(rows)))]
    metadata = {str(row["ticker"]): row for row in rows}
    return [str(row["ticker"]) for row in rows], snapshot, metadata
