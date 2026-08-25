"""Sector-aware discovery and transparent scan diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable, TypeVar

from .setup_archetypes import SETUP_FAMILIES, normalize_setup_family


T = TypeVar("T")


MAJOR_SP500_SECTORS = (
    "Information Technology",
    "Financials",
    "Industrials",
    "Health Care",
    "Consumer Discretionary",
    "Consumer Staples",
    "Communication Services",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
)


def _ticker(item: object) -> str:
    if isinstance(item, str):
        return item.upper()
    if isinstance(item, dict):
        return str(item.get("ticker") or "").upper()
    return str(getattr(item, "ticker", "") or "").upper()


def _sector(ticker: str, metadata_by_ticker: dict[str, dict]) -> str:
    return str((metadata_by_ticker.get(ticker) or {}).get("sector") or "Unknown")


def sector_counts(items: Iterable[object], metadata_by_ticker: dict[str, dict]) -> dict[str, int]:
    """Count tickers by sector with stable inclusion of the major sectors."""

    counts = Counter()
    for item in items:
        ticker = _ticker(item)
        if ticker:
            counts[_sector(ticker, metadata_by_ticker)] += 1
    ordered = {sector: int(counts.pop(sector, 0)) for sector in MAJOR_SP500_SECTORS}
    ordered.update({sector: int(count) for sector, count in sorted(counts.items())})
    return ordered


def setup_family_counts(items: Iterable[object]) -> dict[str, int]:
    """Count canonical setup interpretations at any scanner stage."""

    counts = Counter()
    for item in items:
        raw = item.get("setup_family") if isinstance(item, dict) else getattr(item, "setup_family", None)
        family = normalize_setup_family(raw)
        if family:
            counts[family] += 1
    return {family: int(counts.get(family, 0)) for family in SETUP_FAMILIES}


def build_sector_aware_candidate_order(
    ranked_candidates: list[dict],
    *,
    metadata_by_ticker: dict[str, dict],
    initial_limit: int,
    min_per_sector: int,
) -> list[dict]:
    """Reserve initial discovery access by sector, then preserve global rank.

    This changes only which candidates receive expensive analysis. It does not
    add a sector bonus to final raw setup scores or force leaderboard diversity.
    """

    if not ranked_candidates:
        return []
    initial_limit = max(1, min(int(initial_limit), len(ranked_candidates)))
    min_per_sector = max(0, int(min_per_sector))
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for candidate in ranked_candidates:
        ticker = _ticker(candidate)
        by_sector[_sector(ticker, metadata_by_ticker)].append(candidate)

    selected: list[dict] = []
    selected_tickers: set[str] = set()
    # Round-robin allocation prevents the first alphabetic sector consuming all
    # reserved slots when the requested initial limit is small.
    for depth in range(min_per_sector):
        for sector in sorted(by_sector):
            candidates = by_sector[sector]
            if depth >= len(candidates) or len(selected) >= initial_limit:
                continue
            candidate = candidates[depth]
            ticker = _ticker(candidate)
            if ticker and ticker not in selected_tickers:
                selected.append(candidate)
                selected_tickers.add(ticker)

    for candidate in ranked_candidates:
        if len(selected) >= initial_limit:
            break
        ticker = _ticker(candidate)
        if ticker and ticker not in selected_tickers:
            selected.append(candidate)
            selected_tickers.add(ticker)

    # Expansion preserves the original global pre-scan order.
    selected.extend(candidate for candidate in ranked_candidates if _ticker(candidate) not in selected_tickers)
    return selected


def build_multilane_candidate_order(
    ranked_candidates: list[dict],
    *,
    metadata_by_ticker: dict[str, dict],
    initial_limit: int,
    min_per_sector: int,
    minimum_by_family: dict[str, int],
    minimum_family_score: float,
) -> list[dict]:
    """Reserve discovery capacity by family, then sector, without final-score bonuses."""

    if not ranked_candidates:
        return []
    initial_limit = max(1, min(int(initial_limit), len(ranked_candidates)))
    by_family: dict[str, list[dict]] = {family: [] for family in SETUP_FAMILIES}
    for candidate in ranked_candidates:
        lane_scores = candidate.get("setup_lane_scores") or {}
        for family in SETUP_FAMILIES:
            if float(lane_scores.get(family) or 0.0) >= float(minimum_family_score):
                by_family[family].append(candidate)
    for family in SETUP_FAMILIES:
        by_family[family].sort(
            key=lambda item: (
                float((item.get("setup_lane_scores") or {}).get(family) or 0.0),
                float(item.get("pre_scan_score") or 0.0),
            ),
            reverse=True,
        )

    selected: list[dict] = []
    selected_tickers: set[str] = set()
    max_reserved_depth = max((int(minimum_by_family.get(family, 0)) for family in SETUP_FAMILIES), default=0)
    for depth in range(max_reserved_depth):
        for family in SETUP_FAMILIES:
            if depth >= int(minimum_by_family.get(family, 0)) or len(selected) >= initial_limit:
                continue
            choices = by_family[family]
            candidate = next((item for item in choices if _ticker(item) not in selected_tickers), None)
            if candidate is not None:
                selected.append(candidate)
                selected_tickers.add(_ticker(candidate))

    sector_order = build_sector_aware_candidate_order(
        ranked_candidates,
        metadata_by_ticker=metadata_by_ticker,
        initial_limit=initial_limit,
        min_per_sector=min_per_sector,
    )
    for candidate in sector_order:
        if len(selected) >= initial_limit:
            break
        ticker = _ticker(candidate)
        if ticker and ticker not in selected_tickers:
            selected.append(candidate)
            selected_tickers.add(ticker)

    # Expansion round-robins the remaining lanes so a failed first batch does
    # not simply inspect more candidates from the dominant family.
    family_offsets = {family: 0 for family in SETUP_FAMILIES}
    while len(selected_tickers) < len(ranked_candidates):
        added = False
        for family in SETUP_FAMILIES:
            choices = by_family[family]
            offset = family_offsets[family]
            while offset < len(choices) and _ticker(choices[offset]) in selected_tickers:
                offset += 1
            family_offsets[family] = offset + 1
            if offset < len(choices):
                candidate = choices[offset]
                selected.append(candidate)
                selected_tickers.add(_ticker(candidate))
                added = True
        if not added:
            break

    selected.extend(candidate for candidate in ranked_candidates if _ticker(candidate) not in selected_tickers)
    return selected


def validate_sp500_universe(
    *,
    universe_size: int,
    sector_filter: str | None,
    industry_filter: str | None,
    minimum_broad_size: int,
) -> dict:
    filtered = bool((sector_filter or "").strip() or (industry_filter or "").strip())
    if filtered:
        return {
            "status": "filtered_scope",
            "valid": True,
            "expected_minimum": None,
            "warning": None,
        }
    valid = int(universe_size) >= int(minimum_broad_size)
    return {
        "status": "valid" if valid else "UNIVERSE_VALIDATION_FAILED",
        "valid": valid,
        "expected_minimum": int(minimum_broad_size),
        "warning": None if valid else (
            f"Expected broad SP500 universe but received only {int(universe_size)} symbols."
        ),
    }


def classify_search_exhaustiveness(
    *,
    analyzed: int,
    viable: int,
    initial_limit: int,
    maximum_limit: int,
) -> str:
    if viable <= 0 or analyzed >= viable:
        return "exhaustive"
    ratio = analyzed / max(viable, 1)
    if ratio >= 0.8 or analyzed >= maximum_limit:
        return "near_exhaustive"
    if analyzed > initial_limit:
        return "expanded"
    return "partial"


def classify_best_setup_quality(candidates: list[dict]) -> str:
    if not candidates:
        return "no_quality_setups"
    grades = [str(item.get("grade") or "F") for item in candidates]
    a_count = sum(grade in {"A-", "A", "A+"} for grade in grades)
    top_score = max(float(item.get("raw_setup_score") or 0.0) for item in candidates)
    if a_count >= 3:
        return "strong_scan"
    if a_count >= 1:
        return "normal_scan"
    if top_score >= 6.5:
        return "weak_scan"
    return "no_quality_setups"


def run_adaptive_batches(
    candidate_order: list[dict],
    *,
    initial_limit: int,
    batch_size: int,
    maximum_limit: int,
    target_actionable: int,
    adaptive: bool,
    analyze_batch: Callable[[list[dict]], list[T]],
    count_actionable: Callable[[list[T]], int],
) -> tuple[list[T], list[dict]]:
    """Analyze deterministic batches, expanding breadth without relaxing quality."""

    cap = min(max(int(maximum_limit), 1), len(candidate_order))
    initial = min(max(int(initial_limit), 1), cap)
    batch_size = max(int(batch_size), 1)
    target_actionable = max(int(target_actionable), 1)
    rows: list[T] = []
    history: list[dict] = []
    cursor = 0
    while cursor < cap:
        next_cursor = initial if cursor == 0 else min(cursor + batch_size, cap)
        batch = candidate_order[cursor:next_cursor]
        if not batch:
            break
        rows.extend(analyze_batch(batch))
        cursor = next_cursor
        actionable = int(count_actionable(rows))
        history.append(
            {
                "batch": len(history) + 1,
                "batch_size": len(batch),
                "deep_analyzed": cursor,
                "actionable_count": actionable,
                "batch_setup_family_counts": setup_family_counts(batch),
                "cumulative_setup_family_counts": setup_family_counts(rows),
            }
        )
        if actionable >= target_actionable or not adaptive:
            break
    return rows, history


__all__ = [
    "MAJOR_SP500_SECTORS",
    "build_sector_aware_candidate_order",
    "build_multilane_candidate_order",
    "classify_best_setup_quality",
    "classify_search_exhaustiveness",
    "run_adaptive_batches",
    "sector_counts",
    "setup_family_counts",
    "validate_sp500_universe",
]
