"""Deterministic historical statistics and transparent similarity weighting."""

from __future__ import annotations

from collections import Counter
from typing import Any


def evidence_strength(sample_size: int) -> str:
    if sample_size < 8:
        return "INSUFFICIENT"
    if sample_size < 20:
        return "LOW"
    if sample_size < 50:
        return "MODERATE"
    return "STRONG"


def hierarchical_weights(*, ticker_samples: int, setup_samples: int, sector_samples: int) -> dict[str, float]:
    """Shrink sparse stock evidence toward setup, sector, and global priors."""
    ticker = ticker_samples / (ticker_samples + 20.0)
    remaining = 1.0 - ticker
    setup = remaining * (setup_samples / (setup_samples + 30.0))
    remaining -= setup
    sector = remaining * (sector_samples / (sector_samples + 40.0))
    global_weight = 1.0 - ticker - setup - sector
    return {key: round(value, 4) for key, value in {"ticker": ticker, "setup": setup, "sector": sector, "global": global_weight}.items()}


def aggregate_attempts(attempts: list[dict], trades: list[dict] | None = None) -> dict[str, Any]:
    trades = trades or []
    outcomes = Counter(str(row.get("outcome") or "UNKNOWN") for row in attempts)
    completed = sum(outcomes.values())
    rejected = sum(value for key, value in outcomes.items() if "REJECT" in key)
    approved = sum(value for key, value in outcomes.items() if key in {"APPROVED", "STRONGLY_CONFIRMED"})
    r_values = [float(row["r_multiple"]) for row in trades if row.get("r_multiple") is not None]
    return {
        "observation_count": completed,
        "attempt_outcomes": dict(outcomes),
        "false_breakout_rate": None if not completed else round(rejected / completed, 4),
        "approval_rate": None if not completed else round(approved / completed, 4),
        "executed_trade_count": len(trades),
        "average_r_multiple": None if not r_values else round(sum(r_values) / len(r_values), 4),
        "evidence_strength": evidence_strength(completed),
    }


def similar_case_score(current: dict, candidate: dict) -> dict[str, Any]:
    fields = {
        "ticker": 2.0,
        "setup_type": 2.0,
        "sector": 1.0,
        "market_regime": 1.0,
        "confirmation_method": 1.5,
        "attempt_number": 0.5,
    }
    matched: list[str] = []
    possible = 0.0
    earned = 0.0
    for field, weight in fields.items():
        if current.get(field) is None:
            continue
        possible += weight
        if current.get(field) == candidate.get(field):
            earned += weight
            matched.append(field)
    return {
        "similarity_score": round(earned / possible, 4) if possible else 0.0,
        "matched_features": matched,
    }
