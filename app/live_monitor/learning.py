"""Deterministic historical statistics, similarity, and bounded adaptation.

Decision-time features and later outcomes stay separate. Historical evidence
changes interpretation weights only; it never rewrites OHLCV or chart facts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


FORMULA_VERSION = "historical-memory-v1"

QUALITY_FLAG_WEIGHTS = {
    "STALE_DATA": 0.25,
    "MISSING_VOLUME": 0.60,
    "EXTENDED_HOURS": 0.75,
    "PARTIAL_HISTORY": 0.50,
    "MANUAL_LEVEL_OVERRIDE": 0.70,
    "NO_EXECUTION_CONFIRMATION": 0.80,
}


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def evidence_strength(
    sample_size: float,
    thresholds: tuple[float, float, float, float] = (8.0, 15.0, 30.0, 60.0),
) -> str:
    """Label effective observations, not merely raw row count."""
    weak, emerging, moderate, strong = thresholds
    if sample_size < weak:
        return "INSUFFICIENT"
    if sample_size < emerging:
        return "WEAK"
    if sample_size < moderate:
        return "EMERGING"
    if sample_size < strong:
        return "MODERATE"
    return "STRONG"


def sample_reliability(weighted_sample_size: float, prior_strength: float = 20.0) -> float:
    """Empirical-Bayes-style reliability n_eff / (n_eff + prior strength)."""
    sample = max(0.0, float(weighted_sample_size))
    prior = max(1e-9, float(prior_strength))
    return round(sample / (sample + prior), 6)


def recency_weight(
    occurred_at: Any,
    *,
    as_of: datetime | None = None,
    half_life_days: float = 120.0,
) -> float:
    timestamp = _as_datetime(occurred_at)
    if timestamp is None:
        return 0.75
    reference = as_of or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (reference - timestamp).total_seconds() / 86400.0)
    return max(0.10, 0.5 ** (age_days / max(1.0, half_life_days)))


def data_quality_weight(flags: Iterable[str] | None) -> float:
    weight = 1.0
    for flag in set(str(item).upper() for item in (flags or [])):
        weight *= QUALITY_FLAG_WEIGHTS.get(flag, 1.0)
    return max(0.05, min(1.0, weight))


def observation_weight(
    row: dict[str, Any],
    *,
    as_of: datetime | None = None,
    half_life_days: float = 120.0,
) -> float:
    return recency_weight(
        row.get("occurred_at") or row.get("created_at") or row.get("trading_date"),
        as_of=as_of,
        half_life_days=half_life_days,
    ) * data_quality_weight(row.get("data_quality_flags") or [])


def hierarchical_weights(
    *,
    ticker_samples: float,
    setup_samples: float,
    sector_samples: float,
    regime_samples: float = 0.0,
    ticker_prior: float = 20.0,
    setup_prior: float = 30.0,
    sector_prior: float = 40.0,
    regime_prior: float = 40.0,
) -> dict[str, float]:
    """Sequentially shrink sparse specific evidence toward broader priors."""
    ticker = sample_reliability(ticker_samples, ticker_prior)
    remaining = 1.0 - ticker
    setup = remaining * sample_reliability(setup_samples, setup_prior)
    remaining -= setup
    sector = remaining * sample_reliability(sector_samples, sector_prior)
    remaining -= sector
    regime = remaining * sample_reliability(regime_samples, regime_prior)
    global_weight = max(0.0, 1.0 - ticker - setup - sector - regime)
    return {
        key: round(value, 4)
        for key, value in {
            "ticker": ticker,
            "setup": setup,
            "sector": sector,
            "regime": regime,
            "global": global_weight,
        }.items()
    }


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in pairs)
    if denominator <= 0:
        return None
    return round(sum(value * weight for value, weight in pairs) / denominator, 4)


def _success(row: dict[str, Any]) -> bool:
    if row.get("r_multiple") is not None:
        return float(row["r_multiple"]) > 0
    outcome = str(row.get("outcome") or "").upper()
    return any(token in outcome for token in ("APPROVED", "CONFIRMED", "TP", "WIN", "PROFIT"))


def _group_stats(
    rows: list[dict[str, Any]],
    field: str,
    thresholds: tuple[float, float, float, float],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "UNKNOWN")].append(row)
    output: dict[str, Any] = {}
    for key, items in grouped.items():
        weighted_n = sum(float(item["_weight"]) for item in items)
        success_weight = sum(float(item["_weight"]) for item in items if _success(item))
        r_pairs = [
            (float(item["r_multiple"]), float(item["_weight"]))
            for item in items
            if item.get("r_multiple") is not None
        ]
        output[key] = {
            "sample_size": len(items),
            "weighted_sample_size": round(weighted_n, 4),
            "success_rate": None if weighted_n <= 0 else round(success_weight / weighted_n, 4),
            "expectancy_r": _weighted_mean(r_pairs),
            "evidence_strength": evidence_strength(weighted_n, thresholds),
        }
    return output


def aggregate_observations(
    observations: list[dict[str, Any]],
    *,
    as_of: datetime | None = None,
    half_life_days: float = 120.0,
    prior_strength: float = 20.0,
    evidence_thresholds: tuple[float, float, float, float] = (8.0, 15.0, 30.0, 60.0),
) -> dict[str, Any]:
    """Aggregate completed decisions/outcomes with recency and quality weights."""
    rows: list[dict[str, Any]] = []
    for source in observations:
        row = dict(source)
        row["_weight"] = observation_weight(row, as_of=as_of, half_life_days=half_life_days)
        rows.append(row)
    weighted_n = sum(float(row["_weight"]) for row in rows)
    outcomes = Counter(str(row.get("outcome") or "UNKNOWN") for row in rows)
    rejection_weight = sum(
        float(row["_weight"])
        for row in rows
        if "REJECT" in str(row.get("outcome") or "").upper()
        or "FALSE_BREAK" in str(row.get("outcome") or "").upper()
    )
    success_weight = sum(float(row["_weight"]) for row in rows if _success(row))
    numeric_fields = (
        "r_multiple", "mfe_atr", "mae_atr", "rvol_1m", "rvol_5m", "hold_days", "entry_distance_pct",
    )
    means = {
        field: _weighted_mean([
            (float(row[field]), float(row["_weight"]))
            for row in rows
            if row.get(field) is not None
        ])
        for field in numeric_fields
    }
    target_rates: dict[str, float | None] = {}
    for target in ("tp1_reached", "tp2_reached", "tp3_reached"):
        eligible = [row for row in rows if row.get(target) is not None]
        denominator = sum(float(row["_weight"]) for row in eligible)
        target_rates[target.replace("_reached", "_hit_rate")] = (
            None
            if denominator <= 0
            else round(sum(float(row["_weight"]) for row in eligible if row.get(target)) / denominator, 4)
        )
    return {
        "formula_version": FORMULA_VERSION,
        "observation_count": len(rows),
        "weighted_sample_size": round(weighted_n, 4),
        "reliability": sample_reliability(weighted_n, prior_strength),
        "evidence_strength": evidence_strength(weighted_n, evidence_thresholds),
        "attempt_outcomes": dict(outcomes),
        "false_breakout_rate": None if weighted_n <= 0 else round(rejection_weight / weighted_n, 4),
        "success_rate": None if weighted_n <= 0 else round(success_weight / weighted_n, 4),
        "expectancy_r": means["r_multiple"],
        "avg_mfe_atr": means["mfe_atr"],
        "avg_mae_atr": means["mae_atr"],
        "avg_rvol_1m": means["rvol_1m"],
        "avg_rvol_5m": means["rvol_5m"],
        "avg_hold_days": means["hold_days"],
        "average_entry_distance_from_trigger": means["entry_distance_pct"],
        **target_rates,
        "confirmation_method_stats": _group_stats(rows, "confirmation_method", evidence_thresholds),
        "attempt_number_stats": _group_stats(rows, "attempt_number", evidence_thresholds),
        "setup_type_stats": _group_stats(rows, "setup_type", evidence_thresholds),
        "market_regime_stats": _group_stats(rows, "market_regime", evidence_thresholds),
        "level_source_stats": _group_stats(rows, "level_source", evidence_thresholds),
        "session_bucket_stats": _group_stats(rows, "session_bucket", evidence_thresholds),
    }


def aggregate_attempts(attempts: list[dict], trades: list[dict] | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper used by older monitor code/tests."""
    trades = trades or []
    trade_by_attempt = {str(row.get("attempt_id")): row for row in trades if row.get("attempt_id")}
    observations: list[dict[str, Any]] = []
    for attempt in attempts:
        row = dict(attempt)
        trade = trade_by_attempt.get(str(row.get("attempt_id")))
        if trade:
            row.update({key: value for key, value in trade.items() if value is not None})
        observations.append(row)
    for trade in trades:
        if trade.get("attempt_id") is None:
            observations.append(dict(trade))
    stats = aggregate_observations(observations)
    stats["approval_rate"] = stats.pop("success_rate")
    stats["executed_trade_count"] = len(trades)
    stats["average_r_multiple"] = stats.get("expectancy_r")
    return stats


def similar_case_score(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    continuous_weights: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Explainable mixed categorical/continuous similarity in [0, 1]."""
    categorical = weights or {
        "ticker": 2.0,
        "broader_structure": 1.25,
        "setup_type": 2.0,
        "execution_structure": 1.5,
        "sector": 0.75,
        "market_regime": 1.0,
        "confirmation_method": 1.25,
        "attempt_number": 0.50,
        "qqq_condition": 0.50,
        "sector_condition": 0.50,
    }
    continuous = continuous_weights or {
        "atr_pct": (1.0, 0.03),
        "rsi": (0.75, 25.0),
        "distance_from_support_atr": (1.0, 2.0),
        "primary_trigger_distance_atr": (1.0, 2.5),
        "rvol_5m": (1.0, 1.5),
    }
    matched: list[str] = []
    contributions: dict[str, float] = {}
    possible = 0.0
    earned = 0.0
    for field, weight in categorical.items():
        if current.get(field) is None or candidate.get(field) is None:
            continue
        possible += weight
        contribution = weight if str(current[field]) == str(candidate[field]) else 0.0
        earned += contribution
        contributions[field] = round(contribution, 4)
        if contribution:
            matched.append(field)
    for field, (weight, scale) in continuous.items():
        if current.get(field) is None or candidate.get(field) is None:
            continue
        possible += weight
        difference = abs(float(current[field]) - float(candidate[field]))
        contribution = weight * max(0.0, 1.0 - difference / max(scale, 1e-9))
        earned += contribution
        contributions[field] = round(contribution, 4)
    return {
        "similarity_score": round(earned / possible, 4) if possible else 0.0,
        "matched_features": matched,
        "similarity_contributions": contributions,
    }


def derive_bounded_adjustments(profile: dict[str, Any], current: dict[str, Any], config: Any) -> list[dict[str, Any]]:
    """Produce explainable bounded changes only when evidence is meaningful."""
    statistics = profile.get("statistics") or {}
    weighted_n = float(statistics.get("weighted_sample_size") or 0.0)
    strength = str(profile.get("evidence_strength") or statistics.get("evidence_strength") or "INSUFFICIENT")
    reliability = float(statistics.get("reliability") or 0.0)
    if strength in {"INSUFFICIENT", "WEAK"} or reliability <= 0:
        return []

    adjustments: list[dict[str, Any]] = []
    false_breakout_rate = statistics.get("false_breakout_rate")
    if false_breakout_rate is not None:
        centered = (float(false_breakout_rate) - 0.35) * -2.0
        delta = max(
            -config.max_historical_score_adjustment,
            min(config.max_historical_score_adjustment, centered * reliability),
        )
        if abs(delta) >= 0.05:
            adjustments.append({
                "adjustment_type": "FALSE_BREAKOUT_PENALTY",
                "base_value": 0.0,
                "learned_value": round(delta, 4),
                "adjustment_value": round(delta, 4),
                "adjustment_strength": round(reliability, 4),
                "evidence_strength": strength,
                "sample_size": int(profile.get("observation_count") or 0),
                "weighted_sample_size": round(weighted_n, 4),
                "reason": "Historical false-breakout frequency changes actionability within the configured score bound.",
                "supporting_stats": {"false_breakout_rate": false_breakout_rate},
                "bounds": {
                    "minimum": -config.max_historical_score_adjustment,
                    "maximum": config.max_historical_score_adjustment,
                },
            })

    method_stats = statistics.get("confirmation_method_stats") or {}
    retest = next((value for key, value in method_stats.items() if "RETEST" in key.upper()), None)
    first_touch = next((value for key, value in method_stats.items() if "FIRST" in key.upper() or "5M_CLOSE" in key.upper()), None)
    retest_expectancy = None if not retest else retest.get("expectancy_r")
    first_expectancy = None if not first_touch else first_touch.get("expectancy_r")
    if retest_expectancy is not None and first_expectancy is not None:
        effect = float(retest_expectancy) - float(first_expectancy)
        if effect >= 0.25 and float(retest.get("weighted_sample_size") or 0) >= 8:
            delta = min(config.max_historical_score_adjustment, effect * reliability)
            adjustments.append({
                "adjustment_type": "CONFIRMATION_PREFERENCE",
                "base_value": 0.0,
                "learned_value": round(delta, 4),
                "adjustment_value": round(delta, 4),
                "adjustment_strength": round(reliability, 4),
                "evidence_strength": strength,
                "sample_size": int(profile.get("observation_count") or 0),
                "weighted_sample_size": round(weighted_n, 4),
                "reason": "Comparable break/retest confirmations have higher expectancy than first-touch confirmations.",
                "supporting_stats": {
                    "retest_expectancy_r": retest_expectancy,
                    "first_touch_expectancy_r": first_expectancy,
                },
                "bounds": {
                    "minimum": -config.max_historical_score_adjustment,
                    "maximum": config.max_historical_score_adjustment,
                },
                "preferred_confirmation": "BREAK_RETEST",
            })

    avg_mfe = statistics.get("avg_mfe_atr")
    current_tp1_atr = current.get("tp1_distance_atr")
    if avg_mfe is not None and current_tp1_atr is not None and float(current_tp1_atr) > float(avg_mfe) + 1.0:
        penalty = -min(
            config.max_target_expectation_adjustment_atr,
            (float(current_tp1_atr) - float(avg_mfe)) * reliability,
        )
        adjustments.append({
            "adjustment_type": "TARGET_REALISM",
            "base_value": float(current_tp1_atr),
            "learned_value": round(float(current_tp1_atr) + penalty, 4),
            "adjustment_value": round(penalty, 4),
            "adjustment_strength": round(reliability, 4),
            "evidence_strength": strength,
            "sample_size": int(profile.get("observation_count") or 0),
            "weighted_sample_size": round(weighted_n, 4),
            "reason": "Current TP1 is materially beyond historical MFE; current resistance still controls the target.",
            "supporting_stats": {"avg_mfe_atr": avg_mfe, "current_tp1_distance_atr": current_tp1_atr},
            "bounds": {"minimum": -config.max_target_expectation_adjustment_atr, "maximum": 0.0},
        })

    current_source = str(current.get("level_source") or "")
    source_stats = (statistics.get("level_source_stats") or {}).get(current_source)
    if source_stats and float(source_stats.get("weighted_sample_size") or 0.0) >= 8:
        source_expectancy = source_stats.get("expectancy_r")
        source_success = source_stats.get("success_rate")
        source_penalty = 0.0
        if source_expectancy is not None and float(source_expectancy) < 0:
            source_penalty = min(config.max_historical_score_adjustment, abs(float(source_expectancy)) * reliability)
        elif source_success is not None and float(source_success) < 0.35:
            source_penalty = min(config.max_historical_score_adjustment, (0.35 - float(source_success)) * reliability * 2.0)
        if source_penalty >= 0.05:
            adjustments.append({
                "adjustment_type": "LEVEL_SOURCE_CONFIDENCE",
                "base_value": 0.0,
                "learned_value": round(-source_penalty, 4),
                "adjustment_value": round(-source_penalty, 4),
                "adjustment_strength": round(reliability, 4),
                "evidence_strength": strength,
                "sample_size": int(source_stats.get("sample_size") or 0),
                "weighted_sample_size": round(float(source_stats.get("weighted_sample_size") or 0.0), 4),
                "reason": (
                    f"Historically resolved {current_source} levels have weak expectancy or success; "
                    "current chart evidence still controls and the score change is bounded."
                ),
                "supporting_stats": {
                    "level_source": current_source,
                    "expectancy_r": source_expectancy,
                    "success_rate": source_success,
                },
                "bounds": {
                    "minimum": -config.max_historical_score_adjustment,
                    "maximum": 0.0,
                },
            })
    return adjustments


def adjustment_breakdown(raw_score: float | None, adjustments: list[dict[str, Any]], max_total: float) -> dict[str, Any]:
    raw = float(raw_score or 0.0)
    score_deltas = [
        float(item.get("adjustment_value") or 0.0)
        for item in adjustments
        if item.get("adjustment_type") in {
            "FALSE_BREAKOUT_PENALTY", "CONFIRMATION_PREFERENCE", "LEVEL_SOURCE_CONFIDENCE",
        }
    ]
    total = max(-max_total, min(max_total, sum(score_deltas)))
    return {
        "raw_setup_score": round(raw, 4),
        "historical_adjustment_score": round(total, 4),
        "learned_actionability_score": round(max(0.0, min(10.0, raw + total)), 4),
        "trade_today_score": round(max(0.0, min(10.0, raw + total)), 4),
        "components": [
            {"type": item.get("adjustment_type"), "value": item.get("adjustment_value"), "reason": item.get("reason")}
            for item in adjustments
        ],
    }


__all__ = [
    "FORMULA_VERSION",
    "QUALITY_FLAG_WEIGHTS",
    "adjustment_breakdown",
    "aggregate_attempts",
    "aggregate_observations",
    "data_quality_weight",
    "derive_bounded_adjustments",
    "evidence_strength",
    "hierarchical_weights",
    "observation_weight",
    "recency_weight",
    "sample_reliability",
    "similar_case_score",
]
