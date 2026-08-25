"""Canonical setup-family scoring and policy helpers.

The functions in this module are deliberately provider- and route-agnostic so
the prescan, planner, ranking, and live monitor share one setup vocabulary.
"""

from __future__ import annotations

from typing import Any, Mapping


HEALTHY_PULLBACK = "healthy_pullback"
MOMENTUM_CONTINUATION = "momentum_continuation"
BREAKOUT_RETEST = "breakout_retest"
BASE_BREAKOUT = "base_breakout"
DEEP_PULLBACK = "deep_pullback"
REVERSAL_ATTEMPT = "reversal_attempt"

SETUP_FAMILIES = (
    HEALTHY_PULLBACK,
    MOMENTUM_CONTINUATION,
    BREAKOUT_RETEST,
    BASE_BREAKOUT,
    DEEP_PULLBACK,
    REVERSAL_ATTEMPT,
)


_ALIASES = {
    "constructive_pullback": HEALTHY_PULLBACK,
    "pullback_in_uptrend": HEALTHY_PULLBACK,
    "strong_continuation_pullback": HEALTHY_PULLBACK,
    "controlled_momentum_continuation": MOMENTUM_CONTINUATION,
    "continuation_breakout": MOMENTUM_CONTINUATION,
    "post_breakout_retest": BREAKOUT_RETEST,
    "breakout": BASE_BREAKOUT,
    "base_building": BASE_BREAKOUT,
    "repair_after_breakdown": REVERSAL_ATTEMPT,
    "range_rebound": REVERSAL_ATTEMPT,
    "deep_rebound_attempt": REVERSAL_ATTEMPT,
    "trend_damage": REVERSAL_ATTEMPT,
    "structural_breakdown": REVERSAL_ATTEMPT,
}


def _clip(value: float) -> float:
    return round(max(0.0, min(10.0, float(value))), 4)


def normalize_setup_family(value: object, default: str | None = None) -> str | None:
    """Return a canonical family while accepting legacy setup labels."""

    normalized = str(value or "").strip().lower().replace(" ", "_")
    if normalized in SETUP_FAMILIES:
        return normalized
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    return default


def score_setup_families(
    components: Mapping[str, float],
    *,
    weights_by_family: Mapping[str, Mapping[str, float]],
    minimum_score: float = 0.0,
) -> dict:
    """Score all setup lanes independently from shared normalized evidence."""

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for family in SETUP_FAMILIES:
        weights = dict(weights_by_family.get(family) or {})
        total_weight = sum(max(float(weight), 0.0) for weight in weights.values())
        weighted = 0.0
        family_contributions: dict[str, float] = {}
        for name, weight in weights.items():
            score = _clip(float(components.get(name, 5.0)))
            contribution = score * max(float(weight), 0.0)
            weighted += contribution
            family_contributions[name] = round(contribution, 4)
        score = _clip(weighted / max(total_weight, 1e-9))
        scores[family] = score
        contributions[family] = family_contributions

    ordered = sorted(scores, key=lambda family: (scores[family], -SETUP_FAMILIES.index(family)), reverse=True)
    primary = ordered[0] if ordered and scores[ordered[0]] >= minimum_score else None
    alternatives = [
        {"setup_family": family, "score": scores[family]}
        for family in ordered[1:]
        if scores[family] >= minimum_score
    ]
    return {
        "setup_family": primary,
        "setup_lane_scores": scores,
        "setup_lane_contributions": contributions,
        "alternative_setup_families": alternatives,
    }


def classify_setup_family(
    *,
    structure_state: str | None,
    trend_state: str | None,
    setup_type: str | None = None,
    prescan_family: str | None = None,
    breakout_level: float | None = None,
    retest_zone: dict | None = None,
    consolidation_range: dict | None = None,
) -> str:
    """Resolve one planner family, allowing rich structure to override prescan."""

    structure = str(structure_state or "").lower()
    trend = str(trend_state or "").lower()
    if structure in {"structural_breakdown", "trend_damage", "reversal_attempt"}:
        return REVERSAL_ATTEMPT
    if structure == "deep_pullback":
        return DEEP_PULLBACK
    if retest_zone and (breakout_level is not None or structure == "breakout"):
        return BREAKOUT_RETEST
    if structure == "base_building" or (consolidation_range and structure == "breakout"):
        return BASE_BREAKOUT
    if structure == "healthy_pullback" or trend == "pullback_in_uptrend":
        return HEALTHY_PULLBACK
    if structure in {"breakout", "extended"} or trend == "uptrend":
        hinted = normalize_setup_family(prescan_family) or normalize_setup_family(setup_type)
        if hinted in {BREAKOUT_RETEST, BASE_BREAKOUT, HEALTHY_PULLBACK}:
            return hinted
        return MOMENTUM_CONTINUATION
    return (
        normalize_setup_family(prescan_family)
        or normalize_setup_family(setup_type)
        or REVERSAL_ATTEMPT
    )


def family_policy(setup_family: str | None) -> dict:
    """Return machine-readable confirmation, risk, and target policy metadata."""

    family = normalize_setup_family(setup_family, REVERSAL_ATTEMPT)
    policies = {
        HEALTHY_PULLBACK: {
            "confirmation_style": "support_hold_or_higher_low_reclaim",
            "stop_style": "below_pullback_swing_or_support",
            "target_style": "prior_high_then_runner",
            "requires_strong_volume": False,
            "runner_eligible": True,
        },
        MOMENTUM_CONTINUATION: {
            "confirmation_style": "short_consolidation_break",
            "stop_style": "below_consolidation_or_higher_low",
            "target_style": "prior_high_then_trend_extension",
            "requires_strong_volume": True,
            "runner_eligible": True,
        },
        BREAKOUT_RETEST: {
            "confirmation_style": "retest_hold_and_reclaim",
            "stop_style": "below_failed_retest_structure",
            "target_style": "next_resistance_then_runner",
            "requires_strong_volume": False,
            "runner_eligible": True,
        },
        BASE_BREAKOUT: {
            "confirmation_style": "base_resistance_break",
            "stop_style": "below_base_support_or_higher_low",
            "target_style": "measured_move_with_resistance_checks",
            "requires_strong_volume": True,
            "runner_eligible": True,
        },
        DEEP_PULLBACK: {
            "confirmation_style": "support_base_and_local_reclaim",
            "stop_style": "below_major_support_invalidation",
            "target_style": "nearest_recovery_resistance",
            "requires_strong_volume": True,
            "runner_eligible": False,
        },
        REVERSAL_ATTEMPT: {
            "confirmation_style": "base_reclaim_and_broader_stabilization",
            "stop_style": "below_reversal_base_invalidation",
            "target_style": "conservative_structural_repair",
            "requires_strong_volume": True,
            "runner_eligible": False,
        },
    }
    return {"setup_family": family, **policies[family]}


def build_runner_plan(
    *,
    setup_family: str | None,
    tp1: float,
    extension_target: float | None,
    config: Any,
) -> dict:
    """Build a manual partial-profit/runner recommendation for continuation lanes."""

    policy = family_policy(setup_family)
    eligible = bool(policy["runner_eligible"])
    return {
        "runner_eligible": eligible,
        "tp1_partial_profit_min_pct": float(config.continuation_tp1_partial_min_pct) if eligible else None,
        "tp1_partial_profit_max_pct": float(config.continuation_tp1_partial_max_pct) if eligible else None,
        "runner_activation_level": round(float(tp1), 6) if eligible else None,
        "runner_trailing_methods": list(config.continuation_runner_trailing_methods) if eligible else [],
        "extension_target": round(float(extension_target), 6) if eligible and extension_target is not None else None,
        "runner_state": "awaiting_tp1_breakout" if eligible else "not_applicable",
        "manual_execution_only": True,
    }


def evaluate_runner_state(
    *,
    setup_family: str | None,
    tp1: float | None,
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_price: float | None = None,
    relative_volume: float | None = None,
) -> dict:
    """Distinguish TP1 rejection from a confirmed continuation breakout."""

    policy = family_policy(setup_family)
    if not policy["runner_eligible"] or tp1 is None:
        return {"runner_state": "not_applicable", "tp1_reached": False, "breakout_rejected": False, "breakout_confirmed": False}
    target = float(tp1)
    close_value = float(close)
    high_value = float(high if high is not None else close_value)
    low_value = float(low if low is not None else close_value)
    open_value = float(open_price if open_price is not None else close_value)
    rvol = float(relative_volume or 0.0)
    reached = high_value >= target
    candle_range = max(high_value - low_value, 1e-9)
    upper_wick = high_value - max(open_value, close_value)
    rejected = bool(reached and close_value < target and upper_wick / candle_range >= 0.35 and rvol >= 1.15)
    confirmed = bool(close_value > target and rvol >= 1.1)
    if rejected:
        state = "tp1_reached_breakout_rejected"
    elif confirmed:
        state = "trend_extension_runner"
    elif reached:
        state = "tp1_reached_unresolved"
    else:
        state = "awaiting_tp1_breakout"
    return {
        "runner_state": state,
        "tp1_reached": reached,
        "breakout_rejected": rejected,
        "breakout_confirmed": confirmed,
    }


__all__ = [
    "BASE_BREAKOUT",
    "BREAKOUT_RETEST",
    "DEEP_PULLBACK",
    "HEALTHY_PULLBACK",
    "MOMENTUM_CONTINUATION",
    "REVERSAL_ATTEMPT",
    "SETUP_FAMILIES",
    "build_runner_plan",
    "classify_setup_family",
    "evaluate_runner_state",
    "family_policy",
    "normalize_setup_family",
    "score_setup_families",
]
