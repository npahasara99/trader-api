from __future__ import annotations

from .config import DEFAULT_PLANNING_CONFIG
from .llm_reasoning import classify_final_action, reconcile_actions


# Lightweight validation fixtures for the final BUY / WAIT / AVOID classifier.
# These are profile-based examples, not ticker-specific rules.
ACTION_CLASSIFICATION_FIXTURES = [
    {
        "name": "constructive_pullback_wait",
        "expected_action": "WAIT",
        "payload": {
            "trend_state": "pullback_in_uptrend",
            "market_regime": "risk_off",
            "buy_threshold": 6,
            "entry_quality_score": 5.9,
            "entry_requires_confirmation": True,
            "support_quality_score": 6.4,
            "relative_strength_score": 7.1,
            "volume_confirmation_score": 4.7,
            "reward_risk": {"tp1": 1.18, "tp2": 1.72, "final": 2.4},
            "earnings": {"days_to_earnings": 22, "earnings_risk_flag": False},
            "volume_context": {"selloff_volume_state": "heavy_distribution", "reversal_volume_state": "weak_bounce"},
            "composite_score": 5.89,
            "expected_return": 0.011,
            "prob_tp": 0.42,
            "prob_sl": 0.38,
        },
    },
    {
        "name": "weak_breakdown_positive_offsets_wait",
        "expected_action": "WAIT",
        "payload": {
            "trend_state": "weak_breakdown_risk",
            "market_regime": "risk_off",
            "buy_threshold": 6,
            "entry_quality_score": 5.1,
            "entry_requires_confirmation": True,
            "support_quality_score": 5.3,
            "relative_strength_score": 6.1,
            "volume_confirmation_score": 4.6,
            "reward_risk": {"tp1": 1.04, "tp2": 1.58, "final": 2.1},
            "earnings": {"days_to_earnings": 18, "earnings_risk_flag": False},
            "volume_context": {"selloff_volume_state": "normal_pullback", "reversal_volume_state": "weak_bounce"},
            "composite_score": 4.46,
            "expected_return": 0.004,
            "prob_tp": 0.39,
            "prob_sl": 0.36,
        },
    },
    {
        "name": "weak_breakdown_negative_offsets_avoid",
        "expected_action": "AVOID",
        "payload": {
            "trend_state": "weak_breakdown_risk",
            "market_regime": "risk_off",
            "buy_threshold": 6,
            "entry_quality_score": 4.1,
            "entry_requires_confirmation": True,
            "support_quality_score": 4.4,
            "relative_strength_score": 4.2,
            "volume_confirmation_score": 3.9,
            "reward_risk": {"tp1": 0.82, "tp2": 1.2, "final": 1.5},
            "earnings": {"days_to_earnings": 12, "earnings_risk_flag": False},
            "volume_context": {"selloff_volume_state": "normal_pullback", "reversal_volume_state": "no_confirmation"},
            "composite_score": 4.30,
            "expected_return": -0.003,
            "prob_tp": 0.31,
            "prob_sl": 0.43,
        },
    },
    {
        "name": "borderline_weak_breakdown_orcl_like_avoid",
        "expected_action": "AVOID",
        "payload": {
            "trend_state": "weak_breakdown_risk",
            "market_regime": "risk_off",
            "buy_threshold": 6,
            "entry_quality_score": 4.9,
            "entry_requires_confirmation": True,
            "support_quality_score": 5.2,
            "relative_strength_score": 4.6,
            "volume_confirmation_score": 4.2,
            "reward_risk": {"tp1": 1.02, "tp2": 1.64, "final": 2.2},
            "earnings": {"days_to_earnings": 20, "earnings_risk_flag": False},
            "volume_context": {"selloff_volume_state": "normal_pullback", "reversal_volume_state": "weak_bounce"},
            "composite_score": 4.29,
            "expected_return": 0.002,
            "prob_tp": 0.37,
            "prob_sl": 0.39,
        },
    },
    {
        "name": "confirmed_buy_setup",
        "expected_action": "BUY",
        "payload": {
            "trend_state": "pullback_in_uptrend",
            "market_regime": "neutral",
            "buy_threshold": 6,
            "entry_quality_score": 7.3,
            "entry_requires_confirmation": False,
            "support_quality_score": 7.0,
            "relative_strength_score": 6.8,
            "volume_confirmation_score": 6.1,
            "reward_risk": {"tp1": 1.52, "tp2": 2.2, "final": 3.1},
            "earnings": {"days_to_earnings": 25, "earnings_risk_flag": False},
            "volume_context": {"selloff_volume_state": "light_pullback", "reversal_volume_state": "confirmed_bounce"},
            "composite_score": 6.85,
            "expected_return": 0.017,
            "prob_tp": 0.54,
            "prob_sl": 0.29,
        },
    },
]


def evaluate_action_classification_fixtures() -> list[dict]:
    results: list[dict] = []
    for fixture in ACTION_CLASSIFICATION_FIXTURES:
        outcome = classify_final_action(payload=fixture["payload"], config=DEFAULT_PLANNING_CONFIG)
        results.append(
            {
                "name": fixture["name"],
                "expected_action": fixture["expected_action"],
                "actual_action": outcome["quant_action"],
                "pass": outcome["quant_action"] == fixture["expected_action"],
                "reason_bucket": outcome["action_reason_bucket"],
                "avoid_severity_score": outcome["avoid_severity_score"],
            }
        )
    return results


def evaluate_reconciliation_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "monitorable_wait_overrides_llm_avoid",
            "quant_action": "WAIT",
            "llm_action": "AVOID",
            "monitorable_setup": True,
            "avoid_severity_score": 4.2,
            "constructive_traits": ["pullback_in_uptrend", "relative_strength_supportive", "support_confluence_present"],
            "trend_state": "pullback_in_uptrend",
            "relative_strength_score": 6.4,
            "expected_action": "WAIT",
        },
        {
            "name": "weak_breakdown_borderline_resolves_avoid",
            "quant_action": "WAIT",
            "llm_action": "AVOID",
            "monitorable_setup": False,
            "avoid_severity_score": 5.2,
            "constructive_traits": ["support_confluence_present", "entry_not_terrible"],
            "trend_state": "weak_breakdown_risk",
            "relative_strength_score": 4.6,
            "expected_action": "AVOID",
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        outcome = reconcile_actions(
            quant_action=fixture["quant_action"],
            llm_action=fixture["llm_action"],
            monitorable_setup=fixture["monitorable_setup"],
            avoid_severity_score=fixture["avoid_severity_score"],
            constructive_traits=fixture["constructive_traits"],
            trend_state=fixture["trend_state"],
            relative_strength_score=fixture["relative_strength_score"],
            config=DEFAULT_PLANNING_CONFIG,
        )
        results.append(
            {
                "name": fixture["name"],
                "expected_action": fixture["expected_action"],
                "actual_action": outcome["reconciled_action"],
                "pass": outcome["reconciled_action"] == fixture["expected_action"],
                "alignment": outcome["action_alignment"],
            }
        )
    return results
