from __future__ import annotations

from types import SimpleNamespace

from .config import DEFAULT_PLANNING_CONFIG
from .llm_reasoning import classify_final_action, reconcile_actions
from .monitoring import build_wait_monitoring_plan
from .suitability import build_swing_trade_suitability


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


def evaluate_wait_monitoring_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "csco_like_wait_monitoring",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=79.25,
                trend_state="pullback_in_uptrend",
                market_regime="risk_off",
                entry_requires_confirmation=True,
                entry_distance_from_current_price_pct=0.8,
                composite_score=5.89,
                relative_strength_score=7.1,
                atr=2.05,
                moving_averages={"ema20": 79.41, "sma50": 78.94},
                earnings={"days_to_earnings": 22},
                support_zone_1={"lower": 78.75, "upper": 80.07, "source_tags": ["ema20"]},
                support_zone_2={"lower": 78.28, "upper": 79.60, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 79.39, "upper": 80.71, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 79.83, "upper": 81.16, "source_tags": ["pivot_high", "fib_382"]},
            ),
            "expected_wait_type": "WAIT_CONFIRMATION",
            "expected_watch_priority": "high",
        },
        {
            "name": "intc_like_wait_monitoring",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=43.2,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                entry_requires_confirmation=True,
                entry_distance_from_current_price_pct=0.5,
                composite_score=4.46,
                relative_strength_score=6.1,
                atr=1.84,
                moving_averages={"ema20": 43.75, "sma50": 42.85},
                earnings={"days_to_earnings": 18},
                support_zone_1={"lower": 42.55, "upper": 43.25, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 41.75, "upper": 42.35, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 43.95, "upper": 44.55, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 44.85, "upper": 45.45, "source_tags": ["gap_fill"]},
            ),
            "expected_wait_type": "WAIT_STRUCTURE_REPAIR",
            "expected_watch_priority": "medium",
        },
        {
            "name": "nvda_like_avoid_no_monitoring",
            "row": SimpleNamespace(
                final_action="AVOID",
                current_price=167.5,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                entry_requires_confirmation=True,
                entry_distance_from_current_price_pct=1.7,
                composite_score=4.26,
                relative_strength_score=4.1,
                atr=5.8,
                moving_averages={"ema20": 171.2},
                earnings={"days_to_earnings": 35},
                support_zone_1={"lower": 161.83, "upper": 166.2, "source_tags": ["pivot_low"]},
                support_zone_2={"lower": 156.4, "upper": 160.0, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 176.1, "upper": 178.3, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 181.2, "upper": 184.8, "source_tags": ["gap_fill"]},
            ),
            "expected_wait_type": None,
            "expected_watch_priority": None,
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        plan = build_wait_monitoring_plan(fixture["row"], config=DEFAULT_PLANNING_CONFIG)
        results.append(
            {
                "name": fixture["name"],
                "has_plan": bool(plan),
                "wait_type": None if not plan else plan.get("wait_type"),
                "watch_priority": None if not plan else plan.get("watch_priority"),
                "has_upgrade_triggers": bool(plan and plan.get("upgrade_triggers")),
                "has_failure_triggers": bool(plan and plan.get("failure_triggers")),
                "has_support_summary": bool(plan and plan.get("support_zone_summary")),
                "pass": bool(
                    (
                        not plan
                        if fixture["expected_wait_type"] is None
                        else (
                            plan
                            and plan.get("wait_type") == fixture["expected_wait_type"]
                            and plan.get("watch_priority") == fixture["expected_watch_priority"]
                        )
                    )
                ),
            }
        )
    return results


def evaluate_suitability_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "csco_like_medium_or_high_suitability",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=79.25,
                trend_state="pullback_in_uptrend",
                market_regime="risk_off",
                trend_quality_score=7.1,
                support_quality_score=6.5,
                entry_quality_score=5.9,
                reward_risk={"tp1": 1.18, "tp2": 1.72, "final": 2.4},
                atr_pct=0.026,
                stop_too_tight_flag=False,
                volume_confirmation_score=4.7,
                relative_strength_score=7.1,
                earnings_risk_score=7.0,
                max_hold_days=9,
                monitor_window_days=5,
                support_zone_1={"lower": 78.75, "upper": 80.07},
                support_zone_2={"lower": 78.28, "upper": 79.60},
                stop_basis="below support_zone_1 and ATR buffer",
                volume_context={"selloff_volume_state": "heavy_distribution", "reversal_volume_state": "weak_bounce"},
                expected_return=0.011,
                prob_tp=0.42,
                prob_sl=0.38,
                earnings={"days_to_earnings": 22},
                monitorable_setup=True,
            ),
            "expected_labels": {"medium", "high"},
            "expected_watchlist": True,
        },
        {
            "name": "intc_like_medium_suitability",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=43.2,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                trend_quality_score=4.8,
                support_quality_score=5.3,
                entry_quality_score=5.1,
                reward_risk={"tp1": 1.04, "tp2": 1.58, "final": 2.1},
                atr_pct=0.042,
                stop_too_tight_flag=False,
                volume_confirmation_score=4.6,
                relative_strength_score=6.1,
                earnings_risk_score=6.0,
                max_hold_days=9,
                monitor_window_days=4,
                support_zone_1={"lower": 42.55, "upper": 43.25},
                support_zone_2={"lower": 41.75, "upper": 42.35},
                stop_basis="below support_zone_1 and ATR buffer",
                volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "weak_bounce"},
                expected_return=0.004,
                prob_tp=0.39,
                prob_sl=0.36,
                earnings={"days_to_earnings": 18},
                monitorable_setup=True,
            ),
            "expected_labels": {"medium"},
            "expected_watchlist": True,
        },
        {
            "name": "orcl_like_low_suitability",
            "row": SimpleNamespace(
                final_action="AVOID",
                current_price=137.5,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                trend_quality_score=3.9,
                support_quality_score=5.2,
                entry_quality_score=4.9,
                reward_risk={"tp1": 1.02, "tp2": 1.64, "final": 2.2},
                atr_pct=0.041,
                stop_too_tight_flag=False,
                volume_confirmation_score=4.2,
                relative_strength_score=4.6,
                earnings_risk_score=6.8,
                max_hold_days=8,
                monitor_window_days=None,
                support_zone_1={"lower": 134.0, "upper": 138.5},
                support_zone_2={"lower": 130.5, "upper": 133.5},
                stop_basis="below support_zone_1 and ATR buffer",
                volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "weak_bounce"},
                expected_return=0.002,
                prob_tp=0.37,
                prob_sl=0.39,
                earnings={"days_to_earnings": 20},
                monitorable_setup=False,
            ),
            "expected_labels": {"low"},
            "expected_watchlist": False,
        },
        {
            "name": "nvda_like_unsuitable_or_low",
            "row": SimpleNamespace(
                final_action="AVOID",
                current_price=167.5,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                trend_quality_score=3.2,
                support_quality_score=4.3,
                entry_quality_score=3.9,
                reward_risk={"tp1": 0.88, "tp2": 1.25, "final": 1.7},
                atr_pct=0.052,
                stop_too_tight_flag=False,
                volume_confirmation_score=3.6,
                relative_strength_score=4.1,
                earnings_risk_score=6.5,
                max_hold_days=9,
                monitor_window_days=None,
                support_zone_1={"lower": 161.8, "upper": 166.2},
                support_zone_2={"lower": 156.4, "upper": 160.0},
                stop_basis="below support_zone_1 and ATR buffer",
                volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "no_confirmation"},
                expected_return=-0.006,
                prob_tp=0.31,
                prob_sl=0.44,
                earnings={"days_to_earnings": 35},
                monitorable_setup=False,
            ),
            "expected_labels": {"low", "unsuitable"},
            "expected_watchlist": False,
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        suitability = build_swing_trade_suitability(fixture["row"], config=DEFAULT_PLANNING_CONFIG)
        results.append(
            {
                "name": fixture["name"],
                "label": suitability["suitability_label"],
                "score": suitability["suitability_score"],
                "watchlist_only": suitability["suitable_for_watchlist_only"],
                "pass": bool(
                    suitability["suitability_label"] in fixture["expected_labels"]
                    and suitability["suitable_for_watchlist_only"] == fixture["expected_watchlist"]
                ),
            }
        )
    return results
