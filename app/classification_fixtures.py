from __future__ import annotations

from types import SimpleNamespace
import pandas as pd

from .config import DEFAULT_PLANNING_CONFIG
from .actionability import build_actionability_soon
from .context_scenarios import build_market_context
from .execution_view import build_chart_execution_view
from .llm_reasoning import classify_final_action, reconcile_actions
from .monitoring import build_wait_monitoring_plan
from .ranking import build_ranking_profile
from .risk_engine import build_stop_loss, build_take_profits, estimate_hold_window
from .scanner import build_pre_scan_profile
from .suitability import build_swing_trade_suitability
from .live_plan_consistency import evaluate_live_plan_consistency
from .what_to_watch import build_what_to_watch
from .watchlist import build_watchlist_profile


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


def evaluate_watchlist_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "constructive_wait_medium_suitability_primary",
            "row": SimpleNamespace(
                final_action="WAIT",
                trend_state="pullback_in_uptrend",
                market_regime="risk_off",
                composite_score=5.89,
                relative_strength_score=7.1,
                monitorable_setup=True,
                watch_priority="high",
                constructive_traits=["pullback_in_uptrend", "relative_strength_supportive", "support_confluence_present"],
                buy_blockers=["confirmation_missing"],
                avoid_reason=None,
                swing_trade_suitability={
                    "suitability_score": 5.8,
                    "suitability_label": "medium",
                    "suitable_for_long_swing": True,
                    "suitable_for_watchlist_only": True,
                    "not_suitable_reason": None,
                },
            ),
            "expected_tier": "primary",
            "expected_bucket": "high_priority_watchlist",
        },
        {
            "name": "weaker_wait_low_suitability_secondary",
            "row": SimpleNamespace(
                final_action="WAIT",
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                composite_score=4.46,
                relative_strength_score=6.1,
                monitorable_setup=True,
                watch_priority="medium",
                constructive_traits=["support_confluence_present", "positive_expectancy", "relative_strength_supportive"],
                buy_blockers=["confirmation_missing", "structure_repair_needed"],
                avoid_reason=None,
                swing_trade_suitability={
                    "suitability_score": 4.1,
                    "suitability_label": "low",
                    "suitable_for_long_swing": False,
                    "suitable_for_watchlist_only": True,
                    "not_suitable_reason": None,
                },
            ),
            "expected_tier": "secondary",
            "expected_bucket": "secondary_watchlist",
        },
        {
            "name": "avoid_not_watchlist_worthy_none",
            "row": SimpleNamespace(
                final_action="AVOID",
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                composite_score=4.26,
                relative_strength_score=4.1,
                monitorable_setup=False,
                watch_priority=None,
                constructive_traits=["support_confluence_present"],
                buy_blockers=["negative_expectancy", "no_confirmation"],
                avoid_reason="Weak structure plus insufficient offsets.",
                swing_trade_suitability={
                    "suitability_score": 3.1,
                    "suitability_label": "unsuitable",
                    "suitable_for_long_swing": False,
                    "suitable_for_watchlist_only": False,
                    "not_suitable_reason": "The stock is not currently a practical swing-trade candidate.",
                },
            ),
            "expected_tier": "none",
            "expected_bucket": "avoid",
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        profile = build_watchlist_profile(fixture["row"], config=DEFAULT_PLANNING_CONFIG)
        results.append(
            {
                "name": fixture["name"],
                "tier": profile["watchlist_tier"],
                "bucket": profile["watchlist_bucket"],
                "primary": profile["is_primary_watchlist_candidate"],
                "secondary": profile["is_secondary_watchlist_candidate"],
                "pass": bool(
                    profile["watchlist_tier"] == fixture["expected_tier"]
                    and profile["watchlist_bucket"] == fixture["expected_bucket"]
                ),
            }
        )
    return results


def evaluate_actionability_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "constructive_primary_wait_near_execution_area",
            "row": SimpleNamespace(
                final_action="WAIT",
                monitorable_setup=True,
                trend_state="pullback_in_uptrend",
                market_regime="neutral",
                current_price=503.0,
                preferred_entry=497.8,
                preferred_entry_type="pullback",
                entry_quality_score=6.1,
                entry_requires_confirmation=True,
                confirmation_trigger="Need continuation confirmation above the current range",
                composite_score=6.4,
                relative_strength_score=6.7,
                support_quality_score=6.3,
                volume_confirmation_score=5.1,
                reward_risk={"tp1": 1.08, "tp2": 1.74},
                atr=6.2,
                watch_priority="high",
                wait_type="WAIT_CONFIRMATION",
                monitor_window_days=5,
                max_hold_days=10,
                support_zone_1={"lower": 494.4, "upper": 499.3, "source_tags": ["ema20"]},
                support_zone_2={"lower": 486.0, "upper": 491.2, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 503.2, "upper": 505.0, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 505.2, "upper": 507.3, "source_tags": ["range_high"]},
                consolidation_range={"lower": 492.5, "upper": 505.4, "source_tags": ["consolidation"]},
                breakout_level=505.0,
                volume_context={"reversal_volume_state": "weak_bounce"},
                relative_strength={"vs_spy": 0.06, "vs_qqq": 0.04},
            ),
            "expected_label": "ready_soon",
            "expected_active_watch": True,
            "expected_urgency": "high",
        },
        {
            "name": "valid_wait_needing_reset",
            "row": SimpleNamespace(
                final_action="WAIT",
                monitorable_setup=True,
                trend_state="range",
                market_regime="neutral",
                current_price=221.4,
                preferred_entry=214.2,
                preferred_entry_type="pullback",
                entry_quality_score=5.8,
                entry_requires_confirmation=True,
                confirmation_trigger="Need follow-through after the recent advance",
                composite_score=5.7,
                relative_strength_score=6.1,
                support_quality_score=5.5,
                volume_confirmation_score=4.8,
                reward_risk={"tp1": 1.12, "tp2": 1.64},
                atr=5.4,
                watch_priority="medium",
                wait_type="WAIT_BETTER_ENTRY",
                monitor_window_days=4,
                max_hold_days=9,
                support_zone_1={"lower": 212.8, "upper": 216.6, "source_tags": ["ema20", "pivot_low"]},
                support_zone_2={"lower": 206.4, "upper": 210.9, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 217.0, "upper": 219.5, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 219.9, "upper": 223.0, "source_tags": ["range_high"]},
                prior_breakout_retest_zone={"lower": 214.4, "upper": 219.4, "source_tags": ["breakout_retest"]},
                consolidation_range={"lower": 213.0, "upper": 223.0, "source_tags": ["consolidation"]},
                breakout_level=219.4,
                volume_context={"reversal_volume_state": "weak_bounce"},
                relative_strength={"vs_spy": 0.04, "vs_qqq": 0.03},
            ),
            "expected_label": "monitor",
            "expected_active_watch": True,
            "expected_urgency": "medium",
        },
        {
            "name": "weak_repair_wait_background",
            "row": SimpleNamespace(
                final_action="WAIT",
                monitorable_setup=True,
                trend_state="weak_breakdown_risk",
                market_regime="risk_off",
                current_price=176.2,
                preferred_entry=174.9,
                preferred_entry_type="pullback",
                entry_quality_score=5.0,
                entry_requires_confirmation=True,
                confirmation_trigger="Need reclaim and structure repair",
                composite_score=4.4,
                relative_strength_score=4.6,
                support_quality_score=4.9,
                volume_confirmation_score=4.2,
                reward_risk={"tp1": 1.03, "tp2": 1.35},
                atr=4.4,
                watch_priority="low",
                wait_type="WAIT_STRUCTURE_REPAIR",
                monitor_window_days=4,
                max_hold_days=7,
                support_zone_1={"lower": 174.6, "upper": 177.4, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 174.0, "upper": 177.0, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 179.2, "upper": 181.1, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 182.0, "upper": 184.0, "source_tags": ["gap_fill"]},
                prior_breakout_retest_zone={"lower": 171.8, "upper": 175.4, "source_tags": ["breakout_retest"]},
                breakout_level=175.4,
                volume_context={"reversal_volume_state": "weak_bounce"},
                relative_strength={"vs_spy": -0.03, "vs_qqq": -0.04},
            ),
            "expected_label": "background",
            "expected_active_watch": False,
            "expected_urgency": "low",
        },
        {
            "name": "non_wait_setup_does_not_receive_wait_semantics",
            "row": SimpleNamespace(
                final_action="BUY",
                monitorable_setup=True,
                trend_state="pullback_in_uptrend",
                market_regime="neutral",
                current_price=78.9,
                preferred_entry=78.8,
                preferred_entry_type="pullback",
                entry_quality_score=7.1,
                entry_requires_confirmation=False,
                composite_score=7.2,
                relative_strength_score=6.5,
                support_quality_score=6.2,
                volume_confirmation_score=6.0,
                reward_risk={"tp1": 1.45, "tp2": 2.1},
                atr=1.6,
                support_zone_1={"lower": 78.75, "upper": 80.07, "source_tags": ["ema20"]},
                resistance_zone_1={"lower": 82.0, "upper": 83.2, "source_tags": ["pivot_high"]},
                breakout_level=83.0,
                volume_context={"reversal_volume_state": "confirmed_bounce"},
            ),
            "expect_none": True,
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        row = fixture["row"]
        row.chart_execution_view = build_chart_execution_view(row, config=DEFAULT_PLANNING_CONFIG)
        row.swing_trade_suitability = build_swing_trade_suitability(row, config=DEFAULT_PLANNING_CONFIG)
        watchlist_profile = build_watchlist_profile(row, config=DEFAULT_PLANNING_CONFIG)
        row.watchlist_tier = watchlist_profile["watchlist_tier"]
        row.watchlist_bucket = watchlist_profile["watchlist_bucket"]
        row.watchlist_summary = watchlist_profile["watchlist_summary"]
        row.watchlist_reason = watchlist_profile["watchlist_reason"]
        row.is_primary_watchlist_candidate = watchlist_profile["is_primary_watchlist_candidate"]
        row.is_secondary_watchlist_candidate = watchlist_profile["is_secondary_watchlist_candidate"]
        profile = build_actionability_soon(row, config=DEFAULT_PLANNING_CONFIG)
        if fixture.get("expect_none"):
            results.append(
                {
                    "name": fixture["name"],
                    "actionability": profile,
                    "pass": profile is None,
                }
            )
            continue
        results.append(
            {
                "name": fixture["name"],
                "actionability_label": None if not profile else profile.get("actionability_label"),
                "active_watch": None if not profile else profile.get("active_watch"),
                "watch_urgency": None if not profile else profile.get("watch_urgency"),
                "days_to_action_estimate": None if not profile else profile.get("days_to_action_estimate"),
                "closest_trigger_type": None if not profile else profile.get("closest_trigger_type"),
                "has_summary": bool(profile and profile.get("actionability_summary")),
                "pass": bool(
                    profile
                    and profile.get("actionability_label") == fixture["expected_label"]
                    and bool(profile.get("active_watch")) == fixture["expected_active_watch"]
                    and profile.get("watch_urgency") == fixture["expected_urgency"]
                    and float(profile.get("actionability_score") or 0.0) >= 0.0
                    and bool(profile.get("actionability_summary"))
                ),
            }
        )
    return results


def evaluate_what_to_watch_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "continuation_setup_gets_hold_fail_continue_lines",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=503.0,
                trend_state="pullback_in_uptrend",
                preferred_entry=497.8,
                preferred_entry_type="pullback",
                entry_quality_score=6.1,
                entry_requires_confirmation=True,
                confirmation_trigger="Need continuation confirmation above the current range",
                reward_risk={"tp1": 1.08},
                atr=6.2,
                support_zone_1={"lower": 494.4, "upper": 499.3, "source_tags": ["ema20"]},
                support_zone_2={"lower": 486.0, "upper": 491.2, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 503.2, "upper": 505.0, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 505.2, "upper": 507.3, "source_tags": ["range_high"]},
                breakout_level=505.0,
                consolidation_range={"lower": 492.5, "upper": 505.4, "source_tags": ["consolidation"]},
                volume_context={"reversal_volume_state": "weak_bounce"},
            ),
            "expected_phrases": ["holds", "loses", "Continuation"],
            "expect_deeper": True,
            "expect_continuation": True,
        },
        {
            "name": "pullback_preferred_setup_gets_constructive_pullback_lines",
            "row": SimpleNamespace(
                final_action="BUY",
                current_price=78.9,
                trend_state="pullback_in_uptrend",
                preferred_entry=78.8,
                preferred_entry_type="pullback",
                entry_quality_score=7.1,
                entry_requires_confirmation=False,
                reward_risk={"tp1": 1.45},
                atr=1.6,
                support_zone_1={"lower": 78.75, "upper": 80.07, "source_tags": ["ema20"]},
                support_zone_2={"lower": 78.28, "upper": 79.60, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 82.0, "upper": 83.2, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 84.0, "upper": 85.4, "source_tags": ["pivot_high"]},
                breakout_level=83.0,
                volume_context={"reversal_volume_state": "confirmed_bounce"},
            ),
            "expected_phrases": ["pullback remains constructive", "look for a deeper reset", "Continuation"],
            "expect_deeper": True,
            "expect_continuation": True,
        },
        {
            "name": "repair_setup_gets_repair_specific_lines",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=43.2,
                trend_state="weak_breakdown_risk",
                preferred_entry=42.9,
                preferred_entry_type="pullback",
                entry_quality_score=5.0,
                entry_requires_confirmation=True,
                confirmation_trigger="Need stabilization and reclaim of resistance",
                reward_risk={"tp1": 1.0},
                atr=1.1,
                support_zone_1={"lower": 42.55, "upper": 43.25, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 41.75, "upper": 42.35, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 43.95, "upper": 44.55, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 44.85, "upper": 45.45, "source_tags": ["gap_fill"]},
                prior_breakout_retest_zone={"lower": 42.8, "upper": 43.7, "source_tags": ["breakout_retest"]},
                breakout_level=43.7,
                volume_context={"reversal_volume_state": "weak_bounce"},
            ),
            "expected_phrases": ["repair attempt stays alive", "structure stays weak", "Repair improves"],
            "expect_deeper": True,
            "expect_continuation": True,
        },
        {
            "name": "no_fake_deeper_or_continuation_zone_forced",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=176.2,
                trend_state="weak_breakdown_risk",
                preferred_entry=174.9,
                preferred_entry_type="pullback",
                entry_quality_score=5.2,
                entry_requires_confirmation=True,
                confirmation_trigger="Need reclaim and stabilization",
                reward_risk={"tp1": 1.03},
                atr=4.4,
                support_zone_1={"lower": 174.6, "upper": 177.4, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 174.0, "upper": 177.0, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 179.2, "upper": 181.1, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 182.0, "upper": 184.0, "source_tags": ["gap_fill"]},
                prior_breakout_retest_zone={"lower": 171.8, "upper": 175.4, "source_tags": ["breakout_retest"]},
                breakout_level=175.4,
                volume_context={"reversal_volume_state": "weak_bounce"},
            ),
            "expected_phrases": ["repair attempt stays alive", "structure stays weak"],
            "expect_deeper": False,
            "expect_continuation": False,
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        row = fixture["row"]
        row.chart_execution_view = build_chart_execution_view(row, config=DEFAULT_PLANNING_CONFIG)
        profile = build_what_to_watch(row, config=DEFAULT_PLANNING_CONFIG)
        summary_lines = [] if not profile else list(profile.get("watch_summary") or [])
        summary_short = "" if not profile else str(profile.get("watch_summary_short") or "")
        full_text = " ".join(summary_lines) + " " + summary_short
        results.append(
            {
                "name": fixture["name"],
                "has_hold_zone": bool(profile and profile.get("bullish_hold_zone")),
                "has_deeper_zone": bool(profile and profile.get("deeper_reset_target_zone")),
                "has_continuation_zone": bool(profile and profile.get("continuation_trigger_zone")),
                "watch_summary_count": len(summary_lines),
                "pass": bool(
                    profile
                    and bool(profile.get("bullish_hold_zone"))
                    and bool(summary_short)
                    and all(phrase in full_text for phrase in fixture["expected_phrases"])
                    and bool(profile.get("deeper_reset_target_zone")) == fixture["expect_deeper"]
                    and bool(profile.get("continuation_trigger_zone")) == fixture["expect_continuation"]
                ),
            }
        )
    return results


def _bars_from_closes(closes: list[float], *, base_volume: float = 1_000_000.0) -> list[dict]:
    bars: list[dict] = []
    for idx, close in enumerate(closes):
        prev = closes[idx - 1] if idx > 0 else close
        bars.append(
            {
                "date": f"2025-01-{(idx % 28) + 1:02d}",
                "open": prev,
                "high": max(prev, close) * 1.01,
                "low": min(prev, close) * 0.99,
                "close": close,
                "volume": base_volume * (1.0 + (0.08 if idx % 7 == 0 else 0.0)),
            }
        )
    return bars


def evaluate_pre_scan_fixtures() -> list[dict]:
    constructive = [100 + (i * 0.55) for i in range(65)] + [134.5, 133.8, 133.1, 132.6, 132.9, 133.4, 134.2, 135.0]
    weak = [140 - (i * 0.42) for i in range(73)]

    strong_profile = build_pre_scan_profile(
        ticker="GOOD",
        current_price=135.0,
        bars=_bars_from_closes(constructive, base_volume=2_400_000),
        benchmark_bars={
            "SPY": _bars_from_closes([100 + (i * 0.18) for i in range(len(constructive))]),
            "QQQ": _bars_from_closes([100 + (i * 0.22) for i in range(len(constructive))]),
            "XLK": _bars_from_closes([100 + (i * 0.20) for i in range(len(constructive))]),
        },
        sector_benchmark_symbol="XLK",
        earnings_context={"days_to_earnings": 22},
        config=DEFAULT_PLANNING_CONFIG,
    )
    weak_profile = build_pre_scan_profile(
        ticker="WEAK",
        current_price=109.34,
        bars=_bars_from_closes(weak, base_volume=700_000),
        benchmark_bars={
            "SPY": _bars_from_closes([100 + (i * 0.12) for i in range(len(weak))]),
            "QQQ": _bars_from_closes([100 + (i * 0.16) for i in range(len(weak))]),
            "XLK": _bars_from_closes([100 + (i * 0.15) for i in range(len(weak))]),
        },
        sector_benchmark_symbol="XLK",
        earnings_context={"days_to_earnings": 4},
        config=DEFAULT_PLANNING_CONFIG,
    )

    sector_good = build_pre_scan_profile(
        ticker="SECTOR_GOOD",
        current_price=135.0,
        bars=_bars_from_closes(constructive, base_volume=2_000_000),
        benchmark_bars={
            "SPY": _bars_from_closes([100 + (i * 0.18) for i in range(len(constructive))]),
            "QQQ": _bars_from_closes([100 + (i * 0.22) for i in range(len(constructive))]),
            "XLK": _bars_from_closes([100 + (i * 0.08) for i in range(len(constructive))]),
        },
        sector_benchmark_symbol="XLK",
        earnings_context={"days_to_earnings": 25},
        config=DEFAULT_PLANNING_CONFIG,
    )
    sector_bad = build_pre_scan_profile(
        ticker="SECTOR_BAD",
        current_price=135.0,
        bars=_bars_from_closes(constructive, base_volume=2_000_000),
        benchmark_bars={
            "SPY": _bars_from_closes([100 + (i * 0.18) for i in range(len(constructive))]),
            "QQQ": _bars_from_closes([100 + (i * 0.22) for i in range(len(constructive))]),
            "XLK": _bars_from_closes([100 + (i * 0.42) for i in range(len(constructive))]),
        },
        sector_benchmark_symbol="XLK",
        earnings_context={"days_to_earnings": 25},
        config=DEFAULT_PLANNING_CONFIG,
    )

    return [
        {
            "name": "constructive_pullback_scores_strong",
            "score": strong_profile["pre_scan_score"],
            "pass": strong_profile["pre_scan_score"] > weak_profile["pre_scan_score"],
            "tags": strong_profile["pre_scan_reason_tags"][:4],
        },
        {
            "name": "weak_breakdown_scores_lower",
            "score": weak_profile["pre_scan_score"],
            "pass": weak_profile["pre_scan_score"] < 5.0,
            "tags": weak_profile["pre_scan_reason_tags"][:4],
        },
        {
            "name": "sector_relative_strength_changes_score",
            "good_score": sector_good["pre_scan_score"],
            "bad_score": sector_bad["pre_scan_score"],
            "pass": sector_good["pre_scan_score"] > sector_bad["pre_scan_score"],
        },
    ]


def evaluate_split_ranking_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "buy_ready_name_lands_immediate",
            "row": SimpleNamespace(
                final_action="BUY",
                watchlist_tier="primary",
                market_regime="neutral",
                trend_state="pullback_in_uptrend",
                composite_score=7.0,
                entry_quality_score=7.2,
                reward_risk={"tp1": 1.6},
                expected_return=0.018,
                prob_tp=0.55,
                prob_sl=0.27,
                confidence=0.72,
                pre_scan_score=7.4,
                sector_relative_strength=0.05,
                swing_trade_suitability={"suitability_score": 7.1},
            ),
            "expected_bucket": "best_immediate_setups",
        },
        {
            "name": "watchlist_wait_name_lands_watchlist",
            "row": SimpleNamespace(
                final_action="WAIT",
                watchlist_tier="primary",
                market_regime="risk_off",
                trend_state="pullback_in_uptrend",
                composite_score=5.9,
                entry_quality_score=5.8,
                reward_risk={"tp1": 1.12},
                expected_return=0.008,
                prob_tp=0.42,
                prob_sl=0.36,
                confidence=0.6,
                pre_scan_score=6.5,
                sector_relative_strength=0.03,
                swing_trade_suitability={"suitability_score": 5.8},
            ),
            "expected_bucket": "best_watchlist_setups",
        },
        {
            "name": "weak_avoid_name_lands_rejected",
            "row": SimpleNamespace(
                final_action="AVOID",
                watchlist_tier="none",
                market_regime="risk_off",
                trend_state="weak_breakdown_risk",
                composite_score=3.8,
                entry_quality_score=4.0,
                reward_risk={"tp1": 0.85},
                expected_return=-0.004,
                prob_tp=0.31,
                prob_sl=0.43,
                confidence=0.57,
                pre_scan_score=3.7,
                sector_relative_strength=-0.05,
                swing_trade_suitability={"suitability_score": 3.1},
            ),
            "expected_bucket": "rejected_or_low_priority",
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        profile = build_ranking_profile(fixture["row"], config=DEFAULT_PLANNING_CONFIG)
        results.append(
            {
                "name": fixture["name"],
                "bucket": profile["ranking_bucket"],
                "scanner_rank_score": profile["scanner_rank_score"],
                "pass": profile["ranking_bucket"] == fixture["expected_bucket"],
            }
        )
    return results


def evaluate_chart_execution_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "constructive_wait_near_resistance_breakout_or_pullback",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=503.0,
                trend_state="pullback_in_uptrend",
                preferred_entry=497.8,
                preferred_entry_type="pullback",
                entry_quality_score=6.1,
                entry_requires_confirmation=True,
                confirmation_trigger="Wait for breakout close above the recent range high",
                reward_risk={"tp1": 1.08},
                atr=6.2,
                moving_averages={"ema20": 498.4, "sma50": 494.1},
                volume_context={"reversal_volume_state": "weak_bounce"},
                support_zone_1={"lower": 489.0, "upper": 498.8, "source_tags": ["ema20"]},
                support_zone_2={"lower": 473.44, "upper": 500.34, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 490.37, "upper": 500.51, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 499.8, "upper": 504.1, "source_tags": ["range_high"]},
                breakout_level=500.2,
                prior_breakout_retest_zone={"lower": 494.4, "upper": 500.2, "source_tags": ["breakout_retest"]},
                consolidation_range={"lower": 489.8, "upper": 504.1, "source_tags": ["consolidation"]},
            ),
            "expected_shape": {"continuation_pullback_preferred", "near_resistance_wait", "breakout_or_pullback"},
            "expected_enter": {"no", "only_on_confirmation"},
            "expected_location": {"continuation_near_range_high", "near_resistance"},
            "expected_bias": {"pullback_preferred", "avoid_chasing"},
            "expected_breakout_type": {"reclaim_trigger", "breakout_trigger"},
            "expected_prior_status": {"context_only"},
            "expected_current_anchor_type": {"continuation_support", "pullback_support"},
            "require_breakout_not_below_prior": True,
        },
        {
            "name": "constructive_support_retest_pullback_candidate",
            "row": SimpleNamespace(
                final_action="BUY",
                current_price=78.9,
                trend_state="pullback_in_uptrend",
                preferred_entry=78.8,
                preferred_entry_type="pullback",
                entry_quality_score=7.1,
                entry_requires_confirmation=False,
                confirmation_trigger="Optional hold above support",
                reward_risk={"tp1": 1.45},
                atr=1.6,
                moving_averages={"ema20": 79.1, "sma50": 78.4},
                volume_context={"reversal_volume_state": "confirmed_bounce"},
                support_zone_1={"lower": 78.75, "upper": 80.07, "source_tags": ["ema20"]},
                support_zone_2={"lower": 78.28, "upper": 79.60, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 82.0, "upper": 83.2, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 84.0, "upper": 85.4, "source_tags": ["pivot_high"]},
                breakout_level=83.0,
            ),
            "expected_shape": "pullback_candidate",
            "expected_enter": {"yes", "only_on_confirmation"},
            "expected_location": {"near_support"},
            "expected_bias": {"pullback_preferred"},
            "expected_breakout_type": {"breakout_trigger"},
            "expected_prior_status": {"active", None},
            "expected_current_anchor_type": {"pullback_support"},
        },
        {
            "name": "weak_structure_repair_needed",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=43.2,
                trend_state="weak_breakdown_risk",
                preferred_entry=42.9,
                preferred_entry_type="pullback",
                entry_quality_score=5.0,
                entry_requires_confirmation=True,
                confirmation_trigger="Need stabilization and reclaim of resistance",
                reward_risk={"tp1": 1.0},
                atr=1.1,
                moving_averages={"ema20": 43.75, "sma50": 42.85},
                volume_context={"reversal_volume_state": "weak_bounce"},
                support_zone_1={"lower": 42.55, "upper": 43.25, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 41.75, "upper": 42.35, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 43.95, "upper": 44.55, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 44.85, "upper": 45.45, "source_tags": ["gap_fill"]},
                prior_breakout_retest_zone={"lower": 42.8, "upper": 43.7, "source_tags": ["breakout_retest"]},
                breakout_level=43.7,
            ),
            "expected_shape": "structure_repair_needed",
            "expected_enter": {"no"},
            "expected_location": {"repair_band_still_active"},
            "expected_bias": {"wait_for_repair"},
            "expected_breakout_type": {"repair_trigger", "none"},
            "expected_prior_status": {"active"},
            "expected_current_anchor_type": {"repair_band"},
        },
        {
            "name": "post_trigger_case_uses_non_generic_label",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=221.4,
                trend_state="range",
                preferred_entry=214.2,
                preferred_entry_type="pullback",
                entry_quality_score=5.8,
                entry_requires_confirmation=True,
                confirmation_trigger="Need follow-through above the reclaimed range high",
                reward_risk={"tp1": 1.12},
                atr=5.4,
                moving_averages={"ema20": 216.8, "sma50": 212.1},
                volume_context={"reversal_volume_state": "weak_bounce"},
                support_zone_1={"lower": 212.8, "upper": 216.6, "source_tags": ["ema20", "pivot_low"]},
                support_zone_2={"lower": 206.4, "upper": 210.9, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 217.0, "upper": 219.5, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 219.9, "upper": 223.0, "source_tags": ["range_high"]},
                breakout_level=219.4,
                prior_breakout_retest_zone={"lower": 214.4, "upper": 219.4, "source_tags": ["breakout_retest"]},
                consolidation_range={"lower": 213.0, "upper": 223.0, "source_tags": ["consolidation"]},
            ),
            "expected_shape": {"post_breakout_retest", "continuation_pullback_preferred"},
            "expected_enter": {"only_on_confirmation"},
            "expected_location": {"above_first_trigger_not_confirmed", "continuation_near_range_high", "post_breakout_retest", "continuation_above_old_trigger"},
            "expected_bias": {"pullback_preferred"},
            "expected_breakout_type": {"reclaim_trigger", "none"},
            "expected_prior_status": {"context_only"},
            "expected_current_anchor_type": {"continuation_support"},
            "require_breakout_not_below_prior": True,
        },
        {
            "name": "deeper_pullback_can_be_null_when_not_distinct",
            "row": SimpleNamespace(
                final_action="WAIT",
                current_price=176.2,
                trend_state="weak_breakdown_risk",
                preferred_entry=174.9,
                preferred_entry_type="pullback",
                entry_quality_score=5.2,
                entry_requires_confirmation=True,
                confirmation_trigger="Need reclaim and stabilization",
                reward_risk={"tp1": 1.03},
                atr=4.4,
                volume_context={"reversal_volume_state": "weak_bounce"},
                support_zone_1={"lower": 174.6, "upper": 177.4, "source_tags": ["pivot_low", "ema20"]},
                support_zone_2={"lower": 174.0, "upper": 177.0, "source_tags": ["sma50"]},
                resistance_zone_1={"lower": 179.2, "upper": 181.1, "source_tags": ["pivot_high"]},
                resistance_zone_2={"lower": 182.0, "upper": 184.0, "source_tags": ["gap_fill"]},
                prior_breakout_retest_zone={"lower": 171.8, "upper": 175.4, "source_tags": ["breakout_retest"]},
                breakout_level=175.4,
            ),
            "expected_shape": "structure_repair_needed",
            "expected_enter": {"no"},
            "expected_location": {"repair_reclaimed_but_not_clean"},
            "expected_bias": {"wait_for_repair"},
            "expected_breakout_type": {"repair_trigger"},
            "expected_prior_status": {"context_only"},
            "expected_current_anchor_type": {"repair_band"},
            "expect_deeper_null": True,
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        view = build_chart_execution_view(fixture["row"], config=DEFAULT_PLANNING_CONFIG)
        breakout = None if not view else view.get("breakout_point")
        pullback = None if not view else view.get("pullback_entry_zone")
        deeper = None if not view else view.get("deeper_pullback_zone")
        prior_anchor = None if not view else view.get("prior_trigger_anchor")
        current_anchor = None if not view else view.get("current_execution_anchor")
        breakout_width_pct = (
            None
            if not breakout
            else round((float(breakout["upper"]) - float(breakout["lower"])) / max(float(fixture["row"].current_price), 0.01), 4)
        )
        overlap_ratio = 0.0
        if pullback and deeper:
            overlap_lower = max(float(pullback["lower"]), float(deeper["lower"]))
            overlap_upper = min(float(pullback["upper"]), float(deeper["upper"]))
            overlap_width = max(0.0, overlap_upper - overlap_lower)
            overlap_base = min(float(pullback["upper"]) - float(pullback["lower"]), float(deeper["upper"]) - float(deeper["lower"]))
            overlap_ratio = 0.0 if overlap_base <= 0 else overlap_width / overlap_base
        breakout_not_below_prior = True
        if view and fixture.get("require_breakout_not_below_prior") and view.get("prior_trigger_anchor_status") == "context_only":
            prior_upper = None if not prior_anchor else float(prior_anchor["upper"])
            breakout_lower = None if not breakout else float(breakout["lower"])
            breakout_not_below_prior = breakout is None or prior_upper is None or breakout_lower >= prior_upper - 0.05
        results.append(
            {
                "name": fixture["name"],
                "trade_shape": None if not view else view.get("trade_shape"),
                "enter_now": None if not view else view.get("enter_now"),
                "current_price_location": None if not view else view.get("current_price_location"),
                "execution_bias": None if not view else view.get("execution_bias"),
                "breakout_point_type": None if not view else view.get("breakout_point_type"),
                "execution_zone_quality": None if not view else view.get("execution_zone_quality"),
                "range_position_pct": None if not view else view.get("range_position_pct"),
                "deeper_pullback_available": None if not view else view.get("deeper_pullback_available"),
                "prior_trigger_anchor_status": None if not view else view.get("prior_trigger_anchor_status"),
                "current_execution_anchor_type": None if not view else view.get("current_execution_anchor_type"),
                "has_current_execution_anchor": bool(view and view.get("current_execution_anchor")),
                "breakout_not_below_prior": breakout_not_below_prior,
                "has_breakout": bool(view and view.get("breakout_point")),
                "has_pullback": bool(view and view.get("pullback_entry_zone")),
                "has_summary": bool(view and view.get("chart_execution_summary")),
                "breakout_width_pct": breakout_width_pct,
                "deeper_overlap_ratio": round(overlap_ratio, 4),
                "pass": bool(
                    view
                    and (
                        (isinstance(fixture["expected_shape"], set) and view.get("trade_shape") in fixture["expected_shape"])
                        or view.get("trade_shape") == fixture["expected_shape"]
                    )
                    and view.get("enter_now") in fixture["expected_enter"]
                    and view.get("current_price_location") in fixture["expected_location"]
                    and view.get("execution_bias") in fixture["expected_bias"]
                    and view.get("breakout_point_type") in fixture["expected_breakout_type"]
                    and view.get("prior_trigger_anchor_status") in fixture["expected_prior_status"]
                    and view.get("current_execution_anchor_type") in fixture["expected_current_anchor_type"]
                    and view.get("current_price_location") != "above_breakout"
                    and (view.get("range_position_pct") is None or 0.0 <= float(view.get("range_position_pct")) <= 1.25)
                    and (breakout_width_pct is None or breakout_width_pct <= DEFAULT_PLANNING_CONFIG.execution_zone_max_width_pct + 0.002)
                    and (not deeper or not pullback or overlap_ratio <= DEFAULT_PLANNING_CONFIG.execution_zone_overlap_max_pct + 0.05)
                    and (
                        "expect_deeper_null" not in fixture
                        or fixture["expect_deeper_null"] is False
                        or not view.get("deeper_pullback_available")
                    )
                    and breakout_not_below_prior
                    and bool(view.get("current_execution_anchor"))
                    and bool(view.get("chart_execution_summary"))
                ),
            }
        )
    return results


def evaluate_swing_realism_fixtures() -> list[dict]:
    config = DEFAULT_PLANNING_CONFIG

    wide_stop = build_stop_loss(
        preferred_entry=100.0,
        support_zone_1={"lower": 71.0, "upper": 73.0, "source_tags": ["pivot_low"]},
        support_zone_2={"lower": 66.0, "upper": 68.0, "source_tags": ["sma50"]},
        recent_swing_low=69.5,
        atr=3.0,
        current_price=101.0,
        trend_state="pullback_in_uptrend",
        config=config,
    )

    repair_stop = build_stop_loss(
        preferred_entry=100.0,
        support_zone_1={"lower": 84.8, "upper": 86.2, "source_tags": ["repair_band"]},
        support_zone_2={"lower": 82.5, "upper": 84.0, "source_tags": ["sma50"]},
        recent_swing_low=85.2,
        atr=3.0,
        current_price=99.5,
        trend_state="weak_breakdown_risk",
        sl_tolerance="moderate_to_wide",
        setup_scenario="rebound_repair_candidate",
        config=config,
    )
    continuation_stop = build_stop_loss(
        preferred_entry=100.0,
        support_zone_1={"lower": 93.6, "upper": 94.8, "source_tags": ["ema20", "pivot_low"]},
        support_zone_2={"lower": 90.8, "upper": 92.4, "source_tags": ["sma50"]},
        recent_swing_low=94.1,
        atr=2.6,
        current_price=101.5,
        trend_state="uptrend",
        sl_tolerance="tight_to_moderate",
        setup_scenario="supported_high_range_continuation",
        config=config,
    )

    initial_hold = estimate_hold_window(
        preferred_entry=100.0,
        take_profit_1=118.0,
        atr=2.0,
        recent_swing_bars=12,
        historical_hold_days=10,
        config=config,
    )
    capped_tp = build_take_profits(
        preferred_entry=100.0,
        stop_loss=93.0,
        resistance_zone_1={"lower": 128.0, "upper": 132.0, "source_tags": ["pivot_high"]},
        resistance_zone_2={"lower": 136.0, "upper": 140.0, "source_tags": ["gap_fill"]},
        recent_swing_high=129.5,
        atr=2.0,
        hold_days_hint=6,
        trend_state="pullback_in_uptrend",
        tp_aggressiveness="conservative",
        expected_move_profile="repair_bounce_not_full_recovery",
        price_location_context="weak_near_low",
        config=config,
    )
    broad_tp = build_take_profits(
        preferred_entry=100.0,
        stop_loss=94.0,
        resistance_zone_1={"lower": 106.0, "upper": 108.0, "source_tags": ["pivot_high"]},
        resistance_zone_2={"lower": 111.0, "upper": 114.0, "source_tags": ["gap_fill"]},
        recent_swing_high=107.0,
        atr=2.4,
        hold_days_hint=12,
        trend_state="uptrend",
        tp_aggressiveness="moderate_to_high",
        expected_move_profile="breakout_can_extend_if_confirmed",
        price_location_context="near_high_but_supported",
        config=config,
    )

    return [
        {
            "name": "normal_swing_stop_is_capped_when_structure_is_too_wide",
            "pass": (
                wide_stop["stop_width_pct"] <= config.max_stop_width_pct_default * 100.0 + 0.01
                and wide_stop["risk_width_flag"] == "capped_for_swing"
            ),
            "stop_width_pct": wide_stop["stop_width_pct"],
            "risk_width_flag": wide_stop["risk_width_flag"],
        },
        {
            "name": "repair_setups_can_run_wider_but_stay_bounded",
            "pass": (
                repair_stop["stop_width_pct"] <= config.max_stop_width_pct_repair * 100.0 + 0.01
                and repair_stop["stop_width_pct"] > continuation_stop["stop_width_pct"]
            ),
            "stop_width_pct": repair_stop["stop_width_pct"],
            "swing_realism_flag": repair_stop["swing_realism_flag"],
        },
        {
            "name": "continuation_setup_uses_tighter_stop_tolerance_than_repair",
            "pass": continuation_stop["stop_width_pct"] < repair_stop["stop_width_pct"],
            "continuation_stop_width_pct": continuation_stop["stop_width_pct"],
            "repair_stop_width_pct": repair_stop["stop_width_pct"],
        },
        {
            "name": "hold_window_reachability_compresses_far_tp1",
            "pass": (
                capped_tp["target_reachability_flag"] == "capped_to_hold_window"
                and capped_tp["tp1_distance_pct"] < 20.0
            ),
            "tp1_distance_pct": capped_tp["tp1_distance_pct"],
            "reachability_flag": capped_tp["target_reachability_flag"],
            "reachability_score": capped_tp["hold_window_reachability_score"],
        },
        {
            "name": "tp1_prefers_first_swing_target_not_distant_structure",
            "pass": (
                broad_tp["take_profit_1"] < 109.0
                and broad_tp["tp1_generation_reason"].startswith("tp1 near first actionable resistance slice")
            ),
            "take_profit_1": broad_tp["take_profit_1"],
            "tp1_generation_reason": broad_tp["tp1_generation_reason"],
        },
        {
            "name": "hold_window_estimate_stays_consistent_with_near_term_target",
            "pass": initial_hold["max_hold_days"] <= config.max_hold_days_max,
            "max_hold_days": initial_hold["max_hold_days"],
        },
    ]


def evaluate_market_context_fixtures() -> list[dict]:
    config = DEFAULT_PLANNING_CONFIG

    def _frame(closes: list[float], volumes: list[int]) -> pd.DataFrame:
        return pd.DataFrame({"close": closes, "volume": volumes})

    lin_context = build_market_context(
        ticker="LIN",
        current_price=486.0,
        frame=_frame(
            closes=[440.0, 444.0, 448.0, 451.0, 455.0, 459.0, 462.0, 466.0, 470.0, 473.0, 476.0, 480.0, 483.0, 485.5, 486.0],
            volumes=[100, 102, 98, 104, 103, 106, 101, 109, 110, 108, 112, 115, 116, 114, 113],
        ),
        trend_state="pullback_in_uptrend",
        moving_averages={"ema20": 474.0, "sma50": 466.0, "sma100": 452.0, "sma200": 430.0},
        atr=7.0,
        volume_context={"selloff_volume_state": "light_pullback", "reversal_volume_state": "confirmed_bounce"},
        relative_strength={"vs_spy": 0.06, "vs_qqq": 0.04},
        market_regime="neutral",
        news_items=[{"headline": "LIN upgraded after strong demand outlook", "summary": "Analysts raised targets after constructive guidance.", "datetime": "2026-06-30T12:00:00+00:00"}],
        news_score=5,
        earnings={"days_to_earnings": 25, "earnings_risk_flag": False},
        ticker_meta={"sector": "materials", "industry": "industrial gases"},
        sector_relative_strength=0.05,
        config=config,
    )

    extended_context = build_market_context(
        ticker="AMD",
        current_price=226.0,
        frame=_frame(
            closes=[182.0, 186.0, 191.0, 197.0, 202.0, 207.0, 212.0, 217.0, 221.0, 224.0, 226.0],
            volumes=[100, 99, 102, 105, 107, 111, 113, 115, 116, 118, 117],
        ),
        trend_state="uptrend",
        moving_averages={"ema20": 208.0, "sma50": 198.0, "sma100": 184.0, "sma200": 170.0},
        atr=4.3,
        volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "confirmed_bounce"},
        relative_strength={"vs_spy": 0.03, "vs_qqq": -0.01},
        market_regime="risk_off",
        news_items=[{"headline": "AMD faces target cut after stretched rally", "summary": "Analysts warned valuation looks rich after the recent move.", "datetime": "2026-06-30T12:00:00+00:00"}],
        news_score=-4,
        earnings={"days_to_earnings": 29, "earnings_risk_flag": False},
        ticker_meta={"sector": "technology", "industry": "semiconductors"},
        sector_relative_strength=-0.02,
        config=config,
    )

    rebound_context = build_market_context(
        ticker="NVDA",
        current_price=132.0,
        frame=_frame(
            closes=[171.0, 166.0, 159.0, 151.0, 145.0, 139.0, 133.0, 128.0, 124.0, 126.0, 129.0, 132.0],
            volumes=[118, 120, 124, 126, 128, 132, 135, 130, 122, 118, 116, 114],
        ),
        trend_state="weak_breakdown_risk",
        moving_averages={"ema20": 140.0, "sma50": 149.0, "sma100": 161.0, "sma200": 175.0},
        atr=6.2,
        volume_context={"selloff_volume_state": "normal_pullback", "reversal_volume_state": "confirmed_bounce"},
        relative_strength={"vs_spy": -0.02, "vs_qqq": 0.01},
        market_regime="neutral",
        news_items=[{"headline": "NVDA wins new AI platform contract", "summary": "The deal supports demand recovery after a weak period.", "datetime": "2026-06-30T12:00:00+00:00"}],
        news_score=4,
        earnings={"days_to_earnings": 41, "earnings_risk_flag": False},
        ticker_meta={"sector": "technology", "industry": "semiconductors"},
        sector_relative_strength=0.03,
        config=config,
    )

    weak_low_context = build_market_context(
        ticker="PFE",
        current_price=24.8,
        frame=_frame(
            closes=[32.0, 31.4, 30.8, 29.9, 28.7, 27.8, 26.9, 26.0, 25.4, 25.0, 24.8],
            volumes=[100, 101, 99, 102, 104, 107, 109, 111, 114, 116, 118],
        ),
        trend_state="weak_breakdown_risk",
        moving_averages={"ema20": 27.4, "sma50": 28.9, "sma100": 30.6, "sma200": 33.1},
        atr=1.1,
        volume_context={"selloff_volume_state": "heavy_distribution", "reversal_volume_state": "no_confirmation"},
        relative_strength={"vs_spy": -0.05, "vs_qqq": -0.04},
        market_regime="risk_off",
        news_items=[],
        news_score=-1,
        earnings={"days_to_earnings": 18, "earnings_risk_flag": False},
        ticker_meta={"sector": "health care", "industry": "biopharma"},
        sector_relative_strength=-0.03,
        config=config,
    )

    return [
        {
            "name": "high_range_constructive_setup_can_still_be_continuation_favored",
            "pass": (
                lin_context["price_location_context"] == "near_high_but_supported"
                and lin_context["continuation_vs_reversion_bias"] == "continuation_favored"
                and lin_context["setup_scenario"] in {"strong_continuation_pullback", "supported_high_range_continuation", "controlled_high_range_continuation"}
            ),
            "context": lin_context,
        },
        {
            "name": "high_range_conflicted_extension_gets_downgraded",
            "pass": (
                extended_context["price_location_context"] == "extended_near_high"
                and extended_context["continuation_vs_reversion_bias"] == "mean_reversion_favored"
                and extended_context["news_regime_alignment"] in {"aligned_bearish", "conflicted"}
            ),
            "context": extended_context,
        },
        {
            "name": "low_range_with_reversal_and_catalyst_can_be_rebound_candidate",
            "pass": (
                rebound_context["setup_type"] in {"deep_rebound_attempt", "repair_after_breakdown"}
                and rebound_context["continuation_vs_reversion_bias"] == "rebound_candidate"
                and rebound_context["news_regime_alignment"] == "aligned_bullish"
            ),
            "context": rebound_context,
        },
        {
            "name": "low_range_without_supportive_context_is_not_false_bullish",
            "pass": (
                weak_low_context["price_location_context"] in {"weak_near_low", "deep_in_lower_range"}
                and weak_low_context["news_regime_alignment"] != "aligned_bullish"
                and weak_low_context["continuation_vs_reversion_bias"] != "continuation_favored"
            ),
            "context": weak_low_context,
        },
    ]


def evaluate_live_plan_consistency_fixtures() -> list[dict]:
    fixtures = [
        {
            "name": "continuation_modestly_above_entry_stays_live_but_extended",
            "payload": {
                "live_price": 102.2,
                "preferred_entry": 100.0,
                "stop_loss": 95.0,
                "take_profit_1": 107.5,
                "setup_scenario": "supported_high_range_continuation",
                "continuation_vs_reversion_bias": "continuation_favored",
            },
            "expected": {
                "entry_status": "above_entry_zone",
                "plan_freshness_status": "live_but_extended",
                "live_vs_plan_alignment": "continuation_extended",
                "replan_needed": False,
            },
        },
        {
            "name": "continuation_above_tp1_marks_target_already_hit",
            "payload": {
                "live_price": 110.8,
                "preferred_entry": 100.0,
                "stop_loss": 95.0,
                "take_profit_1": 107.0,
                "setup_scenario": "supported_high_range_continuation",
                "continuation_vs_reversion_bias": "continuation_favored",
            },
            "expected": {
                "tp1_status": "tp1_exceeded",
                "plan_freshness_status": "stale_for_live_price",
                "live_vs_plan_alignment": "target_already_hit",
                "replan_needed": True,
            },
        },
        {
            "name": "pullback_entry_far_above_becomes_entry_missed",
            "payload": {
                "live_price": 106.5,
                "preferred_entry": 100.0,
                "stop_loss": 95.0,
                "take_profit_1": 111.0,
                "setup_scenario": "pullback_candidate",
                "continuation_vs_reversion_bias": "mean_reversion_balanced",
            },
            "expected": {
                "entry_status": "extended_beyond_entry",
                "plan_freshness_status": "stale_for_live_price",
                "live_vs_plan_alignment": "entry_missed",
                "replan_needed": True,
            },
        },
        {
            "name": "live_price_near_stop_flags_near_invalidation",
            "payload": {
                "live_price": 95.6,
                "preferred_entry": 100.0,
                "stop_loss": 95.0,
                "take_profit_1": 107.0,
                "setup_scenario": "constructive_pullback",
            },
            "expected": {
                "stop_status": "at_risk_of_invalidation",
                "plan_freshness_status": "partially_stale",
                "live_vs_plan_alignment": "near_invalidation",
                "replan_needed": False,
            },
        },
        {
            "name": "repair_bounce_beyond_target_needs_refresh",
            "payload": {
                "live_price": 109.4,
                "preferred_entry": 100.0,
                "stop_loss": 94.0,
                "take_profit_1": 106.5,
                "setup_scenario": "rebound_repair_candidate",
                "continuation_vs_reversion_bias": "rebound_candidate",
            },
            "expected": {
                "tp1_status": "tp1_exceeded",
                "plan_freshness_status": "stale_for_live_price",
                "live_vs_plan_alignment": "rebound_already_moved",
                "replan_needed": True,
            },
        },
    ]

    results: list[dict] = []
    for fixture in fixtures:
        outcome = evaluate_live_plan_consistency(fixture["payload"])
        checks = {key: outcome.get(key) == value for key, value in fixture["expected"].items()}
        results.append(
            {
                "name": fixture["name"],
                "pass": all(checks.values()),
                "checks": checks,
                "actual": {key: outcome.get(key) for key in fixture["expected"].keys()},
            }
        )
    return results
