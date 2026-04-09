from __future__ import annotations

from types import SimpleNamespace

from .config import DEFAULT_PLANNING_CONFIG
from .execution_view import build_chart_execution_view
from .llm_reasoning import classify_final_action, reconcile_actions
from .monitoring import build_wait_monitoring_plan
from .ranking import build_ranking_profile
from .scanner import build_pre_scan_profile
from .suitability import build_swing_trade_suitability
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
            "expected_location": {"near_support", "structure_below_trigger"},
            "expected_bias": {"wait_for_repair"},
            "expected_breakout_type": {"repair_trigger"},
            "expected_prior_status": {"active", "context_only"},
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
            "expected_location": {"above_first_trigger_not_confirmed", "continuation_near_range_high", "post_breakout_retest"},
            "expected_bias": {"pullback_preferred"},
            "expected_breakout_type": {"reclaim_trigger"},
            "expected_prior_status": {"context_only"},
            "expected_current_anchor_type": {"continuation_support"},
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
            "expected_location": {"near_support"},
            "expected_bias": {"wait_for_repair"},
            "expected_breakout_type": {"repair_trigger"},
            "expected_prior_status": {"context_only", "active"},
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
                    and bool(view.get("current_execution_anchor"))
                    and bool(view.get("chart_execution_summary"))
                ),
            }
        )
    return results
