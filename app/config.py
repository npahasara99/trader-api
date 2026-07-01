from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanningConfig:
    """Centralized tunables for structured swing planning."""

    history_lookback_days: int = 320
    pivot_lookback: int = 4
    pivot_max_points: int = 6
    atr_window: int = 14
    atr_stop_buffer_mult: float = 1.15
    atr_zone_width_mult: float = 0.35
    atr_target_window_mult: float = 0.9
    consolidation_window: int = 20
    consolidation_range_atr_mult: float = 2.8
    min_reward_risk_for_buy: float = 1.35
    min_reward_risk_for_wait: float = 0.9
    min_reward_risk_tp2_for_buy: float = 1.8
    max_entry_distance_pct: float = 0.045
    deep_entry_distance_pct: float = 0.12
    immediate_entry_atr_mult: float = 0.2
    pullback_buffer_atr_mult: float = 0.18
    deeper_pullback_buffer_atr_mult: float = 0.32
    stop_buffer_atr_mult: float = 1.2
    stop_below_zone_buffer_pct: float = 0.004
    tp1_atr_mult: float = 1.6
    tp2_atr_mult: float = 3.0
    max_hold_days_min: int = 5
    max_hold_days_max: int = 30
    max_stop_width_pct_default: float = 0.12
    max_stop_width_pct_repair: float = 0.18
    max_stop_width_atr_default: float = 3.8
    max_stop_width_atr_repair: float = 5.2
    max_tp1_distance_pct_default: float = 0.16
    max_tp1_distance_pct_repair: float = 0.22
    max_tp1_distance_atr_default: float = 5.5
    max_tp1_distance_atr_repair: float = 7.0
    hold_window_reachability_factor: float = 0.85
    earnings_penalty_near_days: int = 7
    earnings_penalty_mid_days: int = 14
    earnings_hard_block_days: int = 3
    volume_window: int = 20
    relative_strength_window: int = 40
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ")
    pre_scan_shortlist_size: int = 30
    pre_scan_min_history_bars: int = 60
    pre_scan_min_avg_dollar_volume: float = 20_000_000.0
    buy_min_entry_quality: float = 6.2
    buy_min_relative_strength_score: float = 5.2
    buy_min_support_quality_score: float = 5.2
    buy_min_volume_confirmation_score: float = 4.8
    wait_min_composite_score: float = 4.2
    wait_min_entry_quality: float = 4.4
    wait_min_relative_strength_score: float = 5.0
    weak_breakdown_wait_min_traits: int = 4
    weak_breakdown_wait_min_support_quality_score: float = 5.0
    weak_breakdown_wait_min_prob_edge: float = -0.02
    weak_breakdown_wait_max_severity: float = 4.6
    wait_monitor_days_pullback: int = 6
    wait_monitor_days_structure_repair: int = 4
    wait_monitor_days_other: int = 3
    wait_monitor_days_min: int = 2
    wait_monitor_days_max: int = 7
    wait_watch_priority_high_composite: float = 5.6
    wait_watch_priority_medium_composite: float = 4.6
    suitability_high_threshold: float = 7.0
    suitability_medium_threshold: float = 5.2
    suitability_low_threshold: float = 3.6
    watchlist_primary_min_suitability_score: float = 5.2
    watchlist_secondary_min_suitability_score: float = 3.4
    watchlist_primary_min_relative_strength_score: float = 5.8
    watchlist_primary_min_composite_score: float = 5.0
    actionability_ready_soon_threshold: float = 6.6
    actionability_monitor_threshold: float = 4.6
    actionability_ready_soon_min_trigger: float = 6.0
    actionability_ready_soon_min_timing: float = 5.6
    actionability_ready_soon_min_structure: float = 6.0
    execution_zone_max_width_pct: float = 0.028
    execution_zone_min_width_pct: float = 0.0045
    execution_breakout_zone_fraction: float = 0.38
    execution_pullback_zone_fraction: float = 0.44
    execution_deeper_zone_fraction: float = 0.34
    execution_zone_overlap_max_pct: float = 0.5
    execution_near_trigger_buffer_pct: float = 0.006
    execution_extended_above_trigger_pct: float = 0.022
    execution_range_near_high_pct: float = 0.16
    execution_range_near_low_pct: float = 0.18
    execution_breakout_zone_max_width_pct: float = 0.018
    execution_pullback_zone_max_width_pct: float = 0.024
    execution_deeper_zone_max_width_pct: float = 0.018
    execution_deeper_zone_min_gap_pct: float = 0.005
    execution_deeper_zone_drop_overlap_pct: float = 0.72
    execution_reanchor_above_prior_trigger_pct: float = 0.014
    execution_reanchor_above_range_pct: float = 0.018
    context_range_window_1m: int = 21
    context_range_window_3m: int = 63
    context_range_window_12m: int = 252
    context_local_range_window: int = 20
    context_expansion_window: int = 10
    context_near_high_position: float = 0.82
    context_near_low_position: float = 0.22
    context_extended_from_ema20_pct: float = 0.06
    context_compression_range_atr: float = 2.4
    context_expansion_range_atr: float = 4.5
    context_strong_catalyst_score: float = 6.2
    context_macro_alignment_supportive: float = 5.8
    context_macro_alignment_conflict: float = 4.2
    scenario_tp_aggressive_mult: float = 1.12
    scenario_tp_moderate_high_mult: float = 1.05
    scenario_tp_conservative_mult: float = 0.9
    scenario_stop_tight_mult: float = 0.9
    scenario_stop_tight_to_moderate_mult: float = 0.96
    scenario_stop_moderate_to_wide_mult: float = 1.08
    avoid_severity_threshold: float = 5.0
    avoid_severity_threshold_risk_off: float = 4.8
    avoid_bad_composite_gap: float = 1.7
    avoid_negative_expectancy_penalty: float = 1.35
    avoid_prob_penalty: float = 1.25
    avoid_no_confirmation_penalty: float = 0.95
    avoid_weak_bounce_penalty: float = 0.45
    avoid_negative_rs_penalty: float = 1.45
    avoid_poor_entry_penalty: float = 1.25
    avoid_weak_support_penalty: float = 0.85
    avoid_poor_rr_penalty: float = 0.95
    avoid_risk_off_weak_trend_penalty: float = 0.55
    avoid_downtrend_penalty: float = 2.6
    avoid_weak_breakdown_penalty: float = 1.55
    pre_scan_weights: dict[str, float] = field(
        default_factory=lambda: {
            "trend": 1.2,
            "relative_strength": 1.2,
            "sector_relative": 0.8,
            "pullback": 1.0,
            "volatility": 0.7,
            "volume": 0.55,
            "earnings": 0.75,
            "liquidity": 0.7,
        }
    )
    immediate_rank_weights: dict[str, float] = field(
        default_factory=lambda: {
            "composite": 1.7,
            "entry_quality": 0.45,
            "reward_risk_tp1": 1.4,
            "expected_return": 110.0,
            "probability_edge": 2.4,
            "confidence": 1.2,
            "suitability": 1.0,
            "pre_scan": 0.55,
            "sector_relative": 28.0,
        }
    )
    watchlist_rank_weights: dict[str, float] = field(
        default_factory=lambda: {
            "composite": 1.25,
            "entry_quality": 0.3,
            "reward_risk_tp1": 0.8,
            "expected_return": 70.0,
            "probability_edge": 1.25,
            "confidence": 0.7,
            "suitability": 1.45,
            "pre_scan": 0.8,
            "sector_relative": 24.0,
        }
    )
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "trend_quality": 1.25,
            "pullback_quality": 1.0,
            "support_quality": 1.05,
            "volatility_quality": 0.85,
            "relative_strength": 0.9,
            "volume_confirmation": 0.7,
            "earnings_risk": 0.85,
            "reward_risk": 1.25,
            "historical_analogue": 0.75,
            "entry_quality": 1.1,
            "llm_quality": 0.8,
            "context_score": 0.85,
            "catalyst_score": 0.7,
            "macro_score": 0.65,
            "scenario_score": 0.8,
        }
    )


DEFAULT_PLANNING_CONFIG = PlanningConfig()
