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
    earnings_penalty_near_days: int = 7
    earnings_penalty_mid_days: int = 14
    earnings_hard_block_days: int = 3
    volume_window: int = 20
    relative_strength_window: int = 40
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ")
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
        }
    )


DEFAULT_PLANNING_CONFIG = PlanningConfig()
