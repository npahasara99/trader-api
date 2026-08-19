from __future__ import annotations

from dataclasses import dataclass

from app.settings import settings


@dataclass(frozen=True)
class LiveMonitorConfig:
    enabled: bool = True
    poll_interval_seconds: int = 60
    stale_data_seconds: int = 180
    near_trigger_distance_pct: float = 0.005
    volume_lookback_bars: int = 20
    constructive_rvol: float = 1.20
    strong_rvol: float = 1.50
    max_upper_wick_ratio: float = 0.35
    min_close_location: float = 0.60
    minimum_current_rr: float = 1.25
    max_chase_pct: float = 0.005
    max_chase_atr_fraction: float = 0.35
    retest_tolerance_atr_fraction: float = 0.15
    auto_llm_min_setup_score: float = 8.0
    learning_min_observations: int = 12
    ticker_maturity_samples: int = 20
    setup_maturity_samples: int = 30
    sector_maturity_samples: int = 40
    trigger_max_atr: float = 3.0
    trigger_max_range_fraction: float = 0.65
    chart_max_stop_atr: float = 3.5
    chart_max_target_atr: float = 5.0
    plan_max_age_days: int = 15
    auto_propose_reanalysis_on_stale: bool = True
    chart_review_on_add: bool = True
    chart_review_model: str = "gpt-5"
    chart_review_cooldown_seconds: int = 900
    max_auto_chart_reviews_per_day: int = 3
    chart_snapshot_dir: str = "chart_snapshots"
    chart_max_bars: int = 180
    chart_retention_days: int = 90


def load_live_monitor_config() -> LiveMonitorConfig:
    return LiveMonitorConfig(
        enabled=bool(settings.LIVE_MONITOR_ENABLED),
        poll_interval_seconds=max(15, int(settings.LIVE_MONITOR_POLL_SECONDS)),
        stale_data_seconds=max(30, int(settings.LIVE_MONITOR_STALE_SECONDS)),
        near_trigger_distance_pct=max(0.0005, float(settings.LIVE_MONITOR_NEAR_TRIGGER_PCT)),
        volume_lookback_bars=max(5, int(settings.LIVE_MONITOR_VOLUME_LOOKBACK_BARS)),
        constructive_rvol=max(1.0, float(settings.LIVE_MONITOR_CONSTRUCTIVE_RVOL)),
        strong_rvol=max(1.0, float(settings.LIVE_MONITOR_STRONG_RVOL)),
        max_upper_wick_ratio=min(0.8, max(0.05, float(settings.LIVE_MONITOR_MAX_UPPER_WICK_RATIO))),
        min_close_location=min(0.95, max(0.05, float(settings.LIVE_MONITOR_MIN_CLOSE_LOCATION))),
        minimum_current_rr=max(0.5, float(settings.LIVE_MONITOR_MIN_CURRENT_RR)),
        max_chase_pct=max(0.001, float(settings.LIVE_MONITOR_MAX_CHASE_PCT)),
        max_chase_atr_fraction=max(0.05, float(settings.LIVE_MONITOR_MAX_CHASE_ATR_FRACTION)),
        retest_tolerance_atr_fraction=max(0.01, float(settings.LIVE_MONITOR_RETEST_ATR_FRACTION)),
        auto_llm_min_setup_score=max(0.0, float(settings.LIVE_MONITOR_AUTO_LLM_MIN_SCORE)),
        learning_min_observations=max(5, int(settings.LIVE_MONITOR_LEARNING_MIN_OBSERVATIONS)),
        trigger_max_atr=max(1.0, float(settings.LIVE_MONITOR_TRIGGER_MAX_ATR)),
        trigger_max_range_fraction=max(0.1, min(1.5, float(settings.LIVE_MONITOR_TRIGGER_MAX_RANGE_FRACTION))),
        chart_max_stop_atr=max(1.0, float(settings.LIVE_MONITOR_CHART_MAX_STOP_ATR)),
        chart_max_target_atr=max(1.0, float(settings.LIVE_MONITOR_CHART_MAX_TARGET_ATR)),
        plan_max_age_days=max(1, int(settings.LIVE_MONITOR_PLAN_MAX_AGE_DAYS)),
        auto_propose_reanalysis_on_stale=bool(settings.LIVE_MONITOR_AUTO_PROPOSE_REANALYSIS_ON_STALE),
        chart_review_on_add=bool(settings.LIVE_MONITOR_CHART_REVIEW_ON_ADD),
        chart_review_model=str(settings.LIVE_MONITOR_CHART_REVIEW_MODEL),
        chart_review_cooldown_seconds=max(60, int(settings.LIVE_MONITOR_CHART_REVIEW_COOLDOWN_SECONDS)),
        max_auto_chart_reviews_per_day=max(1, int(settings.LIVE_MONITOR_MAX_AUTO_CHART_REVIEWS_PER_DAY)),
        chart_snapshot_dir=str(settings.LIVE_MONITOR_CHART_SNAPSHOT_DIR),
        chart_max_bars=max(40, int(settings.LIVE_MONITOR_CHART_MAX_BARS)),
        chart_retention_days=max(7, int(settings.LIVE_MONITOR_CHART_RETENTION_DAYS)),
    )
