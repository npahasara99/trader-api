from __future__ import annotations

from dataclasses import dataclass, field
import json

from app.settings import settings


DEFAULT_SIMILARITY_WEIGHTS = {
    "ticker": 2.0, "broader_structure": 1.25, "setup_type": 2.0,
    "execution_structure": 1.5, "sector": 0.75, "market_regime": 1.0,
    "confirmation_method": 1.25, "attempt_number": 0.5,
    "qqq_condition": 0.5, "sector_condition": 0.5,
}
DEFAULT_SIMILARITY_CONTINUOUS = {
    "atr_pct": (1.0, 0.03), "rsi": (0.75, 25.0),
    "distance_from_support_atr": (1.0, 2.0),
    "primary_trigger_distance_atr": (1.0, 2.5), "rvol_5m": (1.0, 1.5),
}


def _json_mapping(raw: str, fallback: dict[str, float]) -> dict[str, float]:
    try:
        parsed = json.loads(raw)
        return {str(key): max(0.0, float(value)) for key, value in parsed.items()}
    except (TypeError, ValueError, AttributeError):
        return dict(fallback)


def _continuous_mapping(raw: str) -> dict[str, tuple[float, float]]:
    try:
        parsed = json.loads(raw)
        return {
            str(key): (max(0.0, float(value[0])), max(1e-9, float(value[1])))
            for key, value in parsed.items()
            if isinstance(value, (list, tuple)) and len(value) == 2
        } or dict(DEFAULT_SIMILARITY_CONTINUOUS)
    except (TypeError, ValueError, AttributeError):
        return dict(DEFAULT_SIMILARITY_CONTINUOUS)


def _evidence_thresholds(raw: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
        if len(values) != 4 or any(value <= 0 for value in values) or list(values) != sorted(values):
            raise ValueError
        return values
    except (TypeError, ValueError, AttributeError):
        return (8.0, 15.0, 30.0, 60.0)


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
    source_plan_max_age_minutes: int = 120
    plan_price_drift_pct: float = 0.03
    plan_price_drift_atr: float = 1.50
    support_failure_atr: float = 0.35
    major_gap_atr: float = 1.50
    new_structure_lookback_bars: int = 12
    market_data_mismatch_pct: float = 0.0125
    market_data_mismatch_atr_fraction: float = 0.50
    auto_propose_reanalysis_on_stale: bool = True
    chart_review_on_add: bool = True
    chart_review_model: str = "gpt-5"
    chart_review_cooldown_seconds: int = 900
    max_auto_chart_reviews_per_day: int = 3
    chart_snapshot_dir: str = "chart_snapshots"
    chart_max_bars: int = 180
    chart_retention_days: int = 90
    level_auto_correct_confidence: float = 0.90
    level_auto_correct_enabled: bool = True
    primary_max_distance_pct: float = 0.06
    target_reachability_atr: float = 4.0
    learning_recency_half_life_days: float = 120.0
    ticker_prior_strength: float = 20.0
    setup_prior_strength: float = 30.0
    sector_prior_strength: float = 40.0
    regime_prior_strength: float = 40.0
    max_historical_score_adjustment: float = 1.0
    max_rvol_threshold_adjustment: float = 0.25
    max_chase_adjustment_pct: float = 0.0025
    max_target_expectation_adjustment_atr: float = 0.75
    similar_case_count: int = 8
    evidence_thresholds: tuple[float, float, float, float] = (8.0, 15.0, 30.0, 60.0)
    similarity_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIMILARITY_WEIGHTS))
    similarity_continuous: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_SIMILARITY_CONTINUOUS))
    profile_refresh_hour_et: int = 20
    profile_refresh_minute_et: int = 15
    bar_retention_days: int = 365
    proposal_min_effect_r: float = 0.30
    paper_test_min_observations: int = 20


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
        source_plan_max_age_minutes=max(1, int(settings.LIVE_MONITOR_SOURCE_PLAN_MAX_AGE_MINUTES)),
        plan_price_drift_pct=max(0.001, float(settings.LIVE_MONITOR_PLAN_PRICE_DRIFT_PCT)),
        plan_price_drift_atr=max(0.25, float(settings.LIVE_MONITOR_PLAN_PRICE_DRIFT_ATR)),
        support_failure_atr=max(0.05, float(settings.LIVE_MONITOR_SUPPORT_FAILURE_ATR)),
        major_gap_atr=max(0.50, float(settings.LIVE_MONITOR_MAJOR_GAP_ATR)),
        new_structure_lookback_bars=max(5, int(settings.LIVE_MONITOR_NEW_STRUCTURE_LOOKBACK_BARS)),
        market_data_mismatch_pct=max(0.0025, float(settings.LIVE_MONITOR_MARKET_DATA_MISMATCH_PCT)),
        market_data_mismatch_atr_fraction=max(0.10, float(settings.LIVE_MONITOR_MARKET_DATA_MISMATCH_ATR_FRACTION)),
        auto_propose_reanalysis_on_stale=bool(settings.LIVE_MONITOR_AUTO_PROPOSE_REANALYSIS_ON_STALE),
        chart_review_on_add=bool(settings.LIVE_MONITOR_CHART_REVIEW_ON_ADD),
        chart_review_model=str(settings.LIVE_MONITOR_CHART_REVIEW_MODEL),
        chart_review_cooldown_seconds=max(60, int(settings.LIVE_MONITOR_CHART_REVIEW_COOLDOWN_SECONDS)),
        max_auto_chart_reviews_per_day=max(1, int(settings.LIVE_MONITOR_MAX_AUTO_CHART_REVIEWS_PER_DAY)),
        chart_snapshot_dir=str(settings.LIVE_MONITOR_CHART_SNAPSHOT_DIR),
        chart_max_bars=max(40, int(settings.LIVE_MONITOR_CHART_MAX_BARS)),
        chart_retention_days=max(7, int(settings.LIVE_MONITOR_CHART_RETENTION_DAYS)),
        level_auto_correct_confidence=min(1.0, max(0.5, float(settings.LIVE_MONITOR_LEVEL_AUTO_CORRECT_CONFIDENCE))),
        level_auto_correct_enabled=bool(settings.LIVE_MONITOR_LEVEL_AUTO_CORRECT_ENABLED),
        primary_max_distance_pct=max(0.01, float(settings.LIVE_MONITOR_PRIMARY_MAX_DISTANCE_PCT)),
        target_reachability_atr=max(1.0, float(settings.LIVE_MONITOR_TARGET_REACHABILITY_ATR)),
        learning_recency_half_life_days=max(14.0, float(settings.LIVE_MONITOR_LEARNING_RECENCY_HALF_LIFE_DAYS)),
        ticker_prior_strength=max(1.0, float(settings.LIVE_MONITOR_LEARNING_TICKER_PRIOR_STRENGTH)),
        setup_prior_strength=max(1.0, float(settings.LIVE_MONITOR_LEARNING_SETUP_PRIOR_STRENGTH)),
        sector_prior_strength=max(1.0, float(settings.LIVE_MONITOR_LEARNING_SECTOR_PRIOR_STRENGTH)),
        regime_prior_strength=max(1.0, float(settings.LIVE_MONITOR_LEARNING_REGIME_PRIOR_STRENGTH)),
        max_historical_score_adjustment=max(0.0, float(settings.LIVE_MONITOR_MAX_HISTORICAL_SCORE_ADJUSTMENT)),
        max_rvol_threshold_adjustment=max(0.0, float(settings.LIVE_MONITOR_MAX_RVOL_THRESHOLD_ADJUSTMENT)),
        max_chase_adjustment_pct=max(0.0, float(settings.LIVE_MONITOR_MAX_CHASE_ADJUSTMENT_PCT)),
        max_target_expectation_adjustment_atr=max(0.0, float(settings.LIVE_MONITOR_MAX_TARGET_EXPECTATION_ADJUSTMENT_ATR)),
        similar_case_count=max(1, min(25, int(settings.LIVE_MONITOR_SIMILAR_CASE_COUNT))),
        evidence_thresholds=_evidence_thresholds(settings.LIVE_MONITOR_EVIDENCE_THRESHOLDS),
        similarity_weights=_json_mapping(settings.LIVE_MONITOR_SIMILARITY_WEIGHTS_JSON, DEFAULT_SIMILARITY_WEIGHTS),
        similarity_continuous=_continuous_mapping(settings.LIVE_MONITOR_SIMILARITY_CONTINUOUS_JSON),
        profile_refresh_hour_et=min(23, max(0, int(settings.LIVE_MONITOR_PROFILE_REFRESH_HOUR_ET))),
        profile_refresh_minute_et=min(59, max(0, int(settings.LIVE_MONITOR_PROFILE_REFRESH_MINUTE_ET))),
        bar_retention_days=max(30, int(settings.LIVE_MONITOR_BAR_RETENTION_DAYS)),
        proposal_min_effect_r=max(0.05, float(settings.LIVE_MONITOR_PROPOSAL_MIN_EFFECT_R)),
        paper_test_min_observations=max(5, int(settings.LIVE_MONITOR_PAPER_TEST_MIN_OBSERVATIONS)),
    )
