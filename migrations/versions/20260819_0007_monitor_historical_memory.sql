-- Append-only evidence and aggregate versions for the advisory live monitor.
-- These tables are analytical records only and are never broker order queues.
CREATE TABLE IF NOT EXISTS monitor_bar_summaries (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL,
    market_snapshot_id VARCHAR(80), ticker VARCHAR(32) NOT NULL, timeframe VARCHAR(20) NOT NULL,
    bar_timestamp TIMESTAMPTZ NOT NULL, open_price DOUBLE PRECISION NOT NULL,
    high_price DOUBLE PRECISION NOT NULL, low_price DOUBLE PRECISION NOT NULL,
    close_price DOUBLE PRECISION NOT NULL, volume DOUBLE PRECISION, indicators_json TEXT,
    data_quality_flags_json TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_monitor_bar_summary UNIQUE (setup_id, timeframe, bar_timestamp)
);
CREATE INDEX IF NOT EXISTS ix_monitor_bar_summaries_ticker_time ON monitor_bar_summaries (ticker, bar_timestamp);

CREATE TABLE IF NOT EXISTS monitor_daily_summaries (
    id VARCHAR(80) PRIMARY KEY, trading_date DATE NOT NULL, watch_id VARCHAR(80) NOT NULL,
    setup_id VARCHAR(80) NOT NULL, market_snapshot_id VARCHAR(80), ticker VARCHAR(32) NOT NULL,
    open_price DOUBLE PRECISION, high_price DOUBLE PRECISION, low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION, starting_monitor_price DOUBLE PRECISION,
    ending_monitor_price DOUBLE PRECISION, broader_structure VARCHAR(80), setup_type VARCHAR(80),
    execution_structure VARCHAR(80), market_regime VARCHAR(60), sector VARCHAR(100),
    levels_json TEXT NOT NULL, indicators_json TEXT NOT NULL, context_json TEXT NOT NULL,
    decisions_json TEXT NOT NULL, outcome_json TEXT NOT NULL, data_quality_flags_json TEXT NOT NULL,
    number_of_trigger_attempts INTEGER NOT NULL DEFAULT 0,
    number_of_rejections INTEGER NOT NULL DEFAULT 0, highest_state_reached VARCHAR(40),
    mfe_atr DOUBLE PRECISION, mae_atr DOUBLE PRECISION,
    recommendation_r_multiple DOUBLE PRECISION, actual_trade_executed BOOLEAN NOT NULL DEFAULT FALSE,
    actual_trade_r_multiple DOUBLE PRECISION, finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_monitor_daily_setup_date UNIQUE (setup_id, trading_date)
);
CREATE INDEX IF NOT EXISTS ix_monitor_daily_ticker_date ON monitor_daily_summaries (ticker, trading_date);

CREATE TABLE IF NOT EXISTS behavior_profile_versions (
    id VARCHAR(80) PRIMARY KEY, profile_id VARCHAR(80), scope_type VARCHAR(40) NOT NULL,
    scope_value VARCHAR(160) NOT NULL, version INTEGER NOT NULL, observation_count INTEGER NOT NULL DEFAULT 0,
    weighted_observation_count DOUBLE PRECISION NOT NULL DEFAULT 0,
    evidence_strength VARCHAR(30) NOT NULL DEFAULT 'INSUFFICIENT', reliability DOUBLE PRECISION NOT NULL DEFAULT 0,
    statistics_json TEXT NOT NULL, formula_version VARCHAR(60) NOT NULL DEFAULT 'historical-memory-v1',
    source_cutoff_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_behavior_profile_version UNIQUE (scope_type, scope_value, version)
);
CREATE INDEX IF NOT EXISTS ix_behavior_profile_scope_created ON behavior_profile_versions (scope_type, scope_value, created_at);

CREATE TABLE IF NOT EXISTS learned_adjustments (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL,
    market_snapshot_id VARCHAR(80), ticker VARCHAR(32) NOT NULL, adjustment_type VARCHAR(80) NOT NULL,
    base_value DOUBLE PRECISION, learned_value DOUBLE PRECISION,
    adjustment_value DOUBLE PRECISION NOT NULL DEFAULT 0, adjustment_strength DOUBLE PRECISION NOT NULL DEFAULT 0,
    evidence_strength VARCHAR(30) NOT NULL DEFAULT 'INSUFFICIENT', sample_size INTEGER NOT NULL DEFAULT 0,
    weighted_sample_size DOUBLE PRECISION NOT NULL DEFAULT 0, reason TEXT NOT NULL,
    supporting_stats_json TEXT NOT NULL, bounds_json TEXT NOT NULL, profile_version_id VARCHAR(80),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_learned_adjustments_setup_created ON learned_adjustments (setup_id, created_at);

CREATE TABLE IF NOT EXISTS level_revisions (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL,
    chart_review_id VARCHAR(80), market_snapshot_id VARCHAR(80), ticker VARCHAR(32) NOT NULL,
    level_name VARCHAR(60) NOT NULL, level_role VARCHAR(60) NOT NULL,
    planner_price DOUBLE PRECISION, llm_proposed_price DOUBLE PRECISION,
    validated_price DOUBLE PRECISION, manual_price DOUBLE PRECISION, final_active_price DOUBLE PRECISION,
    source VARCHAR(40) NOT NULL, validation_result VARCHAR(40) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0, reason TEXT,
    anomaly_flags_json TEXT NOT NULL, outcome_json TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_level_revisions_setup_level ON level_revisions (setup_id, level_name, created_at);

CREATE TABLE IF NOT EXISTS learning_job_runs (
    id VARCHAR(80) PRIMARY KEY, trading_date DATE NOT NULL, status VARCHAR(30) NOT NULL,
    summaries_finalized INTEGER NOT NULL DEFAULT 0, profiles_updated INTEGER NOT NULL DEFAULT 0,
    observations_created INTEGER NOT NULL DEFAULT 0, details_json TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_learning_job_runs_date ON learning_job_runs (trading_date, started_at);

CREATE TABLE IF NOT EXISTS llm_decision_postmortems (
    id VARCHAR(80) PRIMARY KEY, llm_review_id VARCHAR(80) NOT NULL UNIQUE,
    watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, ticker VARCHAR(32) NOT NULL,
    outcome_type VARCHAR(40) NOT NULL, original_decision VARCHAR(20) NOT NULL,
    outcome_json TEXT NOT NULL, rationale_tags_json TEXT NOT NULL, lessons_json TEXT NOT NULL,
    model VARCHAR(100), prompt_version VARCHAR(60) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_llm_postmortems_ticker ON llm_decision_postmortems (ticker, created_at);

-- Existing paper-test rows remain valid; these fields keep production and
-- hypothetical outcomes explicitly separate when the shadow test resolves.
ALTER TABLE shadow_rule_evaluations
    ADD COLUMN IF NOT EXISTS production_outcome VARCHAR(80);
ALTER TABLE shadow_rule_evaluations
    ADD COLUMN IF NOT EXISTS shadow_hypothetical_outcome VARCHAR(80);
ALTER TABLE shadow_rule_evaluations
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS ix_shadow_rule_production_outcome
    ON shadow_rule_evaluations (production_outcome);
CREATE INDEX IF NOT EXISTS ix_shadow_rule_hypothetical_outcome
    ON shadow_rule_evaluations (shadow_hypothetical_outcome);
