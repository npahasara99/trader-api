-- Persistent advisory-only live swing monitor. No table is an execution queue.
CREATE TABLE IF NOT EXISTS live_watches (
    id VARCHAR(80) PRIMARY KEY, ticker VARCHAR(32) NOT NULL UNIQUE, source VARCHAR(40) NOT NULL DEFAULT 'manual',
    monitor_active BOOLEAN NOT NULL DEFAULT TRUE, state VARCHAR(40) NOT NULL DEFAULT 'WATCHING',
    current_setup_id VARCHAR(80), current_price DOUBLE PRECISION, market_data_as_of TIMESTAMPTZ,
    session_label VARCHAR(24), last_event TEXT, latest_evaluation_json TEXT, last_polled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), removed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_live_watches_active_state ON live_watches (monitor_active, state);

CREATE TABLE IF NOT EXISTS monitor_setups (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, ticker VARCHAR(32) NOT NULL, version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(40) NOT NULL DEFAULT 'active', valid_setup BOOLEAN NOT NULL DEFAULT TRUE,
    setup_quality_score DOUBLE PRECISION, broader_structure VARCHAR(80), setup_type VARCHAR(80), execution_structure VARCHAR(80),
    sector VARCHAR(100), industry VARCHAR(160), market_regime VARCHAR(60), planner_baseline_json TEXT NOT NULL,
    planner_levels_json TEXT NOT NULL, active_levels_json TEXT NOT NULL, manual_overrides_json TEXT,
    trigger_source VARCHAR(20) NOT NULL DEFAULT 'PLANNER', max_chase_price DOUBLE PRECISION, expires_at TIMESTAMPTZ,
    invalidated_at TIMESTAMPTZ, invalidation_price DOUBLE PRECISION, invalidation_reason TEXT,
    replaced_by_setup_id VARCHAR(80), rule_version VARCHAR(60) NOT NULL DEFAULT 'live-monitor-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_monitor_setups_watch_version UNIQUE (watch_id, version)
);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_watch_status ON monitor_setups (watch_id, status);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_ticker ON monitor_setups (ticker);

CREATE TABLE IF NOT EXISTS confirmation_attempts (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, ticker VARCHAR(32) NOT NULL,
    attempt_number INTEGER NOT NULL, started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), ended_at TIMESTAMPTZ,
    trigger_price DOUBLE PRECISION, peak_price DOUBLE PRECISION, lowest_retest_price DOUBLE PRECISION,
    rvol_1m DOUBLE PRECISION, rvol_5m DOUBLE PRECISION, price_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
    volume_confirmation BOOLEAN NOT NULL DEFAULT FALSE, retest_result VARCHAR(60), confirmation_method VARCHAR(60),
    outcome VARCHAR(60), rejection_reason TEXT, evidence_json TEXT,
    CONSTRAINT uq_confirmation_attempt_number UNIQUE (setup_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS ix_confirmation_attempts_setup_started ON confirmation_attempts (setup_id, started_at);

CREATE TABLE IF NOT EXISTS monitor_events (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80), attempt_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL, event_type VARCHAR(80) NOT NULL, from_state VARCHAR(40), to_state VARCHAR(40),
    message TEXT NOT NULL, snapshot_json TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_monitor_events_watch_created ON monitor_events (watch_id, created_at);

CREATE TABLE IF NOT EXISTS monitor_decision_snapshots (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, attempt_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL, snapshot_type VARCHAR(60) NOT NULL, payload_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_monitor_decision_snapshots_setup ON monitor_decision_snapshots (setup_id);

CREATE TABLE IF NOT EXISTS llm_advisory_reviews (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, attempt_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL, model VARCHAR(100), prompt_version VARCHAR(60) NOT NULL DEFAULT 'live-advisor-v1',
    decision VARCHAR(20) NOT NULL, confidence DOUBLE PRECISION NOT NULL DEFAULT 0, status VARCHAR(40) NOT NULL DEFAULT 'available',
    reason_summary TEXT, input_snapshot_json TEXT NOT NULL, output_json TEXT NOT NULL, hard_blockers_json TEXT,
    final_user_action VARCHAR(40), actual_outcome_json TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_llm_advisory_reviews_setup ON llm_advisory_reviews (setup_id, created_at);

CREATE TABLE IF NOT EXISTS manual_monitor_trades (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, attempt_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL, status VARCHAR(40) NOT NULL, quantity DOUBLE PRECISION, planned_entry DOUBLE PRECISION,
    actual_entry DOUBLE PRECISION, stop_price DOUBLE PRECISION, targets_json TEXT, entered_at TIMESTAMPTZ,
    exited_at TIMESTAMPTZ, exit_price DOUBLE PRECISION, realised_pnl DOUBLE PRECISION, r_multiple DOUBLE PRECISION,
    mfe_pct DOUBLE PRECISION, mae_pct DOUBLE PRECISION, notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_manual_monitor_trades_watch_status ON manual_monitor_trades (watch_id, status);

CREATE TABLE IF NOT EXISTS recommendation_outcomes (
    id VARCHAR(80) PRIMARY KEY, watch_id VARCHAR(80) NOT NULL, setup_id VARCHAR(80) NOT NULL, attempt_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL, user_action VARCHAR(40) NOT NULL, outcome VARCHAR(60),
    entry_distance_from_trigger_pct DOUBLE PRECISION, mfe_pct DOUBLE PRECISION, mae_pct DOUBLE PRECISION,
    r_multiple DOUBLE PRECISION, details_json TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_recommendation_outcomes_ticker ON recommendation_outcomes (ticker, created_at);

CREATE TABLE IF NOT EXISTS stock_behavior_profiles (
    id VARCHAR(80) PRIMARY KEY, scope_type VARCHAR(40) NOT NULL, scope_value VARCHAR(160) NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 0, evidence_strength VARCHAR(30) NOT NULL DEFAULT 'INSUFFICIENT',
    statistics_json TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_stock_behavior_scope UNIQUE (scope_type, scope_value)
);

CREATE TABLE IF NOT EXISTS learning_observations (
    id VARCHAR(80) PRIMARY KEY, scope_type VARCHAR(40) NOT NULL, scope_value VARCHAR(160) NOT NULL,
    observation_type VARCHAR(80) NOT NULL, summary TEXT NOT NULL, sample_size INTEGER NOT NULL DEFAULT 0,
    evidence_strength VARCHAR(30) NOT NULL DEFAULT 'INSUFFICIENT', evidence_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_proposals (
    id VARCHAR(80) PRIMARY KEY, observation_id VARCHAR(80), scope_type VARCHAR(40) NOT NULL, scope_value VARCHAR(160) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING', title TEXT NOT NULL, proposed_change_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL, decided_at TIMESTAMPTZ, decided_by VARCHAR(80), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitor_rule_versions (
    id VARCHAR(80) PRIMARY KEY, version VARCHAR(60) NOT NULL UNIQUE, status VARCHAR(30) NOT NULL, proposal_id VARCHAR(80),
    rules_json TEXT NOT NULL, approved_by VARCHAR(80), approved_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shadow_rule_evaluations (
    id VARCHAR(80) PRIMARY KEY, proposal_id VARCHAR(80) NOT NULL, watch_id VARCHAR(80), setup_id VARCHAR(80),
    production_decision VARCHAR(40) NOT NULL, shadow_decision VARCHAR(40) NOT NULL, evidence_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_shadow_rule_evaluations_proposal ON shadow_rule_evaluations (proposal_id, created_at);
