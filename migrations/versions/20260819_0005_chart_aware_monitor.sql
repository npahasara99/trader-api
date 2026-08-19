-- Chart-aware advisory monitor. Images and decisions are evidence only; no table is an order queue.
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS llm_proposed_levels_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS validated_chart_levels_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS level_sources_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS chart_analysis_status VARCHAR(40) NOT NULL DEFAULT 'NOT_RUN';
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS latest_chart_review_id VARCHAR(80);
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_stale_reason TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS proposed_setup_json TEXT;
CREATE INDEX IF NOT EXISTS ix_monitor_setups_chart_analysis_status ON monitor_setups (chart_analysis_status);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_latest_chart_review_id ON monitor_setups (latest_chart_review_id);

CREATE TABLE IF NOT EXISTS chart_snapshots (
    id VARCHAR(80) PRIMARY KEY,
    watch_id VARCHAR(80) NOT NULL,
    setup_id VARCHAR(80) NOT NULL,
    decision_event_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL,
    timeframe VARCHAR(24) NOT NULL,
    event_type VARCHAR(80) NOT NULL,
    image_path TEXT NOT NULL,
    image_data_base64 TEXT,
    content_hash VARCHAR(80) NOT NULL UNIQUE,
    data_source VARCHAR(80),
    data_last_bar_at TIMESTAMPTZ,
    decision_time_boundary TIMESTAMPTZ NOT NULL,
    metadata_json TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retain_permanently BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS ix_chart_snapshots_setup_event ON chart_snapshots (setup_id, event_type, generated_at);
CREATE INDEX IF NOT EXISTS ix_chart_snapshots_watch_id ON chart_snapshots (watch_id);
CREATE INDEX IF NOT EXISTS ix_chart_snapshots_ticker ON chart_snapshots (ticker);

CREATE TABLE IF NOT EXISTS chart_structure_reviews (
    id VARCHAR(80) PRIMARY KEY,
    watch_id VARCHAR(80) NOT NULL,
    setup_id VARCHAR(80) NOT NULL,
    ticker VARCHAR(32) NOT NULL,
    review_type VARCHAR(60) NOT NULL,
    status VARCHAR(40) NOT NULL,
    model VARCHAR(100),
    prompt_version VARCHAR(80) NOT NULL,
    chart_snapshot_ids_json TEXT NOT NULL,
    deterministic_input_json TEXT NOT NULL,
    planner_levels_json TEXT NOT NULL,
    llm_output_json TEXT NOT NULL,
    llm_proposed_levels_json TEXT NOT NULL,
    validated_levels_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    decision VARCHAR(40) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    reason_summary TEXT,
    data_consistency_status VARCHAR(40) NOT NULL DEFAULT 'CONSISTENT',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_chart_structure_reviews_watch_id ON chart_structure_reviews (watch_id);
CREATE INDEX IF NOT EXISTS ix_chart_structure_reviews_setup_id ON chart_structure_reviews (setup_id);
CREATE INDEX IF NOT EXISTS ix_chart_structure_reviews_ticker ON chart_structure_reviews (ticker);
CREATE INDEX IF NOT EXISTS ix_chart_structure_reviews_type_status ON chart_structure_reviews (review_type, status, created_at);

CREATE TABLE IF NOT EXISTS chart_level_decisions (
    id VARCHAR(80) PRIMARY KEY,
    watch_id VARCHAR(80) NOT NULL,
    setup_id VARCHAR(80) NOT NULL,
    chart_review_id VARCHAR(80),
    ticker VARCHAR(32) NOT NULL,
    decision VARCHAR(40) NOT NULL,
    previous_active_levels_json TEXT NOT NULL,
    selected_levels_json TEXT NOT NULL,
    level_sources_json TEXT NOT NULL,
    decided_by VARCHAR(80) NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_chart_level_decisions_watch_id ON chart_level_decisions (watch_id);
CREATE INDEX IF NOT EXISTS ix_chart_level_decisions_setup_id ON chart_level_decisions (setup_id);
CREATE INDEX IF NOT EXISTS ix_chart_level_decisions_review_id ON chart_level_decisions (chart_review_id);
CREATE INDEX IF NOT EXISTS ix_chart_level_decisions_ticker ON chart_level_decisions (ticker);
