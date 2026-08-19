-- Canonical market-data snapshots used to keep planner, chart, LLM, and monitor state synchronized.
CREATE TABLE IF NOT EXISTS market_snapshots (
    id VARCHAR(80) PRIMARY KEY,
    ticker VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quote_timestamp TIMESTAMPTZ,
    reference_price DOUBLE PRECISION,
    data_source VARCHAR(160),
    daily_last_bar_at TIMESTAMPTZ,
    hourly_last_bar_at TIMESTAMPTZ,
    thirty_min_last_bar_at TIMESTAMPTZ,
    five_min_last_bar_at TIMESTAMPTZ,
    one_min_last_bar_at TIMESTAMPTZ,
    consistency_status VARCHAR(40) NOT NULL DEFAULT 'CONSISTENT',
    cache_status VARCHAR(40),
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_market_snapshots_ticker_created ON market_snapshots (ticker, created_at);
CREATE INDEX IF NOT EXISTS ix_market_snapshots_consistency_status ON market_snapshots (consistency_status);

ALTER TABLE live_watches ADD COLUMN IF NOT EXISTS market_snapshot_id VARCHAR(80);
ALTER TABLE live_watches ADD COLUMN IF NOT EXISTS last_backend_evaluation_at TIMESTAMPTZ;
ALTER TABLE live_watches ADD COLUMN IF NOT EXISTS last_market_data_update_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS ix_live_watches_market_snapshot_id ON live_watches (market_snapshot_id);

ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS market_snapshot_id VARCHAR(80);
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_reference_price DOUBLE PRECISION;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_created_at TIMESTAMPTZ;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS market_data_timestamp TIMESTAMPTZ;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_stale BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_stale_reasons_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS previous_setup_id VARCHAR(80);
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS replacement_reason TEXT;
CREATE INDEX IF NOT EXISTS ix_monitor_setups_market_snapshot_id ON monitor_setups (market_snapshot_id);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_plan_stale ON monitor_setups (plan_stale);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_previous_setup_id ON monitor_setups (previous_setup_id);

ALTER TABLE chart_snapshots ADD COLUMN IF NOT EXISTS market_snapshot_id VARCHAR(80);
ALTER TABLE chart_structure_reviews ADD COLUMN IF NOT EXISTS market_snapshot_id VARCHAR(80);
CREATE INDEX IF NOT EXISTS ix_chart_snapshots_market_snapshot_id ON chart_snapshots (market_snapshot_id);
CREATE INDEX IF NOT EXISTS ix_chart_structure_reviews_market_snapshot_id ON chart_structure_reviews (market_snapshot_id);
