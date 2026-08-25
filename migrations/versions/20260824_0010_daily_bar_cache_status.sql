-- Persist provider mapping and incremental S&P 500 daily-bar ingestion state.
CREATE TABLE IF NOT EXISTS daily_bar_cache_status (
    canonical_symbol VARCHAR(20) PRIMARY KEY,
    provider_symbol VARCHAR(30),
    provider VARCHAR(20),
    last_bar_date DATE,
    row_count INTEGER NOT NULL DEFAULT 0,
    data_source VARCHAR(30),
    freshness_status VARCHAR(40) NOT NULL DEFAULT 'CACHE_MISSING',
    history_sufficient BOOLEAN NOT NULL DEFAULT FALSE,
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_attempt_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_error_code VARCHAR(50),
    last_error_detail TEXT
);

CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_provider ON daily_bar_cache_status (provider);
CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_last_bar_date ON daily_bar_cache_status (last_bar_date);
CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_freshness_status ON daily_bar_cache_status (freshness_status);
CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_history_sufficient ON daily_bar_cache_status (history_sufficient);
CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_last_updated_at ON daily_bar_cache_status (last_updated_at);
CREATE INDEX IF NOT EXISTS ix_daily_bar_cache_status_last_error_code ON daily_bar_cache_status (last_error_code);
