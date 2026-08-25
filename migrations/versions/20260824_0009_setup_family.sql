-- Canonical strategy identity for monitor history and setup-family learning.
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS setup_family VARCHAR(80);
ALTER TABLE monitor_daily_summaries ADD COLUMN IF NOT EXISTS setup_family VARCHAR(80);

CREATE INDEX IF NOT EXISTS ix_monitor_setups_setup_family ON monitor_setups (setup_family);
CREATE INDEX IF NOT EXISTS ix_monitor_daily_summaries_setup_family ON monitor_daily_summaries (setup_family);

UPDATE monitor_setups
SET setup_family = CASE
    WHEN setup_type IN ('healthy_pullback', 'constructive_pullback', 'pullback_in_uptrend') THEN 'healthy_pullback'
    WHEN setup_type IN ('controlled_momentum_continuation', 'continuation_breakout') THEN 'momentum_continuation'
    WHEN setup_type IN ('post_breakout_retest', 'breakout_retest') THEN 'breakout_retest'
    WHEN setup_type IN ('base_building', 'breakout', 'base_breakout') THEN 'base_breakout'
    WHEN setup_type = 'deep_pullback' THEN 'deep_pullback'
    ELSE 'reversal_attempt'
END
WHERE setup_family IS NULL;

UPDATE monitor_daily_summaries summary
SET setup_family = setup.setup_family
FROM monitor_setups setup
WHERE summary.setup_id = setup.id AND summary.setup_family IS NULL;
