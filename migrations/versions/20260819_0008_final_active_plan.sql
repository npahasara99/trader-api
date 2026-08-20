-- One authoritative, validated live-monitor plan and explicit reconciliation state.
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS final_active_plan_id VARCHAR(80);
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS final_active_plan_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS final_plan_validation_json TEXT;
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS plan_integrity_status VARCHAR(20) NOT NULL DEFAULT 'INVALID';
ALTER TABLE monitor_setups ADD COLUMN IF NOT EXISTS reconciliation_status VARCHAR(40) NOT NULL DEFAULT 'PLANNER_ACCEPTED';

CREATE INDEX IF NOT EXISTS ix_monitor_setups_final_active_plan_id ON monitor_setups (final_active_plan_id);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_plan_integrity_status ON monitor_setups (plan_integrity_status);
CREATE INDEX IF NOT EXISTS ix_monitor_setups_reconciliation_status ON monitor_setups (reconciliation_status);
