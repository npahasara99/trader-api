-- Preserve descriptive workflow identifiers in operational decision history.
-- Safe to apply repeatedly on PostgreSQL.
alter table if exists swing_decisions
    alter column mode type varchar(80);
