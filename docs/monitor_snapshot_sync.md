# Live Monitor Snapshot Synchronization

## Root Cause

The previous monitor activation path promoted `planner_payload` directly through
`build_monitor_baseline()` and returned an already-active monitor without a
freshness check. Chart generation later fetched OHLCV independently. A scanner
plan based on an older price regime could therefore remain active while the
chart displayed newer bars.

The legacy schema did not store a plan reference price, market-data timestamp,
or snapshot ID. Consequently, the exact production timestamp/provider behind a
historical level such as INTC 107.91 cannot be reconstructed after the fact
unless it survives in a scanner response or decision-journal payload. New plans
are traceable by construction.

## Canonical Flow

Monitor creation and reanalysis now use this order:

1. Bypass the market-data TTL and fetch daily, hourly, 30m, 5m, and 1m bars.
2. Validate session-aware freshness and intraday price consistency.
3. Persist an immutable `market_snapshots` row.
4. Validate a supplied scanner plan against that snapshot, or run a fresh plan.
5. Persist `plan_reference_price`, plan/data timestamps, and snapshot ID.
6. Build charts and chart-review packets from the persisted snapshot bars.
7. Require planner, chart, and baseline snapshot IDs to match.

Scanner plans are reusable context only when price drift, ATR drift, support,
age, and data consistency all pass. Explicit reanalysis never reuses old
geometry.

## Staleness

A plan becomes stale when any configured hard signal applies:

- absolute price drift exceeds `LIVE_MONITOR_PLAN_PRICE_DRIFT_PCT`;
- ATR-normalized drift exceeds `LIVE_MONITOR_PLAN_PRICE_DRIFT_ATR`;
- structural invalidation or support fails;
- a major gap or new non-overlapping local structure appears;
- plan age exceeds its validity window;
- canonical data is inconsistent; or
- a legacy setup lacks a plan reference price.

Distant triggers and targets produce sanity warnings. Support above current
price is reclassified as `OLD_SUPPORT_LOST`. A stale setup exposes old levels
only under `historical_stale_levels`; active levels, executable R:R, order-plan
generation, and LLM approval are disabled until reanalysis.

## Cache Policy

- Creation/reanalysis: provider cache bypass requested (`cache_ttl_seconds=0`).
- Active 1m/5m polling: short provider TTLs remain enabled to avoid rate-limit
  abuse.
- Every canonical snapshot stores cache policy, provider source, quote time,
  timeframe last-bar timestamps, and the exact bars used.
- A payload without a usable timestamp cannot pass snapshot validation.

## Deployment

Apply `migrations/versions/20260819_0006_monitor_market_snapshots.sql` to the
operational database. Startup also performs best-effort additive-column
compatibility updates; the SQL migration remains the auditable deployment path.

The monitor remains advisory-only. No IBKR execution behavior is changed.
