# Chart-Aware Live Monitor

The live swing monitor remains advisory-only. It never imports a broker client and never submits an order.

## Data flow

The trader API builds one canonical chart bundle from normalized OHLCV:

1. Daily bars come from `daily_bars` when available, then the normalized market-data provider.
2. Hourly, 30-minute, and 5-minute bars use `app.market_data.get_bars` through the monitor's injected provider.
3. Every bundle is filtered at `decision_time_boundary`; later candles are excluded.
4. EMA20/50/100/200, intraday VWAP, candles, and volume are derived from those same bars.
5. The interactive chart, PNG renderer, deterministic validator, and multimodal review consume the same bundle.

Each response records `data_source`, `data_timestamp`, `last_bar_timestamp`, and `data_freshness_seconds`.

## Level hierarchy

The monitor distinguishes:

- `near_confirmation`: first local reaction level.
- `primary_entry_trigger`: actionable local swing confirmation.
- `strong_confirmation`: stronger local acceptance level.
- `major_trend_repair`: distant structural recovery level, not a normal primary entry.
- `invalidation_level`: thesis failure.
- `suggested_stop`: executable risk control, kept separate from invalidation.

A planner trigger beyond `LIVE_MONITOR_TRIGGER_MAX_ATR` or outside the configured recent-range fraction is proposed as major trend repair when nearer pivot evidence exists. This proposal never changes active levels by itself.

## Review and reconciliation

`CHART_STRUCTURE_REVIEW` and `CONFIRMED_TRADE_REVIEW` have separate prompts and schemas. The OpenAI Responses API receives decision-time PNGs plus exact structured values. Images are interpretive; numeric OHLCV remains authoritative.

Chart-suggested levels pass deterministic checks for recent OHLC range, pivot/reaction evidence, long-side geometry, ATR distance, and canonical price consistency.

The monitor stores planner, LLM-proposed, validated, manual, and active values independently. Active priority is:

1. Manual override.
2. User-accepted validated chart level.
3. Planner level.

The dashboard provides **Accept Validated**, **Keep Planner**, and manual editing. No chart proposal becomes active without one of those actions.

## R:R semantics

- `planned_rr_at_primary_trigger` is available before confirmation.
- `current_executable_rr` remains null until an approved, strongly-confirmed, or missed executable state.
- Legacy `current_rr_tp1` remains but now mirrors `current_executable_rr`.
- A manual order plan is not shown before an executable confirmation state.

## Monitoring and UI

The backend daemon evaluates active monitors on `LIVE_MONITOR_POLL_SECONDS`, whether or not a browser is open. Dashboard polling only reads state every 2-5 seconds and does not drive evaluation. `Resume` is shown only for paused monitors.

The Chart Analysis tab uses TradingView Lightweight Charts 5.2.0 for rendering only. It receives bars from `GET /live-monitor/{watch_id}/charts`; it does not query TradingView market data or embed the TradingView website.

## API

- `GET /live-monitor/status`
- `GET /live-monitor/{watch_id}/charts`
- `POST /live-monitor/{watch_id}/chart-review`
- `POST /live-monitor/{watch_id}/chart-level-decision`

Level decision examples:

```json
{"decision":"ACCEPT_VALIDATED"}
```

```json
{"decision":"KEEP_PLANNER"}
```

```json
{"decision":"EDIT_MANUALLY","manual_levels":{"primary_entry_trigger":95.25}}
```

## Deployment

Apply `migrations/versions/20260819_0005_chart_aware_monitor.sql` before deploying against an existing database. Fresh databases are covered by SQLAlchemy metadata creation.

The renderer writes local copies under `LIVE_MONITOR_CHART_SNAPSHOT_DIR` and stores decision-critical PNG data in the database row so container filesystem replacement does not erase evidence. Automatic reviews are bounded by cooldown and per-setup daily limits.

Optional multimodal review requires `OPENAI_API_KEY`. If the SDK, model, image, or provider is unavailable, the deterministic monitor remains available and records a conservative fallback status.

## Key configuration

- `LIVE_MONITOR_TRIGGER_MAX_ATR`
- `LIVE_MONITOR_TRIGGER_MAX_RANGE_FRACTION`
- `LIVE_MONITOR_CHART_MAX_STOP_ATR`
- `LIVE_MONITOR_CHART_MAX_TARGET_ATR`
- `LIVE_MONITOR_PLAN_MAX_AGE_DAYS`
- `LIVE_MONITOR_CHART_REVIEW_MODEL`
- `LIVE_MONITOR_CHART_REVIEW_COOLDOWN_SECONDS`
- `LIVE_MONITOR_MAX_AUTO_CHART_REVIEWS_PER_DAY`
- `LIVE_MONITOR_FRONTEND_REFRESH_SECONDS`
- `LIVE_MONITOR_CHART_SNAPSHOT_DIR`

## Current limitations

- Intraday bars inherit the normalized provider's availability and delay characteristics.
- The monitor UI renders daily, 30-minute, and 5-minute panels. Hourly bars remain in the canonical API bundle for broader context.
- A multimodal review cannot repair missing or inconsistent canonical data. `CHART_DATA_MISMATCH` blocks automated chart recommendations.
