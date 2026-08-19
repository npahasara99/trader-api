# Live Swing Monitor

The live monitor is a persistent, manual-execution-only subsystem. It is
separate from the S&P 500 scanner and from `app.bot`; it never imports or calls
a broker order submission method.

## Runtime flow

1. Add a scanner result or ticker to `POST /live-monitor`.
2. A scanner row is reused as the immutable planner baseline. A manual ticker
   runs the structured planner once. Missing market data produces a visible
   `NO_VALID_SWING_SETUP` baseline rather than fabricated levels.
3. The background service loads active rows after process startup and polls only
   those symbols. It requests cached Yahoo 1-minute and 5-minute bars.
4. The deterministic engine evaluates trigger distance, fresh data, 5-minute
   close, relative volume, candle quality, retest, current R:R, and maximum chase.
5. Meaningful state changes append journal events and immutable snapshots.
6. LLM advisory is event-triggered only. Its output cannot override stale data,
   invalidation, chase, or unacceptable live R:R.
7. Any order plan is labeled `MANUAL_ONLY`; the user records their own fills.

## State machine

`WATCHING -> NEAR_TRIGGER -> ARMED -> CONFIRMING -> APPROVED`

Additional states are `STRONGLY_CONFIRMED`, `REJECTED_BREAKOUT`, `MISSED`,
`INVALIDATED`, `DATA_STALE`, `PAUSED`, `STOPPED`, and `EXPIRED`.

A failed breakout closes the current confirmation attempt, not the monitor. A
later cross creates a new attempt. Reanalysis creates a new setup version and
marks the old setup replaced; it never rewrites prior levels or events.

## Deterministic formulas

- `distance_to_trigger_pct = (current - trigger) / trigger`
- `RVOL = current bar volume / median(previous N bar volumes)`; default `N=20`
- Constructive volume defaults to `RVOL >= 1.20`; strong volume defaults to
  `RVOL >= 1.50`.
- `upper_wick_ratio = (high - max(open, close)) / (high - low)`
- `close_location = (close - low) / (high - low)`
- A constructive candle defaults to upper wick `<= 0.35` and close location
  `>= 0.60`.
- `max_chase = trigger + max(trigger * 0.5%, ATR * 0.35)`.
- `current_rr_tp1 = (TP1 - current) / (current - suggested_stop)`.

Approval requires a valid setup, fresh data, a 5-minute close above the active
trigger, constructive RVOL, acceptable candle quality, current R:R of at least
1.25, and price no higher than maximum chase.

## Manual level overlays

`PATCH /live-monitor/{watch_id}/levels` accepts near confirmation, primary
trigger, strong confirmation, major repair, invalidation, support, suggested
stop, and maximum chase. `planner_levels_json` remains unchanged;
`active_levels_json` and `manual_overrides_json` are updated and the trigger
source becomes `MANUAL`.

## APIs

- `GET/POST /live-monitor`
- `GET /live-monitor/{watch_id}`
- `POST /live-monitor/{watch_id}/pause|resume|stop|remove|reanalyze|evaluate`
- `PATCH /live-monitor/{watch_id}/levels`
- `POST /live-monitor/{watch_id}/llm-review`
- `POST /live-monitor/{watch_id}/manual-trades`
- `GET /live-monitor/journal`
- `GET /live-monitor/profiles/{ticker}`
- `GET /live-monitor/learning`
- `POST /live-monitor/learning/observations`
- `POST /live-monitor/learning/proposals`
- `POST /live-monitor/learning/proposals/{id}/decision`

All routes use the API's existing bearer-token dependency.

## Learning controls

Statistics are deterministic. Sparse ticker evidence is shrunk toward setup,
sector, and global priors. Ticker weight is `n/(n+20)`, setup uses `n/(n+30)`,
and sector uses `n/(n+40)` against each remaining share. Similar cases receive
explicit points for ticker, setup, sector, regime, confirmation method, and
attempt number.

Proposals begin `PENDING`. `APPROVE` creates a new immutable rule version;
`REJECT` records rejection; `PAPER_TEST` creates a shadow evaluation while
leaving production behavior unchanged.

## Deployment

Apply `migrations/versions/20260819_0004_live_swing_monitor.sql` using the
existing migration script, then deploy the API and dashboard normally. A single
API worker is recommended for the in-process polling thread. State is database
backed and reloads after restart.

At a 60-second interval, 5, 10, and 20 symbols require approximately 10, 20,
and 40 cached timeframe lookups per minute before cache hits. Provider pacing
and availability remain the practical limit; stale or missing data results in
`DATA_STALE`, never approval.
