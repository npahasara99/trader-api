# TradingView monitoring webhook

The API registers `POST /webhooks/tradingview` and stores accepted alerts in
`tradingview_signal_events` with `processing_status=pending_replan`. It never
submits orders. A separate planner/execution pass remains mandatory.

## API integration

Set a dedicated secret of at least 16 characters:

```text
TRADINGVIEW_WEBHOOK_SECRET=<random-dedicated-webhook-secret>
```

Apply `migrations/versions/20260813_0002_tradingview_events.sql` in production
before enabling alerts. Local SQLite development creates the table through
SQLAlchemy metadata on API startup.

The sink only persists and marks re-evaluation as requested. The webhook
response and normalized event explicitly set `execution_requested` to `false`.
The webhook module does not import any broker, proposal, execution, or order
code.

## TradingView setup

1. Open `tradingview/trader_monitor.pine` in TradingView's Pine Editor.
2. Add the indicator to a shortlisted ticker chart.
3. In indicator settings, enter the same dedicated webhook secret. The source
   has an empty default and contains no hardcoded credential.
4. Optionally enter explicit support and breakout levels. Zero uses the prior
   rolling local low/high.
5. Create an alert and choose `Trader API Watchlist Monitor` then
   `Any alert() function call`.
6. Set the webhook URL to `https://<api-host>/webhooks/tradingview`.

The alert body is generated as valid JSON by Pine. Use HTTPS on port 443 and a
dedicated rotating webhook credential, never a broker or account password.
TradingView requires two-factor authentication for webhook alerts and cancels
requests that take longer than approximately three seconds.

## End-to-end validation

1. Restart the API after setting the secret.
2. Run `POST /plan/swing` for one ticker and verify the response includes
   `chart_context`, `execution_scenarios`, `preferred_scenario`, and
   `execution_action`.
3. Send one Pine alert or a matching JSON request to
   `POST /webhooks/tradingview`.
4. Confirm a row exists in `tradingview_signal_events` with
   `processed=false`, `processing_status=pending_replan`, and
   `execution_requested=false`.
5. Re-run the plan endpoint to refresh structured bars and execution geometry.

## Data limitations

- Daily planning retains the existing Finnhub/Yahoo/Stooq source chain.
- Hourly and 30-minute chart context currently use normalized Yahoo chart bars
  with an in-process cache. Provider failure produces an explicit missing
  timeframe rather than synthetic intraday bars.
- The TradingView widget may display a feed that differs slightly from the API
  provider. It is for human verification only and never supplies planner levels.

## Accepted events

- `SUPPORT_HOLD`
- `SUPPORT_BREAK`
- `EMA20_RECLAIM`
- `EMA50_RECLAIM`
- `BREAKOUT`
- `BREAKOUT_FAILURE`
- `MOMENTUM_IMPROVING`
- `MOMENTUM_WEAKENING`
- `RSI_RECOVERY`
- `RSI_OVEREXTENDED`

Unknown event types, unsupported timeframes, invalid numbers, timezone-naive
timestamps, and extra payload fields are rejected.

## Tests

```powershell
py -m pytest tests/test_tradingview_webhook.py -q
```
