# Trader API + Interactive Brokers Swing Bot

See [Chart-Aware Live Monitor](docs/chart_aware_monitor.md) for canonical chart data, level reconciliation, decision-time snapshots, and dashboard controls.

This repository contains:

- the existing FastAPI trader API
- the existing scanner / planner / watchlist pipeline
- the existing Streamlit dashboard
- a new trading-bot layer that can connect to Interactive Brokers paper trading through a broker abstraction

## Architecture summary

Core app:

- [app/main.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/main.py)
- [app/logic.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/logic.py)
- [app/planner.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/planner.py)
- [app/scanner.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/scanner.py)

Trading bot:

- [app/bot/broker.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/bot/broker.py)
- [app/bot/config.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/bot/config.py)
- [app/bot/service.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/bot/service.py)
- [app/bot/api.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/app/bot/api.py)

Dashboard:

- [dashboard/app.py](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/dashboard/app.py)

## Safety defaults

- `TRADING_MODE=disabled`
- `AUTO_EXECUTION=false`
- `IBKR_REQUIRE_PAPER_ACCOUNT=true`
- live trading remains locked unless every live-specific environment control is explicitly enabled

## Local setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

- `DATABASE_URL`
- `SUPABASE_DATABASE_URL` if you use reporting/dashboard active snapshots
- `API_BEARER_TOKEN`
- `TRADER_API_BASE_URL`
- IBKR connection values for paper trading

## Run the API

```bash
python scripts/start_api.py
```

For local TWS paper testing, set `TRADER_API_PORT=8081` in the ignored
`.env.bot` file if port `8080` is already occupied. The launcher reads `.env`
and `.env.bot` automatically; deployed environment variables still take
precedence.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

For Railway, use `python dashboard/start.py` as the dashboard service start
command so the injected `PORT` is validated without relying on shell expansion.

Railway services using this repository must keep their start commands separate:

| Service | Start command |
| --- | --- |
| Trader API | `python scripts/start_api.py` |
| Streamlit dashboard | `python dashboard/start.py` |

The API launcher validates Railway's `PORT` and enforces a single Uvicorn worker,
which is required by the in-process bot and its persistent IBKR connection.

The dashboard includes `Live Monitor`, `Decision Journal`, and `Learning / Research`
tabs alongside the scanner and active watchlist. The live monitor is advisory-only
and never submits an IBKR order. See `docs/live_swing_monitor.md` for its API,
state machine, persistence, and safety rules.

## Apply schema updates

```bash
python scripts/apply_migrations.py
```

## Run tests

```bash
pytest
```

## IBKR paper trading setup

1. Install TWS or IB Gateway.
2. Use the paper account login first.
3. Enable API connections.
4. Set the API port to `7497` for paper TWS or the matching IB Gateway paper port.
5. Set trusted IPs if needed.
6. Start the API service and use `GET /broker/health` and `POST /broker/reconnect`.
7. Keep `TRADING_MODE=disabled` or `manual_paper` until you have verified reconciliation.

## Deployment notes

- A persistent process is required for the broker socket.
- The provided [Dockerfile](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/Dockerfile) and [docker-compose.yml](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/docker-compose.yml) support local multi-service review.
- In production, the API service and the long-running bot should run on infrastructure that allows persistent outbound socket connections to TWS or IB Gateway.

## Known limitations

- IBKR live execution remains intentionally locked by configuration defaults.
- The mock broker is the primary automated test broker.
- The repo did not previously contain a formal migration framework; a schema sync script and reviewable SQL migration stub were added to fit the existing architecture.
