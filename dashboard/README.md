# Streamlit Dashboard

Dashboard and runner for the Supabase reporting database used by workflow scan outputs.

## What it shows

- Latest run overview
- Latest watchlist table with filters
- Ready-soon WAIT names
- Ticker detail with pretty-rendered JSON blocks
- Scan run history with per-run ticker results
- Latest top-5 active watch names
- Scanner / Runner controls that call the live trader API
- Trading Bot controls and visibility for broker status, candidates, proposals, orders, positions, journal and memory

## Environment

The dashboard reads from:

- `SUPABASE_DATABASE_URL`

The runner calls the API with:

- `TRADER_API_BASE_URL`
- `API_BEARER_TOKEN` (if your API requires bearer auth)

The dashboard remains read-only for reporting data. Workflow execution is done through the existing API only.

## Run locally

From the repo root:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Windows double-click launch

From the repo root you can also use:

- [Launch Dashboard.bat](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/Launch%20Dashboard.bat)
  - double-click to start the dashboard with a visible console window and wait for the local server before opening the browser
- [Launch Dashboard.vbs](C:/Users/nadun/OneDrive/Documents/Stock%20Simulator/trader-api/Launch%20Dashboard.vbs)
  - double-click to start it without keeping the console window in front

Both launchers expect Python and the dashboard requirements to already be installed.

## Local env setup

Set:

```bash
SUPABASE_DATABASE_URL=postgresql://...
TRADER_API_BASE_URL=https://your-api-host
API_BEARER_TOKEN=...
```

Use the same Supabase reporting connection string that the API uses for reporting writes.
For local use, the dashboard will also read `SUPABASE_DATABASE_URL` from the repo-level `.env` file if it is present there.
The runner will also read `TRADER_API_BASE_URL` and `API_BEARER_TOKEN` from the same repo-level `.env` file when present.

## Deploy

For a simple Streamlit deployment:

1. Point the deployment to this repo
2. Set `SUPABASE_DATABASE_URL`
3. Set `TRADER_API_BASE_URL`
4. Set `API_BEARER_TOKEN` if the API requires auth
5. Use the start command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

## Notes

- The dashboard uses cached read queries for speed.
- After a successful API run from the Scanner tab, the dashboard clears cached reads and refreshes from Supabase.
- If Supabase is unavailable, the UI shows a clear error instead of crashing silently.
- The dashboard is intentionally read-only for reporting data and does not touch the main `DATABASE_URL` path.
- Supported runner routes currently include:
  - `/workflow/swing-plan-log`
  - `/workflow/sp100/top10-log`
  - `/plan/swing`
