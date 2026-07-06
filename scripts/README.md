# Planner Validation Harness

Use `scripts/validate_planner.py` to run the live swing planner over a fixed basket and export a reviewable CSV/summary.

## What It Exports

- Detail CSV with per-ticker planner fields
- Summary CSV with aggregate counts and averages
- Summary JSON with count breakdowns
- Optional comparison CSV against a prior export

## Basket Modes

### Manual tickers

```powershell
py scripts/validate_planner.py --tickers LIN,AMD,NVDA
```

### Tickers from a text file

```powershell
py scripts/validate_planner.py --tickers-file .\my_validation_basket.txt
```

### SP100 sector / industry basket

```powershell
py scripts/validate_planner.py --sector technology --industry semiconductors --top-n 12
```

### Latest active watchlist names from Supabase

Requires `SUPABASE_DATABASE_URL` in the environment or repo `.env`.

```powershell
py scripts/validate_planner.py --top-watchlist 15
```

## Optional Comparison

Compare a new run against a previously exported detail CSV:

```powershell
py scripts/validate_planner.py --tickers LIN,AMD,NVDA --compare-csv .\validation_outputs\planner_validation_manual_20260705_120000_details.csv
```

## Output Location

By default outputs are written to:

```text
validation_outputs\
```

Override with:

```powershell
py scripts/validate_planner.py --tickers LIN,AMD,NVDA --output-dir .\tmp\planner_checks
```

## Notes

- The harness calls the real `plan_swing()` path, so it uses the same planner stack as the API.
- If a ticker still hits a planner crash or returns incomplete levels, that state is preserved in the detail CSV via `planner_status` and `scan_rejection_reason`.
