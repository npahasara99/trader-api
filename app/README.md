# Trader API planning notes

## Structured swing-planning upgrade

The swing planner no longer derives `entry`, `stop`, and `take_profit` from naive fixed percentages.

Current planning flow:
- `app/scanner.py` adds a cheap Stage-1 swing pre-scan so SP100/generic scans rank a broad universe before full planning.
- The workflow now pre-scans the broad universe, shortlists the strongest swing candidates, and only then runs the full structured planner.
- `app/planner.py` orchestrates deterministic market-structure analysis.
- `app/indicators.py` computes ATR, moving averages, returns, and volume context inputs.
- `app/structure.py` classifies trend state and recent pivots/breakout context.
- `app/zones.py` builds support/resistance zones from pivots, MAs, fibs, consolidation, gaps, and volume congestion.
- `app/entry_engine.py` generates immediate/pullback/deeper entry candidates and selects the preferred entry.
- `app/risk_engine.py` places stop loss and take-profit targets using structure plus ATR buffers and estimates hold time.
- `app/scoring.py` produces component scores and a composite score.
- `app/llm_reasoning.py` isolates the reasoning layer so an external LLM can review structured data without inventing price levels.
- `app/llm_reasoning.py` also contains the final three-bucket classifier:
  - `BUY` = confirmed and actionable now
  - `WAIT` = constructive but not ready
  - `AVOID` = materially weak / unattractive
- `app/classification_fixtures.py` contains lightweight profile fixtures for validating the intended BUY / WAIT / AVOID behavior.

Backward compatibility:
- Existing routes still return `entry`, `stop`, and `take_profit`.
- `take_profit` now maps to the first realistic target (`take_profit_1`) for compatibility with logging/evaluation.
- Richer structured fields are returned alongside the legacy fields for ranking, logging, and UI use.
- Final API rows now also expose decision nuance:
  - `quant_action`
  - `llm_action`
  - `reconciled_action`
  - `final_action`
  - `action_alignment`
  - `action_reason_bucket`
  - `monitorable_setup`
  - `avoid_severity_score`
  - `wait_reason`
  - `avoid_reason`
  - `buy_blockers`
  - `constructive_traits`
- WAIT rows additionally expose a deterministic monitoring plan:
  - monitor window and stale date
  - trader-readable support / resistance zone displays
  - upgrade triggers
  - failure triggers
  - next check focus
  - `setup_monitoring_summary`
- Final API rows now also expose a separate swing-trade suitability assessment:
  - `swing_trade_suitability.suitability_score`
  - `swing_trade_suitability.suitability_label`
  - suitability subscores for trend / structure / entry / reward-risk / volatility / volume / relative strength / event risk / timing
  - `key_strengths`
  - `key_weaknesses`
  - `disqualifiers`
  - `suitable_for_long_swing`
  - `suitable_for_watchlist_only`
  - `not_suitable_reason`
- Final API rows also expose a mutually exclusive watchlist prioritization layer:
  - `watchlist_tier` = `primary | secondary | none`
  - `watchlist_bucket` = `high_priority_watchlist | secondary_watchlist | avoid`
  - `watchlist_summary`
  - `watchlist_reason`
  - `is_primary_watchlist_candidate`
  - `is_secondary_watchlist_candidate`
- Final API rows now also expose a chart execution view:
  - `chart_execution_view.trade_shape`
  - `chart_execution_view.enter_now`
  - `chart_execution_view.breakout_point`
  - `chart_execution_view.breakout_point_type`
  - `chart_execution_view.pullback_entry_zone`
  - `chart_execution_view.deeper_pullback_zone`
  - `chart_execution_view.current_price_location`
  - `chart_execution_view.execution_bias`
  - `chart_execution_view.execution_zone_quality`
  - `chart_execution_view.chart_execution_summary`
- Final API rows now also expose scanner/ranking diagnostics:
  - `pre_scan_score`
  - `pre_scan_reason_tags`
  - `sector_relative_strength`
  - `scanner_rank_score`
  - `immediate_rank_score`
  - `watchlist_rank_score`
  - `ranking_bucket`
  - `scan_shortlisted`
  - `scan_rejection_reason`
- SP100 workflow responses now split ranked outputs into:
  - `best_immediate_setups`
  - `best_watchlist_setups`
  - `rejected_or_low_priority`

Fallback behavior:
- If live quote data is unavailable, planning still falls back to recent cached closes.
- If full OHLCV bars are unavailable, the planner degrades gracefully using close-only history, but richer bars improve accuracy materially.
