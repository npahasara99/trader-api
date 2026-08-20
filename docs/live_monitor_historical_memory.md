# Live Monitor Historical Memory

The live monitor stores decision-time facts and later outcomes separately. Historical evidence can adjust advisory scoring and confirmation preference within configured bounds, but it cannot rewrite OHLCV, bypass current hard gates, or submit broker orders.

## Persistent Evidence

- `monitor_bar_summaries`: deduplicated completed 1m/5m bars retained for monitored setups. Repeated quote polls are not retained.
- `monitor_events`, `confirmation_attempts`, `monitor_decision_snapshots`, `chart_snapshots`: existing event and immutable decision-time evidence.
- `monitor_daily_summaries`: one finalized setup/day record, including no-trigger and no-trade days.
- `level_revisions`: planner, chart-LLM, validated, manual, and final active price lineage, plus eventual outcome attribution.
- `llm_advisory_reviews` and `llm_decision_postmortems`: original structured rationale and a separate outcome-time postmortem.
- `recommendation_outcomes` and `manual_monitor_trades`: recommendation and actual-trade results remain distinct.
- `stock_behavior_profiles`: mutable latest aggregates for ticker, setup type, sector, market regime, and global scopes.
- `behavior_profile_versions`: append-only profile versions used to identify the exact historical prior behind a decision.
- `learned_adjustments`: append-only bounded adjustments applied to a current setup.
- `learning_observations`, `learning_proposals`, `monitor_rule_versions`, `shadow_rule_evaluations`: observation, approval, versioning, and paper-test workflow.
- `learning_job_runs`: operational audit of each daily/on-demand learning cycle.

Raw high-frequency quote polls are intentionally not retained. Completed bar retention defaults to 365 days and chart-image retention remains independently configured.

## Statistical Rules

For observation `i`:

```text
recency_i = max(0.10, 0.5 ** (age_days / half_life_days))
quality_i = product(configured quality multipliers)
weight_i = recency_i * quality_i
n_eff = sum(weight_i)
reliability = n_eff / (n_eff + prior_strength)
```

Default evidence tiers use effective, not raw, sample size:

- `< 8`: `INSUFFICIENT`
- `< 15`: `WEAK`
- `< 30`: `EMERGING`
- `< 60`: `MODERATE`
- otherwise: `STRONG`

The thresholds are configurable through `LIVE_MONITOR_EVIDENCE_THRESHOLDS`.

Hierarchical weights are assigned sequentially from ticker to setup, sector, regime, and global evidence. Each scope uses `n_eff / (n_eff + configured_prior)`, and unused mass falls through to broader priors.

Similar cases use a normalized weighted score across ticker, structure, setup type, sector, regime, confirmation method, attempt number, market context, ATR%, RSI, support/trigger distance in ATR, and RVOL. Categorical weights and continuous weights/scales are environment-configurable.

## Bounded Adaptation

The following may be applied automatically only with sufficient evidence:

- false-breakout actionability adjustment;
- break/retest confirmation preference;
- target-realism penalty when current TP1 is materially beyond historical MFE;
- level-source confidence penalty when a currently used source has meaningful weak resolved performance.

`raw_setup_score` remains unchanged. Responses expose `historical_adjustment_score`, `learned_actionability_score`, `trade_today_score`, and component reasons. The total score adjustment is clamped by `LIVE_MONITOR_MAX_HISTORICAL_SCORE_ADJUSTMENT`. Target, RVOL, and chase adjustments have independent bounds.

Current `INVALIDATED`, `DATA_STALE`, `PLAN_STALE`, `MISSED`, and poor-R:R states remain authoritative. Historical evidence alone cannot approve a setup.

## Level Sanity And Correction

`LEVEL_SANITY_ENGINE` evaluates the intended entry regime:

- primary trigger distance in percent and ATR;
- nearer ranked resistance and possible major-repair misclassification;
- stop/invalidation side, structural evidence, and ATR width;
- TP ordering, ATR reachability, and skipped nearer resistance;
- realistic R:R from intended entry.

Default high-confidence automatic current-setup correction requires:

1. a configured pricing anomaly;
2. chart review decision `MODIFY_LEVELS` or `APPROVE_LEVELS`;
3. deterministic validation `VALID` or `PARTIAL`;
4. confidence at or above `0.90`;
5. at least one validated level;
6. no manual level ownership.

Low-confidence or inconclusive disagreement becomes `MANUAL_REVIEW_REQUIRED`. A distant trigger may remain when the review keeps the planner. Structural invalidation and suggested executable stop remain separate; a truly wide invalidation returns untradeable geometry instead of being tightened arbitrarily.

All corrected current levels still require live price, volume, candle-quality, retest, max-chase, and current-R:R confirmation. Execution remains `MANUAL_ONLY`.

## Daily And On-Demand Learning

The backend polling cycle attempts the daily learning job after the configured New York time (default `20:15`). It finalizes monitor-day summaries, links outcomes, creates LLM postmortems, refreshes all profile scopes, versions profiles, generates sparse observations/proposals, and prunes old bars.

Run it explicitly with:

```http
POST /live-monitor/learning/run
Content-Type: application/json

{"trading_date":"2026-08-19"}
```

Opening a ticker profile also refreshes finalized evidence on demand.

## Proposals And Paper Tests

Permanent rule changes require `APPROVE`, `REJECT`, `PAPER_TEST`, or `LATER`. Approval creates a new rule version and never rewrites old recommendations.

Paper-test proposals are evaluated on important live state transitions. Production decisions are stored unchanged beside the shadow decision. At EOD, production and hypothetical shadow outcomes are resolved into separate fields. Shadow results never alter live recommendations or broker behavior.

## Deployment

Apply `migrations/versions/20260819_0007_monitor_historical_memory.sql` to an existing database. Application startup also performs best-effort additive compatibility for the new shadow-result columns. New tables are created by the existing SQLAlchemy startup convention.

The dashboard Learning view exposes profiles, setup/level/LLM performance, observations, proposals, paper tests, rule versions, and job runs. Live ticker detail exposes historical context, score breakdown, learned adjustments, similar cases, profile version, level anomalies, and pricing lineage.

