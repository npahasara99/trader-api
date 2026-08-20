# Final Active Plan Flow

The live swing monitor uses one authoritative plan version per setup.

1. `market_snapshot.build_market_snapshot` captures canonical quote and bars.
2. `baseline.build_monitor_baseline` adapts the deterministic planner output.
3. `LiveMonitorService._create_setup` stores immutable planner originals.
4. `level_sanity.evaluate_level_sanity` detects suspicious roles and geometry.
5. `chart_review.review_chart_packet` runs structured chart review when configured.
6. `chart_levels.validate_chart_levels` validates every proposed chart level.
7. `chart_levels.reconcile_levels` records agreement, rejection, or review required.
8. `final_plan.finalize_active_plan` regenerates targets after entry changes and runs the hard final validator.
9. `MonitorSetup.final_active_plan_json` stores the authoritative version and lineage.
10. `engine.evaluate_monitor`, chart bundles, R:R, learning, and manual-order output consume the corresponding `active_levels_json` flat view and final plan ID.

Planner, LLM-proposed, validated, manual, and final versions remain available in
monitor diagnostics. A rejected critical chart proposal sets
`MANUAL_REVIEW_REQUIRED`; it does not silently promote the planner fallback.

Hard geometry failures set `PLAN_GEOMETRY_INVALID` and disable confirmation,
active R:R, and manual-order plan generation until a corrected plan passes.
