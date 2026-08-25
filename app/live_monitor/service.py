from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from threading import Event, Lock, Thread
import time
from typing import Any, Callable
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.market_data import get_bars
from app.models import (
    ChartLevelDecision,
    ChartSnapshot,
    ChartStructureReview,
    BehaviorProfileVersion,
    ConfirmationAttempt,
    LearnedAdjustment,
    LearningJobRun,
    LearningObservation,
    LearningProposal,
    MarketSnapshot,
    LiveWatch,
    LLMAdvisoryReview,
    LLMDecisionPostmortem,
    LevelRevision,
    ManualMonitorTrade,
    MonitorDecisionSnapshot,
    MonitorEvent,
    MonitorDailySummary,
    MonitorRuleVersion,
    MonitorSetup,
    RecommendationOutcome,
    ShadowRuleEvaluation,
    StockBehaviorProfile,
)

from .advisor import PROMPT_VERSION, build_advisory_packet, review_advisory_packet
from .baseline import build_monitor_baseline
from .chart_data import build_chart_bundle
from .chart_levels import (
    LEVEL_NAMES,
    check_data_consistency,
    derive_chart_level_candidates,
    detect_stale_plan,
    number,
    reconcile_levels,
    validate_level_semantics,
)
from .chart_renderer import cleanup_chart_snapshot_retention, render_chart_snapshots
from .chart_review import CHART_STRUCTURE_PROMPT_VERSION, review_chart_packet
from .config import LiveMonitorConfig, load_live_monitor_config
from .engine import evaluate_monitor
from .enums import ACTIVE_MONITOR_STATES, MonitorState
from .final_plan import finalize_active_plan, validate_final_plan
from .learning import adjustment_breakdown, aggregate_attempts, hierarchical_weights, similar_case_score
from .level_sanity import LEVEL_ROLES, can_auto_apply_chart_correction, evaluate_level_sanity
from .memory import (
    learned_adjustment_payloads,
    load_historical_context,
    past_postmortems,
    persist_adjustments,
    persist_completed_bars,
    refresh_profile,
    run_daily_learning_cycle,
)
from ..setup_archetypes import normalize_setup_family
from .market_snapshot import build_market_snapshot, market_snapshot_payload, persist_market_snapshot


def _id() -> str:
    return str(uuid.uuid4())


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _loads(value: str | None, fallback: Any = None) -> Any:
    if not value:
        return {} if fallback is None else fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {} if fallback is None else fallback


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _llm_level_reasons(output: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    level_payload = output.get("levels") or {}
    target_payload = output.get("targets") or {}
    for name in LEVEL_NAMES:
        raw = (
            level_payload.get("structural_invalidation")
            if name == "invalidation_level"
            else level_payload.get(name)
            if name != "optional_support_level"
            else level_payload.get("support_zone")
        )
        if raw is None:
            raw = target_payload.get(name)
        if isinstance(raw, dict) and raw.get("reason"):
            reasons[name] = str(raw["reason"])
    return reasons


class LiveMonitorService:
    """Persistent lightweight monitor; deliberately contains no broker client."""

    def __init__(
        self,
        *,
        config: LiveMonitorConfig | None = None,
        bars_loader: Callable[[str, str, int | None], list[dict]] | None = None,
        advisory_provider: Callable[[dict], dict] | None = None,
        chart_review_provider: Callable[[dict], dict] | None = None,
    ) -> None:
        self.config = config or load_live_monitor_config()
        self._provided_bars_loader = bars_loader
        self._bars_loader = bars_loader or self._load_live_bars
        self._advisory_provider = advisory_provider
        self._chart_review_provider = chart_review_provider
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._running = False
        self._last_cycle_at: datetime | None = None
        self._last_cycle_error: str | None = None
        self._last_learning_date: date | None = None

    @staticmethod
    def _load_live_bars(ticker: str, timeframe: str, lookback_days: int | None) -> list[dict]:
        ttl = 20 if timeframe == "one_minute" else 30
        return get_bars(ticker, timeframe, lookback_days, cache_ttl_seconds=ttl)

    def _fetch_bars(self, ticker: str, timeframe: str, lookback_days: int | None, *, force_refresh: bool) -> list[dict]:
        if self._provided_bars_loader is not None:
            try:
                return self._provided_bars_loader(ticker, timeframe, lookback_days, force_refresh=force_refresh)  # type: ignore[call-arg]
            except TypeError:
                return self._provided_bars_loader(ticker, timeframe, lookback_days)
        ttl = 0 if force_refresh else (20 if timeframe == "one_minute" else 30)
        return get_bars(ticker, timeframe, lookback_days, cache_ttl_seconds=ttl)

    def _fresh_market_snapshot(self, ticker: str) -> dict[str, Any]:
        return build_market_snapshot(
            ticker,
            bars_loader=lambda symbol, timeframe, lookback: self._fetch_bars(
                symbol, timeframe, lookback, force_refresh=True
            ),
            force_refresh=True,
            consistency_max_pct=self.config.market_data_mismatch_pct,
            consistency_atr_fraction=self.config.market_data_mismatch_atr_fraction,
        )

    @staticmethod
    def _require_usable_snapshot(snapshot: dict[str, Any]) -> None:
        status = str(snapshot.get("consistency_status") or "MARKET_DATA_UNAVAILABLE")
        if status == "MARKET_DATA_UNAVAILABLE" or number(snapshot.get("reference_price")) is None:
            raise ValueError("MARKET_DATA_UNAVAILABLE: fresh quote/OHLCV could not be loaded")
        if status == "MARKET_DATA_MISMATCH":
            raise ValueError("MARKET_DATA_MISMATCH: intraday prices disagree beyond the configured tolerance")
        if snapshot.get("freshness_status") == "MARKET_DATA_STALE":
            raise ValueError("MARKET_DATA_STALE: provider bars are older than the session-aware freshness allowance")

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return {"ok": True, "message": "Live monitor already running"}
            self._stop_event.clear()
            self._running = True
            self._thread = Thread(target=self._run_loop, name="live-swing-monitor", daemon=True)
            self._thread.start()
        return {"ok": True, "message": "Live monitor started"}

    def stop_service(self) -> dict[str, Any]:
        self._stop_event.set()
        self._running = False
        return {"ok": True, "message": "Live monitor service stopped"}

    def status(self) -> dict[str, Any]:
        with SessionLocal() as db:
            active_count = db.query(LiveWatch).filter(LiveWatch.monitor_active.is_(True), LiveWatch.removed_at.is_(None)).count()
            last_market_update = db.query(func.max(LiveWatch.market_data_as_of)).scalar()
        running = self._running and bool(self._thread and self._thread.is_alive())
        health = "LIVE" if running and self._last_cycle_error is None else "DEGRADED" if running else "STOPPED"
        return {
            "running": running,
            "monitor_service_status": health,
            "execution": "MANUAL_ONLY",
            "active_monitor_count": active_count,
            "poll_interval_seconds": self.config.poll_interval_seconds,
            "last_cycle_at": self._last_cycle_at,
            "last_cycle_error": self._last_cycle_error,
            "last_backend_evaluation_at": self._last_cycle_at,
            "last_market_update_at": last_market_update,
            "market_data_age_seconds": (
                None
                if last_market_update is None
                else max(0.0, (_utcnow() - (last_market_update if last_market_update.tzinfo else last_market_update.replace(tzinfo=timezone.utc))).total_seconds())
            ),
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.evaluate_all()
                self._last_cycle_error = None
            except Exception as exc:
                self._last_cycle_error = f"{type(exc).__name__}: {exc}"
            self._last_cycle_at = _utcnow()
            self._stop_event.wait(self.config.poll_interval_seconds)
        self._running = False

    def _schedule_chart_review(self, watch_id: str, review_type: str, event_type: str) -> None:
        """Run expensive image/model work outside the polling transaction."""
        def run() -> None:
            try:
                self.run_chart_review(
                    watch_id,
                    review_type=review_type,
                    automatic=True,
                    snapshot_event_type=event_type,
                )
            except Exception as exc:
                self._record_chart_review_failure(watch_id, review_type=review_type, error=exc)

        Thread(
            target=run,
            name=f"chart-review-{watch_id[:8]}",
            daemon=True,
        ).start()

    def _record_chart_review_failure(self, watch_id: str, *, review_type: str, error: Exception) -> None:
        """Persist unexpected chart-review failures instead of hiding them."""
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            setup = db.get(MonitorSetup, watch.current_setup_id) if watch and watch.current_setup_id else None
            if watch is None or setup is None:
                return
            detail = f"{type(error).__name__}: {error}"
            row = ChartStructureReview(
                id=_id(), watch_id=watch.id, setup_id=setup.id,
                market_snapshot_id=setup.market_snapshot_id, ticker=watch.ticker,
                review_type=str(review_type or "CHART_STRUCTURE_REVIEW").upper(),
                status="VALIDATION_FAILED", model=self.config.chart_review_model,
                prompt_version=CHART_STRUCTURE_PROMPT_VERSION,
                chart_snapshot_ids_json=_dumps([]),
                deterministic_input_json=_dumps({
                    "market_snapshot_id": setup.market_snapshot_id,
                    "setup_id": setup.id,
                    "monitor_id": watch.id,
                    "unexpected_error": detail,
                }),
                planner_levels_json=setup.planner_levels_json,
                llm_output_json=_dumps({}), llm_proposed_levels_json=_dumps({}),
                validated_levels_json=_dumps({}),
                validation_json=_dumps({"status": "SKIPPED", "reason": "unexpected_chart_review_failure", "error": detail}),
                decision="MANUAL_REVIEW", confidence=0.0,
                reason_summary=f"Chart review failed before a valid response could be persisted: {detail}",
                data_consistency_status="INSUFFICIENT_DATA",
            )
            db.add(row)
            setup.latest_chart_review_id = row.id
            setup.chart_analysis_status = "VALIDATION_FAILED"
            setup.updated_at = _utcnow()
            self._event(
                db, watch, setup=setup, event_type="CHART_REVIEW_FAILED",
                message=f"{row.review_type} failed: {detail}",
                snapshot={
                    "review_id": row.id, "model": row.model,
                    "prompt_version": row.prompt_version,
                    "market_snapshot_id": setup.market_snapshot_id,
                    "error": detail,
                },
            )
            db.commit()

    def _schedule_chart_snapshot(self, watch_id: str, event_type: str) -> None:
        def run() -> None:
            try:
                with SessionLocal() as db:
                    watch = db.get(LiveWatch, watch_id)
                    if watch is None or not watch.current_setup_id:
                        return
                    setup = db.get(MonitorSetup, watch.current_setup_id)
                    bundle = self._chart_bundle(db, watch, setup)
                    snapshots = render_chart_snapshots(
                        db,
                        watch=watch,
                        setup=setup,
                        bundle=bundle,
                        event_type=event_type,
                        config=self.config,
                        force=True,
                    )
                    event = self._event(
                        db,
                        watch,
                        setup=setup,
                        event_type="chart_snapshot",
                        message=f"Decision-time chart snapshot captured for {event_type}",
                        snapshot={"event_type": event_type, "snapshot_ids": [row.id for row in snapshots]},
                    )
                    for snapshot in snapshots:
                        snapshot.decision_event_id = event.id
                    db.commit()
            except Exception:
                return

        Thread(target=run, name=f"chart-snapshot-{watch_id[:8]}", daemon=True).start()

    def evaluate_all(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            ids = [
                row.id
                for row in db.query(LiveWatch.id)
                .filter(LiveWatch.monitor_active.is_(True), LiveWatch.removed_at.is_(None))
                .all()
            ]
        results = []
        for watch_id in ids:
            try:
                results.append(self.evaluate_watch(watch_id))
            except Exception as exc:
                with SessionLocal() as db:
                    watch = db.get(LiveWatch, watch_id)
                    if watch:
                        self._event(
                            db,
                            watch,
                            event_type="monitor_error",
                            message=f"Ticker evaluation failed: {type(exc).__name__}: {exc}",
                        )
                        db.commit()
                results.append({"watch_id": watch_id, "error": f"{type(exc).__name__}: {exc}"})
        self._maybe_run_daily_learning_cycle()
        return results

    def _maybe_run_daily_learning_cycle(self) -> None:
        now_et = _utcnow().astimezone(ZoneInfo("America/New_York"))
        refresh_time = (self.config.profile_refresh_hour_et, self.config.profile_refresh_minute_et)
        if (now_et.hour, now_et.minute) < refresh_time or self._last_learning_date == now_et.date():
            return
        try:
            self.run_learning_cycle(now_et.date())
            self._last_learning_date = now_et.date()
        except Exception as exc:
            self._last_cycle_error = f"learning_cycle: {type(exc).__name__}: {exc}"

    def run_learning_cycle(self, trading_date: date | str | None = None) -> dict[str, Any]:
        target = trading_date
        if isinstance(target, str):
            target = date.fromisoformat(target)
        if target is None:
            target = _utcnow().astimezone(ZoneInfo("America/New_York")).date()
        with SessionLocal() as db:
            result = run_daily_learning_cycle(db, target, self.config)
            db.commit()
            return result

    def add_monitor(self, ticker: str, *, source: str = "manual", planner_payload: dict | None = None) -> dict[str, Any]:
        symbol = str(ticker or "").strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("A valid ticker is required")
        snapshot = self._fresh_market_snapshot(symbol)
        self._require_usable_snapshot(snapshot)
        watch_id: str
        should_review = False
        with SessionLocal() as db:
            persist_market_snapshot(db, snapshot)
            watch = db.query(LiveWatch).filter(LiveWatch.ticker == symbol).one_or_none()
            now = _utcnow()
            if watch is None:
                watch = LiveWatch(id=_id(), ticker=symbol, source=source, created_at=now)
                db.add(watch)
                db.flush()
            old_setup = db.get(MonitorSetup, watch.current_setup_id) if watch.current_setup_id else None
            replacement_reason = None
            if old_setup is not None and watch.monitor_active and watch.removed_at is None:
                old_levels = _loads(old_setup.active_levels_json)
                staleness = detect_stale_plan(
                    current_price=snapshot.get("reference_price"),
                    levels=old_levels,
                    atr=old_levels.get("atr"),
                    setup_created_at=old_setup.created_at,
                    plan_reference_price=old_setup.plan_reference_price,
                    plan_created_at=old_setup.plan_created_at,
                    structure_bars=(snapshot.get("bars") or {}).get("thirty_minute") or [],
                    data_consistency_status=snapshot.get("consistency_status"),
                    config=self.config,
                )
                if not staleness["stale"]:
                    watch.current_price = snapshot.get("reference_price")
                    watch.market_data_as_of = _as_datetime(snapshot.get("quote_timestamp"))
                    watch.market_snapshot_id = snapshot.get("market_snapshot_id")
                    watch.last_market_data_update_at = watch.market_data_as_of or now
                    watch.updated_at = now
                    self._event(
                        db,
                        watch,
                        setup=old_setup,
                        event_type="MARKET_DATA_REFRESHED",
                        message="Existing active setup validated against a fresh market snapshot",
                        snapshot={"market_snapshot_id": snapshot.get("market_snapshot_id"), "staleness": staleness},
                    )
                    db.commit()
                    return self._watch_payload(db, watch, include_detail=True)
                replacement_reason = ",".join(staleness["reasons"])
                old_setup.plan_stale = True
                old_setup.plan_stale_reason = replacement_reason
                old_setup.plan_stale_reasons_json = _dumps(staleness["reasons"])
                self._event(
                    db,
                    watch,
                    setup=old_setup,
                    event_type="PLAN_STALE_DETECTED",
                    message="Existing monitor plan became stale during activation refresh",
                    snapshot=staleness,
                )

            baseline = build_monitor_baseline(
                db,
                symbol,
                supplied_plan=planner_payload,
                config=self.config,
                market_snapshot=snapshot,
                allow_source_reuse=old_setup is None,
            )
            watch.source = source
            watch.monitor_active = True
            watch.removed_at = None
            watch.state = MonitorState.WATCHING.value
            watch.current_price = snapshot.get("reference_price")
            watch.market_data_as_of = _as_datetime(snapshot.get("quote_timestamp"))
            watch.market_snapshot_id = snapshot.get("market_snapshot_id")
            watch.last_market_data_update_at = watch.market_data_as_of or now
            watch.updated_at = now
            setup = self._create_setup(
                db,
                watch,
                baseline,
                previous_setup=old_setup,
                replacement_reason=replacement_reason,
            )
            self._attach_historical_context(db, watch, setup)
            if old_setup:
                old_setup.status = "replaced"
                old_setup.replaced_by_setup_id = setup.id
                old_setup.updated_at = now
            self._event(
                db,
                watch,
                setup=setup,
                event_type="MARKET_DATA_REFRESHED",
                message="Fresh quote and OHLCV were captured in the canonical market snapshot",
                snapshot={
                    "market_snapshot_id": setup.market_snapshot_id,
                    "quote_timestamp": snapshot.get("quote_timestamp"),
                    "data_source": snapshot.get("data_source"),
                    "cache_status": snapshot.get("cache_status"),
                    "freshness_status": snapshot.get("freshness_status"),
                },
            )
            self._event(
                db,
                watch,
                setup=setup,
                event_type="MONITOR_CREATED_FROM_FRESH_SNAPSHOT",
                to_state=watch.state,
                message=f"{symbol} activated from canonical snapshot {setup.market_snapshot_id}",
                snapshot={
                    "market_snapshot_id": setup.market_snapshot_id,
                    "plan_reference_price": setup.plan_reference_price,
                    "source_plan_reused": baseline.get("source_plan_reused"),
                    "source_plan_validation": baseline.get("source_plan_validation"),
                },
            )
            db.add(MonitorDecisionSnapshot(
                id=_id(), watch_id=watch.id, setup_id=setup.id, ticker=watch.ticker,
                snapshot_type="MONITOR_CREATED",
                payload_json=_dumps({
                    "market_snapshot_id": setup.market_snapshot_id,
                    "planner_baseline": _loads(setup.planner_baseline_json),
                    "planner_levels": _loads(setup.planner_levels_json),
                    "active_levels": _loads(setup.active_levels_json),
                    "profile_version": (_loads(setup.planner_baseline_json).get("historical_context_at_creation") or {}),
                }),
            ))
            db.commit()
            watch_id = watch.id
            should_review = True
        if should_review and self.config.chart_review_on_add:
            try:
                self.run_chart_review(
                    watch_id,
                    review_type="CHART_STRUCTURE_REVIEW",
                    automatic=True,
                    snapshot_event_type="monitor_added",
                )
            except Exception as exc:
                # A valid deterministic plan remains usable, but the failure is explicit.
                self._record_chart_review_failure(watch_id, review_type="CHART_STRUCTURE_REVIEW", error=exc)
        return self.get_monitor(watch_id)

    def _persist_final_active_plan(
        self,
        setup: MonitorSetup,
        *,
        levels: dict[str, Any],
        sources: dict[str, str],
        current_price: float | None,
        reconciliation_status: str,
        structure_bars: list[dict[str, Any]] | None = None,
        execution_bars: list[dict[str, Any]] | None = None,
        entry_changed: bool = False,
        change_source: str | None = None,
        level_reasons: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist the one plan consumed by every live execution component."""
        final_plan = finalize_active_plan(
            setup_id=setup.id,
            levels=levels,
            sources=sources,
            current_price=current_price,
            market_snapshot_id=setup.market_snapshot_id,
            config=self.config,
            reconciliation_status=reconciliation_status,
            structure_bars=structure_bars,
            execution_bars=execution_bars,
            entry_changed=entry_changed,
            change_source=change_source,
            level_reasons=level_reasons,
        )
        setup.active_levels_json = _dumps(final_plan["flat_levels"])
        setup.level_sources_json = _dumps(final_plan["level_sources"])
        setup.final_active_plan_id = final_plan["plan_id"]
        setup.final_active_plan_json = _dumps(final_plan)
        setup.final_plan_validation_json = _dumps(final_plan["validation"])
        setup.plan_integrity_status = final_plan["plan_integrity_status"]
        setup.reconciliation_status = reconciliation_status
        setup.max_chase_price = final_plan["flat_levels"].get("max_chase_price")
        return final_plan

    @staticmethod
    def _set_plan_gate_state(watch: LiveWatch, setup: MonitorSetup) -> None:
        if setup.plan_integrity_status == "INVALID":
            watch.state = MonitorState.PLAN_GEOMETRY_INVALID.value
        elif setup.reconciliation_status == "MANUAL_REVIEW_REQUIRED":
            watch.state = MonitorState.PLAN_REVIEW_REQUIRED.value
        elif watch.state in {
            MonitorState.PLAN_GEOMETRY_INVALID.value,
            MonitorState.PLAN_REVIEW_REQUIRED.value,
        }:
            watch.state = MonitorState.WATCHING.value

    def _create_setup(
        self,
        db: Session,
        watch: LiveWatch,
        baseline: dict,
        *,
        previous_setup: MonitorSetup | None = None,
        replacement_reason: str | None = None,
    ) -> MonitorSetup:
        current_max = db.query(func.max(MonitorSetup.version)).filter(MonitorSetup.watch_id == watch.id).scalar() or 0
        plan = baseline["plan"]
        levels = baseline["levels"]
        expires_at = _as_datetime(plan.get("max_hold_date"))
        setup = MonitorSetup(
            id=_id(),
            watch_id=watch.id,
            ticker=watch.ticker,
            version=int(current_max) + 1,
            status="active",
            valid_setup=bool(baseline["valid_setup"]),
            setup_quality_score=plan.get("raw_setup_score") or plan.get("composite_score"),
            broader_structure=plan.get("broader_structure") or plan.get("trend_state"),
            setup_type=plan.get("setup_type"),
            setup_family=normalize_setup_family(plan.get("setup_family") or plan.get("setup_type"), "reversal_attempt"),
            execution_structure=plan.get("execution_structure") or plan.get("trade_shape"),
            sector=plan.get("sector"),
            industry=plan.get("industry"),
            market_regime=plan.get("market_regime"),
            planner_baseline_json=_dumps(plan),
            market_snapshot_id=plan.get("market_snapshot_id"),
            plan_reference_price=plan.get("plan_reference_price"),
            plan_created_at=_as_datetime(plan.get("plan_created_at")),
            market_data_timestamp=_as_datetime(plan.get("market_data_timestamp")),
            plan_stale=False,
            plan_stale_reasons_json=_dumps([]),
            previous_setup_id=previous_setup.id if previous_setup else None,
            replacement_reason=replacement_reason,
            planner_levels_json=_dumps(levels),
            active_levels_json=_dumps(levels),
            manual_overrides_json=_dumps({}),
            level_sources_json=_dumps({name: "PLANNER" for name in LEVEL_NAMES if number(levels.get(name)) is not None}),
            chart_analysis_status="NOT_RUN",
            trigger_source="PLANNER",
            max_chase_price=levels.get("max_chase_price"),
            expires_at=expires_at,
        )
        db.add(setup)
        db.flush()
        snapshot = baseline.get("market_snapshot") or {}
        snapshot_bars = snapshot.get("bars") or {}
        final_plan = self._persist_final_active_plan(
            setup,
            levels=levels,
            sources={name: "PLANNER" for name in LEVEL_NAMES if number(levels.get(name)) is not None},
            current_price=setup.plan_reference_price,
            reconciliation_status="PLANNER_ACCEPTED",
            structure_bars=snapshot_bars.get("thirty_minute") or [],
            execution_bars=snapshot_bars.get("five_minute") or [],
        )
        watch.current_setup_id = setup.id
        self._set_plan_gate_state(watch, setup)
        self._record_level_revisions(
            db, watch=watch, setup=setup, review_row=None,
            planner_levels=levels, proposed_levels={}, validated_levels={},
            final_levels=final_plan["flat_levels"],
            sources={name: "PLANNER" for name in LEVEL_NAMES if number(levels.get(name)) is not None},
            sanity={}, validation={}, llm_output={},
        )
        return setup

    def _attach_historical_context(self, db: Session, watch: LiveWatch, setup: MonitorSetup) -> dict[str, Any]:
        """Retrieve history at bootstrap and persist bounded, decision-time adjustments."""
        context = load_historical_context(db, setup, self.config)
        levels = _loads(setup.active_levels_json)
        entry = number(levels.get("primary_entry_trigger"))
        tp1 = number(levels.get("tp1"))
        atr = number(levels.get("atr")) or number(_loads(setup.planner_baseline_json).get("atr"))
        features = {
            "tp1_distance_atr": None if entry is None or tp1 is None or atr is None else (tp1 - entry) / atr,
            "level_source": setup.trigger_source,
        }
        applied = persist_adjustments(
            db, watch=watch, setup=setup, context=context,
            current_features=features, config=self.config,
        )
        baseline = _loads(setup.planner_baseline_json)
        baseline["historical_context_at_creation"] = {
            "profile_version_id": (context.get("ticker_profile") or {}).get("profile_version_id"),
            "profile_version": (context.get("ticker_profile") or {}).get("profile_version"),
            "profile_last_updated_at": (context.get("ticker_profile") or {}).get("profile_last_updated_at"),
            "evidence_strength": (context.get("ticker_profile") or {}).get("evidence_strength"),
            "hierarchical_weights": context.get("hierarchical_weights"),
            "recommendation_breakdown": applied["recommendation_breakdown"],
        }
        setup.planner_baseline_json = _dumps(baseline)
        self._event(
            db, watch, setup=setup, event_type="HISTORICAL_PROFILE_APPLIED",
            message=(
                f"Historical context loaded with "
                f"{(context.get('ticker_profile') or {}).get('evidence_strength', 'INSUFFICIENT')} evidence"
            ),
            snapshot={
                "historical_context": baseline["historical_context_at_creation"],
                "learned_adjustments": applied["adjustments"],
            },
        )
        return {**context, **applied}

    def list_monitors(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            query = db.query(LiveWatch).filter(LiveWatch.removed_at.is_(None))
            if not include_inactive:
                query = query.filter(LiveWatch.monitor_active.is_(True))
            rows = query.all()
            priority = {
                MonitorState.STRONGLY_CONFIRMED.value: 0,
                MonitorState.APPROVED.value: 1,
                MonitorState.CONFIRMING.value: 2,
                MonitorState.ARMED.value: 3,
                MonitorState.NEAR_TRIGGER.value: 4,
                MonitorState.WATCHING.value: 5,
                MonitorState.DATA_STALE.value: 6,
                MonitorState.PLAN_STALE.value: 7,
                MonitorState.PLAN_REVIEW_REQUIRED.value: 8,
                MonitorState.PLAN_GEOMETRY_INVALID.value: 9,
                MonitorState.MISSED.value: 10,
                MonitorState.INVALIDATED.value: 11,
                MonitorState.PAUSED.value: 12,
                MonitorState.STOPPED.value: 13,
            }
            payloads = [self._watch_payload(db, row) for row in rows]
            return sorted(payloads, key=lambda row: (priority.get(row["state"], 99), abs(row.get("distance_to_trigger_pct") or 999), row["ticker"]))

    def get_monitor(self, watch_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None:
                raise LookupError("Monitor not found")
            return self._watch_payload(db, watch, include_detail=True)

    def control(self, watch_id: str, action: str) -> dict[str, Any]:
        normalized = str(action).strip().lower()
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None:
                raise LookupError("Monitor not found")
            previous = watch.state
            if normalized == "pause":
                watch.monitor_active = False
                watch.state = MonitorState.PAUSED.value
            elif normalized == "resume":
                watch.monitor_active = True
                watch.removed_at = None
                watch.state = MonitorState.WATCHING.value
            elif normalized == "stop":
                watch.monitor_active = False
                watch.state = MonitorState.STOPPED.value
            elif normalized == "remove":
                watch.monitor_active = False
                watch.removed_at = _utcnow()
            else:
                raise ValueError(f"Unsupported monitor action: {action}")
            watch.updated_at = _utcnow()
            self._event(db, watch, event_type=f"monitor_{normalized}", from_state=previous, to_state=watch.state, message=f"Monitor {normalized} requested")
            db.commit()
            return self._watch_payload(db, watch, include_detail=True)

    def edit_levels(self, watch_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        allowed = {*LEVEL_NAMES, "max_chase_price"}
        clean: dict[str, float | None] = {}
        for key, value in overrides.items():
            if key not in allowed:
                continue
            if value in (None, ""):
                clean[key] = None
            else:
                parsed_number = float(value)
                if parsed_number <= 0:
                    raise ValueError(f"{key} must be positive")
                clean[key] = parsed_number
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active setup not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            if setup.plan_stale:
                raise ValueError("PLAN_STALE: reanalyze before editing active execution levels")
            active = _loads(setup.active_levels_json)
            manual = _loads(setup.manual_overrides_json)
            sources = _loads(setup.level_sources_json)
            previous_primary = number(active.get("primary_entry_trigger"))
            candidate = dict(active)
            candidate.update(clean)
            manual.update(clean)
            sources.update({name: "MANUAL" for name in clean})
            setup.manual_overrides_json = _dumps(manual)
            setup.trigger_source = "MANUAL"
            setup.chart_analysis_status = "MANUAL_OVERRIDE"
            bundle = self._chart_bundle(db, watch, setup)
            final_plan = self._persist_final_active_plan(
                setup,
                levels=candidate,
                sources=sources,
                current_price=number(bundle.get("reference_price")) or number(watch.current_price) or setup.plan_reference_price,
                reconciliation_status="MANUAL_OVERRIDE",
                structure_bars=((bundle.get("timeframes") or {}).get("structure") or {}).get("bars") or [],
                execution_bars=((bundle.get("timeframes") or {}).get("execution") or {}).get("bars") or [],
                entry_changed=number(candidate.get("primary_entry_trigger")) != previous_primary,
                change_source="MANUAL",
            )
            active = final_plan["flat_levels"]
            self._set_plan_gate_state(watch, setup)
            setup.updated_at = _utcnow()
            self._record_level_revisions(
                db, watch=watch, setup=setup, review_row=None,
                planner_levels=_loads(setup.planner_levels_json), proposed_levels={},
                validated_levels={}, final_levels=active, sources=sources,
                sanity={}, validation={}, llm_output={},
            )
            self._event(
                db, watch, setup=setup, event_type="levels_overridden",
                message="Manual level overlay updated and final plan revalidated",
                snapshot={
                    "manual_overrides": manual,
                    "planner_originals": _loads(setup.planner_levels_json),
                    "final_active_plan_id": final_plan["plan_id"],
                    "final_plan_validation": final_plan["validation"],
                    "target_regeneration": final_plan["target_regeneration"],
                },
            )
            db.commit()
            return self._watch_payload(db, watch, include_detail=True)

    def reanalyze(self, watch_id: str, planner_payload: dict | None = None) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None:
                raise LookupError("Monitor not found")
            self._event(db, watch, event_type="REANALYSIS_STARTED", message="Forced fresh monitor reanalysis started")
            db.commit()
            symbol = watch.ticker
        snapshot = self._fresh_market_snapshot(symbol)
        try:
            self._require_usable_snapshot(snapshot)
        except ValueError:
            with SessionLocal() as db:
                watch = db.get(LiveWatch, watch_id)
                if watch:
                    self._event(db, watch, event_type="DATA_MISMATCH", message="Reanalysis stopped because fresh market data was unavailable or inconsistent", snapshot=snapshot)
                    db.commit()
            raise
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None:
                raise LookupError("Monitor not found")
            persist_market_snapshot(db, snapshot)
            old_setup = db.get(MonitorSetup, watch.current_setup_id) if watch.current_setup_id else None
            baseline = build_monitor_baseline(
                db,
                watch.ticker,
                supplied_plan=planner_payload or (_loads(old_setup.planner_baseline_json) if old_setup else None),
                config=self.config,
                market_snapshot=snapshot,
                allow_source_reuse=False,
            )
            replacement_reason = "FORCED_FRESH_REANALYSIS"
            new_setup = self._create_setup(
                db,
                watch,
                baseline,
                previous_setup=old_setup,
                replacement_reason=replacement_reason,
            )
            self._attach_historical_context(db, watch, new_setup)
            if old_setup:
                old_setup.status = "replaced"
                old_setup.replaced_by_setup_id = new_setup.id
                old_setup.replacement_reason = replacement_reason
                old_setup.updated_at = _utcnow()
            previous = watch.state
            watch.state = MonitorState.WATCHING.value
            watch.monitor_active = True
            watch.current_price = snapshot.get("reference_price")
            watch.market_data_as_of = _as_datetime(snapshot.get("quote_timestamp"))
            watch.market_snapshot_id = snapshot.get("market_snapshot_id")
            watch.last_market_data_update_at = watch.market_data_as_of or _utcnow()
            watch.updated_at = _utcnow()
            self._event(
                db,
                watch,
                setup=new_setup,
                event_type="NEW_PLAN_CREATED",
                from_state=previous,
                to_state=watch.state,
                message="Fresh canonical planner baseline created; prior setup preserved",
                snapshot={
                    "old_setup_id": old_setup.id if old_setup else None,
                    "old_market_snapshot_id": old_setup.market_snapshot_id if old_setup else None,
                    "new_setup_id": new_setup.id,
                    "market_snapshot_id": new_setup.market_snapshot_id,
                    "plan_reference_price": new_setup.plan_reference_price,
                },
            )
            if old_setup:
                self._event(
                    db,
                    watch,
                    setup=new_setup,
                    event_type="OLD_PLAN_REPLACED",
                    message="Old setup retained as history and replaced by fresh reanalysis",
                    snapshot={"old_setup_id": old_setup.id, "new_setup_id": new_setup.id, "replacement_reason": replacement_reason},
                )
            db.commit()
        try:
            self.run_chart_review(
                watch_id,
                review_type="CHART_STRUCTURE_REVIEW",
                automatic=False,
                snapshot_event_type="setup_reanalyzed",
            )
        except Exception as exc:
            self._record_chart_review_failure(watch_id, review_type="CHART_STRUCTURE_REVIEW", error=exc)
        return self.get_monitor(watch_id)

    def _chart_bundle(self, db: Session, watch: LiveWatch, setup: MonitorSetup, *, boundary: datetime | None = None) -> dict[str, Any]:
        attempts = (
            db.query(ConfirmationAttempt)
            .filter(ConfirmationAttempt.setup_id == setup.id)
            .order_by(ConfirmationAttempt.attempt_number.asc())
            .all()
        )
        snapshot_row = db.get(MarketSnapshot, setup.market_snapshot_id) if setup.market_snapshot_id else None
        snapshot = market_snapshot_payload(snapshot_row)
        bundle = build_chart_bundle(
            db,
            ticker=watch.ticker,
            levels=_loads(setup.active_levels_json),
            level_sources=_loads(setup.level_sources_json),
            bars_loader=self._bars_loader,
            decision_time_boundary=boundary or setup.plan_created_at,
            max_bars=self.config.chart_max_bars,
            attempts=attempts,
            market_snapshot=snapshot or None,
        )
        bundle["planner_market_snapshot_id"] = setup.market_snapshot_id
        bundle["final_active_plan_id"] = setup.final_active_plan_id
        bundle["plan_integrity_status"] = setup.plan_integrity_status
        bundle["reconciliation_status"] = setup.reconciliation_status
        bundle["levels_source"] = "FINAL_ACTIVE_PLAN"
        bundle["snapshot_ids_match"] = bool(
            setup.market_snapshot_id
            and bundle.get("market_snapshot_id") == setup.market_snapshot_id
            and _loads(setup.planner_baseline_json).get("market_snapshot_id") == setup.market_snapshot_id
        )
        return bundle

    def chart_bundle(self, watch_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active monitor setup not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            bundle = self._chart_bundle(db, watch, setup)
            bundle["planner_price"] = _loads(setup.planner_baseline_json).get("current_price")
            bundle["monitor_price"] = watch.current_price
            bundle["chart_analysis_status"] = setup.chart_analysis_status
            bundle["plan_stale"] = setup.plan_stale
            if setup.plan_stale:
                bundle["historical_levels"] = bundle.get("levels") or []
                bundle["levels"] = []
                bundle["levels_status"] = "HISTORICAL_STALE"
            return bundle

    def _automatic_chart_review_allowed(self, db: Session, setup: MonitorSetup) -> bool:
        start_of_day = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        automatic_count = (
            db.query(ChartStructureReview)
            .filter(
                ChartStructureReview.setup_id == setup.id,
                ChartStructureReview.created_at >= start_of_day,
            )
            .count()
        )
        if automatic_count >= self.config.max_auto_chart_reviews_per_day:
            return False
        latest = (
            db.query(ChartStructureReview)
            .filter(ChartStructureReview.setup_id == setup.id)
            .order_by(ChartStructureReview.created_at.desc())
            .first()
        )
        return bool(
            latest is None
            or latest.created_at is None
            or _utcnow() - (latest.created_at if latest.created_at.tzinfo else latest.created_at.replace(tzinfo=timezone.utc))
            >= timedelta(seconds=self.config.chart_review_cooldown_seconds)
        )

    def run_chart_review(
        self,
        watch_id: str,
        *,
        review_type: str = "CHART_STRUCTURE_REVIEW",
        automatic: bool = False,
        snapshot_event_type: str | None = None,
    ) -> dict[str, Any]:
        normalized_type = str(review_type or "CHART_STRUCTURE_REVIEW").strip().upper()
        if normalized_type not in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW", "CONFIRMED_TRADE_REVIEW"}:
            raise ValueError("Unsupported chart review type")
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active monitor setup not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            if automatic and not self._automatic_chart_review_allowed(db, setup):
                return {"status": "COOLDOWN", "message": "Automatic chart review cooldown is active"}
            setup.chart_analysis_status = "PENDING"
            cleanup_chart_snapshot_retention(db, config=self.config)
            bundle = self._chart_bundle(db, watch, setup)
            baseline = _loads(setup.planner_baseline_json)
            planner_levels = _loads(setup.planner_levels_json)
            active_levels = _loads(setup.active_levels_json)
            chart_close = bundle.get("latest_chart_close")
            current_price = number(bundle.get("reference_price")) or number(chart_close) or number(setup.plan_reference_price)
            atr = number(active_levels.get("atr")) or number(baseline.get("atr")) or (current_price or 1.0) * 0.02
            consistency = check_data_consistency(
                planner_price=setup.plan_reference_price,
                monitor_price=watch.current_price,
                chart_close=chart_close,
                atr=atr,
            )
            if not bundle.get("snapshot_ids_match"):
                consistency = {
                    **consistency,
                    "status": "MARKET_DATA_MISMATCH",
                    "snapshot_id_mismatch": {
                        "planner": setup.market_snapshot_id,
                        "chart": bundle.get("market_snapshot_id"),
                        "baseline": baseline.get("market_snapshot_id"),
                    },
                }
            stale = detect_stale_plan(
                current_price=watch.current_price or current_price,
                levels=active_levels,
                atr=atr,
                setup_created_at=setup.created_at,
                plan_reference_price=setup.plan_reference_price,
                plan_created_at=setup.plan_created_at,
                structure_bars=((bundle.get("timeframes") or {}).get("structure") or {}).get("bars") or [],
                data_consistency_status=consistency.get("status"),
                config=self.config,
            )
            snapshots: list[ChartSnapshot] = []
            render_error: str | None = None
            try:
                snapshots = render_chart_snapshots(
                    db,
                    watch=watch,
                    setup=setup,
                    bundle=bundle,
                    event_type=snapshot_event_type or ("setup_reanalyzed" if normalized_type == "CHART_STRUCTURE_REVIEW" else "APPROVED"),
                    config=self.config,
                    force=True,
                )
            except Exception as exc:
                render_error = f"{type(exc).__name__}: {exc}"
            structure_bars = ((bundle.get("timeframes") or {}).get("structure") or {}).get("bars") or []
            execution_bars = ((bundle.get("timeframes") or {}).get("execution") or {}).get("bars") or []
            sanity = evaluate_level_sanity(
                current_price=float(current_price or 0.0),
                atr=float(atr), levels=active_levels,
                structure_bars=structure_bars, execution_bars=execution_bars,
                config=self.config,
            ) if current_price else {"status": "UNAVAILABLE", "anomalies": ["MARKET_DATA_UNAVAILABLE"], "review_required": False}
            historical_context = load_historical_context(db, setup, self.config)
            latest_attempt = db.query(ConfirmationAttempt).filter(
                ConfirmationAttempt.setup_id == setup.id,
            ).order_by(ConfirmationAttempt.attempt_number.desc()).first()
            packet = {
                "review_type": normalized_type,
                "ticker": watch.ticker,
                "current_price": current_price,
                "market_snapshot_id": setup.market_snapshot_id,
                "plan_reference_price": setup.plan_reference_price,
                "plan_created_at": setup.plan_created_at,
                "atr": atr,
                "broader_structure": setup.broader_structure,
                "setup_type": setup.setup_type,
                "setup_family": setup.setup_family,
                "execution_structure": setup.execution_structure,
                "market_regime": setup.market_regime,
                "planner_levels": planner_levels if not stale["stale"] else {},
                "active_levels": active_levels if not stale["stale"] else {},
                "historical_stale_plan": {"levels": planner_levels, "status": "HISTORICAL_STALE"} if stale["stale"] else None,
                "stale_plan": stale,
                "data_consistency": consistency,
                "data_source": bundle.get("data_source"),
                "data_timestamp": bundle.get("data_timestamp"),
                "last_bar_timestamp": bundle.get("last_bar_timestamp"),
                "data_freshness_seconds": bundle.get("data_freshness_seconds"),
                "structure_bars": structure_bars,
                "execution_bars": execution_bars,
                "latest_evaluation": _loads(watch.latest_evaluation_json),
                "level_sanity": sanity,
                "pricing_anomalies": sanity.get("anomalies") or [],
                "historical_profile": historical_context.get("ticker_profile"),
                "broader_historical_profiles": historical_context.get("broader_profiles"),
                "similar_historical_cases": self._similar_cases(db, setup, latest_attempt),
                "learned_adjustments": learned_adjustment_payloads(db, setup.id),
                "past_llm_postmortems": past_postmortems(db, setup.ticker),
                "image_paths": [row.image_path for row in snapshots],
            }
            blocked = consistency["status"] in {"CHART_DATA_MISMATCH", "MARKET_DATA_MISMATCH"} or current_price is None or stale["stale"]
            if blocked:
                blocked_status = "PLAN_STALE" if stale["stale"] else (
                    "MARKET_DATA_MISMATCH" if consistency["status"] == "MARKET_DATA_MISMATCH" else "CHART_DATA_MISMATCH"
                )
                review = {
                    "status": blocked_status if current_price is not None else "UNAVAILABLE",
                    "model": None,
                    "prompt_version": CHART_STRUCTURE_PROMPT_VERSION,
                    "output": {},
                    "proposed_levels": {},
                    "validated_levels": {},
                    "validation": {"status": "SKIPPED", "reason": blocked_status, "stale_plan": stale},
                    "decision": "MANUAL_REVIEW",
                    "confidence": 0.0,
                    "reason_summary": "Chart recommendation skipped because the active plan is stale or canonical snapshot IDs/prices are inconsistent.",
                }
            else:
                review = review_chart_packet(packet, provider=self._chart_review_provider, config=self.config)
            proposed = review.get("proposed_levels") or {}
            validated = review.get("validated_levels") or {}
            if normalized_type in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW"}:
                candidate_reconciliation = reconcile_levels(
                    planner_levels=planner_levels,
                    proposed_levels=proposed,
                    validation=review.get("validation") or {"accepted_levels": validated},
                    manual_overrides=_loads(setup.manual_overrides_json),
                )
                entry_changed = number(candidate_reconciliation["final_active_levels"].get("primary_entry_trigger")) != number(active_levels.get("primary_entry_trigger"))
                candidate_final_plan = finalize_active_plan(
                    setup_id=setup.id,
                    levels=candidate_reconciliation["final_active_levels"],
                    sources=candidate_reconciliation["level_sources"],
                    current_price=current_price,
                    market_snapshot_id=setup.market_snapshot_id,
                    config=self.config,
                    reconciliation_status=candidate_reconciliation["reconciliation_status"],
                    structure_bars=structure_bars,
                    execution_bars=execution_bars,
                    entry_changed=entry_changed,
                    change_source="VALIDATED_CHART_LLM",
                    level_reasons=_llm_level_reasons(review.get("output") or {}),
                )
                candidate_reconciliation["candidate_final_active_plan"] = candidate_final_plan
                candidate_reconciliation["final_active_levels"] = candidate_final_plan["flat_levels"]
                candidate_reconciliation["level_sources"] = candidate_final_plan["level_sources"]
                candidate_reconciliation["activation_blocked"] = bool(
                    candidate_reconciliation.get("activation_blocked")
                    or not candidate_final_plan["validation"]["activation_allowed"]
                )
            else:
                candidate_reconciliation = {
                    "final_active_levels": active_levels,
                    "level_sources": _loads(setup.level_sources_json),
                    "status": "TRADE_REVIEW",
                    "has_disagreement": False,
                }
            analysis_status = review.get("status") or "VALIDATION_FAILED"
            if normalized_type in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW"} and analysis_status not in {"CHART_DATA_MISMATCH", "MARKET_DATA_MISMATCH", "PLAN_STALE", "UNAVAILABLE", "VALIDATION_FAILED"}:
                analysis_status = "DISAGREEMENT" if candidate_reconciliation["has_disagreement"] else "AGREES"
            row = ChartStructureReview(
                id=_id(),
                watch_id=watch.id,
                setup_id=setup.id,
                market_snapshot_id=setup.market_snapshot_id,
                ticker=watch.ticker,
                review_type=normalized_type,
                status=analysis_status,
                model=review.get("model"),
                prompt_version=str(review.get("prompt_version") or CHART_STRUCTURE_PROMPT_VERSION),
                chart_snapshot_ids_json=_dumps([snapshot.id for snapshot in snapshots]),
                deterministic_input_json=_dumps({key: value for key, value in packet.items() if key not in {"image_paths"}}),
                planner_levels_json=_dumps(planner_levels),
                llm_output_json=_dumps(review.get("output") or {}),
                llm_proposed_levels_json=_dumps(proposed),
                validated_levels_json=_dumps(validated),
                validation_json=_dumps({**(review.get("validation") or {}), "render_error": render_error, "stale_plan": stale}),
                decision=str(review.get("decision") or "MANUAL_REVIEW"),
                confidence=float(review.get("confidence") or 0.0),
                reason_summary=review.get("reason_summary"),
                data_consistency_status=str(consistency.get("status") or "INSUFFICIENT_DATA"),
            )
            db.add(row)
            auto_policy = {"allowed": False, "blockers": ["not_automatic_level_review"]}
            if automatic and normalized_type in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW"}:
                auto_policy = can_auto_apply_chart_correction(
                    review=review,
                    manual_overrides=_loads(setup.manual_overrides_json),
                    sanity=sanity,
                    config=self.config,
                )
                if auto_policy["allowed"]:
                    if candidate_reconciliation.get("activation_blocked"):
                        auto_policy = {
                            **auto_policy,
                            "allowed": False,
                            "blockers": [*(auto_policy.get("blockers") or []), "candidate_final_plan_not_activatable"],
                        }
                if auto_policy["allowed"]:
                    previous_active = dict(active_levels)
                    final_plan = self._persist_final_active_plan(
                        setup,
                        levels=candidate_reconciliation["final_active_levels"],
                        sources=candidate_reconciliation["level_sources"],
                        current_price=current_price,
                        reconciliation_status="LLM_CORRECTION_ACCEPTED",
                        structure_bars=structure_bars,
                        execution_bars=execution_bars,
                        entry_changed=False,
                        change_source="VALIDATED_CHART_LLM",
                        level_reasons=_llm_level_reasons(review.get("output") or {}),
                    )
                    active_levels = final_plan["flat_levels"]
                    level_sources = final_plan["level_sources"]
                    setup.trigger_source = "VALIDATED_CHART_LLM"
                    analysis_status = "AUTO_CORRECTED"
                    row.status = analysis_status
                    self._set_plan_gate_state(watch, setup)
                    db.add(ChartLevelDecision(
                        id=_id(), watch_id=watch.id, setup_id=setup.id,
                        chart_review_id=row.id, ticker=watch.ticker,
                        decision="AUTO_ACCEPT_VALIDATED",
                        previous_active_levels_json=_dumps(previous_active),
                        selected_levels_json=_dumps(active_levels),
                        level_sources_json=_dumps(level_sources),
                        decided_by="system_high_confidence_validator",
                    ))
                    self._event(
                        db, watch, setup=setup, event_type="LEVEL_CORRECTED",
                        message="High-confidence validated chart levels replaced anomalous current levels",
                        snapshot={
                            "review_id": row.id, "auto_policy": auto_policy,
                            "anomalies": sanity.get("anomalies"),
                            "previous_active_levels": previous_active,
                            "final_active_levels": active_levels,
                        },
                    )
                elif candidate_reconciliation.get("has_disagreement"):
                    analysis_status = "MANUAL_REVIEW_REQUIRED"
                    row.status = analysis_status
                    setup.reconciliation_status = "MANUAL_REVIEW_REQUIRED"
                    self._set_plan_gate_state(watch, setup)
                elif analysis_status not in {"UNAVAILABLE", "PLAN_STALE", "MARKET_DATA_MISMATCH", "CHART_DATA_MISMATCH"}:
                    setup.reconciliation_status = "PLANNER_ACCEPTED"
            if not automatic and normalized_type in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW"}:
                if candidate_reconciliation.get("has_disagreement"):
                    analysis_status = "MANUAL_REVIEW_REQUIRED"
                    row.status = analysis_status
                    setup.reconciliation_status = "MANUAL_REVIEW_REQUIRED"
                    self._set_plan_gate_state(watch, setup)
                elif analysis_status not in {"UNAVAILABLE", "PLAN_STALE", "MARKET_DATA_MISMATCH", "CHART_DATA_MISMATCH"}:
                    setup.reconciliation_status = "PLANNER_ACCEPTED"
            self._record_level_revisions(
                db, watch=watch, setup=setup, review_row=row,
                planner_levels=planner_levels, proposed_levels=proposed,
                validated_levels=validated, final_levels=active_levels,
                sources=(candidate_reconciliation.get("level_sources") if auto_policy.get("allowed") else _loads(setup.level_sources_json)),
                sanity=sanity, validation=review.get("validation") or {},
                llm_output=review.get("output") or {}, auto_policy=auto_policy,
            )
            if normalized_type in {"CHART_STRUCTURE_REVIEW", "CHART_LEVEL_REVIEW"}:
                setup.llm_proposed_levels_json = _dumps(proposed)
                setup.validated_chart_levels_json = _dumps(validated)
                setup.chart_analysis_status = "STALE" if stale["stale"] else analysis_status
            setup.latest_chart_review_id = row.id
            setup.plan_stale_reason = "; ".join(stale["reasons"]) or None
            setup.plan_stale = bool(stale["stale"])
            setup.plan_stale_reasons_json = _dumps(stale["reasons"])
            setup.proposed_setup_json = _dumps(candidate_reconciliation)
            setup.updated_at = _utcnow()
            if normalized_type == "CONFIRMED_TRADE_REVIEW":
                latest_evaluation = _loads(watch.latest_evaluation_json)
                latest_evaluation["chart_llm_review"] = {
                    "review_id": row.id,
                    "status": review.get("status"),
                    "decision": review.get("decision"),
                    "confidence": review.get("confidence"),
                    "reason_summary": review.get("reason_summary"),
                }
                if review.get("status") == "AVAILABLE" and review.get("decision") in {"WAIT", "REJECT"}:
                    latest_evaluation["manual_order_plan"] = None
                    latest_evaluation["llm_confirmation_status"] = f"TECHNICALLY_CONFIRMED_LLM_{review.get('decision')}"
                elif review.get("status") == "AVAILABLE" and review.get("decision") == "APPROVE":
                    latest_evaluation["llm_confirmation_status"] = "TECHNICALLY_CONFIRMED_LLM_APPROVE"
                else:
                    latest_evaluation["llm_confirmation_status"] = "LLM_UNAVAILABLE_DETERMINISTIC_REVIEW_ONLY"
                watch.latest_evaluation_json = _dumps(latest_evaluation)
            review_event = self._event(
                db,
                watch,
                setup=setup,
                event_type="chart_review",
                message=f"{normalized_type} completed with status {analysis_status}",
                snapshot={
                    "review_id": row.id,
                    "status": analysis_status,
                    "consistency": consistency,
                    "stale_plan": stale,
                    "planner_levels": planner_levels,
                    "proposed_levels": proposed,
                    "validated_levels": validated,
                    "level_sanity": sanity,
                    "auto_correction": auto_policy,
                },
            )
            for snapshot in snapshots:
                snapshot.decision_event_id = review_event.id
            db.commit()
            return self._chart_review_payload(row)

    def _record_level_revisions(
        self,
        db: Session,
        *,
        watch: LiveWatch,
        setup: MonitorSetup,
        review_row: ChartStructureReview | None,
        planner_levels: dict[str, Any],
        proposed_levels: dict[str, Any],
        validated_levels: dict[str, Any],
        final_levels: dict[str, Any],
        sources: dict[str, Any],
        sanity: dict[str, Any],
        validation: dict[str, Any],
        llm_output: dict[str, Any],
        auto_policy: dict[str, Any] | None = None,
    ) -> None:
        level_payload = llm_output.get("levels") or {}
        targets = llm_output.get("targets") or {}
        rejected = validation.get("rejected_levels") or {}
        for name in LEVEL_NAMES:
            planner_price = number(planner_levels.get(name))
            proposed_price = number(proposed_levels.get(name))
            validated_price = number(validated_levels.get(name))
            final_price = number(final_levels.get(name))
            if all(value is None for value in (planner_price, proposed_price, validated_price, final_price)):
                continue
            raw_reason = targets.get(name) if name.startswith("tp") or name == "stretch_target" else level_payload.get(name)
            reason = raw_reason.get("reason") if isinstance(raw_reason, dict) else None
            db.add(LevelRevision(
                id=_id(), watch_id=watch.id, setup_id=setup.id,
                chart_review_id=review_row.id if review_row else None,
                market_snapshot_id=setup.market_snapshot_id, ticker=setup.ticker,
                level_name=name, level_role=LEVEL_ROLES.get(name, name.upper()),
                planner_price=planner_price, llm_proposed_price=proposed_price,
                validated_price=validated_price,
                manual_price=number(_loads(setup.manual_overrides_json).get(name)),
                final_active_price=final_price,
                source=str(sources.get(name) or "PLANNER"),
                validation_result=("REJECTED" if name in rejected else "VALIDATED" if validated_price is not None else "NOT_PROPOSED"),
                confidence=float(review_row.confidence if review_row else 1.0),
                reason=reason or (review_row.reason_summary if review_row else "Manual level decision"),
                anomaly_flags_json=_dumps(sanity.get("anomalies") or []),
                outcome_json=_dumps({"auto_policy": auto_policy or {}}),
            ))

    def apply_chart_level_decision(
        self,
        watch_id: str,
        *,
        decision: str,
        manual_levels: dict[str, Any] | None = None,
        decided_by: str = "user",
    ) -> dict[str, Any]:
        normalized = str(decision or "").strip().upper()
        if normalized not in {"ACCEPT_VALIDATED", "KEEP_PLANNER", "EDIT_MANUALLY"}:
            raise ValueError("decision must be ACCEPT_VALIDATED, KEEP_PLANNER, or EDIT_MANUALLY")
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active monitor setup not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            if setup.plan_stale:
                raise ValueError("PLAN_STALE: reanalyze before accepting or retaining active chart levels")
            previous = _loads(setup.active_levels_json)
            planner = _loads(setup.planner_levels_json)
            manual = _loads(setup.manual_overrides_json)
            proposed = _loads(setup.llm_proposed_levels_json)
            validated = _loads(setup.validated_chart_levels_json)
            latest_review = db.get(ChartStructureReview, setup.latest_chart_review_id) if setup.latest_chart_review_id else None
            review_validation = _loads(latest_review.validation_json) if latest_review else {}
            if normalized == "ACCEPT_VALIDATED":
                result = reconcile_levels(
                    planner_levels=planner,
                    proposed_levels=proposed,
                    validation=review_validation or {"accepted_levels": validated},
                    manual_overrides=manual,
                )
                if result.get("critical_rejections"):
                    setup.reconciliation_status = "MANUAL_REVIEW_REQUIRED"
                    setup.chart_analysis_status = "MANUAL_REVIEW_REQUIRED"
                    self._set_plan_gate_state(watch, setup)
                    db.commit()
                    raise ValueError("LLM correction cannot be accepted because a critical proposed level was rejected; manual review is required")
                selected = result["final_active_levels"]
                sources = result["level_sources"]
                setup.trigger_source = "VALIDATED_CHART_LLM"
                setup.chart_analysis_status = "MODIFIED" if result["has_disagreement"] else "AGREES"
                reconciliation_status = "LLM_CORRECTION_ACCEPTED" if result["has_disagreement"] else "PLANNER_ACCEPTED"
            elif normalized == "KEEP_PLANNER":
                selected = dict(planner)
                selected.update(manual)
                sources = {name: "PLANNER" for name in LEVEL_NAMES if number(planner.get(name)) is not None}
                sources.update({name: "MANUAL" for name in LEVEL_NAMES if number(manual.get(name)) is not None})
                setup.trigger_source = "MANUAL" if manual else "PLANNER"
                setup.chart_analysis_status = "AGREES"
                reconciliation_status = "MANUAL_OVERRIDE" if manual else "PLANNER_ACCEPTED"
            else:
                clean: dict[str, float | None] = {}
                for name, value in (manual_levels or {}).items():
                    if name not in {*LEVEL_NAMES, "max_chase_price"}:
                        continue
                    if value in (None, ""):
                        clean[name] = None
                    else:
                        parsed = float(value)
                        if parsed <= 0:
                            raise ValueError(f"{name} must be positive")
                        clean[name] = parsed
                if not clean:
                    raise ValueError("At least one supported manual level is required")
                manual.update(clean)
                selected = dict(previous)
                selected.update(clean)
                sources = _loads(setup.level_sources_json)
                sources.update({name: "MANUAL" for name in clean})
                setup.manual_overrides_json = _dumps(manual)
                setup.trigger_source = "MANUAL"
                setup.chart_analysis_status = "MANUAL_OVERRIDE"
                reconciliation_status = "MANUAL_OVERRIDE"
            previous_primary = number(previous.get("primary_entry_trigger"))
            selected_primary = number(selected.get("primary_entry_trigger"))
            bundle = self._chart_bundle(db, watch, setup)
            final_plan = self._persist_final_active_plan(
                setup,
                levels=selected,
                sources=sources,
                current_price=number(bundle.get("reference_price")) or number(watch.current_price) or setup.plan_reference_price,
                reconciliation_status=reconciliation_status,
                structure_bars=((bundle.get("timeframes") or {}).get("structure") or {}).get("bars") or [],
                execution_bars=((bundle.get("timeframes") or {}).get("execution") or {}).get("bars") or [],
                entry_changed=selected_primary != previous_primary,
                change_source="MANUAL" if normalized == "EDIT_MANUALLY" else "VALIDATED_CHART_LLM" if normalized == "ACCEPT_VALIDATED" else setup.trigger_source,
                level_reasons=_llm_level_reasons(_loads(latest_review.llm_output_json) if latest_review else {}),
            )
            selected = final_plan["flat_levels"]
            sources = final_plan["level_sources"]
            self._set_plan_gate_state(watch, setup)
            setup.updated_at = _utcnow()
            row = ChartLevelDecision(
                id=_id(),
                watch_id=watch.id,
                setup_id=setup.id,
                chart_review_id=setup.latest_chart_review_id,
                ticker=watch.ticker,
                decision=normalized,
                previous_active_levels_json=_dumps(previous),
                selected_levels_json=_dumps(selected),
                level_sources_json=_dumps(sources),
                decided_by=decided_by,
            )
            db.add(row)
            review_input = _loads(latest_review.deterministic_input_json) if latest_review else {}
            self._record_level_revisions(
                db, watch=watch, setup=setup, review_row=latest_review,
                planner_levels=planner, proposed_levels=proposed,
                validated_levels=validated, final_levels=selected, sources=sources,
                sanity=review_input.get("level_sanity") or {},
                validation=review_validation,
                llm_output=_loads(latest_review.llm_output_json) if latest_review else {},
            )
            self._event(
                db,
                watch,
                setup=setup,
                event_type="chart_level_decision",
                message=f"Chart level decision recorded: {normalized}",
                snapshot={
                    "decision_id": row.id, "decision": normalized,
                    "selected_levels": selected, "sources": sources,
                    "final_active_plan_id": final_plan["plan_id"],
                    "final_plan_validation": final_plan["validation"],
                    "target_regeneration": final_plan["target_regeneration"],
                },
            )
            db.commit()
            return self._watch_payload(db, watch, include_detail=True)

    def evaluate_watch(self, watch_id: str, *, bars_1m: list[dict] | None = None, bars_5m: list[dict] | None = None, now: datetime | None = None) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active monitor setup not found")
            if not watch.monitor_active or watch.removed_at is not None:
                return self._watch_payload(db, watch, include_detail=True)
            setup = db.get(MonitorSetup, watch.current_setup_id)
            current_time = now or _utcnow()
            if setup.expires_at and setup.expires_at < current_time:
                previous = watch.state
                watch.state = MonitorState.EXPIRED.value
                watch.monitor_active = False
                setup.status = "expired"
                self._event(db, watch, setup=setup, event_type="setup_expired", from_state=previous, to_state=watch.state, message="Configured setup expiry reached")
                db.commit()
                return self._watch_payload(db, watch, include_detail=True)

            levels = _loads(setup.active_levels_json)
            if bars_1m is None:
                bars_1m = self._fetch_bars(watch.ticker, "one_minute", 1, force_refresh=False)
            if bars_5m is None:
                bars_5m = self._fetch_bars(watch.ticker, "five_minute", 5, force_refresh=False)
            latest_runtime_price = number((bars_1m[-1] if bars_1m else {}).get("close")) or number((bars_5m[-1] if bars_5m else {}).get("close")) or number(watch.current_price)
            persisted_final_plan = _loads(setup.final_active_plan_json)
            if not persisted_final_plan:
                persisted_final_plan = self._persist_final_active_plan(
                    setup,
                    levels=levels,
                    sources=_loads(setup.level_sources_json),
                    current_price=latest_runtime_price or setup.plan_reference_price,
                    reconciliation_status=setup.reconciliation_status or "PLANNER_ACCEPTED",
                    structure_bars=bars_5m,
                    execution_bars=bars_1m,
                )
                levels = persisted_final_plan["flat_levels"]
            final_plan_validation = validate_final_plan(
                levels=levels,
                current_price=latest_runtime_price,
                market_snapshot_id=setup.market_snapshot_id,
                level_metadata=persisted_final_plan.get("levels") or {},
                config=self.config,
            )
            setup.final_plan_validation_json = _dumps(final_plan_validation)
            setup.plan_integrity_status = final_plan_validation["status"]
            attempt_count = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).count()
            evaluation = evaluate_monitor(
                previous_state=watch.state,
                levels={**levels, "_setup_family": setup.setup_family},
                bars_1m=bars_1m,
                bars_5m=bars_5m,
                setup_valid=setup.valid_setup,
                now=current_time,
                config=self.config,
                prior_attempt_count=attempt_count,
            )
            latest_5m_close = number((bars_5m[-1] if bars_5m else {}).get("close"))
            runtime_consistency = check_data_consistency(
                planner_price=None,
                monitor_price=evaluation.get("current_price"),
                chart_close=latest_5m_close,
                atr=levels.get("atr"),
            )
            runtime_reference = number(evaluation.get("current_price")) or latest_5m_close or 1.0
            runtime_tolerance = max(
                self.config.market_data_mismatch_pct,
                (number(levels.get("atr")) or 0.0) / runtime_reference,
            )
            runtime_mismatch = bool(
                runtime_consistency.get("max_difference_pct") is not None
                and float(runtime_consistency["max_difference_pct"]) > runtime_tolerance
            )
            runtime_consistency["configured_tolerance_pct"] = round(runtime_tolerance, 6)
            runtime_consistency["status"] = "MARKET_DATA_MISMATCH" if runtime_mismatch else "CONSISTENT"
            semantic_validation = validate_level_semantics(
                current_price=evaluation.get("current_price"),
                levels=levels,
                config=self.config,
            )
            level_sanity = evaluate_level_sanity(
                current_price=float(evaluation.get("current_price") or runtime_reference),
                atr=float(number(levels.get("atr")) or runtime_reference * 0.02),
                levels=levels, structure_bars=bars_5m,
                execution_bars=bars_1m, config=self.config,
            )
            stale_plan = detect_stale_plan(
                current_price=evaluation.get("current_price"),
                levels=levels,
                atr=levels.get("atr"),
                setup_created_at=setup.created_at,
                plan_reference_price=setup.plan_reference_price,
                plan_created_at=setup.plan_created_at,
                structure_bars=bars_5m,
                data_consistency_status=(
                    "MARKET_DATA_MISMATCH" if runtime_mismatch else None
                ),
                config=self.config,
            )
            evaluation["market_snapshot_id"] = setup.market_snapshot_id
            evaluation["plan_reference_price"] = setup.plan_reference_price
            evaluation["plan_created_at"] = setup.plan_created_at
            evaluation["price_drift_pct"] = stale_plan.get("price_drift_pct")
            evaluation["price_drift_atr"] = stale_plan.get("price_drift_atr")
            evaluation["plan_age_seconds"] = stale_plan.get("plan_age_seconds")
            evaluation["level_semantic_validation"] = semantic_validation
            evaluation["level_sanity"] = level_sanity
            evaluation["runtime_data_consistency"] = runtime_consistency
            evaluation["final_active_plan_id"] = setup.final_active_plan_id
            evaluation["final_active_plan_market_snapshot_id"] = setup.market_snapshot_id
            evaluation["plan_integrity_status"] = setup.plan_integrity_status
            evaluation["final_plan_validation"] = final_plan_validation
            evaluation["reconciliation_status"] = setup.reconciliation_status
            if stale_plan["stale"]:
                if evaluation["state"] not in {MonitorState.INVALIDATED.value, MonitorState.DATA_STALE.value}:
                    evaluation["pre_stale_state"] = evaluation["state"]
                    evaluation["state"] = MonitorState.PLAN_STALE.value
                evaluation["plan_stale"] = True
                evaluation["plan_stale_reasons"] = stale_plan["reasons"]
                evaluation["plan_stale_warnings"] = stale_plan.get("warnings") or []
                evaluation.setdefault("hard_blockers", []).append("plan_stale_reanalysis_required")
                evaluation["manual_order_plan"] = None
                evaluation["current_executable_rr"] = None
                evaluation["current_rr_tp1"] = None
                setup.chart_analysis_status = "STALE"
                setup.plan_stale = True
                setup.plan_stale_reason = "; ".join(stale_plan["reasons"])
                setup.plan_stale_reasons_json = _dumps(stale_plan["reasons"])
                if self.config.auto_propose_reanalysis_on_stale:
                    current = number(evaluation.get("current_price"))
                    atr = number(levels.get("atr")) or (current or 1.0) * 0.02
                    if current is not None:
                        setup.proposed_setup_json = _dumps(
                            derive_chart_level_candidates(
                                current_price=current,
                                atr=atr,
                                planner_levels=_loads(setup.planner_levels_json),
                                structure_bars=bars_5m,
                                execution_bars=bars_1m,
                                config=self.config,
                            )
                        )
            else:
                evaluation["plan_stale"] = False
                evaluation["plan_stale_reasons"] = []
                setup.plan_stale = False
                setup.plan_stale_reason = None
                setup.plan_stale_reasons_json = _dumps([])
                if setup.chart_analysis_status == "STALE":
                    setup.chart_analysis_status = "MODIFIED" if setup.trigger_source in {"MANUAL", "VALIDATED_CHART_LLM"} else "AGREES"
            protected_states = {
                MonitorState.INVALIDATED.value,
                MonitorState.DATA_STALE.value,
                MonitorState.PLAN_STALE.value,
            }
            if evaluation["state"] not in protected_states and setup.plan_integrity_status == "INVALID":
                evaluation["pre_integrity_state"] = evaluation["state"]
                evaluation["state"] = MonitorState.PLAN_GEOMETRY_INVALID.value
                evaluation.setdefault("hard_blockers", []).append("plan_geometry_invalid")
                evaluation["manual_order_plan"] = None
                evaluation["planned_rr_at_primary_trigger"] = None
                evaluation["current_executable_rr"] = None
                evaluation["current_rr_tp1"] = None
            elif evaluation["state"] not in protected_states and setup.reconciliation_status == "MANUAL_REVIEW_REQUIRED":
                evaluation["pre_reconciliation_state"] = evaluation["state"]
                evaluation["state"] = MonitorState.PLAN_REVIEW_REQUIRED.value
                evaluation.setdefault("hard_blockers", []).append("level_disagreement_manual_review_required")
                evaluation["manual_order_plan"] = None
                evaluation["current_executable_rr"] = None
                evaluation["current_rr_tp1"] = None
            evaluation["evaluated_at"] = current_time
            historical_creation = _loads(setup.planner_baseline_json).get("historical_context_at_creation") or {}
            evaluation["rule_version"] = setup.rule_version
            evaluation["learning_profile_version"] = historical_creation.get("profile_version")
            evaluation["learning_profile_version_id"] = historical_creation.get("profile_version_id")
            evaluation["llm_prompt_version"] = PROMPT_VERSION
            learned_adjustments = learned_adjustment_payloads(db, setup.id)
            score_breakdown = adjustment_breakdown(
                setup.setup_quality_score,
                learned_adjustments,
                self.config.max_historical_score_adjustment,
            )
            evaluation["historical_context"] = {
                "profile": (_loads(setup.planner_baseline_json).get("historical_context_at_creation") or {}),
                "learned_adjustments": learned_adjustments,
                "recommendation_breakdown": score_breakdown,
            }
            evaluation["base_deterministic_state"] = evaluation["state"]
            prefers_retest = any(
                item.get("adjustment_type") == "CONFIRMATION_PREFERENCE"
                and item.get("evidence_strength") in {"MODERATE", "STRONG"}
                for item in learned_adjustments
            )
            retest_held = str(evaluation.get("retest_result") or "").upper() in {"HELD", "SUCCESS", "PASSED"}
            hard_state = evaluation["state"] in {
                MonitorState.INVALIDATED.value, MonitorState.DATA_STALE.value,
                MonitorState.PLAN_STALE.value, MonitorState.MISSED.value,
            }
            if (
                not hard_state and prefers_retest and not retest_held and attempt_count == 0
                and evaluation["state"] in {MonitorState.APPROVED.value, MonitorState.STRONGLY_CONFIRMED.value}
            ):
                evaluation["state"] = MonitorState.CONFIRMING.value
                evaluation["historical_recommendation"] = "WAIT_FOR_RETEST"
                evaluation["historical_adjustment_reason"] = (
                    "Evidence-qualified comparable setups favor break/retest confirmation over the first touch."
                )
                evaluation["manual_order_plan"] = None
            else:
                evaluation["historical_recommendation"] = "NO_STATE_OVERRIDE"
            persist_completed_bars(
                db, watch=watch, setup=setup, timeframe="1m",
                bars=bars_1m, evaluation=evaluation,
            )
            persist_completed_bars(
                db, watch=watch, setup=setup, timeframe="5m",
                bars=bars_5m, evaluation=evaluation,
            )
            previous = watch.state
            watch.state = evaluation["state"]
            watch.current_price = evaluation.get("current_price")
            watch.market_data_as_of = evaluation.get("market_data_as_of")
            watch.last_market_data_update_at = evaluation.get("market_data_as_of")
            watch.last_backend_evaluation_at = current_time
            watch.session_label = evaluation.get("market_session")
            watch.latest_evaluation_json = _dumps(evaluation)
            watch.last_polled_at = current_time
            watch.updated_at = current_time
            watch.last_event = evaluation.get("rejection_reason") or watch.state
            attempt = self._update_attempt(db, watch, setup, previous, evaluation)
            if previous != watch.state:
                if watch.state == MonitorState.PLAN_STALE.value:
                    self._event(
                        db,
                        watch,
                        setup=setup,
                        event_type="PLAN_STALE_DETECTED",
                        from_state=previous,
                        to_state=watch.state,
                        message="Active plan is stale; reanalysis is required before approval or order planning",
                        snapshot=stale_plan,
                    )
                self._event(
                    db,
                    watch,
                    setup=setup,
                    attempt=attempt,
                    event_type="state_transition",
                    from_state=previous,
                    to_state=watch.state,
                    message=self._transition_message(watch.ticker, watch.state, evaluation),
                    snapshot=evaluation,
                )
            if watch.state == MonitorState.INVALIDATED.value and not setup.invalidated_at:
                setup.status = "invalidated"
                setup.valid_setup = False
                setup.invalidated_at = current_time
                setup.invalidation_price = evaluation.get("current_price")
                setup.invalidation_reason = "Structural invalidation level lost"
            if previous != watch.state:
                db.add(MonitorDecisionSnapshot(
                    id=_id(), watch_id=watch.id, setup_id=setup.id,
                    attempt_id=attempt.id if attempt else None, ticker=watch.ticker,
                    snapshot_type=watch.state, payload_json=_dumps(evaluation),
                ))
                self._record_shadow_rule_evaluations(db, watch, setup, evaluation)
            if previous != watch.state and watch.state in {
                MonitorState.APPROVED.value,
                MonitorState.STRONGLY_CONFIRMED.value,
                MonitorState.INVALIDATED.value,
                MonitorState.MISSED.value,
                MonitorState.REJECTED_BREAKOUT.value,
            }:
                profile = self._profile_for(db, setup, persist=True)
                self._maybe_generate_observation(db, setup, profile)
            if previous != watch.state and watch.state in {MonitorState.APPROVED.value, MonitorState.STRONGLY_CONFIRMED.value}:
                if float(setup.setup_quality_score or 0.0) >= self.config.auto_llm_min_setup_score:
                    self._request_llm_review(db, watch, setup, attempt, evaluation)
            self._update_open_manual_trades(db, watch, evaluation)
            db.commit()
            payload = self._watch_payload(db, watch, include_detail=True)
            payload["evaluation"] = evaluation
            chart_review_type = None
            if previous != watch.state and watch.state in {
                MonitorState.APPROVED.value,
                MonitorState.STRONGLY_CONFIRMED.value,
            }:
                chart_review_type = "CONFIRMED_TRADE_REVIEW"
            elif previous != watch.state and watch.state in {
                MonitorState.NEAR_TRIGGER.value,
                MonitorState.REJECTED_BREAKOUT.value,
                MonitorState.INVALIDATED.value,
                MonitorState.MISSED.value,
                MonitorState.PLAN_STALE.value,
            }:
                chart_review_type = "CHART_STRUCTURE_REVIEW"
            elif level_sanity.get("review_required"):
                chart_review_type = "CHART_LEVEL_REVIEW"
        if chart_review_type:
            self._schedule_chart_review(watch_id, chart_review_type, str(payload.get("state") or "setup_reanalyzed"))
        return payload

    def _update_attempt(self, db: Session, watch: LiveWatch, setup: MonitorSetup, previous: str, evaluation: dict) -> ConfirmationAttempt | None:
        open_attempt = (
            db.query(ConfirmationAttempt)
            .filter(ConfirmationAttempt.setup_id == setup.id, ConfirmationAttempt.ended_at.is_(None))
            .order_by(ConfirmationAttempt.attempt_number.desc())
            .first()
        )
        active_attempt_states = {MonitorState.ARMED.value, MonitorState.CONFIRMING.value}
        terminal_attempt_states = {
            MonitorState.APPROVED.value,
            MonitorState.STRONGLY_CONFIRMED.value,
            MonitorState.REJECTED_BREAKOUT.value,
            MonitorState.INVALIDATED.value,
            MonitorState.MISSED.value,
        }
        trigger_crossed = bool(((evaluation.get("confirmation_components") or {}).get("trigger_crossed") or {}).get("passed"))
        should_create_attempt = evaluation["state"] in active_attempt_states or (
            evaluation["state"] in terminal_attempt_states and trigger_crossed
        )
        if should_create_attempt and open_attempt is None:
            attempt_number = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).count() + 1
            open_attempt = ConfirmationAttempt(
                id=_id(), watch_id=watch.id, setup_id=setup.id, ticker=watch.ticker,
                attempt_number=attempt_number, trigger_price=evaluation.get("primary_entry_trigger"),
                confirmation_method="5m_close_rvol",
            )
            db.add(open_attempt)
            db.flush()
            self._event(db, watch, setup=setup, attempt=open_attempt, event_type="confirmation_attempt_started", message=f"Confirmation attempt #{attempt_number} started", snapshot=evaluation)
        if open_attempt:
            price = evaluation.get("current_price")
            if price is not None:
                open_attempt.peak_price = max(float(open_attempt.peak_price or price), float(price))
                open_attempt.lowest_retest_price = min(float(open_attempt.lowest_retest_price or price), float(price))
            open_attempt.rvol_1m = evaluation.get("rvol_1m")
            open_attempt.rvol_5m = evaluation.get("rvol_5m")
            open_attempt.price_confirmation = bool(evaluation.get("price_confirmation"))
            open_attempt.volume_confirmation = bool(evaluation.get("volume_confirmation"))
            open_attempt.retest_result = evaluation.get("retest_result")
            open_attempt.evidence_json = _dumps(evaluation)
            if evaluation["state"] in terminal_attempt_states:
                open_attempt.outcome = evaluation["state"]
                open_attempt.rejection_reason = evaluation.get("rejection_reason")
                open_attempt.ended_at = _utcnow()
        return open_attempt

    def request_llm_review(self, watch_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Monitor not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            if setup.plan_stale or watch.state in {MonitorState.PLAN_STALE.value, MonitorState.DATA_STALE.value}:
                raise ValueError("PLAN_STALE: fresh reanalysis is required before LLM review")
            snapshot = (
                db.query(MonitorDecisionSnapshot)
                .filter(MonitorDecisionSnapshot.setup_id == setup.id)
                .order_by(MonitorDecisionSnapshot.created_at.desc())
                .first()
            )
            evaluation = _loads(snapshot.payload_json) if snapshot else {"state": watch.state, "hard_blockers": ["confirmation_not_recorded"]}
            attempt = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).order_by(ConfirmationAttempt.attempt_number.desc()).first()
            review = self._request_llm_review(db, watch, setup, attempt, evaluation)
            db.commit()
            return review

    def _request_llm_review(self, db: Session, watch: LiveWatch, setup: MonitorSetup, attempt: ConfirmationAttempt | None, evaluation: dict) -> dict[str, Any]:
        if setup.plan_stale or evaluation.get("plan_stale"):
            return {
                "status": "blocked",
                "decision": "WAIT",
                "confidence": 0.0,
                "reason_summary": "Fresh reanalysis is required; stale levels were not sent to the LLM.",
            }
        baseline = _loads(setup.planner_baseline_json)
        context = load_historical_context(db, setup, self.config)
        profile = context.get("ticker_profile") or self._profile_for(db, setup)
        similar = self._similar_cases(db, setup, attempt)
        packet = build_advisory_packet(
            baseline=baseline, evaluation=evaluation,
            historical_profile=profile, similar_cases=similar,
            learned_adjustments=learned_adjustment_payloads(db, setup.id),
            broader_profiles=context.get("broader_profiles") or {},
            past_postmortems=past_postmortems(db, setup.ticker),
        )
        review = review_advisory_packet(packet, self._advisory_provider)
        row = LLMAdvisoryReview(
            id=_id(), watch_id=watch.id, setup_id=setup.id,
            attempt_id=attempt.id if attempt else None, ticker=watch.ticker,
            model=review.get("model"), prompt_version=PROMPT_VERSION,
            decision=review["decision"], confidence=review["confidence"],
            status=review.get("status", "available"), reason_summary=review.get("reason_summary"),
            input_snapshot_json=_dumps(packet), output_json=_dumps(review),
            hard_blockers_json=_dumps(review.get("hard_blockers") or []),
        )
        db.add(row)
        self._event(db, watch, setup=setup, attempt=attempt, event_type="llm_review", message=f"LLM advisory: {review['decision']} ({review['confidence']:.0%})", snapshot=review)
        return {"review_id": row.id, **review}

    def journal(self, *, watch_id: str | None = None, ticker: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            query = db.query(MonitorEvent)
            if watch_id:
                query = query.filter(MonitorEvent.watch_id == watch_id)
            if ticker:
                query = query.filter(MonitorEvent.ticker == ticker.upper())
            rows = query.order_by(MonitorEvent.created_at.desc()).limit(max(1, min(limit, 2000))).all()
            return [self._event_payload(row) for row in rows]

    def record_manual_action(self, watch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Monitor not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            if setup.plan_stale and action == "entered":
                raise ValueError("PLAN_STALE: manual order tracking requires a fresh active plan")
            attempt = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).order_by(ConfirmationAttempt.attempt_number.desc()).first()
            if action == "entered":
                trade = ManualMonitorTrade(
                    id=_id(), watch_id=watch.id, setup_id=setup.id,
                    attempt_id=attempt.id if attempt else None, ticker=watch.ticker, status="OPEN",
                    quantity=payload.get("quantity"), planned_entry=payload.get("planned_entry"),
                    actual_entry=payload.get("actual_entry"), stop_price=payload.get("stop_price"),
                    targets_json=_dumps(payload.get("targets") or {}), entered_at=_utcnow(), notes=payload.get("notes"),
                )
                db.add(trade)
                db.flush()
                recommendation = RecommendationOutcome(
                    id=_id(), watch_id=watch.id, setup_id=setup.id,
                    attempt_id=attempt.id if attempt else None, ticker=watch.ticker,
                    user_action="EXECUTED", outcome="tracking",
                    details_json=_dumps({"manual_trade_id": trade.id}),
                )
                db.add(recommendation)
                result = {"trade_id": trade.id, "recommendation_outcome_id": recommendation.id, "status": trade.status}
            elif action == "skipped":
                outcome = RecommendationOutcome(
                    id=_id(), watch_id=watch.id, setup_id=setup.id,
                    attempt_id=attempt.id if attempt else None, ticker=watch.ticker,
                    user_action="SKIPPED", outcome="tracking", details_json=_dumps(payload),
                )
                db.add(outcome)
                result = {"recommendation_outcome_id": outcome.id, "status": "tracking"}
            elif action in {"exited", "partial_exit"}:
                trade_id = str(payload.get("trade_id") or "")
                trade = db.get(ManualMonitorTrade, trade_id)
                if trade is None or trade.watch_id != watch.id:
                    raise LookupError("Manual trade not found")
                exit_price = float(payload.get("exit_price"))
                if action == "exited":
                    trade.status = "CLOSED"
                    trade.exit_price = exit_price
                    trade.exited_at = _utcnow()
                    quantity = float(trade.quantity or 0.0)
                    entry = float(trade.actual_entry or 0.0)
                    trade.realised_pnl = (exit_price - entry) * quantity
                    risk = entry - float(trade.stop_price or entry)
                    trade.r_multiple = None if risk <= 0 else round((exit_price - entry) / risk, 4)
                    recommendation = (
                        db.query(RecommendationOutcome)
                        .filter(
                            RecommendationOutcome.setup_id == setup.id,
                            RecommendationOutcome.user_action == "EXECUTED",
                            RecommendationOutcome.resolved_at.is_(None),
                        )
                        .order_by(RecommendationOutcome.created_at.desc())
                        .first()
                    )
                    if recommendation:
                        recommendation.outcome = "EXECUTED_CLOSED"
                        recommendation.r_multiple = trade.r_multiple
                        recommendation.mfe_pct = trade.mfe_pct
                        recommendation.mae_pct = trade.mae_pct
                        recommendation.resolved_at = _utcnow()
                trade.updated_at = _utcnow()
                result = {"trade_id": trade.id, "status": trade.status}
            else:
                raise ValueError("action must be entered, skipped, partial_exit, or exited")
            latest_review = (
                db.query(LLMAdvisoryReview)
                .filter(LLMAdvisoryReview.setup_id == setup.id)
                .order_by(LLMAdvisoryReview.created_at.desc())
                .first()
            )
            if latest_review:
                latest_review.final_user_action = action.upper()
            self._event(db, watch, setup=setup, attempt=attempt, event_type=f"manual_trade_{action}", message=f"Manual trade action recorded: {action}", snapshot=payload)
            db.commit()
        if action in {"entered", "exited"}:
            self._schedule_chart_snapshot(watch_id, f"manual_trade_{action}")
        return result

    def stock_profile(self, ticker: str) -> dict[str, Any]:
        with SessionLocal() as db:
            setup = db.query(MonitorSetup).filter(MonitorSetup.ticker == ticker.upper()).order_by(MonitorSetup.created_at.desc()).first()
            if setup is None:
                return {"ticker": ticker.upper(), "observation_count": 0, "evidence_strength": "INSUFFICIENT", "statistics": {}}
            payload = self._profile_for(db, setup, persist=True)
            db.commit()
            return payload

    def learning_overview(self) -> dict[str, Any]:
        with SessionLocal() as db:
            profiles = db.query(StockBehaviorProfile).order_by(StockBehaviorProfile.updated_at.desc()).all()
            observations = db.query(LearningObservation).order_by(LearningObservation.created_at.desc()).limit(100).all()
            proposals = db.query(LearningProposal).order_by(LearningProposal.created_at.desc()).limit(100).all()
            rules = db.query(MonitorRuleVersion).order_by(MonitorRuleVersion.created_at.desc()).limit(50).all()
            profile_versions = db.query(BehaviorProfileVersion).order_by(BehaviorProfileVersion.created_at.desc()).limit(200).all()
            jobs = db.query(LearningJobRun).order_by(LearningJobRun.started_at.desc()).limit(50).all()
            shadows = db.query(ShadowRuleEvaluation).order_by(ShadowRuleEvaluation.created_at.desc()).limit(100).all()
            postmortems = db.query(LLMDecisionPostmortem).order_by(LLMDecisionPostmortem.created_at.desc()).limit(200).all()
            level_revisions = db.query(LevelRevision).all()
            helpful = sum("LLM_CORRECTION_HELPFUL" in _loads(row.rationale_tags_json, []) for row in postmortems)
            return {
                "profiles": [{"id": row.id, "scope_type": row.scope_type, "scope_value": row.scope_value, "observation_count": row.observation_count, "evidence_strength": row.evidence_strength, "statistics": _loads(row.statistics_json), "updated_at": row.updated_at} for row in profiles],
                "observations": [{"id": row.id, "scope_type": row.scope_type, "scope_value": row.scope_value, "observation_type": row.observation_type, "summary": row.summary, "sample_size": row.sample_size, "evidence_strength": row.evidence_strength, "evidence": _loads(row.evidence_json), "created_at": row.created_at} for row in observations],
                "proposals": [self._proposal_payload(row) for row in proposals],
                "rule_versions": [{"id": row.id, "version": row.version, "status": row.status, "proposal_id": row.proposal_id, "rules": _loads(row.rules_json), "approved_at": row.approved_at} for row in rules],
                "profile_versions": [{
                    "id": row.id, "scope_type": row.scope_type, "scope_value": row.scope_value,
                    "version": row.version, "observation_count": row.observation_count,
                    "weighted_observation_count": row.weighted_observation_count,
                    "evidence_strength": row.evidence_strength, "reliability": row.reliability,
                    "formula_version": row.formula_version, "source_cutoff_at": row.source_cutoff_at,
                    "created_at": row.created_at,
                } for row in profile_versions],
                "learning_jobs": [{
                    "id": row.id, "trading_date": row.trading_date, "status": row.status,
                    "summaries_finalized": row.summaries_finalized,
                    "profiles_updated": row.profiles_updated,
                    "observations_created": row.observations_created,
                    "details": _loads(row.details_json), "started_at": row.started_at,
                    "completed_at": row.completed_at,
                } for row in jobs],
                "paper_tests": [{
                    "id": row.id, "proposal_id": row.proposal_id,
                    "production_decision": row.production_decision,
                    "shadow_decision": row.shadow_decision,
                    "production_outcome": row.production_outcome,
                    "shadow_hypothetical_outcome": row.shadow_hypothetical_outcome,
                    "resolved_at": row.resolved_at,
                    "evidence": _loads(row.evidence_json), "created_at": row.created_at,
                } for row in shadows],
                "llm_performance": {
                    "evaluated_reviews": len(postmortems),
                    "aligned_with_outcome": helpful,
                    "alignment_rate": None if not postmortems else round(helpful / len(postmortems), 4),
                },
                "level_accuracy": {
                    "revision_count": len(level_revisions),
                    "validated_count": sum(row.validation_result == "VALIDATED" for row in level_revisions),
                    "planner_active_count": sum(row.source == "PLANNER" for row in level_revisions),
                    "chart_llm_active_count": sum(row.source == "VALIDATED_CHART_LLM" for row in level_revisions),
                    "manual_active_count": sum(row.source == "MANUAL" for row in level_revisions),
                },
            }

    def create_learning_observation(self, ticker: str) -> dict[str, Any]:
        with SessionLocal() as db:
            setup = db.query(MonitorSetup).filter(MonitorSetup.ticker == ticker.upper()).order_by(MonitorSetup.created_at.desc()).first()
            if setup is None:
                raise LookupError("No monitor history for ticker")
            profile = self._profile_for(db, setup, persist=True)
            if profile["observation_count"] < self.config.learning_min_observations:
                return {"created": False, "reason": "minimum_observations_not_met", "profile": profile}
            statistics = profile["statistics"]
            observation = LearningObservation(
                id=_id(), scope_type="ticker", scope_value=setup.ticker,
                observation_type="confirmation_performance",
                summary=f"{setup.ticker} confirmation history has {profile['observation_count']} observations; false-breakout rate is {statistics.get('false_breakout_rate')}.",
                sample_size=profile["observation_count"], evidence_strength=profile["evidence_strength"],
                evidence_json=_dumps(statistics),
            )
            db.add(observation)
            db.commit()
            return {"created": True, "observation_id": observation.id, "profile": profile}

    def _maybe_generate_observation(self, db: Session, setup: MonitorSetup, profile: dict[str, Any]) -> None:
        """Create sparse periodic observations, never proposals or active rules."""
        sample_size = int(profile.get("observation_count") or 0)
        if sample_size < self.config.learning_min_observations:
            return
        latest = (
            db.query(LearningObservation)
            .filter(LearningObservation.scope_type == "ticker", LearningObservation.scope_value == setup.ticker)
            .order_by(LearningObservation.created_at.desc())
            .first()
        )
        if latest and sample_size - int(latest.sample_size or 0) < self.config.learning_min_observations:
            return
        statistics = profile.get("statistics") or {}
        db.add(LearningObservation(
            id=_id(), scope_type="ticker", scope_value=setup.ticker,
            observation_type="periodic_confirmation_review",
            summary=(
                f"{setup.ticker} reached {sample_size} confirmation observations; "
                f"false-breakout rate is {statistics.get('false_breakout_rate')}."
            ),
            sample_size=sample_size, evidence_strength=profile.get("evidence_strength") or "INSUFFICIENT",
            evidence_json=_dumps(statistics),
        ))

    def create_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal() as db:
            row = LearningProposal(
                id=_id(), observation_id=payload.get("observation_id"),
                scope_type=str(payload.get("scope_type") or "global"),
                scope_value=str(payload.get("scope_value") or "all"),
                status="PENDING", title=str(payload.get("title") or "Monitor rule proposal"),
                proposed_change_json=_dumps(payload.get("proposed_change") or {}),
                evidence_json=_dumps(payload.get("evidence") or {}),
            )
            db.add(row)
            db.commit()
            return self._proposal_payload(row)

    def decide_proposal(self, proposal_id: str, *, decision: str, decided_by: str = "user") -> dict[str, Any]:
        normalized = decision.strip().upper()
        if normalized not in {"APPROVE", "REJECT", "PAPER_TEST", "LATER"}:
            raise ValueError("decision must be APPROVE, REJECT, PAPER_TEST, or LATER")
        with SessionLocal() as db:
            proposal = db.get(LearningProposal, proposal_id)
            if proposal is None:
                raise LookupError("Proposal not found")
            proposal.status = {
                "APPROVE": "APPROVED", "REJECT": "REJECTED",
                "PAPER_TEST": "PAPER_TESTING", "LATER": "DEFERRED",
            }[normalized]
            proposal.decided_at = _utcnow()
            proposal.decided_by = decided_by
            if normalized == "APPROVE":
                version = f"live-monitor-{_utcnow().strftime('%Y%m%d%H%M%S')}"
                db.add(MonitorRuleVersion(
                    id=_id(), version=version, status="ACTIVE", proposal_id=proposal.id,
                    rules_json=proposal.proposed_change_json, approved_by=decided_by, approved_at=_utcnow(),
                ))
            elif normalized == "PAPER_TEST":
                db.add(ShadowRuleEvaluation(
                    id=_id(), proposal_id=proposal.id,
                    production_decision="UNCHANGED", shadow_decision="PENDING",
                    evidence_json=_dumps({"message": "Shadow mode enabled; production rules remain unchanged."}),
                ))
            db.commit()
            return self._proposal_payload(proposal)

    @staticmethod
    def _proposal_matches_setup(proposal: LearningProposal, setup: MonitorSetup) -> bool:
        scope = str(proposal.scope_type or "").lower()
        value = str(proposal.scope_value or "")
        if scope == "ticker":
            return value.upper() == setup.ticker.upper()
        if scope == "setup_type":
            return value == str(setup.setup_type or "")
        if scope == "setup_family":
            return value == str(setup.setup_family or "")
        if scope == "sector":
            return value == str(setup.sector or "")
        if scope == "market_regime":
            return value == str(setup.market_regime or "")
        return scope in {"global", "all"}

    def _record_shadow_rule_evaluations(
        self,
        db: Session,
        watch: LiveWatch,
        setup: MonitorSetup,
        evaluation: dict[str, Any],
    ) -> None:
        """Evaluate supported paper rules without changing the production state."""
        proposals = db.query(LearningProposal).filter(LearningProposal.status == "PAPER_TESTING").all()
        production = str(evaluation.get("state") or watch.state)
        retest_held = str(evaluation.get("retest_result") or "").upper() in {"HELD", "SUCCESS", "PASSED"}
        for proposal in proposals:
            if not self._proposal_matches_setup(proposal, setup):
                continue
            change = _loads(proposal.proposed_change_json)
            preferred = str(
                change.get("preferred_confirmation")
                or change.get("confirmation_preference")
                or ""
            ).upper()
            requires_retest = preferred == "BREAK_RETEST" or number(change.get("retest_weight")) is not None
            shadow = production
            if (
                requires_retest
                and production in {MonitorState.APPROVED.value, MonitorState.STRONGLY_CONFIRMED.value}
                and not retest_held
            ):
                shadow = "WAIT_FOR_RETEST"
            db.add(ShadowRuleEvaluation(
                id=_id(), proposal_id=proposal.id, watch_id=watch.id, setup_id=setup.id,
                production_decision=production, shadow_decision=shadow,
                evidence_json=_dumps({
                    "proposed_change": change,
                    "production_unchanged": True,
                    "evaluated_at": evaluation.get("evaluated_at"),
                    "market_snapshot_id": setup.market_snapshot_id,
                    "price_confirmation": evaluation.get("price_confirmation"),
                    "volume_confirmation": evaluation.get("volume_confirmation"),
                    "retest_result": evaluation.get("retest_result"),
                    "hard_blockers": evaluation.get("hard_blockers") or [],
                }),
                created_at=_as_datetime(evaluation.get("evaluated_at")) or _utcnow(),
            ))

    def _profile_for(self, db: Session, setup: MonitorSetup, *, persist: bool = False) -> dict[str, Any]:
        profile = refresh_profile(
            db, scope_type="ticker", scope_value=setup.ticker,
            config=self.config, force_version=False,
        )
        chart_reviews = db.query(ChartStructureReview).filter(ChartStructureReview.ticker == setup.ticker).all()
        chart_decisions = db.query(ChartLevelDecision).filter(ChartLevelDecision.ticker == setup.ticker).all()
        level_revisions = db.query(LevelRevision).filter(LevelRevision.ticker == setup.ticker).all()
        statistics = profile["statistics"]
        statistics["chart_level_history"] = {
            "review_count": len(chart_reviews),
            "disagreement_count": sum(row.status == "DISAGREEMENT" for row in chart_reviews),
            "validation_failure_count": sum(row.status == "VALIDATION_FAILED" for row in chart_reviews),
            "accepted_validated_count": sum(row.decision == "ACCEPT_VALIDATED" for row in chart_decisions),
            "auto_corrected_count": sum(row.decision == "AUTO_ACCEPT_VALIDATED" for row in chart_decisions),
            "kept_planner_count": sum(row.decision == "KEEP_PLANNER" for row in chart_decisions),
            "manual_override_count": sum(row.decision == "EDIT_MANUALLY" for row in chart_decisions),
            "level_revision_count": len(level_revisions),
        }
        profile["ticker"] = setup.ticker
        profile["statistics"] = statistics
        return profile

    def _similar_cases(self, db: Session, setup: MonitorSetup, attempt: ConfirmationAttempt | None, limit: int = 8) -> list[dict]:
        effective_limit = max(1, min(limit or self.config.similar_case_count, self.config.similar_case_count))
        baseline = _loads(setup.planner_baseline_json)
        levels = _loads(setup.active_levels_json)
        current_price = number(baseline.get("current_price")) or setup.plan_reference_price
        atr = number(levels.get("atr")) or number(baseline.get("atr"))
        primary = number(levels.get("primary_entry_trigger"))
        support = number(levels.get("optional_support_level"))
        current = {
            "ticker": setup.ticker, "broader_structure": setup.broader_structure,
            "setup_type": setup.setup_type, "setup_family": setup.setup_family,
            "execution_structure": setup.execution_structure,
            "sector": setup.sector, "market_regime": setup.market_regime,
            "confirmation_method": attempt.confirmation_method if attempt else None,
            "attempt_number": attempt.attempt_number if attempt else None,
            "atr_pct": baseline.get("atr_pct"), "rsi": baseline.get("rsi"),
            "distance_from_support_atr": None if not current_price or not atr or support is None else (current_price - support) / atr,
            "primary_trigger_distance_atr": None if not current_price or not atr or primary is None else (primary - current_price) / atr,
            "rvol_5m": attempt.rvol_5m if attempt else None,
            "qqq_condition": baseline.get("qqq_context"),
            "sector_condition": baseline.get("sector_context"),
        }
        rows = db.query(ConfirmationAttempt, MonitorSetup).join(
            MonitorSetup, ConfirmationAttempt.setup_id == MonitorSetup.id,
        ).filter(ConfirmationAttempt.setup_id != setup.id).order_by(
            ConfirmationAttempt.started_at.desc(),
        ).limit(500).all()
        cases = []
        for candidate, candidate_setup in rows:
            evidence = _loads(candidate.evidence_json)
            candidate_baseline = _loads(candidate_setup.planner_baseline_json)
            recommendation = db.query(RecommendationOutcome).filter(
                RecommendationOutcome.attempt_id == candidate.id,
            ).order_by(RecommendationOutcome.created_at.desc()).first()
            case = {
                "attempt_id": candidate.id, "ticker": candidate.ticker,
                "broader_structure": candidate_setup.broader_structure,
                "setup_type": candidate_setup.setup_type,
                "setup_family": candidate_setup.setup_family,
                "execution_structure": candidate_setup.execution_structure,
                "sector": candidate_setup.sector, "market_regime": candidate_setup.market_regime,
                "chart_analysis_status": candidate_setup.chart_analysis_status,
                "level_sources": _loads(candidate_setup.level_sources_json),
                "confirmation_method": candidate.confirmation_method,
                "attempt_number": candidate.attempt_number, "outcome": candidate.outcome,
                "r_multiple": recommendation.r_multiple if recommendation else None,
                "rvol_5m": candidate.rvol_5m, "atr_pct": candidate_baseline.get("atr_pct"),
                "rsi": candidate_baseline.get("rsi"),
                "distance_from_support_atr": evidence.get("distance_from_support_atr"),
                "primary_trigger_distance_atr": evidence.get("distance_to_trigger_atr"),
                "qqq_condition": candidate_baseline.get("qqq_context"),
                "sector_condition": candidate_baseline.get("sector_context"),
                "occurred_at": candidate.ended_at or candidate.started_at,
            }
            case.update(similar_case_score(
                current, case,
                weights=self.config.similarity_weights,
                continuous_weights=self.config.similarity_continuous,
            ))
            cases.append(case)
        daily_rows = db.query(MonitorDailySummary).filter(
            MonitorDailySummary.setup_id != setup.id,
            MonitorDailySummary.number_of_trigger_attempts == 0,
        ).order_by(MonitorDailySummary.trading_date.desc()).limit(200).all()
        for summary in daily_rows:
            indicators = _loads(summary.indicators_json)
            context = _loads(summary.context_json)
            decisions = _loads(summary.decisions_json)
            outcome = _loads(summary.outcome_json)
            case = {
                "daily_summary_id": summary.id, "ticker": summary.ticker,
                "broader_structure": summary.broader_structure,
                "setup_type": summary.setup_type,
                "setup_family": summary.setup_family,
                "execution_structure": summary.execution_structure,
                "sector": summary.sector, "market_regime": summary.market_regime,
                "confirmation_method": decisions.get("confirmation_method") or "NO_TRIGGER",
                "attempt_number": 0,
                "outcome": outcome.get("recommendation_outcome") or summary.highest_state_reached,
                "r_multiple": summary.recommendation_r_multiple,
                "atr_pct": indicators.get("atr_pct"), "rsi": indicators.get("rsi"),
                "rvol_5m": indicators.get("rvol_5m"),
                "qqq_condition": context.get("qqq_context"),
                "sector_condition": context.get("sector_context"),
                "occurred_at": summary.finalized_at,
            }
            case.update(similar_case_score(
                current, case,
                weights=self.config.similarity_weights,
                continuous_weights=self.config.similarity_continuous,
            ))
            cases.append(case)
        return sorted(cases, key=lambda row: row["similarity_score"], reverse=True)[:effective_limit]

    def _update_open_manual_trades(self, db: Session, watch: LiveWatch, evaluation: dict) -> None:
        current = evaluation.get("current_price")
        if current is None:
            return
        trades = db.query(ManualMonitorTrade).filter(ManualMonitorTrade.watch_id == watch.id, ManualMonitorTrade.status == "OPEN").all()
        for trade in trades:
            entry = float(trade.actual_entry or 0.0)
            if entry <= 0:
                continue
            excursion = (float(current) - entry) / entry * 100.0
            trade.mfe_pct = max(float(trade.mfe_pct or 0.0), excursion)
            trade.mae_pct = min(float(trade.mae_pct or 0.0), excursion)
            trade.updated_at = _utcnow()
        recommendations = (
            db.query(RecommendationOutcome)
            .filter(RecommendationOutcome.watch_id == watch.id, RecommendationOutcome.resolved_at.is_(None))
            .all()
        )
        setup = db.get(MonitorSetup, watch.current_setup_id) if watch.current_setup_id else None
        levels = _loads(setup.active_levels_json) if setup else {}
        trigger = levels.get("primary_entry_trigger")
        stop = levels.get("suggested_stop") or levels.get("invalidation_level")
        tp1 = levels.get("tp1")
        if trigger:
            excursion = (float(current) - float(trigger)) / float(trigger) * 100.0
            for recommendation in recommendations:
                recommendation.mfe_pct = max(float(recommendation.mfe_pct or 0.0), excursion)
                recommendation.mae_pct = min(float(recommendation.mae_pct or 0.0), excursion)
                if tp1 and float(current) >= float(tp1):
                    recommendation.outcome = "TP1_REACHED"
                    recommendation.resolved_at = _utcnow()
                elif stop and float(current) <= float(stop):
                    recommendation.outcome = "STOP_OR_INVALIDATION_REACHED"
                    recommendation.resolved_at = _utcnow()

    def _watch_payload(self, db: Session, watch: LiveWatch, include_detail: bool = False) -> dict[str, Any]:
        setup = db.get(MonitorSetup, watch.current_setup_id) if watch.current_setup_id else None
        levels = _loads(setup.active_levels_json) if setup else {}
        evaluation = _loads(watch.latest_evaluation_json)
        plan_stale = bool(setup and (setup.plan_stale or watch.state == MonitorState.PLAN_STALE.value))
        display_levels = {} if plan_stale else levels
        reference_price = setup.plan_reference_price if setup else None
        price_drift_pct = evaluation.get("price_drift_pct")
        if price_drift_pct is None and number(watch.current_price) is not None and number(reference_price) is not None:
            price_drift_pct = (float(watch.current_price) - float(reference_price)) / float(reference_price)
        snapshot_row = db.get(MarketSnapshot, setup.market_snapshot_id) if setup and setup.market_snapshot_id else None
        snapshot = market_snapshot_payload(snapshot_row)
        latest_chart_snapshot = (
            db.query(ChartSnapshot)
            .filter(ChartSnapshot.setup_id == setup.id)
            .order_by(ChartSnapshot.generated_at.desc())
            .first()
            if setup else None
        )
        payload = {
            "id": watch.id,
            "ticker": watch.ticker,
            "source": watch.source,
            "monitor_active": watch.monitor_active,
            "state": watch.state,
            "current_setup_id": watch.current_setup_id,
            "market_snapshot_id": setup.market_snapshot_id if setup else None,
            "current_market_snapshot_id": watch.market_snapshot_id,
            "current_price": watch.current_price,
            "current_price_timestamp": watch.market_data_as_of,
            "plan_reference_price": reference_price,
            "plan_created_at": setup.plan_created_at if setup else None,
            "market_data_timestamp": setup.market_data_timestamp if setup else None,
            "price_drift_pct": price_drift_pct,
            "price_drift_atr": evaluation.get("price_drift_atr"),
            "plan_stale": plan_stale,
            "plan_stale_reasons": _loads(setup.plan_stale_reasons_json, []) if setup else [],
            "action_required": "REANALYZE_REQUIRED" if plan_stale else None,
            "market_data_as_of": watch.market_data_as_of,
            "market_session": watch.session_label,
            "last_event": watch.last_event,
            "last_polled_at": watch.last_polled_at,
            "last_backend_evaluation_at": watch.last_backend_evaluation_at,
            "last_market_data_update_at": watch.last_market_data_update_at,
            "updated_at": watch.updated_at,
            "primary_trigger": display_levels.get("primary_entry_trigger"),
            "near_confirmation": display_levels.get("near_confirmation"),
            "strong_confirmation": display_levels.get("strong_confirmation"),
            "major_trend_repair": display_levels.get("major_trend_repair"),
            "previous_planner_primary_trigger": levels.get("primary_entry_trigger") if plan_stale else None,
            "distance_to_trigger_pct": evaluation.get("distance_to_trigger_pct"),
            "rvol_1m": evaluation.get("rvol_1m"),
            "rvol_5m": evaluation.get("rvol_5m"),
            "price_confirmation": evaluation.get("price_confirmation"),
            "volume_confirmation": evaluation.get("volume_confirmation"),
            "setup_valid": bool(setup and setup.valid_setup and not plan_stale),
            "setup_type": setup.setup_type if setup else None,
            "setup_family": setup.setup_family if setup else None,
            "broader_structure": setup.broader_structure if setup else None,
            "execution_structure": setup.execution_structure if setup else None,
            "trigger_source": setup.trigger_source if setup else None,
            "live_confirmation_score": evaluation.get("live_confirmation_score"),
            "current_rr_tp1": None if plan_stale else evaluation.get("current_rr_tp1"),
            "planned_rr_at_primary_trigger": None if plan_stale else evaluation.get("planned_rr_at_primary_trigger"),
            "current_executable_rr": None if plan_stale else evaluation.get("current_executable_rr"),
            "data_age_seconds": evaluation.get("data_age_seconds"),
            "max_chase_price": display_levels.get("max_chase_price"),
            "chart_analysis_status": setup.chart_analysis_status if setup else "NOT_RUN",
            "reconciliation_status": setup.reconciliation_status if setup else None,
            "plan_integrity_status": setup.plan_integrity_status if setup else None,
            "final_active_plan_id": setup.final_active_plan_id if setup else None,
            "final_plan_validation": _loads(setup.final_plan_validation_json) if setup else {},
            "plan_stale_reason": setup.plan_stale_reason if setup else None,
            "planner_chart_snapshot_ids_match": bool(
                setup
                and setup.market_snapshot_id
                and latest_chart_snapshot
                and latest_chart_snapshot.market_snapshot_id == setup.market_snapshot_id
            ),
        }
        if include_detail and setup:
            final_active_plan = _loads(setup.final_active_plan_json)
            latest_chart_review_row = db.get(ChartStructureReview, setup.latest_chart_review_id) if setup.latest_chart_review_id else None
            adjustment_rows = learned_adjustment_payloads(db, setup.id)
            profile_payload = self._profile_for(db, setup)
            score_breakdown = adjustment_breakdown(
                setup.setup_quality_score, adjustment_rows,
                self.config.max_historical_score_adjustment,
            )
            revision_rows = db.query(LevelRevision).filter(
                LevelRevision.setup_id == setup.id,
            ).order_by(LevelRevision.created_at.desc()).all()
            latest_revision_by_name: dict[str, LevelRevision] = {}
            for revision in revision_rows:
                latest_revision_by_name.setdefault(revision.level_name, revision)
            active_level_metadata = final_active_plan.get("levels") or {
                name: {
                    "price": display_levels.get(name),
                    "level_type": LEVEL_ROLES.get(name, name.upper()),
                    "source": (_loads(setup.level_sources_json).get(name) or "PLANNER"),
                    "reason": latest_revision_by_name[name].reason if name in latest_revision_by_name else "Current planner/monitor level",
                    "confidence": latest_revision_by_name[name].confidence if name in latest_revision_by_name else None,
                    "created_at": latest_revision_by_name[name].created_at if name in latest_revision_by_name else setup.created_at,
                    "market_snapshot_id": setup.market_snapshot_id,
                }
                for name in LEVEL_NAMES if number(display_levels.get(name)) is not None
            }
            payload.update({
                "planner_baseline": _loads(setup.planner_baseline_json),
                "planner_levels": {} if plan_stale else _loads(setup.planner_levels_json),
                "active_levels": display_levels,
                "final_active_plan": final_active_plan,
                "final_active_plan_id": setup.final_active_plan_id,
                "final_plan_validation": _loads(setup.final_plan_validation_json),
                "plan_integrity_status": setup.plan_integrity_status,
                "reconciliation_status": setup.reconciliation_status,
                "reconciliation_details": _loads(setup.proposed_setup_json),
                "plan_versions": {
                    "planner_original_plan": {
                        "setup_id": setup.id,
                        "market_snapshot_id": setup.market_snapshot_id,
                        "created_at": setup.plan_created_at,
                        "source": "PLANNER",
                        "levels": _loads(setup.planner_levels_json),
                    },
                    "llm_proposed_plan": {
                        "setup_id": setup.id,
                        "market_snapshot_id": setup.market_snapshot_id,
                        "created_at": latest_chart_review_row.created_at if latest_chart_review_row else None,
                        "source": "CHART_LLM",
                        "levels": _loads(setup.llm_proposed_levels_json),
                    },
                    "validated_plan": {
                        "setup_id": setup.id,
                        "market_snapshot_id": setup.market_snapshot_id,
                        "created_at": latest_chart_review_row.created_at if latest_chart_review_row else None,
                        "source": "DETERMINISTIC_VALIDATOR",
                        "levels": _loads(setup.validated_chart_levels_json),
                    },
                    "final_active_plan": final_active_plan,
                    "manual_override_plan": {
                        "setup_id": setup.id,
                        "market_snapshot_id": setup.market_snapshot_id,
                        "created_at": setup.updated_at,
                        "source": "MANUAL",
                        "levels": _loads(setup.manual_overrides_json),
                    },
                },
                "historical_stale_levels": levels if plan_stale else {},
                "market_snapshot": {key: value for key, value in snapshot.items() if key != "bars"},
                "manual_overrides": _loads(setup.manual_overrides_json),
                "llm_proposed_levels": _loads(setup.llm_proposed_levels_json),
                "validated_chart_levels": _loads(setup.validated_chart_levels_json),
                "level_sources": _loads(setup.level_sources_json),
                "proposed_setup": _loads(setup.proposed_setup_json),
                "latest_evaluation": evaluation,
                "attempts": [self._attempt_payload(row) for row in db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).order_by(ConfirmationAttempt.attempt_number.asc()).all()],
                "llm_reviews": [self._review_payload(row) for row in db.query(LLMAdvisoryReview).filter(LLMAdvisoryReview.setup_id == setup.id).order_by(LLMAdvisoryReview.created_at.desc()).all()],
                "chart_reviews": [self._chart_review_payload(row) for row in db.query(ChartStructureReview).filter(ChartStructureReview.setup_id == setup.id).order_by(ChartStructureReview.created_at.desc()).all()],
                "chart_snapshots": [self._chart_snapshot_payload(row) for row in db.query(ChartSnapshot).filter(ChartSnapshot.setup_id == setup.id).order_by(ChartSnapshot.generated_at.desc()).all()],
                "chart_level_decisions": [self._chart_level_decision_payload(row) for row in db.query(ChartLevelDecision).filter(ChartLevelDecision.setup_id == setup.id).order_by(ChartLevelDecision.created_at.desc()).all()],
                "manual_trades": [self._trade_payload(row) for row in db.query(ManualMonitorTrade).filter(ManualMonitorTrade.setup_id == setup.id).order_by(ManualMonitorTrade.created_at.desc()).all()],
                "journal": [self._event_payload(row) for row in db.query(MonitorEvent).filter(MonitorEvent.watch_id == watch.id).order_by(MonitorEvent.created_at.desc()).limit(200).all()],
                "historical_profile": profile_payload,
                "learned_adjustments": adjustment_rows,
                "recommendation_breakdown": score_breakdown,
                "similar_historical_cases": self._similar_cases(db, setup, None),
                "past_llm_postmortems": past_postmortems(db, setup.ticker),
                "active_level_metadata": active_level_metadata,
                "level_revisions": [self._level_revision_payload(row) for row in revision_rows[:200]],
                "daily_summaries": [
                    self._daily_summary_payload(row)
                    for row in db.query(MonitorDailySummary).filter(
                        MonitorDailySummary.setup_id == setup.id,
                    ).order_by(MonitorDailySummary.trading_date.desc()).limit(30).all()
                ],
                "setup_history": [
                    {
                        "setup_id": row.id,
                        "version": row.version,
                        "status": row.status,
                        "market_snapshot_id": row.market_snapshot_id,
                        "plan_reference_price": row.plan_reference_price,
                        "plan_created_at": row.plan_created_at,
                        "previous_setup_id": row.previous_setup_id,
                        "replaced_by_setup_id": row.replaced_by_setup_id,
                        "replacement_reason": row.replacement_reason,
                        "primary_entry_trigger": _loads(row.planner_levels_json).get("primary_entry_trigger"),
                        "optional_support_level": _loads(row.planner_levels_json).get("optional_support_level"),
                        "tp1": _loads(row.planner_levels_json).get("tp1"),
                        "tp2": _loads(row.planner_levels_json).get("tp2"),
                        "tp3": _loads(row.planner_levels_json).get("tp3"),
                    }
                    for row in db.query(MonitorSetup).filter(MonitorSetup.watch_id == watch.id).order_by(MonitorSetup.version.desc()).all()
                ],
            })
        return payload

    def _event(self, db: Session, watch: LiveWatch, *, event_type: str, message: str, setup: MonitorSetup | None = None, attempt: ConfirmationAttempt | None = None, from_state: str | None = None, to_state: str | None = None, snapshot: dict | None = None) -> MonitorEvent:
        row = MonitorEvent(id=_id(), watch_id=watch.id, setup_id=setup.id if setup else watch.current_setup_id, attempt_id=attempt.id if attempt else None, ticker=watch.ticker, event_type=event_type, from_state=from_state, to_state=to_state, message=message, snapshot_json=_dumps(snapshot) if snapshot is not None else None)
        db.add(row)
        watch.last_event = message
        return row

    @staticmethod
    def _transition_message(ticker: str, state: str, evaluation: dict) -> str:
        reason = evaluation.get("rejection_reason")
        suffix = f": {reason}" if reason else ""
        return f"{ticker} transitioned to {state}{suffix}"

    @staticmethod
    def _event_payload(row: MonitorEvent) -> dict[str, Any]:
        return {"id": row.id, "watch_id": row.watch_id, "setup_id": row.setup_id, "attempt_id": row.attempt_id, "ticker": row.ticker, "event_type": row.event_type, "from_state": row.from_state, "to_state": row.to_state, "message": row.message, "snapshot": _loads(row.snapshot_json), "created_at": row.created_at}

    @staticmethod
    def _attempt_payload(row: ConfirmationAttempt) -> dict[str, Any]:
        return {"id": row.id, "attempt_number": row.attempt_number, "started_at": row.started_at, "ended_at": row.ended_at, "trigger_price": row.trigger_price, "peak_price": row.peak_price, "lowest_retest_price": row.lowest_retest_price, "rvol_1m": row.rvol_1m, "rvol_5m": row.rvol_5m, "price_confirmation": row.price_confirmation, "volume_confirmation": row.volume_confirmation, "retest_result": row.retest_result, "confirmation_method": row.confirmation_method, "outcome": row.outcome, "rejection_reason": row.rejection_reason, "evidence": _loads(row.evidence_json)}

    @staticmethod
    def _review_payload(row: LLMAdvisoryReview) -> dict[str, Any]:
        return {"id": row.id, "model": row.model, "prompt_version": row.prompt_version, "decision": row.decision, "confidence": row.confidence, "status": row.status, "reason_summary": row.reason_summary, "output": _loads(row.output_json), "hard_blockers": _loads(row.hard_blockers_json, []), "created_at": row.created_at}

    @staticmethod
    def _chart_review_payload(row: ChartStructureReview) -> dict[str, Any]:
        deterministic_input = _loads(row.deterministic_input_json)
        validation = _loads(row.validation_json)
        return {
            "id": row.id,
            "review_type": row.review_type,
            "market_snapshot_id": row.market_snapshot_id,
            "status": row.status,
            "model": row.model,
            "prompt_version": row.prompt_version,
            "chart_snapshot_ids": _loads(row.chart_snapshot_ids_json, []),
            "planner_levels": _loads(row.planner_levels_json),
            "llm_output": _loads(row.llm_output_json),
            "llm_proposed_levels": _loads(row.llm_proposed_levels_json),
            "validated_levels": _loads(row.validated_levels_json),
            "validation": validation,
            "level_sanity": deterministic_input.get("level_sanity") or {},
            "pricing_anomalies": deterministic_input.get("pricing_anomalies") or [],
            "historical_profile_version": ((deterministic_input.get("historical_profile") or {}).get("profile_version")),
            "decision": row.decision,
            "confidence": row.confidence,
            "reason_summary": row.reason_summary,
            "data_consistency_status": row.data_consistency_status,
            "created_at": row.created_at,
        }

    @staticmethod
    def _chart_snapshot_payload(row: ChartSnapshot) -> dict[str, Any]:
        return {
            "id": row.id,
            "timeframe": row.timeframe,
            "event_type": row.event_type,
            "market_snapshot_id": row.market_snapshot_id,
            "image_path": row.image_path,
            "data_source": row.data_source,
            "data_last_bar_at": row.data_last_bar_at,
            "decision_time_boundary": row.decision_time_boundary,
            "metadata": _loads(row.metadata_json),
            "generated_at": row.generated_at,
            "retain_permanently": row.retain_permanently,
        }

    @staticmethod
    def _chart_level_decision_payload(row: ChartLevelDecision) -> dict[str, Any]:
        return {
            "id": row.id,
            "chart_review_id": row.chart_review_id,
            "decision": row.decision,
            "previous_active_levels": _loads(row.previous_active_levels_json),
            "selected_levels": _loads(row.selected_levels_json),
            "level_sources": _loads(row.level_sources_json),
            "decided_by": row.decided_by,
            "created_at": row.created_at,
        }

    @staticmethod
    def _level_revision_payload(row: LevelRevision) -> dict[str, Any]:
        return {
            "id": row.id, "chart_review_id": row.chart_review_id,
            "market_snapshot_id": row.market_snapshot_id,
            "level_name": row.level_name, "level_role": row.level_role,
            "planner_price": row.planner_price, "llm_proposed_price": row.llm_proposed_price,
            "validated_price": row.validated_price, "manual_price": row.manual_price,
            "final_active_price": row.final_active_price, "source": row.source,
            "validation_result": row.validation_result, "confidence": row.confidence,
            "reason": row.reason, "anomaly_flags": _loads(row.anomaly_flags_json, []),
            "outcome": _loads(row.outcome_json), "created_at": row.created_at,
        }

    @staticmethod
    def _daily_summary_payload(row: MonitorDailySummary) -> dict[str, Any]:
        return {
            "id": row.id, "trading_date": row.trading_date,
            "ticker": row.ticker, "setup_id": row.setup_id,
            "open": row.open_price, "high": row.high_price,
            "low": row.low_price, "close": row.close_price,
            "starting_monitor_price": row.starting_monitor_price,
            "ending_monitor_price": row.ending_monitor_price,
            "broader_structure": row.broader_structure, "setup_type": row.setup_type,
            "setup_family": row.setup_family,
            "execution_structure": row.execution_structure, "market_regime": row.market_regime,
            "levels": _loads(row.levels_json), "indicators": _loads(row.indicators_json),
            "context": _loads(row.context_json), "decisions": _loads(row.decisions_json),
            "outcome": _loads(row.outcome_json),
            "data_quality_flags": _loads(row.data_quality_flags_json, []),
            "number_of_trigger_attempts": row.number_of_trigger_attempts,
            "number_of_rejections": row.number_of_rejections,
            "highest_state_reached": row.highest_state_reached,
            "mfe_atr": row.mfe_atr, "mae_atr": row.mae_atr,
            "recommendation_r_multiple": row.recommendation_r_multiple,
            "actual_trade_executed": row.actual_trade_executed,
            "actual_trade_r_multiple": row.actual_trade_r_multiple,
            "finalized_at": row.finalized_at,
        }

    @staticmethod
    def _trade_payload(row: ManualMonitorTrade) -> dict[str, Any]:
        return {"id": row.id, "status": row.status, "quantity": row.quantity, "planned_entry": row.planned_entry, "actual_entry": row.actual_entry, "stop_price": row.stop_price, "targets": _loads(row.targets_json), "entered_at": row.entered_at, "exited_at": row.exited_at, "exit_price": row.exit_price, "realised_pnl": row.realised_pnl, "r_multiple": row.r_multiple, "mfe_pct": row.mfe_pct, "mae_pct": row.mae_pct, "notes": row.notes}

    @staticmethod
    def _proposal_payload(row: LearningProposal) -> dict[str, Any]:
        return {"id": row.id, "observation_id": row.observation_id, "scope_type": row.scope_type, "scope_value": row.scope_value, "status": row.status, "title": row.title, "proposed_change": _loads(row.proposed_change_json), "evidence": _loads(row.evidence_json), "decided_at": row.decided_at, "decided_by": row.decided_by, "created_at": row.created_at}


_SERVICE: LiveMonitorService | None = None


def get_live_monitor_service() -> LiveMonitorService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = LiveMonitorService()
    return _SERVICE
