from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from threading import Event, Lock, Thread
import time
from typing import Any, Callable
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.market_data import get_bars
from app.models import (
    ChartLevelDecision,
    ChartSnapshot,
    ChartStructureReview,
    ConfirmationAttempt,
    LearningObservation,
    LearningProposal,
    LiveWatch,
    LLMAdvisoryReview,
    ManualMonitorTrade,
    MonitorDecisionSnapshot,
    MonitorEvent,
    MonitorRuleVersion,
    MonitorSetup,
    RecommendationOutcome,
    ShadowRuleEvaluation,
    StockBehaviorProfile,
)

from .advisor import PROMPT_VERSION, build_advisory_packet, review_advisory_packet
from .baseline import build_monitor_baseline
from .chart_data import build_chart_bundle
from .chart_levels import LEVEL_NAMES, check_data_consistency, derive_chart_level_candidates, detect_stale_plan, number, reconcile_levels
from .chart_renderer import cleanup_chart_snapshot_retention, render_chart_snapshots
from .chart_review import review_chart_packet
from .config import LiveMonitorConfig, load_live_monitor_config
from .engine import evaluate_monitor
from .enums import ACTIVE_MONITOR_STATES, MonitorState
from .learning import aggregate_attempts, hierarchical_weights, similar_case_score


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
        self._bars_loader = bars_loader or self._load_live_bars
        self._advisory_provider = advisory_provider
        self._chart_review_provider = chart_review_provider
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._running = False
        self._last_cycle_at: datetime | None = None
        self._last_cycle_error: str | None = None

    @staticmethod
    def _load_live_bars(ticker: str, timeframe: str, lookback_days: int | None) -> list[dict]:
        ttl = 20 if timeframe == "one_minute" else 30
        return get_bars(ticker, timeframe, lookback_days, cache_ttl_seconds=ttl)

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
            except Exception:
                return

        Thread(
            target=run,
            name=f"chart-review-{watch_id[:8]}",
            daemon=True,
        ).start()

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
        return results

    def add_monitor(self, ticker: str, *, source: str = "manual", planner_payload: dict | None = None) -> dict[str, Any]:
        symbol = str(ticker or "").strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("A valid ticker is required")
        watch_id: str
        with SessionLocal() as db:
            watch = db.query(LiveWatch).filter(LiveWatch.ticker == symbol).one_or_none()
            if watch and watch.monitor_active and watch.removed_at is None and watch.current_setup_id:
                return self._watch_payload(db, watch, include_detail=True)
            baseline = build_monitor_baseline(db, symbol, supplied_plan=planner_payload, config=self.config)
            now = _utcnow()
            if watch is None:
                watch = LiveWatch(id=_id(), ticker=symbol, source=source, created_at=now)
                db.add(watch)
                db.flush()
            watch.source = source
            watch.monitor_active = True
            watch.removed_at = None
            watch.state = MonitorState.WATCHING.value
            watch.current_price = baseline["plan"].get("current_price")
            watch.updated_at = now
            setup = self._create_setup(db, watch, baseline)
            self._event(db, watch, setup=setup, event_type="monitor_added", to_state=watch.state, message=f"{symbol} added to live monitor")
            db.commit()
            watch_id = watch.id
        if self.config.chart_review_on_add:
            try:
                self.run_chart_review(
                    watch_id,
                    review_type="CHART_STRUCTURE_REVIEW",
                    automatic=True,
                    snapshot_event_type="monitor_added",
                )
            except Exception:
                # The deterministic monitor must remain usable when charts or the LLM are unavailable.
                pass
        return self.get_monitor(watch_id)

    def _create_setup(self, db: Session, watch: LiveWatch, baseline: dict) -> MonitorSetup:
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
            execution_structure=plan.get("execution_structure") or plan.get("trade_shape"),
            sector=plan.get("sector"),
            industry=plan.get("industry"),
            market_regime=plan.get("market_regime"),
            planner_baseline_json=_dumps(plan),
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
        watch.current_setup_id = setup.id
        return setup

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
                MonitorState.MISSED.value: 8,
                MonitorState.INVALIDATED.value: 9,
                MonitorState.PAUSED.value: 10,
                MonitorState.STOPPED.value: 11,
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
                number = float(value)
                if number <= 0:
                    raise ValueError(f"{key} must be positive")
                clean[key] = number
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None or not watch.current_setup_id:
                raise LookupError("Active setup not found")
            setup = db.get(MonitorSetup, watch.current_setup_id)
            active = _loads(setup.active_levels_json)
            manual = _loads(setup.manual_overrides_json)
            sources = _loads(setup.level_sources_json)
            active.update(clean)
            manual.update(clean)
            sources.update({name: "MANUAL" for name in clean})
            setup.active_levels_json = _dumps(active)
            setup.manual_overrides_json = _dumps(manual)
            setup.level_sources_json = _dumps(sources)
            setup.trigger_source = "MANUAL"
            setup.chart_analysis_status = "MANUAL_OVERRIDE"
            setup.max_chase_price = active.get("max_chase_price")
            setup.updated_at = _utcnow()
            self._event(db, watch, setup=setup, event_type="levels_overridden", message="Manual level overlay updated", snapshot={"manual_overrides": manual, "planner_originals": _loads(setup.planner_levels_json)})
            db.commit()
            return self._watch_payload(db, watch, include_detail=True)

    def reanalyze(self, watch_id: str, planner_payload: dict | None = None) -> dict[str, Any]:
        with SessionLocal() as db:
            watch = db.get(LiveWatch, watch_id)
            if watch is None:
                raise LookupError("Monitor not found")
            old_setup = db.get(MonitorSetup, watch.current_setup_id) if watch.current_setup_id else None
            baseline = build_monitor_baseline(db, watch.ticker, supplied_plan=planner_payload, config=self.config)
            new_setup = self._create_setup(db, watch, baseline)
            if old_setup:
                old_setup.status = "replaced"
                old_setup.replaced_by_setup_id = new_setup.id
                old_setup.updated_at = _utcnow()
            previous = watch.state
            watch.state = MonitorState.WATCHING.value
            watch.monitor_active = True
            watch.updated_at = _utcnow()
            self._event(db, watch, setup=new_setup, event_type="setup_reanalyzed", from_state=previous, to_state=watch.state, message="Planner baseline reanalyzed; prior setup preserved")
            db.commit()
        try:
            self.run_chart_review(
                watch_id,
                review_type="CHART_STRUCTURE_REVIEW",
                automatic=False,
                snapshot_event_type="setup_reanalyzed",
            )
        except Exception:
            pass
        return self.get_monitor(watch_id)

    def _chart_bundle(self, db: Session, watch: LiveWatch, setup: MonitorSetup, *, boundary: datetime | None = None) -> dict[str, Any]:
        attempts = (
            db.query(ConfirmationAttempt)
            .filter(ConfirmationAttempt.setup_id == setup.id)
            .order_by(ConfirmationAttempt.attempt_number.asc())
            .all()
        )
        return build_chart_bundle(
            db,
            ticker=watch.ticker,
            levels=_loads(setup.active_levels_json),
            level_sources=_loads(setup.level_sources_json),
            bars_loader=self._bars_loader,
            decision_time_boundary=boundary,
            max_bars=self.config.chart_max_bars,
            attempts=attempts,
        )

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
        if normalized_type not in {"CHART_STRUCTURE_REVIEW", "CONFIRMED_TRADE_REVIEW"}:
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
            current_price = number(watch.current_price) or number(chart_close) or number(baseline.get("current_price"))
            atr = number(active_levels.get("atr")) or number(baseline.get("atr")) or (current_price or 1.0) * 0.02
            consistency = check_data_consistency(
                planner_price=baseline.get("current_price"),
                monitor_price=watch.current_price,
                chart_close=chart_close,
                atr=atr,
            )
            stale = detect_stale_plan(
                current_price=current_price,
                levels=active_levels,
                atr=atr,
                setup_created_at=setup.created_at,
                structure_bars=((bundle.get("timeframes") or {}).get("structure") or {}).get("bars") or [],
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
            packet = {
                "review_type": normalized_type,
                "ticker": watch.ticker,
                "current_price": current_price,
                "atr": atr,
                "broader_structure": setup.broader_structure,
                "setup_type": setup.setup_type,
                "execution_structure": setup.execution_structure,
                "market_regime": setup.market_regime,
                "planner_levels": planner_levels,
                "active_levels": active_levels,
                "stale_plan": stale,
                "data_consistency": consistency,
                "data_source": bundle.get("data_source"),
                "data_timestamp": bundle.get("data_timestamp"),
                "last_bar_timestamp": bundle.get("last_bar_timestamp"),
                "data_freshness_seconds": bundle.get("data_freshness_seconds"),
                "structure_bars": structure_bars,
                "execution_bars": execution_bars,
                "latest_evaluation": _loads(watch.latest_evaluation_json),
                "historical_profile": self._profile_for(db, setup),
                "image_paths": [row.image_path for row in snapshots],
            }
            if consistency["status"] == "CHART_DATA_MISMATCH" or current_price is None:
                review = {
                    "status": "CHART_DATA_MISMATCH" if consistency["status"] == "CHART_DATA_MISMATCH" else "UNAVAILABLE",
                    "model": None,
                    "prompt_version": "chart-structure-review-v1",
                    "output": {},
                    "proposed_levels": {},
                    "validated_levels": {},
                    "validation": {"status": "SKIPPED", "reason": consistency["status"]},
                    "decision": "MANUAL_REVIEW",
                    "confidence": 0.0,
                    "reason_summary": "Chart recommendation skipped because canonical prices are inconsistent or unavailable.",
                }
            else:
                review = review_chart_packet(packet, provider=self._chart_review_provider, config=self.config)
            proposed = review.get("proposed_levels") or {}
            validated = review.get("validated_levels") or {}
            if normalized_type == "CHART_STRUCTURE_REVIEW":
                candidate_reconciliation = reconcile_levels(
                    planner_levels=planner_levels,
                    proposed_levels=proposed,
                    validation={"accepted_levels": validated},
                    manual_overrides=_loads(setup.manual_overrides_json),
                )
            else:
                candidate_reconciliation = {
                    "final_active_levels": active_levels,
                    "level_sources": _loads(setup.level_sources_json),
                    "status": "TRADE_REVIEW",
                    "has_disagreement": False,
                }
            analysis_status = review.get("status") or "VALIDATION_FAILED"
            if normalized_type == "CHART_STRUCTURE_REVIEW" and analysis_status not in {"CHART_DATA_MISMATCH", "UNAVAILABLE", "VALIDATION_FAILED"}:
                analysis_status = "DISAGREEMENT" if candidate_reconciliation["has_disagreement"] else "AGREES"
            row = ChartStructureReview(
                id=_id(),
                watch_id=watch.id,
                setup_id=setup.id,
                ticker=watch.ticker,
                review_type=normalized_type,
                status=analysis_status,
                model=review.get("model"),
                prompt_version=str(review.get("prompt_version") or "chart-structure-review-v1"),
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
            if normalized_type == "CHART_STRUCTURE_REVIEW":
                setup.llm_proposed_levels_json = _dumps(proposed)
                setup.validated_chart_levels_json = _dumps(validated)
                setup.chart_analysis_status = "STALE" if stale["stale"] else analysis_status
            setup.latest_chart_review_id = row.id
            setup.plan_stale_reason = "; ".join(stale["reasons"]) or None
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
                },
            )
            for snapshot in snapshots:
                snapshot.decision_event_id = review_event.id
            db.commit()
            return self._chart_review_payload(row)

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
            previous = _loads(setup.active_levels_json)
            planner = _loads(setup.planner_levels_json)
            manual = _loads(setup.manual_overrides_json)
            proposed = _loads(setup.llm_proposed_levels_json)
            validated = _loads(setup.validated_chart_levels_json)
            if normalized == "ACCEPT_VALIDATED":
                result = reconcile_levels(
                    planner_levels=planner,
                    proposed_levels=proposed,
                    validation={"accepted_levels": validated},
                    manual_overrides=manual,
                )
                selected = result["final_active_levels"]
                sources = result["level_sources"]
                setup.trigger_source = "VALIDATED_CHART_LLM"
                setup.chart_analysis_status = "MODIFIED" if result["has_disagreement"] else "AGREES"
            elif normalized == "KEEP_PLANNER":
                selected = dict(planner)
                selected.update(manual)
                sources = {name: "PLANNER" for name in LEVEL_NAMES if number(planner.get(name)) is not None}
                sources.update({name: "MANUAL" for name in LEVEL_NAMES if number(manual.get(name)) is not None})
                setup.trigger_source = "MANUAL" if manual else "PLANNER"
                setup.chart_analysis_status = "AGREES"
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
            setup.active_levels_json = _dumps(selected)
            setup.level_sources_json = _dumps(sources)
            setup.max_chase_price = selected.get("max_chase_price")
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
            self._event(
                db,
                watch,
                setup=setup,
                event_type="chart_level_decision",
                message=f"Chart level decision recorded: {normalized}",
                snapshot={"decision_id": row.id, "decision": normalized, "selected_levels": selected, "sources": sources},
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
                bars_1m = self._bars_loader(watch.ticker, "one_minute", 1)
            if bars_5m is None:
                bars_5m = self._bars_loader(watch.ticker, "five_minute", 5)
            attempt_count = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.setup_id == setup.id).count()
            evaluation = evaluate_monitor(
                previous_state=watch.state,
                levels=levels,
                bars_1m=bars_1m,
                bars_5m=bars_5m,
                setup_valid=setup.valid_setup,
                now=current_time,
                config=self.config,
                prior_attempt_count=attempt_count,
            )
            stale_plan = detect_stale_plan(
                current_price=evaluation.get("current_price"),
                levels=levels,
                atr=levels.get("atr"),
                setup_created_at=setup.created_at,
                structure_bars=bars_5m,
                config=self.config,
            )
            if stale_plan["stale"] and evaluation["state"] not in {
                MonitorState.INVALIDATED.value,
                MonitorState.DATA_STALE.value,
            }:
                evaluation["pre_stale_state"] = evaluation["state"]
                evaluation["state"] = MonitorState.PLAN_STALE.value
                evaluation["plan_stale"] = True
                evaluation["plan_stale_reasons"] = stale_plan["reasons"]
                evaluation.setdefault("hard_blockers", []).append("plan_stale_reanalysis_required")
                setup.chart_analysis_status = "STALE"
                setup.plan_stale_reason = "; ".join(stale_plan["reasons"])
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
                setup.plan_stale_reason = None
                if setup.chart_analysis_status == "STALE":
                    setup.chart_analysis_status = "MODIFIED" if setup.trigger_source in {"MANUAL", "VALIDATED_CHART_LLM"} else "AGREES"
            previous = watch.state
            watch.state = evaluation["state"]
            watch.current_price = evaluation.get("current_price")
            watch.market_data_as_of = evaluation.get("market_data_as_of")
            watch.session_label = evaluation.get("market_session")
            watch.latest_evaluation_json = _dumps(evaluation)
            watch.last_polled_at = current_time
            watch.updated_at = current_time
            watch.last_event = evaluation.get("rejection_reason") or watch.state
            attempt = self._update_attempt(db, watch, setup, previous, evaluation)
            if previous != watch.state:
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
            if previous != watch.state and watch.state in {
                MonitorState.APPROVED.value,
                MonitorState.STRONGLY_CONFIRMED.value,
                MonitorState.INVALIDATED.value,
                MonitorState.MISSED.value,
                MonitorState.REJECTED_BREAKOUT.value,
            }:
                db.add(MonitorDecisionSnapshot(
                    id=_id(), watch_id=watch.id, setup_id=setup.id,
                    attempt_id=attempt.id if attempt else None, ticker=watch.ticker,
                    snapshot_type=watch.state, payload_json=_dumps(evaluation),
                ))
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
        baseline = _loads(setup.planner_baseline_json)
        profile = self._profile_for(db, setup)
        similar = self._similar_cases(db, setup, attempt)
        packet = build_advisory_packet(baseline=baseline, evaluation=evaluation, historical_profile=profile, similar_cases=similar)
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
            return self._profile_for(db, setup)

    def learning_overview(self) -> dict[str, Any]:
        with SessionLocal() as db:
            profiles = db.query(StockBehaviorProfile).order_by(StockBehaviorProfile.updated_at.desc()).all()
            observations = db.query(LearningObservation).order_by(LearningObservation.created_at.desc()).limit(100).all()
            proposals = db.query(LearningProposal).order_by(LearningProposal.created_at.desc()).limit(100).all()
            rules = db.query(MonitorRuleVersion).order_by(MonitorRuleVersion.created_at.desc()).limit(50).all()
            return {
                "profiles": [{"id": row.id, "scope_type": row.scope_type, "scope_value": row.scope_value, "observation_count": row.observation_count, "evidence_strength": row.evidence_strength, "statistics": _loads(row.statistics_json), "updated_at": row.updated_at} for row in profiles],
                "observations": [{"id": row.id, "scope_type": row.scope_type, "scope_value": row.scope_value, "observation_type": row.observation_type, "summary": row.summary, "sample_size": row.sample_size, "evidence_strength": row.evidence_strength, "evidence": _loads(row.evidence_json), "created_at": row.created_at} for row in observations],
                "proposals": [self._proposal_payload(row) for row in proposals],
                "rule_versions": [{"id": row.id, "version": row.version, "status": row.status, "proposal_id": row.proposal_id, "rules": _loads(row.rules_json), "approved_at": row.approved_at} for row in rules],
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
        if normalized not in {"APPROVE", "REJECT", "PAPER_TEST"}:
            raise ValueError("decision must be APPROVE, REJECT, or PAPER_TEST")
        with SessionLocal() as db:
            proposal = db.get(LearningProposal, proposal_id)
            if proposal is None:
                raise LookupError("Proposal not found")
            proposal.status = {"APPROVE": "APPROVED", "REJECT": "REJECTED", "PAPER_TEST": "PAPER_TESTING"}[normalized]
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

    def _profile_for(self, db: Session, setup: MonitorSetup, *, persist: bool = False) -> dict[str, Any]:
        attempts = db.query(ConfirmationAttempt).filter(ConfirmationAttempt.ticker == setup.ticker).all()
        trades = db.query(ManualMonitorTrade).filter(ManualMonitorTrade.ticker == setup.ticker, ManualMonitorTrade.status == "CLOSED").all()
        chart_reviews = db.query(ChartStructureReview).filter(ChartStructureReview.ticker == setup.ticker).all()
        chart_decisions = db.query(ChartLevelDecision).filter(ChartLevelDecision.ticker == setup.ticker).all()
        attempt_rows = [{"outcome": row.outcome, "confirmation_method": row.confirmation_method, "attempt_number": row.attempt_number} for row in attempts]
        trade_rows = [{"r_multiple": row.r_multiple, "mfe_pct": row.mfe_pct, "mae_pct": row.mae_pct} for row in trades]
        statistics = aggregate_attempts(attempt_rows, trade_rows)
        setup_samples = db.query(ConfirmationAttempt).join(MonitorSetup, ConfirmationAttempt.setup_id == MonitorSetup.id).filter(MonitorSetup.setup_type == setup.setup_type).count() if setup.setup_type else 0
        sector_samples = db.query(ConfirmationAttempt).join(MonitorSetup, ConfirmationAttempt.setup_id == MonitorSetup.id).filter(MonitorSetup.sector == setup.sector).count() if setup.sector else 0
        statistics["hierarchical_weights"] = hierarchical_weights(ticker_samples=len(attempts), setup_samples=setup_samples, sector_samples=sector_samples)
        statistics["chart_level_history"] = {
            "review_count": len(chart_reviews),
            "disagreement_count": sum(row.status == "DISAGREEMENT" for row in chart_reviews),
            "validation_failure_count": sum(row.status == "VALIDATION_FAILED" for row in chart_reviews),
            "accepted_validated_count": sum(row.decision == "ACCEPT_VALIDATED" for row in chart_decisions),
            "kept_planner_count": sum(row.decision == "KEEP_PLANNER" for row in chart_decisions),
            "manual_override_count": sum(row.decision == "EDIT_MANUALLY" for row in chart_decisions),
        }
        payload = {"ticker": setup.ticker, "observation_count": len(attempts), "evidence_strength": statistics["evidence_strength"], "statistics": statistics}
        if persist:
            row = db.query(StockBehaviorProfile).filter(StockBehaviorProfile.scope_type == "ticker", StockBehaviorProfile.scope_value == setup.ticker).one_or_none()
            if row is None:
                row = StockBehaviorProfile(id=_id(), scope_type="ticker", scope_value=setup.ticker, statistics_json="{}")
                db.add(row)
            row.observation_count = len(attempts)
            row.evidence_strength = statistics["evidence_strength"]
            row.statistics_json = _dumps(statistics)
            row.updated_at = _utcnow()
        return payload

    def _similar_cases(self, db: Session, setup: MonitorSetup, attempt: ConfirmationAttempt | None, limit: int = 8) -> list[dict]:
        rows = db.query(ConfirmationAttempt, MonitorSetup).join(MonitorSetup, ConfirmationAttempt.setup_id == MonitorSetup.id).filter(ConfirmationAttempt.setup_id != setup.id).limit(300).all()
        current = {"ticker": setup.ticker, "setup_type": setup.setup_type, "sector": setup.sector, "market_regime": setup.market_regime, "confirmation_method": attempt.confirmation_method if attempt else None, "attempt_number": attempt.attempt_number if attempt else None}
        cases = []
        for candidate, candidate_setup in rows:
            case = {"attempt_id": candidate.id, "ticker": candidate.ticker, "setup_type": candidate_setup.setup_type, "sector": candidate_setup.sector, "market_regime": candidate_setup.market_regime, "chart_analysis_status": candidate_setup.chart_analysis_status, "level_sources": _loads(candidate_setup.level_sources_json), "confirmation_method": candidate.confirmation_method, "attempt_number": candidate.attempt_number, "outcome": candidate.outcome}
            case.update(similar_case_score(current, case))
            cases.append(case)
        return sorted(cases, key=lambda row: row["similarity_score"], reverse=True)[:limit]

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
        payload = {
            "id": watch.id,
            "ticker": watch.ticker,
            "source": watch.source,
            "monitor_active": watch.monitor_active,
            "state": watch.state,
            "current_setup_id": watch.current_setup_id,
            "current_price": watch.current_price,
            "market_data_as_of": watch.market_data_as_of,
            "market_session": watch.session_label,
            "last_event": watch.last_event,
            "last_polled_at": watch.last_polled_at,
            "updated_at": watch.updated_at,
            "primary_trigger": levels.get("primary_entry_trigger"),
            "near_confirmation": levels.get("near_confirmation"),
            "strong_confirmation": levels.get("strong_confirmation"),
            "major_trend_repair": levels.get("major_trend_repair"),
            "distance_to_trigger_pct": evaluation.get("distance_to_trigger_pct"),
            "rvol_1m": evaluation.get("rvol_1m"),
            "rvol_5m": evaluation.get("rvol_5m"),
            "price_confirmation": evaluation.get("price_confirmation"),
            "volume_confirmation": evaluation.get("volume_confirmation"),
            "setup_valid": setup.valid_setup if setup else False,
            "setup_type": setup.setup_type if setup else None,
            "broader_structure": setup.broader_structure if setup else None,
            "execution_structure": setup.execution_structure if setup else None,
            "trigger_source": setup.trigger_source if setup else None,
            "live_confirmation_score": evaluation.get("live_confirmation_score"),
            "current_rr_tp1": evaluation.get("current_rr_tp1"),
            "planned_rr_at_primary_trigger": evaluation.get("planned_rr_at_primary_trigger"),
            "current_executable_rr": evaluation.get("current_executable_rr"),
            "data_age_seconds": evaluation.get("data_age_seconds"),
            "max_chase_price": levels.get("max_chase_price"),
            "chart_analysis_status": setup.chart_analysis_status if setup else "NOT_RUN",
            "plan_stale_reason": setup.plan_stale_reason if setup else None,
        }
        if include_detail and setup:
            payload.update({
                "planner_baseline": _loads(setup.planner_baseline_json),
                "planner_levels": _loads(setup.planner_levels_json),
                "active_levels": levels,
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
                "historical_profile": self._profile_for(db, setup),
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
        return {
            "id": row.id,
            "review_type": row.review_type,
            "status": row.status,
            "model": row.model,
            "prompt_version": row.prompt_version,
            "chart_snapshot_ids": _loads(row.chart_snapshot_ids_json, []),
            "planner_levels": _loads(row.planner_levels_json),
            "llm_output": _loads(row.llm_output_json),
            "llm_proposed_levels": _loads(row.llm_proposed_levels_json),
            "validated_levels": _loads(row.validated_levels_json),
            "validation": _loads(row.validation_json),
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
