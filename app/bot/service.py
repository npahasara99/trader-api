from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time as dt_time
import json
import math
from threading import Event, Lock, Thread
import time
import uuid
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.logic import get_last_price_or_recent_close
from app.market_data import ensure_cached_daily_closes
from app.models import (
    BotConfiguration,
    BotEvent,
    BotHealthSnapshot,
    BotRun,
    BrokerAccount,
    BrokerOrder,
    BrokerOrderEvent,
    CandidateScoreComponent,
    DailyPerformance,
    KillSwitchEvent,
    ManagedPosition,
    PositionEvent,
    ReconciliationRun,
    TradeCandidate,
    TradeMemoryStatistic,
    TradeProposal,
    TradeReview,
)
from app.settings import settings

from .broker import (
    BracketOrderRequest,
    BrokerAccountSummary,
    BrokerError,
    BrokerInterface,
    MockBroker,
    IBKRBroker,
)
from .config import BotConfig, load_bot_config
from .enums import BotRunState, BrokerHealthState, PositionStatus, ProposalStatus, TradingMode


def _build_daily_closes_loader_for_bot(db: Session):
    memo: dict[tuple[str, Any, Any], dict[Any, float]] = {}

    def _loader(symbol, frm, to):
        sym = (symbol or "").strip().upper()
        if not sym:
            return {}
        key = (sym, frm, to)
        if key in memo:
            return memo[key]
        closes = ensure_cached_daily_closes(db, sym, frm, to, auto_fetch=True, commit=False)
        memo[key] = closes
        return closes

    return _loader


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


class TradingBotService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._state = BotRunState.STOPPED
        self._broker_state = BrokerHealthState.DISCONNECTED
        self._kill_switch_active = False
        self._kill_switch_reason: str | None = None
        self._run_id: str | None = None
        self._last_heartbeat: datetime | None = None
        self._last_scan_time: datetime | None = None
        self._last_broker_sync: datetime | None = None
        self._reconciliation_required = True
        self._config = load_bot_config()
        self._broker: BrokerInterface = self._build_broker()

    def _build_broker(self) -> BrokerInterface:
        if self._config.trading_mode in {TradingMode.SHADOW_LIVE, TradingMode.RESTRICTED_LIVE, TradingMode.MANUAL_PAPER, TradingMode.AUTO_PAPER}:
            try:
                return IBKRBroker(
                    host=self._config.ibkr_host,
                    port=self._config.ibkr_port,
                    client_id=self._config.ibkr_client_id,
                    account_id=self._config.ibkr_account_id,
                    read_only=self._config.ibkr_read_only or self._config.trading_mode == TradingMode.SHADOW_LIVE,
                )
            except Exception:
                return MockBroker()
        return MockBroker()

    def get_config(self) -> BotConfig:
        return self._config

    def public_config_dict(self) -> dict[str, Any]:
        payload = asdict(self._config)
        payload["trading_mode"] = self._config.trading_mode.value
        payload["new_entry_start_time"] = self._config.new_entry_start_time.strftime("%H:%M")
        payload["new_entry_end_time"] = self._config.new_entry_end_time.strftime("%H:%M")
        return payload

    def set_config(self, config_patch: dict[str, Any]) -> BotConfig:
        with self._lock:
            merged = {**self.public_config_dict(), **config_patch}
            mode_raw = str(merged.get("trading_mode") or self._config.trading_mode.value).strip().lower()
            merged["trading_mode"] = TradingMode(mode_raw) if mode_raw in {item.value for item in TradingMode} else TradingMode.DISABLED
            for key in ("new_entry_start_time", "new_entry_end_time"):
                value = merged.get(key)
                if isinstance(value, str):
                    try:
                        hour, minute = value.split(":")
                        merged[key] = dt_time(hour=int(hour), minute=int(minute))
                    except Exception:
                        merged[key] = getattr(self._config, key)
            self._config = BotConfig(**merged)
            with SessionLocal() as db:
                self._persist_config(db)
                db.commit()
            return self._config

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._state == BotRunState.RUNNING:
                return {"ok": True, "message": "Bot already running"}
            self._stop_event.clear()
            self._state = BotRunState.RUNNING
            self._run_id = str(uuid.uuid4())
            with SessionLocal() as db:
                db.add(BotRun(run_id=self._run_id, state=self._state.value, trading_mode=self._config.trading_mode.value, heartbeat_at=datetime.now(timezone.utc)))
                self._log_event(db, "bot_started", "ok", "Trading bot started")
                db.commit()
            self._thread = Thread(target=self._run_loop, name="trading-bot-loop", daemon=True)
            self._thread.start()
            return {"ok": True, "message": "Bot started", "run_id": self._run_id}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_event.set()
            self._state = BotRunState.STOPPED
            with SessionLocal() as db:
                self._log_event(db, "bot_stopped", "ok", "Trading bot stopped")
                self._complete_run(db)
                db.commit()
            return {"ok": True, "message": "Bot stopped"}

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._state = BotRunState.PAUSED
            with SessionLocal() as db:
                self._log_event(db, "bot_paused", "ok", "Trading bot paused")
                db.commit()
            return {"ok": True, "message": "Bot paused"}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self._state = BotRunState.RUNNING
            with SessionLocal() as db:
                self._log_event(db, "bot_resumed", "ok", "Trading bot resumed")
                db.commit()
            return {"ok": True, "message": "Bot resumed"}

    def status(self) -> dict[str, Any]:
        return {
            "running": self._state == BotRunState.RUNNING,
            "paused": self._state == BotRunState.PAUSED,
            "state": self._state.value,
            "trading_mode": self._config.trading_mode.value,
            "broker_connected": self._broker_state == BrokerHealthState.CONNECTED,
            "broker_state": self._broker_state.value,
            "execution_mode": self._config.trading_mode.value,
            "auto_execution": self._config.auto_execution,
            "kill_switch_active": self._kill_switch_active,
            "last_heartbeat": self._last_heartbeat,
            "last_scan_time": self._last_scan_time,
            "last_broker_sync": self._last_broker_sync,
            "reconciliation_required": self._reconciliation_required,
            "details": {"run_id": self._run_id, "kill_switch_reason": self._kill_switch_reason},
        }

    def broker_health(self) -> dict[str, Any]:
        try:
            health = self._broker.health_check()
        except Exception as exc:
            health = {"connected": False, "error": str(exc)}
        health["state"] = self._broker_state.value
        return health

    def broker_account(self) -> BrokerAccountSummary:
        return self._broker.account_summary()

    def broker_positions(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._broker.positions()]

    def broker_orders(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._broker.open_orders()]

    def broker_executions(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._broker.executions()]

    def reconnect_broker(self) -> dict[str, Any]:
        try:
            self._broker.reconnect()
            account = self._broker.account_summary()
            self._validate_broker_account(account)
            self._broker_state = BrokerHealthState.CONNECTED
            self._last_broker_sync = datetime.now(timezone.utc)
            return {
                "ok": True,
                "message": "Broker reconnected to verified paper account",
                "account_id_masked": self._mask_account(account.account_id),
                "is_paper": account.is_paper,
            }
        except Exception as exc:
            self._broker_state = BrokerHealthState.DEGRADED
            return {"ok": False, "message": f"Broker reconnect failed: {exc}"}

    def reconcile(self) -> dict[str, Any]:
        with SessionLocal() as db:
            reconciliation_id = str(uuid.uuid4())
            run = ReconciliationRun(reconciliation_id=reconciliation_id, status="running")
            db.add(run)
            db.flush()
            try:
                account = self._broker.account_summary()
                self._validate_broker_account(account)
                db.add(
                    BrokerAccount(
                        account_id_masked=self._mask_account(account.account_id),
                        broker_name="ibkr" if isinstance(self._broker, IBKRBroker) else "mock",
                        account_type=account.account_type,
                        is_paper=account.is_paper,
                        snapshot_json=_json_dumps(asdict(account)),
                    )
                )
                self._reconciliation_required = False
                run.status = "ok"
                run.completed_at = datetime.now(timezone.utc)
                run.summary_json = _json_dumps({"account_id_masked": self._mask_account(account.account_id)})
                self._broker_state = BrokerHealthState.CONNECTED
                self._last_broker_sync = datetime.now(timezone.utc)
                self._log_event(db, "reconciliation_completed", "ok", "Broker reconciliation completed")
                db.commit()
                return {"ok": True, "message": "Reconciliation completed", "reconciliation_id": reconciliation_id}
            except Exception as exc:
                self._reconciliation_required = True
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
                run.summary_json = _json_dumps({"error": str(exc)})
                self._broker_state = BrokerHealthState.RECONCILIATION_REQUIRED
                self._log_event(db, "reconciliation_failed", "error", f"Broker reconciliation failed: {exc}")
                db.commit()
                return {"ok": False, "message": f"Reconciliation failed: {exc}", "reconciliation_id": reconciliation_id}

    def refresh_watchlist(self) -> dict[str, Any]:
        with SessionLocal() as db:
            active_rows = self._load_active_watchlist_rows()
            created = 0
            now = datetime.now(timezone.utc)
            for row in active_rows:
                raw = _safe_json(row.get("raw_result_json"))
                candidate_id = f"{row['ticker']}:{now.strftime('%Y%m%d%H%M%S')}"
                ranking_score = self._rank_candidate(row, raw)
                candidate = TradeCandidate(
                    candidate_id=candidate_id,
                    ticker=row["ticker"],
                    basket="watchlist_snapshots",
                    generated_at=now,
                    status="monitoring",
                    trigger_state=self._trigger_state(row, raw),
                    market_regime=str(raw.get("market_regime") or row.get("market_regime") or "neutral"),
                    setup_type=raw.get("setup_type"),
                    setup_scenario=raw.get("setup_scenario"),
                    actionability_status=(raw.get("actionability_soon") or {}).get("actionability_label"),
                    suitability_status=(raw.get("swing_trade_suitability") or {}).get("suitability_label"),
                    ranking_score=ranking_score,
                    strategy_confidence=float(raw.get("scenario_confidence") or 0.0),
                    current_price=float(raw.get("current_price") or 0.0) if raw.get("current_price") is not None else None,
                    preferred_entry=row.get("preferred_entry"),
                    stop_loss=row.get("stop_loss"),
                    take_profit_1=row.get("take_profit_1"),
                    take_profit_2=raw.get("take_profit_2"),
                    maximum_holding_date=row.get("max_hold_date"),
                    strategy_reason=row.get("short_summary") or raw.get("strategy_reason"),
                    source_snapshot_json=_json_dumps(row),
                )
                db.add(candidate)
                for component_name, component_value, component_weight in [
                    ("actionability_score", float(row.get("actionability_score") or 0.0), 10.0),
                    ("suitability_score", float(row.get("suitability_score") or 0.0), 6.0),
                    ("scenario_confidence", float(raw.get("scenario_confidence") or 0.0), 12.0),
                    ("watchlist_tier_primary_bonus", 4.0 if row.get("watchlist_tier") == "primary" else 0.0, 1.0),
                    ("ready_soon_bonus", 6.0 if row.get("actionability_label") == "ready_soon" else 0.0, 1.0),
                ]:
                    db.add(
                        CandidateScoreComponent(
                            candidate_id=candidate_id,
                            component_name=component_name,
                            component_value=component_value,
                            component_weight=component_weight,
                            evidence_json=_json_dumps({"ticker": row["ticker"]}),
                        )
                    )
                created += 1
            self._last_scan_time = now
            self._log_event(db, "watchlist_refreshed", "ok", f"Loaded {created} trade candidates from active watchlist")
            db.commit()
            return {"ok": True, "message": f"Watchlist refreshed with {created} candidates", "count": created}

    def list_candidates(self, *, only_active: bool = True) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            stmt = select(TradeCandidate).order_by(TradeCandidate.generated_at.desc())
            rows = db.execute(stmt).scalars().all()
            if only_active:
                rows = [row for row in rows if row.status not in {"rejected", "submitted", "closed"}]
            return [self._candidate_to_dict(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            row = db.execute(select(TradeCandidate).where(TradeCandidate.candidate_id == candidate_id)).scalar_one_or_none()
            return None if row is None else self._candidate_to_dict(row)

    def reject_candidate(self, candidate_id: str, *, reason: str = "manual_reject") -> dict[str, Any]:
        with SessionLocal() as db:
            row = db.execute(select(TradeCandidate).where(TradeCandidate.candidate_id == candidate_id)).scalar_one_or_none()
            if row is None:
                return {"ok": False, "message": "Candidate not found"}
            row.status = "rejected"
            row.rejection_code = reason
            row.rejection_reason = reason.replace("_", " ")
            self._log_event(db, "candidate_rejected", "ok", f"Candidate {candidate_id} rejected", candidate_id=candidate_id, ticker=row.ticker)
            db.commit()
            return {"ok": True, "message": "Candidate rejected"}

    def preview_execution(self, candidate_id: str, *, side: str = "BUY", order_type: str = "LIMIT") -> dict[str, Any]:
        with SessionLocal() as db:
            candidate = db.execute(select(TradeCandidate).where(TradeCandidate.candidate_id == candidate_id)).scalar_one_or_none()
            if candidate is None:
                return {"eligible": False, "rejection_codes": ["candidate_not_found"], "rejection_reasons": ["Candidate not found"]}
            preview = self._build_execution_preview(candidate, side=side, order_type=order_type)
            proposal_id = str(uuid.uuid4())
            proposal = TradeProposal(
                proposal_id=proposal_id,
                candidate_id=candidate.candidate_id,
                ticker=candidate.ticker,
                status=ProposalStatus.PREVIEWED.value,
                side=side,
                order_type=order_type,
                idempotency_key=preview["idempotency_key"],
                entry_price=preview.get("entry_price"),
                stop_price=preview.get("stop_price"),
                target_price_1=preview.get("target_prices", [None])[0],
                target_price_2=preview.get("target_prices", [None, None])[1] if len(preview.get("target_prices", [])) > 1 else None,
                quantity=preview.get("quantity"),
                planned_risk_dollars=preview.get("planned_dollar_risk"),
                estimated_max_loss=preview.get("estimated_max_loss"),
                reward_risk_ratio=preview.get("reward_to_risk"),
                rejection_codes_json=_json_dumps(preview.get("rejection_codes") or []),
                warnings_json=_json_dumps(preview.get("warnings") or []),
                preview_json=_json_dumps(preview),
            )
            db.add(proposal)
            db.commit()
            preview["proposal_id"] = proposal_id
            return preview

    def submit_execution(self, *, proposal_id: str | None = None, candidate_id: str | None = None, idempotency_key: str | None = None, auto_approved: bool = False) -> dict[str, Any]:
        with SessionLocal() as db:
            proposal = None
            if proposal_id:
                proposal = db.execute(select(TradeProposal).where(TradeProposal.proposal_id == proposal_id)).scalar_one_or_none()
            elif candidate_id:
                proposal = db.execute(select(TradeProposal).where(TradeProposal.candidate_id == candidate_id).order_by(TradeProposal.created_at.desc())).scalars().first()
            if proposal is None:
                return {"ok": False, "message": "Proposal not found"}
            if idempotency_key and proposal.idempotency_key != idempotency_key:
                return {"ok": False, "message": "Idempotency key mismatch"}
            preview = _safe_json(proposal.preview_json)
            if not preview.get("eligible"):
                proposal.status = ProposalStatus.REJECTED.value
                db.commit()
                return {"ok": False, "message": "Proposal is not eligible", "preview": preview}
            if self._config.trading_mode == TradingMode.DISABLED:
                return {"ok": False, "message": "Trading mode is disabled"}
            if self._config.trading_mode == TradingMode.MANUAL_PAPER and not auto_approved:
                proposal.status = ProposalStatus.APPROVED.value
                db.commit()
                return {"ok": True, "message": "Proposal approved and awaiting submission", "proposal_id": proposal.proposal_id}
            if self._kill_switch_active or self._reconciliation_required:
                return {"ok": False, "message": "Kill switch or reconciliation lock is active"}
            request = BracketOrderRequest(
                ticker=proposal.ticker,
                side=proposal.side,
                quantity=int(proposal.quantity or 0),
                entry_price=float(proposal.entry_price or 0.0),
                stop_price=float(proposal.stop_price or 0.0),
                target_price_1=float(proposal.target_price_1 or 0.0),
                target_price_2=float(proposal.target_price_2) if proposal.target_price_2 is not None else None,
                order_type=proposal.order_type,
                client_order_key=proposal.idempotency_key,
            )
            try:
                orders = self._broker.place_bracket_order(request)
            except Exception as exc:
                proposal.status = ProposalStatus.REJECTED.value
                self._log_event(db, "order_submission_failed", "error", f"Order submission failed: {exc}", proposal_id=proposal.proposal_id, ticker=proposal.ticker)
                db.commit()
                return {"ok": False, "message": f"Order submission failed: {exc}"}
            proposal.status = ProposalStatus.SUBMITTED.value
            proposal.submitted_at = datetime.now(timezone.utc)
            position_id = str(uuid.uuid4())
            db.add(
                ManagedPosition(
                    position_id=position_id,
                    ticker=proposal.ticker,
                    proposal_id=proposal.proposal_id,
                    candidate_id=proposal.candidate_id,
                    status=PositionStatus.OPEN.value,
                    quantity=int(proposal.quantity or 0),
                    average_entry_price=proposal.entry_price,
                    current_stop_price=proposal.stop_price,
                    current_target_price=proposal.target_price_1,
                    opened_at=datetime.now(timezone.utc),
                    position_json=_json_dumps({"submitted_via": proposal.proposal_id}),
                )
            )
            for order in orders:
                db.add(
                    BrokerOrder(
                        broker_order_id=order.broker_order_id,
                        proposal_id=proposal.proposal_id,
                        parent_order_id=order.parent_order_id,
                        child_role=order.child_role,
                        ticker=order.ticker,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        status=order.status,
                        broker_payload_json=_json_dumps(asdict(order)),
                    )
                )
            self._log_event(db, "order_submitted", "ok", f"Bracket order submitted for {proposal.ticker}", proposal_id=proposal.proposal_id, ticker=proposal.ticker)
            db.commit()
            return {"ok": True, "message": "Order submitted", "proposal_id": proposal.proposal_id, "orders": [asdict(order) for order in orders]}

    def approve_order(self, proposal_id: str) -> dict[str, Any]:
        return self.submit_execution(proposal_id=proposal_id, auto_approved=True)

    def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            try:
                cancelled = self._broker.cancel_order(broker_order_id)
            except Exception as exc:
                return {"ok": False, "message": f"Cancel failed: {exc}"}
            order = db.execute(select(BrokerOrder).where(BrokerOrder.broker_order_id == broker_order_id)).scalar_one_or_none()
            if order is not None:
                order.status = cancelled.status
            db.add(BrokerOrderEvent(broker_order_id=broker_order_id, event_type="cancelled", status=cancelled.status, event_payload_json=_json_dumps(asdict(cancelled))))
            db.commit()
            return {"ok": True, "message": "Order cancelled", "order": asdict(cancelled)}

    def close_position(self, position_or_order_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            position = db.execute(select(ManagedPosition).where(ManagedPosition.position_id == position_or_order_id)).scalar_one_or_none()
            if position is None:
                return {"ok": False, "message": "Position not found"}
            orders = self._broker.close_position(position.ticker)
            position.status = PositionStatus.CLOSED.value
            position.closed_at = datetime.now(timezone.utc)
            db.add(PositionEvent(position_id=position.position_id, event_type="close_requested", event_payload_json=_json_dumps({"orders": [asdict(order) for order in orders]})))
            db.commit()
            return {"ok": True, "message": "Position close requested", "orders": [asdict(order) for order in orders]}

    def flatten_all(self) -> dict[str, Any]:
        if self._kill_switch_active:
            return {"ok": False, "message": "Kill switch already active"}
        self._kill_switch_active = True
        self._kill_switch_reason = "flatten_all"
        orders = self._broker.flatten_all_positions()
        with SessionLocal() as db:
            db.add(KillSwitchEvent(active=True, reason="flatten_all", triggered_by="api"))
            self._log_event(db, "flatten_all", "ok", "Flatten-all triggered")
            db.commit()
        return {"ok": True, "message": "Flatten-all requested", "orders": [asdict(order) for order in orders]}

    def activate_kill_switch(self, reason: str) -> dict[str, Any]:
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        with SessionLocal() as db:
            db.add(KillSwitchEvent(active=True, reason=reason, triggered_by="api"))
            self._log_event(db, "kill_switch_activated", "ok", reason)
            db.commit()
        return {"ok": True, "message": "Kill switch activated"}

    def reset_kill_switch(self) -> dict[str, Any]:
        self._kill_switch_active = False
        self._kill_switch_reason = None
        with SessionLocal() as db:
            db.add(KillSwitchEvent(active=False, reason="reset", triggered_by="api"))
            self._log_event(db, "kill_switch_reset", "ok", "Kill switch reset")
            db.commit()
        return {"ok": True, "message": "Kill switch reset"}

    def risk_status(self) -> dict[str, Any]:
        with SessionLocal() as db:
            open_positions = db.execute(select(ManagedPosition).where(ManagedPosition.status == PositionStatus.OPEN.value)).scalars().all()
            open_risk = sum(max((position.average_entry_price or 0.0) - (position.current_stop_price or 0.0), 0.0) * abs(position.quantity or 0) for position in open_positions)
            return {
                "kill_switch_active": self._kill_switch_active,
                "kill_switch_reason": self._kill_switch_reason,
                "reconciliation_required": self._reconciliation_required,
                "open_positions": len(open_positions),
                "max_open_positions": self._config.max_open_positions,
                "open_portfolio_risk": round(open_risk, 2),
                "max_portfolio_risk": round(self._config.trading_budget * self._config.max_portfolio_risk_pct / 100.0, 2),
            }

    def exposure_status(self) -> dict[str, Any]:
        with SessionLocal() as db:
            open_positions = db.execute(select(ManagedPosition).where(ManagedPosition.status == PositionStatus.OPEN.value)).scalars().all()
            capital_in_use = sum(abs(position.quantity or 0) * float(position.average_entry_price or 0.0) for position in open_positions)
            return {
                "capital_in_use": round(capital_in_use, 2),
                "trading_budget": self._config.trading_budget,
                "capital_utilization_pct": round((capital_in_use / max(self._config.trading_budget, 1e-9)) * 100.0, 2),
                "remaining_budget": round(max(self._config.trading_budget - capital_in_use, 0.0), 2),
            }

    def journal_trades(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            positions = db.execute(select(ManagedPosition).order_by(ManagedPosition.opened_at.desc())).scalars().all()
            return [self._position_to_dict(position) for position in positions]

    def journal_trade(self, trade_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            position = db.execute(select(ManagedPosition).where(ManagedPosition.position_id == trade_id)).scalar_one_or_none()
            return None if position is None else self._position_to_dict(position)

    def trade_review(self, trade_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            review = db.execute(select(TradeReview).where(TradeReview.position_id == trade_id)).scalar_one_or_none()
            if review is None:
                return None
            return {
                "position_id": review.position_id,
                "success_category": review.success_category,
                "failure_category": review.failure_category,
                "realised_pnl": review.realised_pnl,
                "net_pnl": review.net_pnl,
                "return_pct": review.return_pct,
                "r_multiple": review.r_multiple,
                "mfe_pct": review.mfe_pct,
                "mae_pct": review.mae_pct,
                "deterministic_review": _safe_json(review.deterministic_review_json),
                "narrative_review": review.narrative_review,
                "reviewed_at": review.reviewed_at,
            }

    def review_trade(self, trade_id: str, *, narrative_review: str | None = None) -> dict[str, Any]:
        with SessionLocal() as db:
            position = db.execute(select(ManagedPosition).where(ManagedPosition.position_id == trade_id)).scalar_one_or_none()
            if position is None:
                return {"ok": False, "message": "Trade not found"}
            deterministic = self._build_trade_review(position)
            review = db.execute(select(TradeReview).where(TradeReview.position_id == trade_id)).scalar_one_or_none()
            if review is None:
                review = TradeReview(position_id=trade_id)
                db.add(review)
            review.success_category = deterministic.get("success_category")
            review.failure_category = deterministic.get("failure_category")
            review.realised_pnl = deterministic.get("realised_pnl")
            review.net_pnl = deterministic.get("net_pnl")
            review.return_pct = deterministic.get("return_pct")
            review.r_multiple = deterministic.get("r_multiple")
            review.mfe_pct = deterministic.get("mfe_pct")
            review.mae_pct = deterministic.get("mae_pct")
            review.deterministic_review_json = _json_dumps(deterministic)
            review.narrative_review = narrative_review
            db.commit()
            return {"ok": True, "message": "Trade review saved", "review": deterministic}

    def memory_statistics(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = db.execute(select(TradeMemoryStatistic).order_by(TradeMemoryStatistic.updated_at.desc())).scalars().all()
            return [
                {
                    "memory_key": row.memory_key,
                    "scope_type": row.scope_type,
                    "scope_value": row.scope_value,
                    "sample_size": row.sample_size,
                    "win_rate": row.win_rate,
                    "avg_r_multiple": row.avg_r_multiple,
                    "bounded_adjustment": row.bounded_adjustment,
                    "evidence": _safe_json(row.evidence_json),
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]

    def similar_trades(self, candidate_id: str) -> list[dict[str, Any]]:
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            return []
        with SessionLocal() as db:
            rows = db.execute(select(TradeReview)).scalars().all()
            out = []
            for row in rows:
                evidence = _safe_json(row.deterministic_review_json)
                if evidence.get("setup_scenario") == candidate.get("setup_scenario"):
                    out.append({"position_id": row.position_id, "review": evidence})
            return out[:10]

    def rebuild_memory(self) -> dict[str, Any]:
        with SessionLocal() as db:
            reviews = db.execute(select(TradeReview)).scalars().all()
            grouped: dict[tuple[str, str], list[TradeReview]] = {}
            for review in reviews:
                evidence = _safe_json(review.deterministic_review_json)
                for scope_type, scope_value in [
                    ("setup_scenario", evidence.get("setup_scenario")),
                    ("success_category", review.success_category),
                    ("failure_category", review.failure_category),
                ]:
                    if not scope_value:
                        continue
                    grouped.setdefault((scope_type, str(scope_value)), []).append(review)
            created = 0
            for (scope_type, scope_value), items in grouped.items():
                sample_size = len(items)
                if sample_size <= 0:
                    continue
                wins = sum(1 for item in items if (item.net_pnl or 0.0) > 0)
                avg_r = sum(float(item.r_multiple or 0.0) for item in items) / max(sample_size, 1)
                bounded_adjustment = 0.0
                if sample_size >= 5:
                    bounded_adjustment = max(-1.5, min(1.5, (wins / sample_size - 0.5) * 2.5 + avg_r * 0.15))
                memory_key = f"{scope_type}:{scope_value}"
                row = db.execute(select(TradeMemoryStatistic).where(TradeMemoryStatistic.memory_key == memory_key)).scalar_one_or_none()
                if row is None:
                    row = TradeMemoryStatistic(memory_key=memory_key, scope_type=scope_type, scope_value=scope_value)
                    db.add(row)
                    created += 1
                row.sample_size = sample_size
                row.win_rate = wins / sample_size
                row.avg_r_multiple = avg_r
                row.bounded_adjustment = bounded_adjustment
                row.evidence_json = _json_dumps({"sample_size": sample_size, "wins": wins})
            db.commit()
            return {"ok": True, "message": "Trade memory rebuilt", "statistics_updated": len(grouped), "statistics_created": created}

    def performance(self) -> dict[str, Any]:
        with SessionLocal() as db:
            latest = db.execute(select(DailyPerformance).order_by(DailyPerformance.performance_date.desc())).scalars().first()
            if latest is None:
                return {"daily_reports": []}
            return {
                "latest": {
                    "performance_date": latest.performance_date,
                    "realised_pnl": latest.realised_pnl,
                    "unrealised_pnl": latest.unrealised_pnl,
                    "net_pnl": latest.net_pnl,
                    "open_positions": latest.open_positions,
                    "closed_positions": latest.closed_positions,
                    "details": _safe_json(latest.details_json),
                }
            }

    def bot_events(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = db.execute(select(BotEvent).order_by(BotEvent.created_at.desc())).scalars().all()
            return [
                {
                    "id": row.id,
                    "event_type": row.event_type,
                    "outcome": row.outcome,
                    "message": row.message,
                    "level": row.level,
                    "ticker": row.ticker,
                    "candidate_id": row.candidate_id,
                    "proposal_id": row.proposal_id,
                    "broker_order_id": row.broker_order_id,
                    "created_at": row.created_at,
                    "details": _safe_json(row.details_json),
                }
                for row in rows
            ]

    def bot_runs(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = db.execute(select(BotRun).order_by(BotRun.started_at.desc())).scalars().all()
            return [
                {
                    "run_id": row.run_id,
                    "state": row.state,
                    "trading_mode": row.trading_mode,
                    "started_at": row.started_at,
                    "ended_at": row.ended_at,
                    "heartbeat_at": row.heartbeat_at,
                    "details": _safe_json(row.details_json),
                }
                for row in rows
            ]

    def daily_report(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = db.execute(select(DailyPerformance).order_by(DailyPerformance.performance_date.desc())).scalars().all()
            return [
                {
                    "performance_date": row.performance_date,
                    "realised_pnl": row.realised_pnl,
                    "unrealised_pnl": row.unrealised_pnl,
                    "net_pnl": row.net_pnl,
                    "open_positions": row.open_positions,
                    "closed_positions": row.closed_positions,
                    "details": _safe_json(row.details_json),
                }
                for row in rows
            ]

    def _run_loop(self) -> None:
        try:
            self._broker_state = BrokerHealthState.CONNECTING
            self._broker.connect()
            self._validate_broker_account(self._broker.account_summary())
            self._broker_state = BrokerHealthState.CONNECTED
        except Exception:
            self._broker_state = BrokerHealthState.DEGRADED
        last_scan = 0.0
        last_trigger = 0.0
        last_monitor = 0.0
        last_reconcile = 0.0
        while not self._stop_event.is_set():
            if self._state == BotRunState.PAUSED:
                time.sleep(1.0)
                continue
            now = time.time()
            self._last_heartbeat = datetime.now(timezone.utc)
            if now - last_reconcile >= self._config.bot_reconcile_interval_seconds:
                self.reconcile()
                last_reconcile = now
            if now - last_scan >= self._config.bot_scan_interval_seconds:
                self.refresh_watchlist()
                last_scan = now
            if now - last_trigger >= self._config.bot_trigger_check_interval_seconds:
                self._auto_preview_top_candidates()
                last_trigger = now
            if now - last_monitor >= self._config.bot_position_monitor_interval_seconds:
                self._refresh_daily_performance()
                last_monitor = now
            with SessionLocal() as db:
                self._snapshot_health(db)
                self._touch_run(db)
                db.commit()
            time.sleep(1.0)

    def _auto_preview_top_candidates(self) -> None:
        candidates = self.list_candidates(only_active=True)
        for candidate in candidates[: max(self._config.max_open_positions * 2, 5)]:
            if candidate.get("trigger_state") != "triggered":
                continue
            self.preview_execution(candidate["candidate_id"])
            if self._config.trading_mode == TradingMode.AUTO_PAPER and self._config.auto_execution:
                latest = self.preview_execution(candidate["candidate_id"])
                proposal_id = latest.get("proposal_id")
                if proposal_id:
                    self.submit_execution(proposal_id=proposal_id, auto_approved=True)

    def _build_execution_preview(self, candidate: TradeCandidate, *, side: str, order_type: str) -> dict[str, Any]:
        rejection_codes: list[str] = []
        rejection_reasons: list[str] = []
        warnings: list[str] = []
        quote = None
        try:
            quote = self._broker.current_quote(candidate.ticker)
        except Exception:
            session = SessionLocal()
            try:
                last = get_last_price_or_recent_close(candidate.ticker, daily_closes_loader=_build_daily_closes_loader_for_bot(session))
            finally:
                session.close()
            if last is not None:
                from .broker import BrokerQuote
                quote = BrokerQuote(ticker=candidate.ticker, last=float(last), bid=float(last) * 0.998, ask=float(last) * 1.002, source="fallback")

        if self._kill_switch_active:
            rejection_codes.append("kill_switch_active")
            rejection_reasons.append("Kill switch is active.")
        if self._reconciliation_required:
            rejection_codes.append("reconciliation_required")
            rejection_reasons.append("Broker reconciliation is required before trading.")
        if self._config.trading_mode == TradingMode.DISABLED:
            rejection_codes.append("trading_disabled")
            rejection_reasons.append("Trading mode is disabled.")
        if self._config.ibkr_read_only:
            rejection_codes.append("broker_read_only")
            rejection_reasons.append("IBKR is connected in read-only mode; order submission is intentionally blocked.")
        if candidate.preferred_entry is None or candidate.stop_loss is None or candidate.take_profit_1 is None:
            rejection_codes.append("invalid_levels")
            rejection_reasons.append("Candidate does not have valid entry, stop and target levels.")
        if quote is None or quote.last is None:
            rejection_codes.append("missing_market_data")
            rejection_reasons.append("Latest market data is unavailable.")
        elif (datetime.now(timezone.utc) - quote.timestamp).total_seconds() > self._config.stale_quote_seconds:
            rejection_codes.append("stale_quote")
            rejection_reasons.append("Quote is stale.")
        reward_risk = None
        quantity = 0
        planned_risk = 0.0
        estimated_max_loss = 0.0
        if candidate.preferred_entry and candidate.stop_loss and candidate.take_profit_1:
            risk_per_share = candidate.preferred_entry - candidate.stop_loss
            if risk_per_share <= 0:
                rejection_codes.append("invalid_stop")
                rejection_reasons.append("Stop loss must be below entry for long trades.")
            else:
                reward = candidate.take_profit_1 - candidate.preferred_entry
                reward_risk = reward / max(risk_per_share, 1e-9)
                if reward_risk < self._config.min_reward_risk:
                    rejection_codes.append("low_reward_risk")
                    rejection_reasons.append("Reward-to-risk is below the configured minimum.")
                risk_amount = self._config.trading_budget * self._config.risk_per_trade_pct / 100.0
                risk_quantity = math.floor(risk_amount / max(risk_per_share, 1e-9))
                max_position_value = self._config.trading_budget * self._config.max_position_pct / 100.0
                budget_quantity = math.floor(max_position_value / max(candidate.preferred_entry, 1e-9))
                account = self._broker.account_summary()
                available_cash_quantity = math.floor(max(account.cash_balance, 0.0) / max(candidate.preferred_entry, 1e-9))
                quantity = min(risk_quantity, budget_quantity, available_cash_quantity, self._config.max_open_positions * 10_000)
                if quantity < 1:
                    rejection_codes.append("quantity_below_one")
                    rejection_reasons.append("Calculated quantity is below one share.")
                planned_risk = risk_per_share * max(quantity, 0)
                estimated_max_loss = planned_risk * (1.0 + self._config.max_slippage_pct / 100.0)
        if quote and quote.spread_pct is not None and quote.spread_pct > self._config.max_spread_pct:
            rejection_codes.append("excessive_spread")
            rejection_reasons.append("Bid-ask spread is too wide for execution.")
        eligible = len(rejection_codes) == 0
        idempotency_key = f"{candidate.candidate_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
        return {
            "eligible": eligible,
            "rejection_codes": rejection_codes,
            "rejection_reasons": rejection_reasons,
            "ticker": candidate.ticker,
            "side": side,
            "order_type": order_type,
            "entry_price": candidate.preferred_entry,
            "stop_price": candidate.stop_loss,
            "target_prices": [candidate.take_profit_1, candidate.take_profit_2] if candidate.take_profit_2 is not None else [candidate.take_profit_1],
            "quantity": quantity,
            "position_value": None if not candidate.preferred_entry else round(candidate.preferred_entry * quantity, 2),
            "planned_dollar_risk": round(planned_risk, 2),
            "estimated_max_loss": round(estimated_max_loss, 2),
            "risk_percentage": round((planned_risk / max(self._config.trading_budget, 1e-9)) * 100.0, 2),
            "reward_to_risk": round(reward_risk, 4) if reward_risk is not None else None,
            "buying_power_before": self._broker.account_summary().buying_power if eligible else None,
            "buying_power_after": None if not eligible or candidate.preferred_entry is None else round(self._broker.account_summary().buying_power - candidate.preferred_entry * quantity, 2),
            "capital_utilization_before": round(self.exposure_status()["capital_utilization_pct"], 2),
            "capital_utilization_after": round(((self.exposure_status()["capital_in_use"] + (candidate.preferred_entry or 0.0) * quantity) / max(self._config.trading_budget, 1e-9)) * 100.0, 2) if candidate.preferred_entry is not None else None,
            "portfolio_risk_before": round(self.risk_status()["open_portfolio_risk"], 2),
            "portfolio_risk_after": round(self.risk_status()["open_portfolio_risk"] + planned_risk, 2),
            "sector_exposure_before": None,
            "sector_exposure_after": None,
            "spread": quote.spread_pct if quote is not None else None,
            "quote_age_seconds": None if quote is None else round((datetime.now(timezone.utc) - quote.timestamp).total_seconds(), 2),
            "market_session": "regular",
            "earnings_date": None,
            "execution_mode": self._config.trading_mode.value,
            "broker_account_type": self._broker.account_summary().account_type if eligible else None,
            "warnings": warnings,
            "preview_expiration_time": datetime.now(timezone.utc) + timedelta(minutes=5),
            "idempotency_key": idempotency_key,
        }

    def _snapshot_health(self, db: Session) -> None:
        db.add(
            BotHealthSnapshot(
                bot_run_id=self._run_id,
                broker_state=self._broker_state.value,
                bot_state=self._state.value,
                kill_switch_active=self._kill_switch_active,
                heartbeat_at=self._last_heartbeat or datetime.now(timezone.utc),
                snapshot_json=_json_dumps(self.status()),
            )
        )

    def _persist_config(self, db: Session) -> None:
        row = db.execute(select(BotConfiguration).where(BotConfiguration.config_key == "active")).scalar_one_or_none()
        payload = self.public_config_dict()
        if row is None:
            row = BotConfiguration(config_key="active", config_json=_json_dumps(payload), updated_by="api")
            db.add(row)
        else:
            row.config_json = _json_dumps(payload)
            row.updated_at = datetime.now(timezone.utc)
            row.updated_by = "api"

    def _touch_run(self, db: Session) -> None:
        if not self._run_id:
            return
        row = db.execute(select(BotRun).where(BotRun.run_id == self._run_id)).scalar_one_or_none()
        if row is not None:
            row.heartbeat_at = self._last_heartbeat or datetime.now(timezone.utc)
            row.state = self._state.value

    def _complete_run(self, db: Session) -> None:
        if not self._run_id:
            return
        row = db.execute(select(BotRun).where(BotRun.run_id == self._run_id)).scalar_one_or_none()
        if row is not None:
            row.ended_at = datetime.now(timezone.utc)
            row.state = self._state.value

    def _log_event(self, db: Session, event_type: str, outcome: str, message: str, *, ticker: str | None = None, candidate_id: str | None = None, proposal_id: str | None = None) -> None:
        db.add(
            BotEvent(
                bot_run_id=self._run_id,
                correlation_id=str(uuid.uuid4()),
                event_type=event_type,
                outcome=outcome,
                message=message,
                ticker=ticker,
                candidate_id=candidate_id,
                proposal_id=proposal_id,
            )
        )

    def _mask_account(self, account_id: str) -> str:
        if not account_id:
            return "unknown"
        return f"***{account_id[-4:]}"

    def _validate_broker_account(self, account: BrokerAccountSummary) -> None:
        configured_account = (self._config.ibkr_account_id or "").strip().upper()
        connected_account = (account.account_id or "").strip().upper()
        if configured_account and connected_account != configured_account:
            raise BrokerError("Connected IBKR account does not match IBKR_ACCOUNT_ID")
        if self._config.ibkr_require_paper_account and not account.is_paper:
            raise BrokerError("Paper-account safety check failed; connected account is not an IBKR paper account")

    def _load_active_watchlist_rows(self) -> list[dict[str, Any]]:
        db_url = settings.SUPABASE_DATABASE_URL
        if not db_url:
            return []
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        db_url = db_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        engine = create_engine(db_url, pool_pre_ping=True)
        sql = """
        with ranked_snapshots as (
            select
                ticker,
                updated_at,
                source_run_id,
                final_action,
                watchlist_tier,
                watch_priority,
                actionability_label,
                actionability_score,
                suitability_label,
                suitability_score,
                trend_state,
                preferred_entry,
                stop_loss,
                take_profit_1,
                max_hold_date,
                short_summary,
                raw_result_json,
                row_number() over (
                    partition by ticker
                    order by updated_at desc nulls last, source_run_id desc nulls last
                ) as snapshot_rank
            from public.watchlist_snapshots
            where max_hold_date is null or max_hold_date >= now()
        )
        select *
        from ranked_snapshots
        where snapshot_rank = 1
        order by actionability_score desc nulls last, updated_at desc nulls last
        """
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(sql).mappings().all()
        return [dict(row) for row in rows]

    def _rank_candidate(self, row: dict[str, Any], raw: dict[str, Any]) -> float:
        score = float(row.get("actionability_score") or 0.0) * 10.0
        score += float(row.get("suitability_score") or 0.0) * 6.0
        score += float(raw.get("scenario_confidence") or 0.0) * 12.0
        if row.get("watchlist_tier") == "primary":
            score += 4.0
        if row.get("actionability_label") == "ready_soon":
            score += 6.0
        return round(score, 4)

    def _trigger_state(self, row: dict[str, Any], raw: dict[str, Any]) -> str:
        current_price = raw.get("current_price")
        preferred_entry = row.get("preferred_entry")
        if current_price is None or preferred_entry in (None, 0):
            return "waiting_for_data"
        distance_pct = ((float(current_price) - float(preferred_entry)) / float(preferred_entry)) * 100.0
        if abs(distance_pct) <= 1.0:
            return "triggered"
        if distance_pct > 1.0:
            return "extended"
        return "monitoring"

    def _candidate_to_dict(self, row: TradeCandidate) -> dict[str, Any]:
        return {
            "candidate_id": row.candidate_id,
            "ticker": row.ticker,
            "basket": row.basket,
            "generated_at": row.generated_at,
            "status": row.status,
            "trigger_state": row.trigger_state,
            "market_regime": row.market_regime,
            "setup_type": row.setup_type,
            "setup_scenario": row.setup_scenario,
            "actionability_status": row.actionability_status,
            "suitability_status": row.suitability_status,
            "ranking_score": row.ranking_score,
            "strategy_confidence": row.strategy_confidence,
            "current_price": row.current_price,
            "preferred_entry": row.preferred_entry,
            "stop_loss": row.stop_loss,
            "take_profit_1": row.take_profit_1,
            "take_profit_2": row.take_profit_2,
            "maximum_holding_date": row.maximum_holding_date,
            "rejection_code": row.rejection_code,
            "rejection_reason": row.rejection_reason,
            "strategy_reason": row.strategy_reason,
            "source_snapshot": _safe_json(row.source_snapshot_json),
        }

    def _position_to_dict(self, row: ManagedPosition) -> dict[str, Any]:
        return {
            "position_id": row.position_id,
            "ticker": row.ticker,
            "proposal_id": row.proposal_id,
            "candidate_id": row.candidate_id,
            "status": row.status,
            "quantity": row.quantity,
            "average_entry_price": row.average_entry_price,
            "current_stop_price": row.current_stop_price,
            "current_target_price": row.current_target_price,
            "realised_pnl": row.realised_pnl,
            "unrealised_pnl": row.unrealised_pnl,
            "opened_at": row.opened_at,
            "closed_at": row.closed_at,
            "position": _safe_json(row.position_json),
        }

    def _build_trade_review(self, position: ManagedPosition) -> dict[str, Any]:
        entry = float(position.average_entry_price or 0.0)
        stop = float(position.current_stop_price or entry)
        realized = float(position.realised_pnl or 0.0)
        risk_per_share = max(entry - stop, 1e-9)
        r_multiple = realized / max(risk_per_share * abs(position.quantity or 1), 1e-9)
        success_category = "correct_setup_and_execution" if realized > 0 else None
        failure_category = None if realized > 0 else "stop_or_invalidation"
        return {
            "position_id": position.position_id,
            "setup_scenario": _safe_json(position.position_json).get("setup_scenario"),
            "realised_pnl": realized,
            "net_pnl": realized,
            "return_pct": 0.0 if entry == 0 else round((realized / max(entry * abs(position.quantity or 1), 1e-9)) * 100.0, 4),
            "r_multiple": round(r_multiple, 4),
            "mfe_pct": None,
            "mae_pct": None,
            "success_category": success_category,
            "failure_category": failure_category,
        }

    def _refresh_daily_performance(self) -> None:
        with SessionLocal() as db:
            positions = db.execute(select(ManagedPosition).where(ManagedPosition.status == PositionStatus.OPEN.value)).scalars().all()
            today = datetime.now(timezone.utc).date()
            row = db.execute(select(DailyPerformance).where(DailyPerformance.performance_date == today)).scalar_one_or_none()
            if row is None:
                row = DailyPerformance(performance_date=today)
                db.add(row)
            row.open_positions = len(positions)
            row.closed_positions = db.execute(select(ManagedPosition).where(ManagedPosition.status == PositionStatus.CLOSED.value)).scalars().all().__len__()
            row.realised_pnl = sum(float(position.realised_pnl or 0.0) for position in positions)
            row.unrealised_pnl = sum(float(position.unrealised_pnl or 0.0) for position in positions)
            row.net_pnl = float(row.realised_pnl or 0.0) + float(row.unrealised_pnl or 0.0)
            row.details_json = _json_dumps({"generated_at": datetime.now(timezone.utc)})


_BOT_SERVICE: TradingBotService | None = None


def get_bot_service() -> TradingBotService:
    global _BOT_SERVICE
    if _BOT_SERVICE is None:
        _BOT_SERVICE = TradingBotService()
    return _BOT_SERVICE
