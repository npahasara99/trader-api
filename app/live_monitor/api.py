from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .service import LiveMonitorService, get_live_monitor_service


router = APIRouter(prefix="/live-monitor", tags=["live-monitor"])


class AddMonitorRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    source: str = "manual"
    planner_payload: dict[str, Any] | None = None


class LevelOverrideRequest(BaseModel):
    levels: dict[str, float | None]


class ReanalyzeRequest(BaseModel):
    planner_payload: dict[str, Any] | None = None


class ChartReviewRequest(BaseModel):
    review_type: str = "CHART_STRUCTURE_REVIEW"


class ChartLevelDecisionRequest(BaseModel):
    decision: str
    manual_levels: dict[str, float | None] | None = None
    decided_by: str = "dashboard_user"


class ManualActionRequest(BaseModel):
    action: str
    trade_id: str | None = None
    quantity: float | None = None
    planned_entry: float | None = None
    actual_entry: float | None = None
    stop_price: float | None = None
    targets: dict[str, float] | None = None
    exit_price: float | None = None
    notes: str | None = None


class LearningObservationRequest(BaseModel):
    ticker: str


class LearningProposalRequest(BaseModel):
    observation_id: str | None = None
    scope_type: str = "global"
    scope_value: str = "all"
    title: str
    proposed_change: dict[str, Any]
    evidence: dict[str, Any] = Field(default_factory=dict)


class ProposalDecisionRequest(BaseModel):
    decision: str
    decided_by: str = "user"


class LearningCycleRequest(BaseModel):
    trading_date: str | None = None


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Live monitor operation failed: {type(exc).__name__}: {exc}")


@router.get("/status")
def monitor_status(service: LiveMonitorService = Depends(get_live_monitor_service)):
    return service.status()


@router.get("")
def list_monitors(
    include_inactive: bool = Query(False),
    service: LiveMonitorService = Depends(get_live_monitor_service),
):
    return {"rows": service.list_monitors(include_inactive=include_inactive)}


@router.post("")
def add_monitor(request: AddMonitorRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.add_monitor(request.ticker, source=request.source, planner_payload=request.planner_payload)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.get("/journal")
def journal(
    watch_id: str | None = None,
    ticker: str | None = None,
    limit: int = Query(500, ge=1, le=2000),
    service: LiveMonitorService = Depends(get_live_monitor_service),
):
    return {"rows": service.journal(watch_id=watch_id, ticker=ticker, limit=limit)}


@router.get("/learning")
def learning_overview(service: LiveMonitorService = Depends(get_live_monitor_service)):
    return service.learning_overview()


@router.post("/learning/run")
def run_learning_cycle(request: LearningCycleRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.run_learning_cycle(request.trading_date)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/learning/observations")
def create_observation(request: LearningObservationRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.create_learning_observation(request.ticker)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/learning/proposals")
def create_proposal(request: LearningProposalRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.create_proposal(request.model_dump())
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/learning/proposals/{proposal_id}/decision")
def decide_proposal(proposal_id: str, request: ProposalDecisionRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.decide_proposal(proposal_id, decision=request.decision, decided_by=request.decided_by)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.get("/profiles/{ticker}")
def stock_profile(ticker: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    return service.stock_profile(ticker)


@router.get("/{watch_id}")
def monitor_detail(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.get_monitor(watch_id)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.get("/{watch_id}/charts")
def monitor_charts(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.chart_bundle(watch_id)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/chart-review")
def request_chart_review(
    watch_id: str,
    request: ChartReviewRequest,
    service: LiveMonitorService = Depends(get_live_monitor_service),
):
    try:
        return service.run_chart_review(watch_id, review_type=request.review_type, automatic=False)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/chart-level-decision")
def chart_level_decision(
    watch_id: str,
    request: ChartLevelDecisionRequest,
    service: LiveMonitorService = Depends(get_live_monitor_service),
):
    try:
        return service.apply_chart_level_decision(
            watch_id,
            decision=request.decision,
            manual_levels=request.manual_levels,
            decided_by=request.decided_by,
        )
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/evaluate")
def evaluate_monitor(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.evaluate_watch(watch_id)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/pause")
def pause_monitor(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.control(watch_id, "pause")
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/resume")
def resume_monitor(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.control(watch_id, "resume")
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/stop")
def stop_monitor(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.control(watch_id, "stop")
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/remove")
def remove_monitor(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.control(watch_id, "remove")
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/reanalyze")
def reanalyze_monitor(watch_id: str, request: ReanalyzeRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.reanalyze(watch_id, planner_payload=request.planner_payload)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.patch("/{watch_id}/levels")
def edit_monitor_levels(watch_id: str, request: LevelOverrideRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.edit_levels(watch_id, request.levels)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/llm-review")
def request_llm_review(watch_id: str, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.request_llm_review(watch_id)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/{watch_id}/manual-trades")
def record_manual_action(watch_id: str, request: ManualActionRequest, service: LiveMonitorService = Depends(get_live_monitor_service)):
    try:
        return service.record_manual_action(watch_id, request.model_dump(exclude_none=True))
    except Exception as exc:
        raise _translate_error(exc) from exc
