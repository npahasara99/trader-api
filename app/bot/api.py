from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from .schemas import (
    ActionResponse,
    BotConfigResponse,
    BotConfigUpdateRequest,
    BotStatusResponse,
    KillSwitchRequest,
    PreviewRequest,
    SubmitRequest,
    TradeReviewRequest,
)
from .service import get_bot_service


router = APIRouter(tags=["trading-bot"])


@router.get("/bot/status", response_model=BotStatusResponse)
def get_bot_status(service=Depends(get_bot_service)):
    return BotStatusResponse(**service.status())


@router.post("/bot/start", response_model=ActionResponse)
def start_bot(service=Depends(get_bot_service)):
    return ActionResponse(**service.start())


@router.post("/bot/stop", response_model=ActionResponse)
def stop_bot(service=Depends(get_bot_service)):
    return ActionResponse(**service.stop())


@router.post("/bot/pause", response_model=ActionResponse)
def pause_bot(service=Depends(get_bot_service)):
    return ActionResponse(**service.pause())


@router.post("/bot/resume", response_model=ActionResponse)
def resume_bot(service=Depends(get_bot_service)):
    return ActionResponse(**service.resume())


@router.get("/bot/config", response_model=BotConfigResponse)
def get_bot_config(service=Depends(get_bot_service)):
    return BotConfigResponse(config=service.public_config_dict())


@router.put("/bot/config", response_model=BotConfigResponse)
def update_bot_config(req: BotConfigUpdateRequest, service=Depends(get_bot_service)):
    service.set_config(req.config)
    return BotConfigResponse(config=service.public_config_dict())


@router.get("/bot/events")
def get_bot_events(service=Depends(get_bot_service)):
    return {"rows": service.bot_events()}


@router.get("/bot/runs")
def get_bot_runs(service=Depends(get_bot_service)):
    return {"rows": service.bot_runs()}


@router.get("/bot/performance")
def get_bot_performance(service=Depends(get_bot_service)):
    return service.performance()


@router.get("/bot/daily-report")
def get_daily_report(service=Depends(get_bot_service)):
    return {"rows": service.daily_report()}


@router.get("/broker/health")
def get_broker_health(service=Depends(get_bot_service)):
    return service.broker_health()


@router.get("/broker/account")
def get_broker_account(service=Depends(get_bot_service)):
    return asdict(service.broker_account())


@router.get("/broker/positions")
def get_broker_positions(service=Depends(get_bot_service)):
    return {"rows": service.broker_positions()}


@router.get("/broker/orders")
def get_broker_orders(service=Depends(get_bot_service)):
    return {"rows": service.broker_orders()}


@router.get("/broker/executions")
def get_broker_executions(service=Depends(get_bot_service)):
    return {"rows": service.broker_executions()}


@router.post("/broker/reconnect", response_model=ActionResponse)
def reconnect_broker(service=Depends(get_bot_service)):
    return ActionResponse(**service.reconnect_broker())


@router.post("/broker/reconcile", response_model=ActionResponse)
def reconcile_broker(service=Depends(get_bot_service)):
    return ActionResponse(**service.reconcile())


@router.get("/watchlist/active")
def get_active_watchlist(service=Depends(get_bot_service)):
    return {"rows": service.list_candidates()}


@router.post("/watchlist/refresh", response_model=ActionResponse)
def refresh_watchlist(service=Depends(get_bot_service)):
    return ActionResponse(**service.refresh_watchlist())


@router.get("/candidates")
def list_candidates(service=Depends(get_bot_service)):
    return {"rows": service.list_candidates()}


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, service=Depends(get_bot_service)):
    row = service.get_candidate(candidate_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return row


@router.post("/candidates/{candidate_id}/refresh")
def refresh_candidate(candidate_id: str, service=Depends(get_bot_service)):
    row = service.get_candidate(candidate_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return service.preview_execution(candidate_id)


@router.post("/candidates/{candidate_id}/reject", response_model=ActionResponse)
def reject_candidate(candidate_id: str, service=Depends(get_bot_service)):
    return ActionResponse(**service.reject_candidate(candidate_id))


@router.post("/execution/preview")
def execution_preview(req: PreviewRequest, service=Depends(get_bot_service)):
    return service.preview_execution(req.candidate_id, side=req.side, order_type=req.order_type)


@router.post("/execution/submit")
def execution_submit(req: SubmitRequest, service=Depends(get_bot_service)):
    return service.submit_execution(proposal_id=req.proposal_id, candidate_id=req.candidate_id, idempotency_key=req.idempotency_key)


@router.post("/execution/{order_id}/approve")
def execution_approve(order_id: str, service=Depends(get_bot_service)):
    return service.approve_order(order_id)


@router.post("/execution/{order_id}/cancel")
def execution_cancel(order_id: str, service=Depends(get_bot_service)):
    return service.cancel_order(order_id)


@router.post("/execution/{order_id}/close")
def execution_close(order_id: str, service=Depends(get_bot_service)):
    return service.close_position(order_id)


@router.post("/execution/flatten")
def execution_flatten(service=Depends(get_bot_service)):
    return service.flatten_all()


@router.post("/risk/kill-switch")
def activate_kill_switch(req: KillSwitchRequest, service=Depends(get_bot_service)):
    return service.activate_kill_switch(req.reason)


@router.post("/risk/kill-switch/reset")
def reset_kill_switch(service=Depends(get_bot_service)):
    return service.reset_kill_switch()


@router.get("/risk/status")
def get_risk_status(service=Depends(get_bot_service)):
    return service.risk_status()


@router.get("/risk/exposure")
def get_risk_exposure(service=Depends(get_bot_service)):
    return service.exposure_status()


@router.get("/journal/trades")
def get_journal_trades(service=Depends(get_bot_service)):
    return {"rows": service.journal_trades()}


@router.get("/journal/trades/{trade_id}")
def get_journal_trade(trade_id: str, service=Depends(get_bot_service)):
    row = service.journal_trade(trade_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return row


@router.get("/journal/trades/{trade_id}/review")
def get_trade_review(trade_id: str, service=Depends(get_bot_service)):
    row = service.trade_review(trade_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Trade review not found")
    return row


@router.post("/journal/trades/{trade_id}/review")
def post_trade_review(trade_id: str, req: TradeReviewRequest, service=Depends(get_bot_service)):
    return service.review_trade(trade_id, narrative_review=req.narrative_review)


@router.get("/memory/statistics")
def get_memory_statistics(service=Depends(get_bot_service)):
    return {"rows": service.memory_statistics()}


@router.get("/memory/similar-trades/{candidate_id}")
def get_similar_trades(candidate_id: str, service=Depends(get_bot_service)):
    return {"rows": service.similar_trades(candidate_id)}


@router.post("/memory/rebuild")
def rebuild_memory(service=Depends(get_bot_service)):
    return service.rebuild_memory()
