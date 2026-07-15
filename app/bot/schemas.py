from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any


class BotStatusResponse(BaseModel):
    running: bool
    paused: bool
    state: str
    trading_mode: str
    broker_connected: bool
    broker_state: str
    execution_mode: str
    auto_execution: bool
    kill_switch_active: bool
    last_heartbeat: datetime | None = None
    last_scan_time: datetime | None = None
    last_broker_sync: datetime | None = None
    reconciliation_required: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class BotConfigResponse(BaseModel):
    config: dict[str, Any]


class BotConfigUpdateRequest(BaseModel):
    config: dict[str, Any]


class ActionResponse(BaseModel):
    ok: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PreviewRequest(BaseModel):
    candidate_id: str
    side: str = "BUY"
    order_type: str = "LIMIT"


class SubmitRequest(BaseModel):
    proposal_id: str | None = None
    candidate_id: str | None = None
    idempotency_key: str | None = None


class KillSwitchRequest(BaseModel):
    reason: str


class TradeReviewRequest(BaseModel):
    narrative_review: str | None = None

