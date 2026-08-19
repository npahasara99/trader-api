from __future__ import annotations

from enum import StrEnum


class MonitorState(StrEnum):
    WATCHING = "WATCHING"
    NEAR_TRIGGER = "NEAR_TRIGGER"
    ARMED = "ARMED"
    CONFIRMING = "CONFIRMING"
    APPROVED = "APPROVED"
    STRONGLY_CONFIRMED = "STRONGLY_CONFIRMED"
    REJECTED_BREAKOUT = "REJECTED_BREAKOUT"
    INVALIDATED = "INVALIDATED"
    MISSED = "MISSED"
    DATA_STALE = "DATA_STALE"
    PLAN_STALE = "PLAN_STALE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    EXPIRED = "EXPIRED"


ACTIVE_MONITOR_STATES = {
    MonitorState.WATCHING,
    MonitorState.NEAR_TRIGGER,
    MonitorState.ARMED,
    MonitorState.CONFIRMING,
    MonitorState.APPROVED,
    MonitorState.STRONGLY_CONFIRMED,
    MonitorState.REJECTED_BREAKOUT,
    MonitorState.MISSED,
    MonitorState.DATA_STALE,
    MonitorState.PLAN_STALE,
}


class AdvisoryDecision(StrEnum):
    APPROVE = "APPROVE"
    WAIT = "WAIT"
    REJECT = "REJECT"
