from __future__ import annotations

from enum import Enum


class TradingMode(str, Enum):
    DISABLED = "disabled"
    MANUAL_PAPER = "manual_paper"
    AUTO_PAPER = "auto_paper"
    SHADOW_LIVE = "shadow_live"
    RESTRICTED_LIVE = "live"


class BotRunState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class BrokerHealthState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    PREVIEWED = "previewed"
    APPROVED = "approved"
    QUEUED = "queued"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CLOSING = "closing"
    CLOSED = "closed"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    UNKNOWN = "unknown"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXTERNAL = "external"
    RECONCILIATION_REQUIRED = "reconciliation_required"

