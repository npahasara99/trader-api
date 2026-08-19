"""US equity-session labels for scan-time execution context."""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


def classify_market_session(value: datetime | None = None) -> str:
    now = value or datetime.now(timezone.utc)
    local = now.astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return "closed"
    clock = local.time()
    if time(4, 0) <= clock < time(9, 30):
        return "premarket"
    if time(9, 30) <= clock < time(16, 0):
        return "regular"
    if time(16, 0) <= clock < time(20, 0):
        return "after_hours"
    return "overnight"


__all__ = ["classify_market_session"]
