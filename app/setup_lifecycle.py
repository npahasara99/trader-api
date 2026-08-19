"""Deterministic lifecycle metadata for regenerated swing setups."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json


def _number(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def build_setup_lifecycle(
    *,
    ticker: str,
    current_price: float,
    structure_state: str,
    entry_status: str,
    invalidation_level: float | None,
    primary_trigger: float | None,
    previous_setup: dict | None = None,
    validated_at: datetime | None = None,
) -> dict:
    """Build current lifecycle state and invalidate a broken prior thesis."""

    now = validated_at or datetime.now(timezone.utc)
    signature = {
        "ticker": ticker.upper(),
        "structure": structure_state,
        "invalidation": _number(invalidation_level),
        "primary_trigger": _number(primary_trigger),
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    status_map = {
        "too_early": "forming",
        "in_price_zone": "awaiting_confirmation",
        "awaiting_confirmation": "awaiting_confirmation",
        "confirmed": "confirmed",
        "extended": "missed",
        "missed": "missed",
        "invalidated": "invalidated",
    }
    setup_status = status_map.get(entry_status, "valid")
    invalidated_at = now.isoformat() if setup_status == "invalidated" else None
    invalidation_reason = "current_price_below_invalidation" if setup_status == "invalidated" else None

    replaced_setup = None
    prior = previous_setup or {}
    prior_invalidation = _number(prior.get("invalidation_level") or prior.get("stop_loss") or prior.get("stop"))
    if prior and prior_invalidation is not None and float(current_price) <= prior_invalidation:
        replaced_setup = {
            "setup_id": prior.get("setup_id"),
            "setup_status": "invalidated",
            "setup_invalidated_at": now.isoformat(),
            "invalidation_reason": "prior_thesis_invalidation_lost",
            "prior_invalidation_level": prior_invalidation,
            "prior_primary_trigger": prior.get("primary_entry_trigger") or prior.get("confirmation_trigger"),
        }

    created_at = now.isoformat()
    if previous_setup and not replaced_setup and previous_setup.get("setup_id") == f"{ticker.upper()}-{digest}":
        created_at = str(previous_setup.get("setup_created_at") or created_at)
    return {
        "setup_id": f"{ticker.upper()}-{digest}",
        "setup_created_at": created_at,
        "setup_last_validated_at": now.isoformat(),
        "setup_status": setup_status,
        "setup_invalidated_at": invalidated_at,
        "setup_invalidation_reason": invalidation_reason,
        "replaced_setup": replaced_setup,
    }


__all__ = ["build_setup_lifecycle"]
