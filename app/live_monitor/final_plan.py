"""Authoritative long-side plan finalization and geometry validation.

This module is intentionally deterministic. It never creates arbitrary price
levels: regenerated targets must come from current reaction evidence or an
existing structured level that remains above the finalized entry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from .chart_levels import LEVEL_NAMES, number, ranked_reaction_levels
from .config import LiveMonitorConfig
from .level_sanity import LEVEL_ROLES


TARGET_NAMES = ("tp1", "tp2", "tp3", "stretch_target")
CRITICAL_EXECUTION_LEVELS = {
    "primary_entry_trigger",
    "invalidation_level",
    "suggested_stop",
}


def _unique_candidates(rows: list[dict[str, Any]], *, tolerance: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["price"])):
        if not selected or abs(float(row["price"]) - float(selected[-1]["price"])) > tolerance:
            selected.append(row)
    return selected


def regenerate_targets(
    *,
    entry: float,
    levels: dict[str, Any],
    structure_bars: list[dict[str, Any]],
    execution_bars: list[dict[str, Any]],
    current_price: float,
    atr: float,
) -> dict[str, Any]:
    """Regenerate targets from structured levels above the finalized entry."""
    atr_value = max(float(atr), float(current_price) * 0.0025, 1e-9)
    reactions = ranked_reaction_levels(
        (structure_bars or [])[-160:] + (execution_bars or [])[-120:],
        current_price=float(current_price),
        atr=atr_value,
    )
    candidates: list[dict[str, Any]] = []
    for row in reactions.get("resistance") or []:
        price = number(row.get("price"))
        if price is not None and price > entry:
            candidates.append({
                "price": price,
                "reason": f"Current reaction resistance with {int(row.get('reaction_count') or 0)} recorded reactions",
                "evidence": "ranked_resistance",
            })
    # Confirmation bands are entry evidence, not profit objectives. Preserve
    # major repair and existing target levels as structured target candidates.
    for name in ("major_trend_repair", *TARGET_NAMES):
        price = number(levels.get(name))
        if price is not None and price > entry:
            candidates.append({
                "price": price,
                "reason": f"Existing structured {name.replace('_', ' ')} remains above finalized entry",
                "evidence": name,
            })
    candidates = _unique_candidates(candidates, tolerance=max(atr_value * 0.10, entry * 0.001, 0.01))
    targets = {name: None for name in TARGET_NAMES}
    reasons: dict[str, str] = {}
    for name, candidate in zip(TARGET_NAMES, candidates[:4]):
        targets[name] = round(float(candidate["price"]), 6)
        reasons[name] = str(candidate["reason"])
    return {
        "targets": targets,
        "target_reasons": reasons,
        "candidate_levels": candidates,
        "status": "REGENERATED" if targets["tp1"] is not None else "NO_VALID_TP1",
    }


def validate_final_plan(
    *,
    levels: dict[str, Any],
    current_price: float | None,
    market_snapshot_id: str | None,
    level_metadata: dict[str, Any] | None = None,
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    """Enforce hard long-plan invariants after every possible level mutation."""
    primary = number(levels.get("primary_entry_trigger"))
    invalidation = number(levels.get("invalidation_level"))
    stop = number(levels.get("suggested_stop"))
    major_repair = number(levels.get("major_trend_repair"))
    support = number(levels.get("optional_support_level"))
    targets = {name: number(levels.get(name)) for name in TARGET_NAMES}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def fail(code: str, message: str, **evidence: Any) -> None:
        failures.append({"code": code, "message": message, "evidence": evidence})

    def warn(code: str, message: str, **evidence: Any) -> None:
        warnings.append({"code": code, "message": message, "evidence": evidence})

    if primary is None:
        fail("NO_PRIMARY_ENTRY_TRIGGER", "A long monitor requires a finalized primary entry trigger.")
    if invalidation is None:
        fail("NO_STRUCTURAL_INVALIDATION", "A long monitor requires structural invalidation.")
    elif primary is not None and invalidation >= primary:
        fail("INVALIDATION_NOT_BELOW_ENTRY", "Structural invalidation must be below long entry.", invalidation=invalidation, entry=primary)
    if stop is None:
        fail("NO_SUGGESTED_STOP", "A long monitor requires an executable suggested stop.")
    elif primary is not None and stop >= primary:
        fail("STOP_NOT_BELOW_ENTRY", "Suggested stop must be below long entry.", stop=stop, entry=primary)
    if targets["tp1"] is None:
        fail("NO_VALID_TP1", "No structured target above the finalized entry is available.")
    elif primary is not None and targets["tp1"] <= primary:
        fail("TP1_NOT_ABOVE_ENTRY", "TP1 must be above the finalized long entry.", tp1=targets["tp1"], entry=primary)
    previous = primary
    for name in ("tp1", "tp2", "tp3"):
        target = targets[name]
        if target is None:
            continue
        if previous is not None and target <= previous:
            fail("TARGET_ORDER_INVALID", f"{name.upper()} must be above the preceding long-plan level.", target=name, price=target, preceding=previous)
        previous = target
    if targets["stretch_target"] is not None and targets["tp3"] is not None and targets["stretch_target"] < targets["tp3"]:
        fail("STRETCH_BELOW_TP3", "Stretch target cannot be below TP3.", stretch=targets["stretch_target"], tp3=targets["tp3"])
    if major_repair is not None and primary is not None and major_repair < primary:
        fail("MAJOR_REPAIR_BELOW_ENTRY", "Major trend repair cannot be below the primary long trigger.", major_repair=major_repair, entry=primary)
    if support is not None and current_price is not None and support > float(current_price):
        support_role = str(levels.get("optional_support_role") or "SUPPORT").upper()
        if support_role not in {"LOST_SUPPORT", "RESISTANCE", "HISTORICAL_SUPPORT"}:
            fail("SUPPORT_MISLABELED_ABOVE_PRICE", "A level above current price cannot remain active support.", support=support, current_price=current_price)

    if primary is not None and current_price is not None and primary > current_price:
        atr = number(levels.get("atr")) or max(float(current_price) * 0.02, 1e-9)
        distance_atr = (primary - float(current_price)) / atr
        if distance_atr > config.trigger_max_atr:
            warn("PRIMARY_TRIGGER_SUSPICIOUS", "Primary trigger is unusually distant for a short swing.", distance_atr=round(distance_atr, 4))
    if primary is not None and stop is not None:
        atr = number(levels.get("atr")) or max((current_price or primary) * 0.02, 1e-9)
        stop_width_atr = (primary - stop) / atr
        if stop_width_atr > config.chart_max_stop_atr:
            warn("STOP_TOO_WIDE", "Executable stop width exceeds the configured chart threshold.", distance_atr=round(stop_width_atr, 4))
    if primary is not None and stop is not None and targets["tp1"] is not None and primary > stop:
        rr = (targets["tp1"] - primary) / (primary - stop)
        if rr < config.minimum_current_rr:
            warn("PLANNED_RR_WEAK", "Planned TP1 reward/risk is below the configured minimum.", planned_rr=round(rr, 4))

    for name, metadata in (level_metadata or {}).items():
        if name not in LEVEL_NAMES or not isinstance(metadata, dict):
            continue
        level_snapshot = metadata.get("market_snapshot_id")
        if market_snapshot_id and level_snapshot and level_snapshot != market_snapshot_id:
            fail(
                "MARKET_SNAPSHOT_MISMATCH",
                "Active levels must belong to the current setup market snapshot.",
                level=name,
                expected=market_snapshot_id,
                actual=level_snapshot,
            )

    status = "INVALID" if failures else "WARNING" if warnings else "VALID"
    return {
        "status": status,
        "code": "PLAN_GEOMETRY_INVALID" if failures else "PLAN_GEOMETRY_WARNING" if warnings else "PLAN_GEOMETRY_VALID",
        "activation_allowed": not failures,
        "hard_failures": failures,
        "warnings": warnings,
    }


def finalize_active_plan(
    *,
    setup_id: str,
    levels: dict[str, Any],
    sources: dict[str, str],
    current_price: float | None,
    market_snapshot_id: str | None,
    config: LiveMonitorConfig,
    reconciliation_status: str,
    structure_bars: list[dict[str, Any]] | None = None,
    execution_bars: list[dict[str, Any]] | None = None,
    entry_changed: bool = False,
    change_source: str | None = None,
    level_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the single plan consumed by monitor, chart, R:R, and order UI."""
    finalized = dict(levels)
    clean_sources = dict(sources)
    reasons = dict(level_reasons or {})
    target_regeneration = {"status": "NOT_REQUIRED", "target_reasons": {}, "candidate_levels": []}
    entry = number(finalized.get("primary_entry_trigger"))
    atr = number(finalized.get("atr")) or ((current_price or entry or 1.0) * 0.02)
    if entry_changed and entry is not None and current_price is not None:
        target_regeneration = regenerate_targets(
            entry=entry,
            levels=finalized,
            structure_bars=structure_bars or [],
            execution_bars=execution_bars or [],
            current_price=float(current_price),
            atr=float(atr),
        )
        finalized.update(target_regeneration["targets"])
        for name in TARGET_NAMES:
            if finalized.get(name) is not None:
                clean_sources[name] = change_source or clean_sources.get("primary_entry_trigger") or "PLANNER"
                reasons[name] = target_regeneration["target_reasons"].get(name, "Regenerated after primary entry changed")
            else:
                clean_sources.pop(name, None)
        finalized["max_chase_price"] = None

    created_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {}
    for name in LEVEL_NAMES:
        price = number(finalized.get(name))
        if price is None:
            continue
        metadata[name] = {
            "price": price,
            "level_type": LEVEL_ROLES.get(name, name.upper()),
            "source": clean_sources.get(name) or "PLANNER",
            "reason": reasons.get(name) or "Current structured planner/monitor level",
            "confidence": None,
            "market_snapshot_id": market_snapshot_id,
            "created_at": created_at,
        }
    validation = validate_final_plan(
        levels=finalized,
        current_price=current_price,
        market_snapshot_id=market_snapshot_id,
        level_metadata=metadata,
        config=config,
    )
    plan_id = str(uuid.uuid4())
    finalized["_active_plan_id"] = plan_id
    finalized["_market_snapshot_id"] = market_snapshot_id
    finalized["_plan_integrity_status"] = validation["status"]
    return {
        "plan_id": plan_id,
        "setup_id": setup_id,
        "side": "LONG",
        "market_snapshot_id": market_snapshot_id,
        "created_at": created_at,
        "source": "FINAL_ACTIVE_PLAN",
        "reconciliation_status": reconciliation_status,
        "plan_integrity_status": validation["status"],
        "validation": validation,
        "levels": metadata,
        "flat_levels": finalized,
        "level_sources": clean_sources,
        "target_regeneration": target_regeneration,
    }


__all__ = [
    "CRITICAL_EXECUTION_LEVELS",
    "TARGET_NAMES",
    "finalize_active_plan",
    "regenerate_targets",
    "validate_final_plan",
]
