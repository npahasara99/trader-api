"""Deterministic sanity checks for chart-aware entry, stop, and target levels."""

from __future__ import annotations

from typing import Any

from .chart_levels import number, ranked_reaction_levels
from .config import LiveMonitorConfig


LEVEL_ROLES = {
    "optional_support_level": "SUPPORT",
    "near_confirmation": "NEAR_CONFIRMATION",
    "primary_entry_trigger": "PRIMARY_ENTRY_TRIGGER",
    "strong_confirmation": "STRONG_CONFIRMATION",
    "major_trend_repair": "MAJOR_TREND_REPAIR",
    "invalidation_level": "STRUCTURAL_INVALIDATION",
    "suggested_stop": "EXECUTABLE_STOP",
    "tp1": "TP1",
    "tp2": "TP2",
    "tp3": "TP3",
    "stretch_target": "STRETCH_TARGET",
}

PRICING_ANOMALIES = {
    "PRIMARY_TRIGGER_TOO_DISTANT",
    "PRIMARY_TRIGGER_BEYOND_MULTIPLE_RESISTANCES",
    "POSSIBLE_MAJOR_REPAIR_MISCLASSIFIED_AS_PRIMARY",
    "TP1_TOO_DISTANT",
    "TP_SKIPS_NEAR_RESISTANCE",
    "STOP_TOO_WIDE",
    "STOP_NOT_STRUCTURAL",
    "SUPPORT_ABOVE_CURRENT_PRICE",
    "STALE_LEVEL_GEOMETRY",
}


def _distance(value: float | None, reference: float, atr: float) -> dict[str, float | None]:
    if value is None:
        return {"pct": None, "atr": None}
    return {
        "pct": round((value - reference) / max(reference, 1e-9), 6),
        "atr": round((value - reference) / max(atr, 1e-9), 4),
    }


def evaluate_level_sanity(
    *,
    current_price: float,
    atr: float,
    levels: dict[str, Any],
    structure_bars: list[dict[str, Any]],
    execution_bars: list[dict[str, Any]],
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    """Inspect long-side geometry without inventing replacement prices."""
    current = float(current_price)
    atr_value = max(float(atr), current * 0.0025, 1e-9)
    combined = (structure_bars or [])[-160:] + (execution_bars or [])[-120:]
    reactions = ranked_reaction_levels(combined, current_price=current, atr=atr_value)
    resistances = sorted(reactions["resistance"], key=lambda row: row["price"])
    supports = sorted(reactions["support"], key=lambda row: row["price"], reverse=True)

    primary = number(levels.get("primary_entry_trigger"))
    support = number(levels.get("optional_support_level"))
    invalidation = number(levels.get("invalidation_level"))
    stop = number(levels.get("suggested_stop"))
    tp1 = number(levels.get("tp1"))
    tp2 = number(levels.get("tp2"))
    tp3 = number(levels.get("tp3"))
    entry = primary or current
    anomalies: list[str] = []

    primary_distance = _distance(primary, current, atr_value)
    nearer_resistances = [row for row in resistances if primary is not None and current < row["price"] < primary - atr_value * 0.10]
    if primary is not None and (
        float(primary_distance["atr"] or 0.0) > config.trigger_max_atr
        or float(primary_distance["pct"] or 0.0) > config.primary_max_distance_pct
    ):
        anomalies.append("PRIMARY_TRIGGER_TOO_DISTANT")
    if len(nearer_resistances) >= 2:
        anomalies.append("PRIMARY_TRIGGER_BEYOND_MULTIPLE_RESISTANCES")
    if primary is not None and nearer_resistances and float(primary_distance["atr"] or 0.0) > max(1.5, config.trigger_max_atr * 0.70):
        anomalies.append("POSSIBLE_MAJOR_REPAIR_MISCLASSIFIED_AS_PRIMARY")

    if support is not None and support > current + atr_value * 0.10:
        anomalies.append("SUPPORT_ABOVE_CURRENT_PRICE")
    if invalidation is not None and invalidation >= entry:
        anomalies.append("STALE_LEVEL_GEOMETRY")

    stop_reference = stop or invalidation
    stop_distance = None if stop_reference is None else (entry - stop_reference) / atr_value
    structural_tolerance = max(atr_value * 0.35, current * 0.004)
    stop_has_structure = bool(
        stop_reference is not None
        and (
            any(abs(stop_reference - row["price"]) <= structural_tolerance for row in supports)
            or (invalidation is not None and abs(stop_reference - invalidation) <= structural_tolerance)
        )
    )
    if stop_distance is not None and stop_distance > config.chart_max_stop_atr:
        anomalies.append("STOP_TOO_WIDE")
    if stop_reference is not None and not stop_has_structure:
        anomalies.append("STOP_NOT_STRUCTURAL")

    target_diagnostics: dict[str, Any] = {}
    previous = entry
    for name, target in (("tp1", tp1), ("tp2", tp2), ("tp3", tp3)):
        distance = _distance(target, entry, atr_value)
        ordered = target is None or target > previous
        target_diagnostics[name] = {
            "price": target,
            "distance_pct": distance["pct"],
            "distance_atr": distance["atr"],
            "reachable_2_10_days": bool(
                target is not None and 0 < float(distance["atr"] or 0.0) <= config.target_reachability_atr
            ),
            "ordered": ordered,
        }
        if target is not None:
            previous = target
    resistance_before_tp1 = [row for row in resistances if tp1 is not None and entry < row["price"] < tp1 - atr_value * 0.10]
    if tp1 is not None and float(target_diagnostics["tp1"]["distance_atr"] or 0.0) > config.target_reachability_atr:
        anomalies.append("TP1_TOO_DISTANT")
    if resistance_before_tp1:
        anomalies.append("TP_SKIPS_NEAR_RESISTANCE")
    if not all(item["ordered"] for item in target_diagnostics.values()):
        anomalies.append("STALE_LEVEL_GEOMETRY")

    risk = entry - stop_reference if stop_reference is not None else None
    reward = tp1 - entry if tp1 is not None else None
    rr = None if risk is None or reward is None or risk <= 0 else round(reward / risk, 4)
    anomalies = list(dict.fromkeys(anomalies))
    return {
        "status": "ANOMALY" if anomalies else "VALID",
        "anomalies": anomalies,
        "review_required": any(code in PRICING_ANOMALIES for code in anomalies),
        "current_price": current,
        "atr": atr_value,
        "primary_entry": {
            "price": primary,
            "distance_pct": primary_distance["pct"],
            "distance_atr": primary_distance["atr"],
            "nearer_resistance_count": len(nearer_resistances),
            "nearer_resistances": nearer_resistances[:6],
        },
        "stop_invalidation": {
            "suggested_stop": stop,
            "structural_invalidation": invalidation,
            "distance_atr": None if stop_distance is None else round(stop_distance, 4),
            "has_structural_evidence": stop_has_structure,
            "tradeable_geometry": bool(stop_distance is not None and 0 < stop_distance <= config.chart_max_stop_atr),
        },
        "targets": target_diagnostics,
        "resistance_before_tp1": resistance_before_tp1[:6],
        "planned_rr_at_entry": rr,
        "ranked_resistance_evidence": resistances[:12],
        "ranked_support_evidence": supports[:12],
        "level_roles": LEVEL_ROLES,
    }


def can_auto_apply_chart_correction(
    *,
    review: dict[str, Any],
    manual_overrides: dict[str, Any],
    sanity: dict[str, Any],
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    """Require anomaly, validation, confidence, and no manual ownership."""
    validated = review.get("validated_levels") or {}
    confidence = float(review.get("confidence") or 0.0)
    decision = str(review.get("decision") or "").upper()
    validation_status = str((review.get("validation") or {}).get("status") or "")
    blockers: list[str] = []
    if not config.level_auto_correct_enabled:
        blockers.append("auto_correction_disabled")
    if not sanity.get("review_required"):
        blockers.append("no_pricing_anomaly")
    if confidence < config.level_auto_correct_confidence:
        blockers.append("confidence_below_threshold")
    if decision not in {"MODIFY_LEVELS", "APPROVE_LEVELS"}:
        blockers.append("review_did_not_approve_modification")
    if validation_status not in {"VALID", "PARTIAL"}:
        blockers.append("deterministic_validation_not_sufficient")
    if not validated:
        blockers.append("no_validated_levels")
    if manual_overrides:
        blockers.append("manual_levels_own_active_setup")
    return {
        "allowed": not blockers,
        "blockers": blockers,
        "confidence": confidence,
        "threshold": config.level_auto_correct_confidence,
    }


__all__ = [
    "LEVEL_ROLES",
    "PRICING_ANOMALIES",
    "can_auto_apply_chart_correction",
    "evaluate_level_sanity",
]
