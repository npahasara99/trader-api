"""Chart-aware level candidates, stale-plan checks, and deterministic validation."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from .config import LiveMonitorConfig


LEVEL_NAMES = (
    "near_confirmation",
    "primary_entry_trigger",
    "strong_confirmation",
    "major_trend_repair",
    "invalidation_level",
    "suggested_stop",
    "optional_support_level",
    "tp1",
    "tp2",
    "tp3",
    "stretch_target",
)


def number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("price") or value.get("high") or value.get("upper") or value.get("low") or value.get("lower")
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _bar_number(bar: dict, name: str) -> float | None:
    return number(bar.get(name))


def _pivot_levels(bars: list[dict], *, kind: str, radius: int = 2) -> list[float]:
    field = "high" if kind == "resistance" else "low"
    values = [_bar_number(bar, field) for bar in bars]
    pivots: list[float] = []
    for index in range(radius, len(values) - radius):
        value = values[index]
        if value is None:
            continue
        neighbors = [item for item in values[index - radius:index + radius + 1] if item is not None]
        if not neighbors:
            continue
        if kind == "resistance" and value >= max(neighbors):
            pivots.append(value)
        elif kind == "support" and value <= min(neighbors):
            pivots.append(value)
    return pivots


def _cluster_levels(values: list[float], *, tolerance: float) -> list[dict[str, Any]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        matching = next((cluster for cluster in clusters if abs(value - sum(cluster) / len(cluster)) <= tolerance), None)
        if matching is None:
            clusters.append([value])
        else:
            matching.append(value)
    return [
        {
            "price": round(sum(cluster) / len(cluster), 6),
            "reaction_count": len(cluster),
            "range_low": round(min(cluster), 6),
            "range_high": round(max(cluster), 6),
        }
        for cluster in clusters
    ]


def ranked_reaction_levels(bars: list[dict], *, current_price: float, atr: float) -> dict[str, list[dict[str, Any]]]:
    tolerance = max(current_price * 0.003, atr * 0.20, 0.01)
    resistances = _cluster_levels(_pivot_levels(bars, kind="resistance"), tolerance=tolerance)
    supports = _cluster_levels(_pivot_levels(bars, kind="support"), tolerance=tolerance)
    for level in resistances:
        level["distance_atr"] = round((level["price"] - current_price) / max(atr, 1e-9), 4)
        level["kind"] = "RESISTANCE"
    for level in supports:
        level["distance_atr"] = round((current_price - level["price"]) / max(atr, 1e-9), 4)
        level["kind"] = "SUPPORT"
    resistances = sorted(
        [item for item in resistances if item["price"] > current_price],
        key=lambda item: (-item["reaction_count"], item["price"]),
    )
    supports = sorted(
        [item for item in supports if item["price"] < current_price],
        key=lambda item: (-item["reaction_count"], -item["price"]),
    )
    return {"resistance": resistances, "support": supports}


def derive_chart_level_candidates(
    *,
    current_price: float,
    atr: float,
    planner_levels: dict,
    structure_bars: list[dict],
    execution_bars: list[dict],
    config: LiveMonitorConfig,
) -> dict[str, Any]:
    """Separate local swing confirmation from distant major trend repair."""
    combined = structure_bars[-160:] + execution_bars[-120:]
    ranked = ranked_reaction_levels(combined, current_price=current_price, atr=atr)
    local_resistance = sorted(
        [item for item in ranked["resistance"] if 0 < item["distance_atr"] <= config.trigger_max_atr],
        key=lambda item: (item["price"], -item["reaction_count"]),
    )
    local_support = sorted(ranked["support"], key=lambda item: (-item["price"], -item["reaction_count"]))
    planner_primary = number(planner_levels.get("primary_entry_trigger"))
    planner_strong = number(planner_levels.get("strong_confirmation"))
    planner_major = number(planner_levels.get("major_trend_repair"))
    planner_distance_atr = None if planner_primary is None else (planner_primary - current_price) / max(atr, 1e-9)
    recent_highs = [value for value in (_bar_number(bar, "high") for bar in combined) if value is not None]
    recent_lows = [value for value in (_bar_number(bar, "low") for bar in combined) if value is not None]
    recent_range = (max(recent_highs) - min(recent_lows)) if recent_highs and recent_lows else None
    distant_planner = bool(
        planner_primary is not None
        and (
            (planner_distance_atr is not None and planner_distance_atr > config.trigger_max_atr)
            or (
                recent_range is not None
                and planner_primary - current_price > recent_range * config.trigger_max_range_fraction
            )
        )
    )

    near = local_resistance[0]["price"] if local_resistance else number(planner_levels.get("near_confirmation"))
    primary = (
        local_resistance[1]["price"]
        if len(local_resistance) > 1 and local_resistance[1]["price"] - near <= max(atr * 1.5, current_price * 0.04)
        else near
    )
    if primary is None and not distant_planner:
        primary = planner_primary
    strong_candidates = [item["price"] for item in local_resistance if primary is not None and item["price"] > primary]
    strong = strong_candidates[0] if strong_candidates else (planner_strong if planner_strong and (primary is None or planner_strong > primary) else None)
    major = planner_major
    if distant_planner:
        major = max(value for value in (planner_primary, planner_major) if value is not None)
    elif major is None and planner_primary and primary and planner_primary > primary + max(atr * 2.0, current_price * 0.05):
        major = planner_primary
    support = local_support[0]["price"] if local_support else number(planner_levels.get("optional_support_level"))

    return {
        "near_confirmation": near,
        "primary_entry_trigger": primary,
        "strong_confirmation": strong,
        "major_trend_repair": major,
        "optional_support_level": support,
        "invalidation_level": number(planner_levels.get("invalidation_level")),
        "suggested_stop": number(planner_levels.get("suggested_stop")),
        "tp1": number(planner_levels.get("tp1")),
        "tp2": number(planner_levels.get("tp2")),
        "tp3": number(planner_levels.get("tp3")),
        "stretch_target": number(planner_levels.get("stretch_target")),
        "planner_primary_distance_atr": None if planner_distance_atr is None else round(planner_distance_atr, 4),
        "planner_primary_reclassified_as_major_repair": distant_planner,
        "ranked_resistance_evidence": ranked["resistance"][:12],
        "ranked_support_evidence": ranked["support"][:12],
    }


def check_data_consistency(
    *,
    planner_price: float | None,
    monitor_price: float | None,
    chart_close: float | None,
    atr: float | None,
) -> dict[str, Any]:
    values = {"planner": number(planner_price), "monitor": number(monitor_price), "chart": number(chart_close)}
    present = [value for value in values.values() if value is not None]
    if len(present) < 2:
        return {"status": "INSUFFICIENT_DATA", "prices": values, "max_difference_pct": None}
    reference = sum(present) / len(present)
    max_diff = max(present) - min(present)
    max_diff_pct = max_diff / max(reference, 1e-9)
    tolerance = max(0.01, (number(atr) or 0.0) * 0.50 / max(reference, 1e-9))
    return {
        "status": "CHART_DATA_MISMATCH" if max_diff_pct > tolerance else "CONSISTENT",
        "prices": values,
        "max_difference_pct": round(max_diff_pct, 6),
        "tolerance_pct": round(tolerance, 6),
    }


def detect_stale_plan(
    *,
    current_price: float | None,
    levels: dict,
    atr: float | None,
    setup_created_at: datetime,
    structure_bars: list[dict],
    config: LiveMonitorConfig,
    plan_reference_price: float | None = None,
    plan_created_at: datetime | None = None,
    data_consistency_status: str | None = None,
) -> dict[str, Any]:
    current = number(current_price)
    atr_value = number(atr) or (current or 1.0) * 0.02
    reasons: list[str] = []
    warnings: list[str] = []
    invalidation = number(levels.get("invalidation_level"))
    support = number(levels.get("optional_support_level"))
    primary = number(levels.get("primary_entry_trigger"))
    reference = number(plan_reference_price)
    drift = None if current is None or reference is None else current - reference
    drift_pct = None if drift is None else drift / reference
    drift_atr = None if drift is None else drift / max(atr_value, 1e-9)
    now = datetime.now(timezone.utc)
    created_source = plan_created_at or setup_created_at
    created = created_source if created_source.tzinfo else created_source.replace(tzinfo=timezone.utc)
    if reference is None:
        reasons.append("PLAN_REFERENCE_MISSING")
    if drift_pct is not None and abs(drift_pct) > config.plan_price_drift_pct:
        reasons.append("PRICE_DRIFT")
    if drift_atr is not None and abs(drift_atr) > config.plan_price_drift_atr:
        reasons.append("ATR_DRIFT")
    if current is not None and invalidation is not None and current <= invalidation:
        reasons.append("STRUCTURE_INVALIDATED")
    if current is not None and support is not None and current < support - atr_value * config.support_failure_atr:
        reasons.append("SUPPORT_FAILED")
        warnings.append("OLD_SUPPORT_LOST")
    if current is not None and primary is not None and (primary - current) / atr_value > config.trigger_max_atr:
        warnings.append("PRIMARY_TRIGGER_SANITY_WARNING")
    if (now - created).days > config.plan_max_age_days:
        reasons.append("PLAN_TOO_OLD")
    if len(structure_bars) >= 2:
        prior_close = _bar_number(structure_bars[-2], "close")
        last_open = _bar_number(structure_bars[-1], "open")
        if prior_close and last_open and abs(last_open - prior_close) > atr_value * config.major_gap_atr:
            reasons.append("MAJOR_GAP")
    recent = structure_bars[-config.new_structure_lookback_bars:]
    recent_lows = [value for value in (_bar_number(bar, "low") for bar in recent) if value is not None]
    recent_highs = [value for value in (_bar_number(bar, "high") for bar in recent) if value is not None]
    if reference is not None and recent_lows and recent_highs:
        if reference < min(recent_lows) - atr_value * 0.5 or reference > max(recent_highs) + atr_value * 0.5:
            reasons.append("NEW_STRUCTURE")
    if data_consistency_status == "MARKET_DATA_MISMATCH":
        reasons.append("DATA_MISMATCH")
    reasons = list(dict.fromkeys(reasons))
    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "warnings": list(dict.fromkeys(warnings)),
        "status": "PLAN_STALE" if reasons else "CURRENT",
        "plan_reference_price": reference,
        "current_price": current,
        "price_drift_pct": None if drift_pct is None else round(drift_pct, 6),
        "price_drift_atr": None if drift_atr is None else round(drift_atr, 4),
        "plan_age_seconds": max(0.0, (now - created).total_seconds()),
    }


def validate_level_semantics(*, current_price: float | None, levels: dict, config: LiveMonitorConfig) -> dict[str, Any]:
    """Validate long-side labels and short-swing geometry without inventing levels."""
    current = number(current_price)
    atr = number(levels.get("atr")) or (current or 1.0) * 0.02
    primary = number(levels.get("primary_entry_trigger"))
    support = number(levels.get("optional_support_level"))
    invalidation = number(levels.get("invalidation_level"))
    warnings: list[str] = []
    reclassified: dict[str, str] = {}
    if current is not None and support is not None and support > current + atr * 0.10:
        warnings.append("OLD_SUPPORT_LOST")
        reclassified["optional_support_level"] = "RESISTANCE_OR_HISTORICAL_SUPPORT"
    if primary is not None and current is not None and (primary - current) / max(atr, 1e-9) > config.trigger_max_atr:
        warnings.append("PRIMARY_TRIGGER_SANITY_WARNING")
        reclassified["primary_entry_trigger"] = "MAJOR_TREND_REPAIR_CANDIDATE"
    if invalidation is not None and primary is not None and invalidation >= primary:
        warnings.append("INVALIDATION_SEMANTIC_ERROR")
    target_diagnostics: dict[str, Any] = {}
    for name in ("tp1", "tp2", "tp3", "stretch_target"):
        target = number(levels.get(name))
        if target is None or primary is None:
            continue
        distance_atr = (target - primary) / max(atr, 1e-9)
        target_diagnostics[name] = {
            "distance_pct": round((target - primary) / primary, 6),
            "distance_atr": round(distance_atr, 4),
            "reachable_2_10_days": bool(0 < distance_atr <= config.chart_max_target_atr),
        }
        if name == "tp1" and distance_atr > config.chart_max_target_atr:
            warnings.append("TARGET_REACHABILITY_WARNING")
    return {
        "status": "WARNING" if warnings else "VALID",
        "warnings": list(dict.fromkeys(warnings)),
        "reclassified_levels": reclassified,
        "target_diagnostics": target_diagnostics,
    }


def validate_chart_levels(
    *,
    current_price: float,
    atr: float,
    proposed_levels: dict,
    planner_levels: dict,
    candidate_evidence: dict,
    structure_bars: list[dict],
    trigger_max_atr: float = 3.0,
    stop_max_atr: float = 3.5,
    target_max_atr: float = 5.0,
) -> dict[str, Any]:
    lows = [value for value in (_bar_number(bar, "low") for bar in structure_bars) if value is not None]
    highs = [value for value in (_bar_number(bar, "high") for bar in structure_bars) if value is not None]
    range_low = min(lows) if lows else current_price - atr * 5
    range_high = max(highs) if highs else current_price + atr * 5
    evidence_prices = [
        number(item.get("price"))
        for group in (candidate_evidence.get("ranked_resistance_evidence") or [], candidate_evidence.get("ranked_support_evidence") or [])
        for item in group
    ]
    evidence_prices = [value for value in evidence_prices if value is not None]
    tolerance = max(atr * 0.25, current_price * 0.003)
    accepted: dict[str, float] = {}
    rejected: dict[str, list[str]] = {}
    for name in LEVEL_NAMES:
        value = number(proposed_levels.get(name))
        if value is None:
            continue
        reasons: list[str] = []
        in_range = range_low - atr <= value <= range_high + atr
        supported = any(abs(value - evidence) <= tolerance for evidence in evidence_prices)
        planner_match = number(planner_levels.get(name)) is not None and abs(value - number(planner_levels.get(name))) <= tolerance
        if name == "major_trend_repair" and number(planner_levels.get("primary_entry_trigger")) is not None:
            planner_match = planner_match or abs(value - number(planner_levels.get("primary_entry_trigger"))) <= tolerance
        if not in_range and name not in {"major_trend_repair", "stretch_target"}:
            reasons.append("outside_recent_ohlc_range")
        if not supported and not planner_match and name not in {"suggested_stop", "invalidation_level"}:
            reasons.append("no_pivot_or_reaction_evidence")
        if reasons:
            rejected[name] = reasons
        else:
            accepted[name] = value

    entry = accepted.get("primary_entry_trigger")
    invalidation = accepted.get("invalidation_level") or number(planner_levels.get("invalidation_level"))
    targets = [accepted.get(name) for name in ("tp1", "tp2", "tp3")]
    if entry is not None and invalidation is not None and invalidation >= entry:
        rejected["invalidation_level"] = ["invalidation_must_be_below_entry"]
        accepted.pop("invalidation_level", None)
    previous_target = entry
    for name, target in zip(("tp1", "tp2", "tp3"), targets):
        if target is None:
            continue
        if previous_target is not None and target <= previous_target:
            rejected[name] = ["targets_must_increase_above_long_entry"]
            accepted.pop(name, None)
        else:
            previous_target = target
    primary = accepted.get("primary_entry_trigger")
    distance_atr = None if primary is None else (primary - current_price) / max(atr, 1e-9)
    flags = {
        "trigger_too_distant": bool(distance_atr is not None and distance_atr > trigger_max_atr),
        "stop_too_wide": bool(primary and invalidation and (primary - invalidation) / max(atr, 1e-9) > stop_max_atr),
        "target_unrealistic": bool(primary and accepted.get("tp1") and (accepted["tp1"] - primary) / max(atr, 1e-9) > target_max_atr),
    }
    if flags["trigger_too_distant"] and primary is not None:
        rejected["primary_entry_trigger"] = ["trigger_too_distant_for_short_swing"]
        accepted.pop("primary_entry_trigger", None)
    return {
        "status": "VALIDATION_FAILED" if rejected and not accepted else "PARTIAL" if rejected else "VALID",
        "accepted_levels": accepted,
        "rejected_levels": rejected,
        "flags": flags,
        "distance_to_primary_trigger_pct": None if primary is None else round((primary - current_price) / current_price, 6),
        "distance_to_primary_trigger_atr": None if distance_atr is None else round(distance_atr, 4),
        "ohlc_range": {"low": range_low, "high": range_high},
        "evidence_tolerance": tolerance,
    }


def reconcile_levels(
    *,
    planner_levels: dict,
    proposed_levels: dict,
    validation: dict,
    manual_overrides: dict,
) -> dict[str, Any]:
    validated = validation.get("accepted_levels") or {}
    rejected = validation.get("rejected_levels") or {}
    final = dict(planner_levels)
    for name in LEVEL_NAMES:
        if name in final:
            final[name] = number(final.get(name))
    sources = {name: "PLANNER" for name in LEVEL_NAMES if number(final.get(name)) is not None}
    for name, value in validated.items():
        final[name] = value
        sources[name] = "VALIDATED_CHART_LLM"
    if validated.get("primary_entry_trigger") is not None:
        # The old chase boundary belongs to the old trigger regime.
        final["max_chase_price"] = None
    for name, value in manual_overrides.items():
        parsed = number(value)
        if parsed is not None:
            final[name] = parsed
            sources[name] = "MANUAL"
    disagreement = any(
        number(planner_levels.get(name)) is not None
        and number(proposed_levels.get(name)) is not None
        and abs(number(planner_levels.get(name)) - number(proposed_levels.get(name))) > max(0.01, number(planner_levels.get(name)) * 0.002)
        for name in LEVEL_NAMES
    )
    rejected_disagreements = {
        name: {
            "planner_level": number(planner_levels.get(name)),
            "llm_proposed_level": number(proposed_levels.get(name)),
            "validator_result": "REJECTED",
            "reasons": reasons,
        }
        for name, reasons in rejected.items()
        if number(proposed_levels.get(name)) is not None
        and (
            number(planner_levels.get(name)) is None
            or abs(number(planner_levels.get(name)) - number(proposed_levels.get(name)))
            > max(0.01, (number(planner_levels.get(name)) or 1.0) * 0.002)
        )
    }
    critical_rejections = {
        name: detail
        for name, detail in rejected_disagreements.items()
        if name in {"primary_entry_trigger", "invalidation_level", "suggested_stop"}
    }
    reconciliation_status = (
        "MANUAL_REVIEW_REQUIRED"
        if critical_rejections
        else "LLM_CORRECTION_PENDING"
        if disagreement and validated
        else "LLM_CORRECTION_REJECTED"
        if rejected_disagreements
        else "PLANNER_ACCEPTED"
    )
    return {
        "final_active_levels": final,
        "level_sources": sources,
        "status": "MANUAL_REVIEW_REQUIRED" if critical_rejections else "DISAGREEMENT" if disagreement else "AGREES",
        "has_disagreement": disagreement,
        "reconciliation_status": reconciliation_status,
        "activation_blocked": bool(critical_rejections),
        "rejected_level_disagreements": rejected_disagreements,
        "critical_rejections": critical_rejections,
    }
