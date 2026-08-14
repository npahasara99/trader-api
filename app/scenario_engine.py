"""Deterministic executable scenarios derived from normalized chart structure.

The engine never calls an LLM or market-data vendor. It receives quant context
and creates bounded scenario levels from support, resistance, ATR and ranges.
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import DEFAULT_PLANNING_CONFIG, PlanningConfig


SCENARIO_KEYS = ("enter_now", "pullback", "breakout", "repair")


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _zone(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    lower = _number(value.get("lower"))
    upper = _number(value.get("upper"))
    if lower is None or upper is None:
        return None
    lower, upper = sorted((lower, upper))
    if lower <= 0:
        return None
    return {
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "display": str(value.get("display") or f"{lower:.2f} to {upper:.2f}"),
        "source_tags": list(value.get("source_tags") or []),
    }


def _midpoint(zone: dict[str, Any] | None) -> float | None:
    if not zone:
        return None
    return (float(zone["lower"]) + float(zone["upper"])) / 2.0


def _first_zone(*values: Any) -> dict[str, Any] | None:
    return next((normalized for value in values if (normalized := _zone(value))), None)


def _bounded_stop(
    *,
    entry: float,
    structural_level: float,
    atr: float,
    repair: bool,
    config: PlanningConfig,
) -> float:
    buffer = max(atr * (0.55 if repair else 0.35), entry * 0.003)
    structural_stop = structural_level - buffer
    width_pct = config.max_stop_width_pct_repair if repair else config.max_stop_width_pct_default
    width_atr = config.max_stop_width_atr_repair if repair else config.max_stop_width_atr_default
    floor = max(entry * (1.0 - width_pct), entry - atr * width_atr)
    return round(max(structural_stop, floor), 6)


def _bounded_target(
    *,
    entry: float,
    candidate: float | None,
    atr: float,
    repair: bool,
    config: PlanningConfig,
    atr_fallback: float,
    distance_scale: float = 1.0,
) -> float:
    proposed = candidate if candidate is not None and candidate > entry else entry + atr * atr_fallback
    max_pct = config.max_tp1_distance_pct_repair if repair else config.max_tp1_distance_pct_default
    max_atr = config.max_tp1_distance_atr_repair if repair else config.max_tp1_distance_atr_default
    ceiling = min(entry * (1.0 + max_pct * distance_scale), entry + atr * max_atr * distance_scale)
    return round(max(entry + max(atr * 0.35, entry * 0.003), min(proposed, ceiling)), 6)


def _reward_risk(entry: float, stop: float, target: float) -> float | None:
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 4)


def _scenario(
    *,
    name: str,
    eligible: bool,
    activated: bool,
    entry_zone: dict[str, Any] | None,
    entry_price: float | None,
    stop_loss: float | None,
    tp1: float | None,
    tp2: float | None,
    confirmation: str,
    score: float,
    reasons: list[str],
) -> dict[str, Any]:
    rr = None
    if entry_price is not None and stop_loss is not None and tp1 is not None:
        rr = _reward_risk(entry_price, stop_loss, tp1)
    return {
        "scenario_type": name,
        "eligible": bool(eligible),
        "activated": bool(activated),
        "entry_zone": entry_zone,
        "entry_price": None if entry_price is None else round(entry_price, 6),
        "structural_invalidation": stop_loss,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "confirmation_requirement": confirmation,
        "reward_risk_to_tp1": rr,
        "scenario_score": round(max(0.0, min(10.0, score)), 3),
        "reasons": reasons,
    }


def generate_execution_scenarios(
    *,
    chart_context: Mapping[str, Any],
    current_price: float,
    atr: float,
    support_zone_1: Mapping[str, Any] | None = None,
    support_zone_2: Mapping[str, Any] | None = None,
    resistance_zone_1: Mapping[str, Any] | None = None,
    resistance_zone_2: Mapping[str, Any] | None = None,
    trend_state: str | None = None,
    relative_strength_score: float | None = None,
    macro_alignment_score: float | None = None,
    news_regime_alignment: str | None = None,
    config: PlanningConfig = DEFAULT_PLANNING_CONFIG,
) -> dict[str, Any]:
    """Generate four quant-grounded alternatives and a preferred execution action."""
    price = float(current_price)
    atr_value = max(float(atr), price * 0.002, 0.01)
    trend = str(trend_state or chart_context.get("dominant_trend") or "range")
    structure = str(chart_context.get("current_structure") or "")
    extension = str(chart_context.get("extension_state") or "balanced")
    breakout_state = str(chart_context.get("breakout_state") or "inside_range")
    preferred_shape = str(chart_context.get("preferred_trade_shape") or "no_clean_trade")
    rsi = _number(chart_context.get("rsi"))
    volume_state = str(chart_context.get("volume_state") or "unknown")
    support = _first_zone(chart_context.get("nearest_support_zone"), support_zone_1)
    secondary_support = _first_zone(chart_context.get("secondary_support_zone"), support_zone_2)
    resistance = _first_zone(chart_context.get("nearest_resistance_zone"), resistance_zone_1)
    major_resistance = _first_zone(chart_context.get("major_resistance_zone"), resistance_zone_2)
    breakout_zone = _first_zone(chart_context.get("breakout_trigger_zone"), resistance)
    damaged = trend in {"downtrend", "weak_breakdown_risk"} or structure in {"damaged_structure", "constructive_recovery"}
    strong_context = (
        float(relative_strength_score or 5.0) >= 5.5
        and float(macro_alignment_score or 5.0) >= 5.0
        and str(news_regime_alignment or "neutral") not in {"conflicted_bearish", "aligned_bearish"}
    )

    near_support = bool(support) and float(support["lower"]) - atr_value * 0.25 <= price <= float(support["upper"]) + atr_value * 0.8
    overextended = extension == "overextended" or (rsi is not None and rsi >= 72.0)

    enter_stop = _bounded_stop(
        entry=price,
        structural_level=float(support["lower"]) if support else price - atr_value,
        atr=atr_value,
        repair=False,
        config=config,
    )
    enter_tp1 = _bounded_target(
        entry=price,
        candidate=float(resistance["lower"]) if resistance else None,
        atr=atr_value,
        repair=False,
        config=config,
        atr_fallback=1.6,
    )
    enter_tp2 = _bounded_target(
        entry=price,
        candidate=float((major_resistance or resistance or {}).get("upper", 0.0)) or None,
        atr=atr_value,
        repair=False,
        config=config,
        atr_fallback=3.0,
        distance_scale=1.6,
    )
    enter_rr = _reward_risk(price, enter_stop, enter_tp1) or 0.0
    enter_eligible = near_support and not damaged and not overextended and enter_rr >= config.min_reward_risk_for_wait
    enter_now = _scenario(
        name="enter_now",
        eligible=enter_eligible,
        activated=enter_eligible,
        entry_zone={"lower": round(price - atr_value * 0.12, 6), "upper": round(price + atr_value * 0.12, 6), "display": f"{price - atr_value * 0.12:.2f} to {price + atr_value * 0.12:.2f}", "source_tags": ["live_price", "atr"]},
        entry_price=price,
        stop_loss=enter_stop,
        tp1=enter_tp1,
        tp2=enter_tp2,
        confirmation="Hold the active support area with stable or improving volume.",
        score=4.0 + (2.0 if near_support else 0.0) + min(2.0, enter_rr) + (0.7 if strong_context else 0.0) - (2.5 if overextended else 0.0),
        reasons=["Current price is near active support." if near_support else "Current price is not near enough to support.", "Immediate entry is blocked when price is extended." if overextended else "Extension is controlled."],
    )

    pullback_entry = _midpoint(support)
    pullback_stop = None
    pullback_tp1 = None
    pullback_tp2 = None
    if support and pullback_entry:
        pullback_stop = _bounded_stop(entry=pullback_entry, structural_level=float(support["lower"]), atr=atr_value, repair=False, config=config)
        pullback_tp1 = _bounded_target(entry=pullback_entry, candidate=float(resistance["lower"]) if resistance else None, atr=atr_value, repair=False, config=config, atr_fallback=1.8)
        pullback_tp2 = _bounded_target(entry=pullback_entry, candidate=float((major_resistance or resistance or {}).get("upper", 0.0)) or None, atr=atr_value, repair=False, config=config, atr_fallback=3.2, distance_scale=1.6)
    pullback_eligible = bool(support and pullback_entry and not (damaged and structure != "constructive_recovery"))
    pullback = _scenario(
        name="pullback",
        eligible=pullback_eligible,
        activated=bool(pullback_eligible and float(support["lower"]) <= price <= float(support["upper"])),
        entry_zone=support,
        entry_price=pullback_entry,
        stop_loss=pullback_stop,
        tp1=pullback_tp1,
        tp2=pullback_tp2,
        confirmation="Hold or reclaim the support zone, then print a constructive close.",
        score=4.8 + (1.8 if trend in {"uptrend", "pullback_in_uptrend"} else 0.0) + (1.0 if preferred_shape in {"pullback_preferred", "continuation_pullback"} else 0.0) + (0.8 if overextended else 0.0),
        reasons=["Entry is tied to the nearest quantified support zone.", "Pullback geometry is preferred when immediate price is extended." if overextended else "Support offers cleaner geometry than a mid-range chase."],
    )

    breakout_entry = float(breakout_zone["upper"]) if breakout_zone else None
    breakout_stop = None
    breakout_tp1 = None
    breakout_tp2 = None
    if breakout_entry is not None:
        breakout_entry = round(breakout_entry + max(atr_value * 0.05, breakout_entry * 0.001), 6)
        structural = float(breakout_zone["lower"])
        breakout_stop = _bounded_stop(entry=breakout_entry, structural_level=structural, atr=atr_value, repair=False, config=config)
        next_resistance = float(major_resistance["lower"]) if major_resistance and float(major_resistance["lower"]) > breakout_entry else None
        breakout_tp1 = _bounded_target(entry=breakout_entry, candidate=next_resistance, atr=atr_value, repair=False, config=config, atr_fallback=1.7)
        breakout_tp2 = _bounded_target(entry=breakout_entry, candidate=float(major_resistance["upper"]) if major_resistance and float(major_resistance["upper"]) > breakout_entry else None, atr=atr_value, repair=False, config=config, atr_fallback=3.0, distance_scale=1.6)
    breakout_activated = bool(breakout_zone and price > float(breakout_zone["upper"]) and breakout_state == "confirmed_breakout")
    breakout_eligible = bool(breakout_zone and not damaged and (not overextended or strong_context))
    breakout = _scenario(
        name="breakout",
        eligible=breakout_eligible,
        activated=breakout_activated,
        entry_zone=breakout_zone,
        entry_price=breakout_entry,
        stop_loss=breakout_stop,
        tp1=breakout_tp1,
        tp2=breakout_tp2,
        confirmation="Close and hold above the trigger zone with improving volume; a successful retest is preferred.",
        score=4.5 + (2.2 if breakout_activated else 0.0) + (1.0 if strong_context else 0.0) + (0.8 if preferred_shape == "breakout_preferred" else 0.0) - (1.5 if overextended and not strong_context else 0.0),
        reasons=["Trigger comes from the active resistance/range regime.", "Breakout is already confirmed." if breakout_activated else "Breakout still requires acceptance above resistance."],
    )

    repair_zone = support or secondary_support
    repair_entry = _midpoint(repair_zone)
    repair_stop = None
    repair_tp1 = None
    repair_tp2 = None
    if repair_zone and repair_entry:
        repair_stop = _bounded_stop(entry=repair_entry, structural_level=float(repair_zone["lower"]), atr=atr_value, repair=True, config=config)
        repair_tp1 = _bounded_target(entry=repair_entry, candidate=float(resistance["lower"]) if resistance else None, atr=atr_value, repair=True, config=config, atr_fallback=1.3)
        repair_tp2 = _bounded_target(entry=repair_entry, candidate=float((major_resistance or resistance or {}).get("lower", 0.0)) or None, atr=atr_value, repair=True, config=config, atr_fallback=2.2, distance_scale=1.35)
    repair_eligible = bool(damaged and repair_zone and resistance and chart_context.get("short_term_reversal_state") == "confirmed")
    repair = _scenario(
        name="repair",
        eligible=repair_eligible,
        activated=False,
        entry_zone=repair_zone,
        entry_price=repair_entry,
        stop_loss=repair_stop,
        tp1=repair_tp1,
        tp2=repair_tp2,
        confirmation="Reclaim the repair trigger and hold it; a low price alone is not sufficient.",
        score=3.0 + (2.8 if repair_eligible else 0.0) + (0.8 if structure == "constructive_recovery" else 0.0),
        reasons=["Repair is available only with quantified reversal evidence.", "Targets stop at the first meaningful resistance rather than assuming full recovery."],
    )

    scenarios = {"enter_now": enter_now, "pullback": pullback, "breakout": breakout, "repair": repair}
    eligible = {name: value for name, value in scenarios.items() if value["eligible"]}
    if damaged:
        preferred = "repair" if repair_eligible else "none"
    elif breakout_activated and breakout_eligible:
        preferred = "breakout"
    elif preferred_shape in {"pullback_preferred", "continuation_pullback"} and pullback_eligible:
        preferred = "pullback"
    elif enter_eligible:
        preferred = "enter_now"
    elif preferred_shape == "breakout_preferred" and breakout_eligible:
        preferred = "breakout"
    elif eligible:
        preferred = max(eligible, key=lambda key: float(eligible[key]["scenario_score"]))
    else:
        preferred = "none"

    action_map = {
        "enter_now": "BUY_NOW",
        "pullback": "WAIT_FOR_PULLBACK",
        "breakout": "BUY_NOW" if breakout_activated else "WAIT_FOR_BREAKOUT",
        "repair": "WAIT_FOR_REPAIR",
        "none": "AVOID" if damaged and not repair_eligible else "MONITOR",
    }
    selected = scenarios.get(preferred)
    confidence = float(selected["scenario_score"]) / 10.0 if selected else 0.3
    return {
        "execution_scenarios": scenarios,
        "enter_now_scenario": enter_now,
        "pullback_scenario": pullback,
        "breakout_scenario": breakout,
        "repair_scenario": repair,
        "preferred_scenario": preferred,
        "execution_action": action_map[preferred],
        "scenario_confidence": round(max(0.0, min(0.95, confidence)), 3),
        "scenario_selection_reason": (
            "No clean executable scenario is available from the current structure."
            if selected is None
            else f"{preferred.replace('_', ' ')} has the strongest valid structure and execution geometry."
        ),
        "pullback_entry_zone": pullback.get("entry_zone"),
        "breakout_trigger_zone": breakout.get("entry_zone"),
        "repair_trigger_zone": resistance if damaged else None,
    }


def evaluate_live_scenario_status(plan: Mapping[str, Any], live_price: float | None) -> dict[str, Any]:
    """Evaluate scenario activation/invalidation without mutating the saved plan."""
    price = _number(live_price)
    scenarios = plan.get("execution_scenarios") if isinstance(plan.get("execution_scenarios"), Mapping) else {}
    preferred = str(plan.get("preferred_scenario") or "none")
    selected = scenarios.get(preferred) if isinstance(scenarios, Mapping) else None
    if price is None or not isinstance(selected, Mapping):
        return {"live_scenario_status": "unavailable", "preferred_scenario_changed": False, "replan_needed": False}
    stop = _number(selected.get("stop_loss"))
    tp1 = _number(selected.get("take_profit_1"))
    entry_zone = _zone(selected.get("entry_zone"))
    breakout = scenarios.get("breakout") if isinstance(scenarios, Mapping) else None
    breakout_zone = _zone((breakout or {}).get("entry_zone")) if isinstance(breakout, Mapping) else None
    if stop is not None and price <= stop:
        return {"live_scenario_status": "scenario_invalidated", "preferred_scenario_changed": False, "replan_needed": True}
    if tp1 is not None and price >= tp1:
        return {"live_scenario_status": "tp1_hit_replan", "preferred_scenario_changed": False, "replan_needed": True}
    if breakout_zone and price > float(breakout_zone["upper"]) and preferred != "breakout":
        return {"live_scenario_status": "breakout_activated", "preferred_scenario_changed": True, "replan_needed": True}
    if entry_zone and float(entry_zone["lower"]) <= price <= float(entry_zone["upper"]):
        return {"live_scenario_status": "preferred_entry_active", "preferred_scenario_changed": False, "replan_needed": False}
    if entry_zone and price > float(entry_zone["upper"]) * 1.04:
        return {"live_scenario_status": "preferred_entry_missed", "preferred_scenario_changed": False, "replan_needed": True}
    return {"live_scenario_status": "scenario_still_valid", "preferred_scenario_changed": False, "replan_needed": False}


__all__ = ["SCENARIO_KEYS", "evaluate_live_scenario_status", "generate_execution_scenarios"]
