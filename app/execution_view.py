"""Trader-readable execution framing built from structured planner fields."""

from __future__ import annotations

from .config import PlanningConfig
from .monitoring import build_zone_display


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _zone_payload(zone: dict | None, *, current_price: float | None, label: str) -> dict | None:
    display = build_zone_display(zone, current_price=current_price, zone_label=label)
    if display["display"] is None:
        return None
    return {
        "lower": float(zone["lower"]),
        "upper": float(zone["upper"]),
        "display": display["display"],
    }


def _current_price_location(*, current_price: float, support_zone_1: dict | None, resistance_zone_1: dict | None) -> str:
    if support_zone_1 and current_price < float(support_zone_1.get("lower", current_price)):
        return "below_support"
    if resistance_zone_1 and current_price > float(resistance_zone_1.get("upper", current_price)):
        return "above_breakout"
    if support_zone_1 and resistance_zone_1:
        support_upper = float(support_zone_1.get("upper", current_price))
        resistance_lower = float(resistance_zone_1.get("lower", current_price))
        range_width = max(resistance_lower - support_upper, current_price * 0.01, 0.01)
        if current_price <= support_upper + range_width * 0.35:
            return "near_support"
        if current_price >= resistance_lower - range_width * 0.35:
            return "near_resistance"
        return "mid_range"
    if support_zone_1:
        return "near_support"
    if resistance_zone_1:
        return "near_resistance"
    return "mid_range"


def build_chart_execution_view(row, *, config: PlanningConfig) -> dict | None:
    """Frame the current setup in trader execution terms using existing zones."""

    current_price = _safe_float(getattr(row, "current_price", None), 0.0)
    if current_price <= 0:
        return None

    final_action = str(getattr(row, "final_action", None) or "").upper()
    trend_state = str(getattr(row, "trend_state", None) or "")
    preferred_entry = _safe_float(getattr(row, "preferred_entry", None), current_price)
    preferred_entry_type = str(getattr(row, "preferred_entry_type", None) or "")
    entry_quality = _safe_float(getattr(row, "entry_quality_score", None))
    entry_requires_confirmation = bool(getattr(row, "entry_requires_confirmation", False))
    confirmation_trigger = str(getattr(row, "confirmation_trigger", None) or "")
    reward_risk = getattr(row, "reward_risk", None) or {}
    rr1 = _safe_float(reward_risk.get("tp1"))
    support_zone_1 = getattr(row, "support_zone_1", None)
    support_zone_2 = getattr(row, "support_zone_2", None)
    resistance_zone_1 = getattr(row, "resistance_zone_1", None)
    resistance_zone_2 = getattr(row, "resistance_zone_2", None)
    moving_averages = getattr(row, "moving_averages", None) or {}
    volume_context = getattr(row, "volume_context", None) or {}

    breakout_point = _zone_payload(resistance_zone_1, current_price=current_price, label="Breakout Point")
    pullback_entry_zone = _zone_payload(support_zone_1, current_price=current_price, label="Pullback Entry Zone")
    deeper_pullback_zone = _zone_payload(support_zone_2, current_price=current_price, label="Deeper Pullback Zone")
    location = _current_price_location(
        current_price=current_price,
        support_zone_1=support_zone_1,
        resistance_zone_1=resistance_zone_1,
    )

    near_resistance = location in {"near_resistance", "above_breakout"}
    near_support = location == "near_support"
    weak_reversal = str(volume_context.get("reversal_volume_state") or "") in {"weak_bounce", "no_confirmation"}
    weak_structure = trend_state in {"weak_breakdown_risk", "downtrend"}

    if weak_structure:
        trade_shape = "structure_repair_needed"
    elif near_resistance and breakout_point and pullback_entry_zone:
        trade_shape = "breakout_or_pullback"
    elif near_resistance:
        trade_shape = "extended_near_resistance"
    elif near_support or preferred_entry_type in {"pullback", "deeper_pullback"}:
        trade_shape = "pullback_candidate" if not weak_structure else "support_retest_setup"
    elif breakout_point and trend_state in {"uptrend", "pullback_in_uptrend"}:
        trade_shape = "breakout_candidate"
    else:
        trade_shape = "no_clear_setup"

    if final_action == "BUY" and near_support and not near_resistance and not entry_requires_confirmation and rr1 >= config.min_reward_risk_for_buy:
        enter_now = "yes"
    elif weak_structure or final_action == "AVOID":
        enter_now = "no"
    elif entry_requires_confirmation or near_resistance:
        enter_now = "only_on_confirmation"
    else:
        enter_now = "no"

    if trade_shape == "breakout_candidate":
        execution_bias = "breakout_preferred"
    elif trade_shape in {"pullback_candidate", "support_retest_setup"}:
        execution_bias = "pullback_preferred"
    elif trade_shape == "structure_repair_needed":
        execution_bias = "wait_for_confirmation"
    else:
        execution_bias = "avoid_chasing" if near_resistance else "wait_for_confirmation"

    if enter_now == "yes":
        enter_now_reason = "Current price sits near actionable support with acceptable trade geometry."
    elif enter_now == "only_on_confirmation":
        enter_now_reason = (
            f"Current price is only acceptable if confirmation appears. {confirmation_trigger}".strip()
        )
    else:
        if weak_structure:
            enter_now_reason = "Structure is still damaged enough that the setup should not be entered immediately."
        elif near_resistance:
            enter_now_reason = "Price is too close to resistance to justify chasing without a clean breakout."
        else:
            enter_now_reason = "A better execution point is likely on a pullback or after stronger confirmation."

    breakout_reason = None
    if breakout_point:
        breakout_reason = (
            "This is the nearest resistance area that would need to be reclaimed or cleared to improve the setup."
        )

    pullback_reason = None
    if pullback_entry_zone:
        pullback_reason = (
            "This is the first meaningful support shelf and the most natural pullback area for improved risk/reward."
        )

    deeper_pullback_reason = None
    if deeper_pullback_zone:
        deeper_pullback_reason = (
            "This is the stronger lower support area if the first pullback zone fails to hold."
        )

    summary_parts: list[str] = []
    if location == "near_resistance":
        summary_parts.append("Price is near the upper end of the active range")
    elif location == "near_support":
        summary_parts.append("Price is sitting closer to active support than resistance")
    elif location == "mid_range":
        summary_parts.append("Price is trading in the middle of the active range")
    elif location == "above_breakout":
        summary_parts.append("Price is already above the first breakout zone")
    elif location == "below_support":
        summary_parts.append("Price is below the first support zone")

    if breakout_point and trade_shape in {"breakout_candidate", "breakout_or_pullback", "extended_near_resistance"}:
        summary_parts.append(f"Better breakout execution would be above {breakout_point['display']}")
    if pullback_entry_zone and trade_shape in {"pullback_candidate", "breakout_or_pullback", "support_retest_setup", "structure_repair_needed"}:
        summary_parts.append(f"Preferred pullback watching area is {pullback_entry_zone['display']}")
    if deeper_pullback_zone:
        summary_parts.append(f"deeper support sits near {deeper_pullback_zone['display']}")

    if weak_structure:
        summary_parts.append("so the chart still needs structure repair before it becomes a clean execution setup")
    elif weak_reversal or entry_requires_confirmation:
        summary_parts.append("and confirmation is still needed before treating it as an active swing entry")

    chart_execution_summary = ". ".join(part.rstrip(".") for part in summary_parts if part).strip()
    if chart_execution_summary:
        chart_execution_summary += "."
    else:
        chart_execution_summary = "The chart does not currently offer a clean swing execution pattern."

    return {
        "trade_shape": trade_shape,
        "enter_now": enter_now,
        "enter_now_reason": enter_now_reason,
        "breakout_point": breakout_point,
        "breakout_reason": breakout_reason,
        "pullback_entry_zone": pullback_entry_zone,
        "pullback_reason": pullback_reason,
        "deeper_pullback_zone": deeper_pullback_zone,
        "deeper_pullback_reason": deeper_pullback_reason,
        "current_price_location": location,
        "execution_bias": execution_bias,
        "chart_execution_summary": chart_execution_summary,
    }
