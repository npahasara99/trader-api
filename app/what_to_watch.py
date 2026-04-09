"""Trader-readable watch statements derived from existing structured zones."""

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


def _payload(zone: dict | None, *, current_price: float, label: str) -> dict | None:
    if not isinstance(zone, dict):
        return None
    lower = _safe_float(zone.get("lower"), -1.0)
    upper = _safe_float(zone.get("upper"), -1.0)
    if lower <= 0 or upper <= 0 or upper < lower:
        return None
    display = str(zone.get("display") or "")
    if not display:
        display = build_zone_display(zone, current_price=current_price, zone_label=label)["display"]
    if not display:
        return None
    return {
        "lower": lower,
        "upper": upper,
        "display": display,
    }


def _overlap_ratio(zone_a: dict | None, zone_b: dict | None) -> float:
    if not zone_a or not zone_b:
        return 0.0
    lower = max(_safe_float(zone_a.get("lower")), _safe_float(zone_b.get("lower")))
    upper = min(_safe_float(zone_a.get("upper")), _safe_float(zone_b.get("upper")))
    overlap = max(0.0, upper - lower)
    width_a = max(0.0, _safe_float(zone_a.get("upper")) - _safe_float(zone_a.get("lower")))
    width_b = max(0.0, _safe_float(zone_b.get("upper")) - _safe_float(zone_b.get("lower")))
    base = min(width_a, width_b)
    if base <= 0:
        return 0.0
    return overlap / base


def build_what_to_watch(row, *, config: PlanningConfig) -> dict | None:
    """Translate structured execution levels into practical daily watch lines."""

    final_action = str(getattr(row, "final_action", None) or "").upper()
    if final_action not in {"BUY", "WAIT"}:
        return None

    execution_view = getattr(row, "chart_execution_view", None) or {}
    trade_shape = str(execution_view.get("trade_shape") or "")
    execution_bias = str(execution_view.get("execution_bias") or "")
    current_price = _safe_float(getattr(row, "current_price", None), 0.0)

    bullish_hold_zone = _payload(
        execution_view.get("current_execution_anchor")
        or execution_view.get("pullback_entry_zone")
        or getattr(row, "support_zone_1", None),
        current_price=current_price,
        label="Bullish Hold Zone",
    )
    deeper_reset_trigger_zone = bullish_hold_zone

    deeper_reset_target_zone = _payload(
        execution_view.get("deeper_pullback_zone") or getattr(row, "support_zone_2", None),
        current_price=current_price,
        label="Deeper Reset Target",
    )
    if _overlap_ratio(bullish_hold_zone, deeper_reset_target_zone) > config.execution_zone_overlap_max_pct:
        deeper_reset_target_zone = None

    continuation_trigger_zone = _payload(
        execution_view.get("breakout_point"),
        current_price=current_price,
        label="Continuation Trigger",
    )
    breakout_type = str(execution_view.get("breakout_point_type") or "")
    if breakout_type == "none":
        continuation_trigger_zone = None

    if trade_shape == "structure_repair_needed":
        bullish_hold_reason = (
            f"If it holds {bullish_hold_zone['display']}, the repair attempt stays alive."
            if bullish_hold_zone
            else "Holding the active repair area keeps the repair attempt alive."
        )
        if deeper_reset_trigger_zone and deeper_reset_target_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, the structure stays weak and may reset toward {deeper_reset_target_zone['display']}."
            )
        elif deeper_reset_trigger_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, the structure stays weak and may reset lower."
            )
        else:
            deeper_reset_reason = "If the repair area fails, the structure likely remains weak."
        continuation_reason = (
            f"Repair improves if price reclaims {continuation_trigger_zone['display']}."
            if continuation_trigger_zone
            else "Repair still lacks a clean continuation trigger."
        )
    elif trade_shape in {"continuation_pullback_preferred", "near_resistance_wait", "breakout_or_pullback"} or execution_bias in {"pullback_preferred", "avoid_chasing"}:
        bullish_hold_reason = (
            f"If it holds {bullish_hold_zone['display']}, the pullback remains constructive."
            if bullish_hold_zone
            else "Holding the active support shelf keeps the pullback constructive."
        )
        if deeper_reset_trigger_zone and deeper_reset_target_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, look for a deeper reset toward {deeper_reset_target_zone['display']}."
            )
        elif deeper_reset_trigger_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, expect the setup to need a deeper reset."
            )
        else:
            deeper_reset_reason = "If the active support area fails, expect the setup to weaken."
        continuation_reason = (
            f"Continuation strengthens if price reclaims {continuation_trigger_zone['display']}."
            if continuation_trigger_zone
            else "Continuation still lacks a clean current trigger."
        )
    else:
        bullish_hold_reason = (
            f"If it holds {bullish_hold_zone['display']}, the breakout stays healthy."
            if bullish_hold_zone
            else "Holding the active support area keeps the breakout healthy."
        )
        if deeper_reset_trigger_zone and deeper_reset_target_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, expect a deeper reset toward {deeper_reset_target_zone['display']}."
            )
        elif deeper_reset_trigger_zone:
            deeper_reset_reason = (
                f"If it loses {deeper_reset_trigger_zone['display']}, expect the setup to weaken and reset lower."
            )
        else:
            deeper_reset_reason = "If the active support area fails, expect the setup to weaken."
        continuation_reason = (
            f"If it pushes and holds above {continuation_trigger_zone['display']}, continuation remains alive."
            if continuation_trigger_zone
            else "There is no clean continuation trigger right now."
        )

    watch_summary = [bullish_hold_reason, deeper_reset_reason]
    if continuation_trigger_zone or continuation_reason.endswith("right now."):
        watch_summary.append(continuation_reason)
    watch_summary = [line for line in watch_summary if line][:3]
    watch_summary_short = "; ".join(line.rstrip(".") for line in watch_summary) + ("." if watch_summary else "")

    return {
        "bullish_hold_zone": bullish_hold_zone,
        "bullish_hold_reason": bullish_hold_reason,
        "deeper_reset_trigger_zone": deeper_reset_trigger_zone,
        "deeper_reset_reason": deeper_reset_reason,
        "deeper_reset_target_zone": deeper_reset_target_zone,
        "continuation_trigger_zone": continuation_trigger_zone,
        "continuation_reason": continuation_reason,
        "watch_summary": watch_summary,
        "watch_summary_short": watch_summary_short,
    }
