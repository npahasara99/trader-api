from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import PlanningConfig


def _price_decimals(price: float | None) -> int:
    p = abs(float(price or 0.0))
    if p >= 1000:
        return 2
    if p >= 1:
        return 2
    if p >= 0.1:
        return 3
    return 4


def _fmt_price(price: float | None) -> str | None:
    if price is None:
        return None
    decimals = _price_decimals(price)
    return f"{float(price):.{decimals}f}"


def _add_trading_days(start_dt: datetime, trading_days: int) -> datetime:
    cur = start_dt
    left = max(0, int(trading_days))
    while left > 0:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:
            left -= 1
    return cur


def build_zone_display(zone: dict | None, *, current_price: float | None = None, zone_label: str = "Zone") -> dict:
    if not zone:
        return {
            "display": None,
            "midpoint": None,
            "width_pct": None,
            "note": None,
            "summary_line": None,
        }

    lower = float(zone.get("lower"))
    upper = float(zone.get("upper"))
    midpoint = (lower + upper) / 2.0
    ref = max(float(current_price or midpoint), 1e-9)
    width_pct = ((upper - lower) / ref) * 100.0
    tags = list(zone.get("source_tags") or [])
    tag_text = ", ".join(tags) if tags else "structure support"
    display = f"{_fmt_price(lower)} to {_fmt_price(upper)}"
    note = f"{tag_text} zone."
    summary = f"{zone_label}: {display} ({tag_text})"
    return {
        "display": display,
        "midpoint": round(midpoint, _price_decimals(midpoint)),
        "width_pct": round(width_pct, 3),
        "note": note,
        "summary_line": summary,
    }


def _wait_type(row, config: PlanningConfig) -> str:
    trend_state = str(getattr(row, "trend_state", None) or "")
    requires_confirmation = bool(getattr(row, "entry_requires_confirmation", False))
    entry_distance_pct = abs(float(getattr(row, "entry_distance_from_current_price_pct", 0.0) or 0.0))

    if trend_state == "weak_breakdown_risk":
        return "WAIT_STRUCTURE_REPAIR"
    if requires_confirmation:
        return "WAIT_CONFIRMATION"
    if entry_distance_pct >= 2.5:
        return "WAIT_BETTER_ENTRY"
    return "WAIT_CONFIRMATION"


def _monitor_window_days(row, config: PlanningConfig) -> int:
    wait_type = _wait_type(row, config)
    regime = str(getattr(row, "market_regime", None) or "neutral")
    earnings = getattr(row, "earnings", None) or {}

    if wait_type == "WAIT_CONFIRMATION":
        days = config.wait_monitor_days_pullback
    elif wait_type == "WAIT_STRUCTURE_REPAIR":
        days = config.wait_monitor_days_structure_repair
    else:
        days = config.wait_monitor_days_other

    if regime == "risk_off":
        days -= 1

    days_to_earnings = earnings.get("days_to_earnings")
    if days_to_earnings is not None:
        try:
            dte = int(days_to_earnings)
            if dte <= days:
                days = min(days, max(config.wait_monitor_days_min, dte - 1))
        except Exception:
            pass

    return max(config.wait_monitor_days_min, min(config.wait_monitor_days_max, days))


def _watch_priority(row, config: PlanningConfig) -> str:
    composite = float(getattr(row, "composite_score", 0.0) or 0.0)
    rs_score = float(getattr(row, "relative_strength_score", 0.0) or 0.0)
    trend_state = str(getattr(row, "trend_state", None) or "")

    if trend_state == "pullback_in_uptrend" and composite >= config.wait_watch_priority_high_composite and rs_score >= 6.0:
        return "high"
    if composite >= config.wait_watch_priority_medium_composite and rs_score >= 5.2:
        return "medium"
    return "low"


def _days_to_trigger_estimate(row) -> float | None:
    atr = float(getattr(row, "atr", 0.0) or 0.0)
    current_price = float(getattr(row, "current_price", 0.0) or 0.0)
    if atr <= 0 or current_price <= 0:
        return None

    moving_averages = getattr(row, "moving_averages", None) or {}
    resistance_zone_1 = getattr(row, "resistance_zone_1", None)
    trigger_ref = None
    if moving_averages.get("ema20") is not None:
        trigger_ref = float(moving_averages["ema20"])
    elif resistance_zone_1 and resistance_zone_1.get("lower") is not None:
        trigger_ref = float(resistance_zone_1["lower"])
    if trigger_ref is None:
        return None
    return round(abs(trigger_ref - current_price) / max(atr, 1e-9), 2)


def build_wait_monitoring_plan(row, *, config: PlanningConfig) -> dict | None:
    if str(getattr(row, "final_action", None) or "").upper() != "WAIT":
        return None

    current_price = float(getattr(row, "current_price", 0.0) or 0.0)
    support1 = build_zone_display(getattr(row, "support_zone_1", None), current_price=current_price, zone_label="Support Zone 1")
    support2 = build_zone_display(getattr(row, "support_zone_2", None), current_price=current_price, zone_label="Support Zone 2")
    resistance1 = build_zone_display(getattr(row, "resistance_zone_1", None), current_price=current_price, zone_label="Resistance Zone 1")
    resistance2 = build_zone_display(getattr(row, "resistance_zone_2", None), current_price=current_price, zone_label="Resistance Zone 2")

    wait_type = _wait_type(row, config)
    monitor_window_days = _monitor_window_days(row, config)
    now_dt = datetime.now(timezone.utc)
    monitor_until_dt = _add_trading_days(now_dt, monitor_window_days)
    stale_after_dt = monitor_until_dt
    days_to_trigger_estimate = _days_to_trigger_estimate(row)

    moving_averages = getattr(row, "moving_averages", None) or {}
    ema20 = moving_averages.get("ema20")
    support_zone_summary = [x for x in [support1["summary_line"], support2["summary_line"]] if x]
    resistance_zone_summary = [x for x in [resistance1["summary_line"], resistance2["summary_line"]] if x]

    upgrade_triggers: list[str] = []
    if ema20 is not None:
        upgrade_triggers.append(f"Daily close back above EMA20 near {_fmt_price(float(ema20))}")
    if support1["display"] is not None:
        upgrade_triggers.append(f"Support Zone 1 ({support1['display']}) holds with stronger reversal volume")
    if resistance1["display"] is not None:
        upgrade_triggers.append(f"Reclaim of Resistance Zone 1 ({resistance1['display']})")
    if getattr(row, "relative_strength_score", None) is not None:
        upgrade_triggers.append("Relative strength improves versus SPY and QQQ")
    upgrade_triggers = upgrade_triggers[:4]

    failure_triggers: list[str] = []
    if support2["display"] is not None:
        failure_triggers.append(f"Close below Support Zone 2 ({support2['display']}) with weak follow-through")
    elif support1["display"] is not None:
        failure_triggers.append(f"Loss of Support Zone 1 ({support1['display']}) without bounce confirmation")
    failure_triggers.append(f"No confirmation by {stale_after_dt.date().isoformat()}")
    failure_triggers.append("Relative strength weakens materially versus SPY and QQQ")
    earnings = getattr(row, "earnings", None) or {}
    if earnings.get("days_to_earnings") is not None:
        failure_triggers.append("Earnings becomes too close for a normal swing entry")
    failure_triggers = failure_triggers[:4]

    next_check_focus: list[str] = []
    if support1["display"] is not None:
        next_check_focus.append("Support hold")
    next_check_focus.append("Reversal volume quality")
    if ema20 is not None:
        next_check_focus.append("EMA20 reclaim")
    next_check_focus.append("Relative strength improvement")
    next_check_focus = next_check_focus[:4]

    if support1["display"] is not None:
        support1_note = f"{support1['summary_line']}. First area to hold for a constructive bounce."
    else:
        support1_note = None
    if support2["display"] is not None:
        support2_note = f"{support2['summary_line']}. Deeper pullback area and stronger invalidation reference."
    else:
        support2_note = None

    summary_parts = [
        f"Watch this setup through {monitor_until_dt.date().isoformat()}.",
    ]
    if support1["display"] is not None:
        summary_parts.append(f"Support Zone 1 is {support1['display']}")
    if support2["display"] is not None:
        summary_parts.append(f"and Support Zone 2 is {support2['display']}.")
    if upgrade_triggers:
        summary_parts.append(f"Upgrade the setup if {upgrade_triggers[0].lower()}.")
    if failure_triggers:
        summary_parts.append(f"Expire the setup if {failure_triggers[0].lower()} or no confirmation appears by the monitoring deadline.")

    return {
        "wait_type": wait_type,
        "monitor_window_days": monitor_window_days,
        "monitor_until_date": monitor_until_dt,
        "stale_after_date": stale_after_dt,
        "watch_priority": _watch_priority(row, config),
        "days_to_trigger_estimate": days_to_trigger_estimate,
        "support_zone_1_display": support1["display"],
        "support_zone_2_display": support2["display"],
        "resistance_zone_1_display": resistance1["display"],
        "resistance_zone_2_display": resistance2["display"],
        "support_zone_1_midpoint": support1["midpoint"],
        "support_zone_2_midpoint": support2["midpoint"],
        "support_zone_1_width_pct": support1["width_pct"],
        "support_zone_2_width_pct": support2["width_pct"],
        "support_zone_1_note": support1_note,
        "support_zone_2_note": support2_note,
        "support_zone_summary": support_zone_summary,
        "resistance_zone_summary": resistance_zone_summary,
        "upgrade_triggers": upgrade_triggers,
        "failure_triggers": failure_triggers,
        "next_check_focus": next_check_focus,
        "setup_monitoring_summary": " ".join(summary_parts).replace(" .", "."),
    }
