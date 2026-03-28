from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import PlanningConfig


def _zone_mid(zone: dict | None) -> float | None:
    if not zone:
        return None
    lower = zone.get("lower")
    upper = zone.get("upper")
    if lower is None or upper is None:
        return None
    return (float(lower) + float(upper)) / 2.0


def build_stop_loss(
    *,
    preferred_entry: float,
    support_zone_1: dict | None,
    support_zone_2: dict | None,
    recent_swing_low: float | None,
    atr: float,
    current_price: float,
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(current_price * 0.01, 0.01))
    buffer = atr * config.stop_buffer_atr_mult
    max_valid_stop = preferred_entry - max(atr * 0.35, current_price * 0.0035)

    candidates: list[tuple[float, str]] = []
    if support_zone_1:
        level = float(support_zone_1["lower"]) - buffer - current_price * config.stop_below_zone_buffer_pct
        if level < max_valid_stop:
            candidates.append((level, "below support_zone_1 and ATR buffer"))
    if support_zone_2:
        level = float(support_zone_2["lower"]) - buffer * 0.8
        if level < max_valid_stop:
            candidates.append((level, "below support_zone_2 and ATR buffer"))
    if recent_swing_low is not None:
        level = float(recent_swing_low) - buffer
        if level < max_valid_stop:
            candidates.append((level, "below recent swing low and ATR buffer"))

    if not candidates:
        candidates.append((preferred_entry - atr * 1.6, "fallback ATR invalidation"))

    stop_loss, basis = min(candidates, key=lambda item: item[0])
    if stop_loss >= preferred_entry:
        stop_loss = preferred_entry - atr * 1.6
        basis = "fallback ATR invalidation"
    stop_distance_pct = (preferred_entry - stop_loss) / max(preferred_entry, 1e-9) * 100.0
    stop_too_tight = (preferred_entry - stop_loss) < atr * 0.9
    return {
        "stop_loss": float(round(stop_loss, 6)),
        "stop_basis": basis,
        "stop_distance_pct": float(round(stop_distance_pct, 3)),
        "stop_too_tight_flag": bool(stop_too_tight),
    }


def build_take_profits(
    *,
    preferred_entry: float,
    stop_loss: float,
    resistance_zone_1: dict | None,
    resistance_zone_2: dict | None,
    recent_swing_high: float | None,
    atr: float,
    hold_days_hint: int,
    trend_state: str,
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(preferred_entry * 0.01, 0.01))
    risk_per_share = max(preferred_entry - stop_loss, atr * 0.6)
    reachable_move = atr * max(1.0, min(float(hold_days_hint), 20.0) * config.atr_target_window_mult)

    tp1 = _zone_mid(resistance_zone_1) or recent_swing_high or preferred_entry + max(risk_per_share * 1.2, atr * config.tp1_atr_mult)
    tp2 = _zone_mid(resistance_zone_2) or preferred_entry + max(risk_per_share * 2.0, atr * config.tp2_atr_mult)
    min_tp1 = preferred_entry + max(atr * 0.45, risk_per_share * 0.9)
    min_tp2 = preferred_entry + max(atr * 0.95, risk_per_share * 1.5)

    if tp1 <= preferred_entry:
        tp1 = min_tp1
    else:
        tp1 = max(tp1, min_tp1)
    if tp2 <= tp1:
        tp2 = max(min_tp2, tp1 + max(atr * 0.8, risk_per_share * 0.5))

    trend_bonus = 1.4 if trend_state == "uptrend" else 1.0
    tp_final = max(tp2, preferred_entry + min(reachable_move * trend_bonus, atr * 8.0))
    if tp_final <= tp2:
        tp_final = tp2 + max(atr * 0.5, risk_per_share * 0.35)

    rr1 = (tp1 - preferred_entry) / max(preferred_entry - stop_loss, 1e-9)
    rr2 = (tp2 - preferred_entry) / max(preferred_entry - stop_loss, 1e-9)
    rr_final = (tp_final - preferred_entry) / max(preferred_entry - stop_loss, 1e-9)
    optimistic_flag = (tp_final - preferred_entry) > reachable_move * 1.35 and trend_state != "uptrend"

    basis_parts = []
    if resistance_zone_1:
        basis_parts.append("tp1 near resistance_zone_1")
    else:
        basis_parts.append("tp1 via ATR/risk multiple")
    if resistance_zone_2:
        basis_parts.append("tp2 near resistance_zone_2")
    else:
        basis_parts.append("tp2 via swing projection")
    basis_parts.append("final target capped by ATR reachability and trend strength")

    return {
        "take_profit_1": float(round(tp1, 6)),
        "take_profit_2": float(round(tp2, 6)),
        "take_profit_final": float(round(tp_final, 6)),
        "tp_basis": "; ".join(basis_parts),
        "expected_reward_risk_to_tp1": float(round(rr1, 3)),
        "expected_reward_risk_to_tp2": float(round(rr2, 3)),
        "expected_reward_risk_to_final": float(round(rr_final, 3)),
        "tp_too_optimistic_flag": bool(optimistic_flag),
    }


def estimate_hold_window(
    *,
    preferred_entry: float,
    take_profit_1: float,
    atr: float,
    recent_swing_bars: int | None,
    historical_hold_days: int | None,
    config: PlanningConfig,
) -> dict:
    atr = max(float(atr or 0.0), max(preferred_entry * 0.01, 0.01))
    distance = max(0.0, take_profit_1 - preferred_entry)
    atr_days = int(round(distance / max(atr, 1e-9)))
    base = atr_days + 4
    if recent_swing_bars is not None:
        base = int(round((base * 0.6) + (recent_swing_bars * 0.4)))
    if historical_hold_days is not None:
        base = int(round((base * 0.7) + (historical_hold_days * 0.3)))

    hold_days = max(config.max_hold_days_min, min(config.max_hold_days_max, base))
    max_hold_date = datetime.now(timezone.utc) + timedelta(days=hold_days)
    return {
        "max_hold_days": int(hold_days),
        "max_hold_date": max_hold_date,
    }
