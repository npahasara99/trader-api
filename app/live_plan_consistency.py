from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .scenario_engine import evaluate_live_scenario_status


@dataclass(frozen=True)
class LivePlanConsistencyConfig:
    entry_zone_pct: float = 1.0
    continuation_extension_pct: float = 3.0
    pullback_extension_pct: float = 2.0
    hard_extension_pct: float = 6.0
    near_tp1_pct: float = 2.5
    at_tp1_pct: float = 0.5
    tp1_exceeded_pct: float = 1.5
    near_stop_pct: float = 4.0
    stop_risk_pct: float = 1.0


DEFAULT_LIVE_PLAN_CONSISTENCY_CONFIG = LivePlanConsistencyConfig()


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _pct_from_level(price: Any, level: Any) -> float | None:
    price_val = _safe_float(price)
    level_val = _safe_float(level)
    if price_val is None or level_val is None or level_val == 0:
        return None
    return round(((price_val - level_val) / level_val) * 100.0, 2)


def _pct_to_target(price: Any, target: Any) -> float | None:
    price_val = _safe_float(price)
    target_val = _safe_float(target)
    if price_val is None or target_val is None or price_val == 0:
        return None
    return round(((target_val - price_val) / price_val) * 100.0, 2)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(value: str, needles: set[str]) -> bool:
    return any(needle in value for needle in needles)


def _scenario_traits(row: Mapping[str, Any]) -> tuple[bool, bool]:
    scenario = _normalize_text(row.get("setup_scenario"))
    setup_type = _normalize_text(row.get("setup_type"))
    bias = _normalize_text(row.get("continuation_vs_reversion_bias"))
    trade_shape = _normalize_text(row.get("trade_shape"))
    combo = " ".join([scenario, setup_type, bias, trade_shape])

    continuation = bias == "continuation_favored" or _contains_any(
        combo,
        {
            "continuation",
            "breakout",
            "supported_high_range",
            "momentum_expansion",
            "constructive_pullback",
            "macro_sensitive_continuation",
            "news_driven_expansion",
            "post_breakout",
        },
    )
    rebound_or_repair = bias == "rebound_candidate" or _contains_any(
        combo,
        {
            "rebound",
            "repair",
            "range_rebound",
            "deep_rebound",
            "breakdown",
            "weak_rally",
        },
    )
    return continuation, rebound_or_repair


def evaluate_live_plan_consistency(
    row: Mapping[str, Any],
    *,
    config: LivePlanConsistencyConfig = DEFAULT_LIVE_PLAN_CONSISTENCY_CONFIG,
) -> dict[str, Any]:
    raw_payload = row.get("raw_result_json") if isinstance(row.get("raw_result_json"), dict) else {}
    chart_execution = raw_payload.get("chart_execution_view") if isinstance(raw_payload, dict) else {}
    live_price = _safe_float(row.get("live_price"))
    live_price_asof = row.get("live_price_asof")
    preferred_entry = _safe_float(row.get("preferred_entry"))
    stop_loss = _safe_float(row.get("stop_loss"))
    tp1 = _safe_float(row.get("take_profit_1"))
    tp2 = _safe_float(row.get("take_profit_2") if row.get("take_profit_2") is not None else raw_payload.get("take_profit_2"))

    distance_to_entry_pct = _pct_from_level(live_price, preferred_entry)
    distance_to_stop_pct = _pct_from_level(live_price, stop_loss)
    distance_to_tp1_pct = _pct_to_target(live_price, tp1)
    distance_to_tp2_pct = _pct_to_target(live_price, tp2)
    live_scenario = evaluate_live_scenario_status(
        {
            **raw_payload,
            "preferred_scenario": row.get("preferred_scenario") or raw_payload.get("preferred_scenario"),
            "execution_scenarios": row.get("execution_scenarios") or raw_payload.get("execution_scenarios"),
        },
        live_price,
    )

    continuation_setup, rebound_or_repair_setup = _scenario_traits(
        {
            **row,
            "setup_scenario": row.get("setup_scenario") or raw_payload.get("setup_scenario"),
            "setup_type": row.get("setup_type") or raw_payload.get("setup_type"),
            "continuation_vs_reversion_bias": row.get("continuation_vs_reversion_bias") or raw_payload.get("continuation_vs_reversion_bias"),
            "trade_shape": row.get("trade_shape") or (chart_execution or {}).get("trade_shape"),
        }
    )
    entry_extension_limit = config.continuation_extension_pct if continuation_setup else config.pullback_extension_pct

    entry_status = None
    if distance_to_entry_pct is not None:
        if abs(distance_to_entry_pct) <= config.entry_zone_pct:
            entry_status = "in_entry_zone"
        elif distance_to_entry_pct < -config.entry_zone_pct:
            entry_status = "below_entry_zone"
        elif distance_to_entry_pct <= entry_extension_limit:
            entry_status = "above_entry_zone"
        else:
            entry_status = "extended_beyond_entry"

    tp1_status = None
    if distance_to_tp1_pct is not None:
        if abs(distance_to_tp1_pct) <= config.at_tp1_pct:
            tp1_status = "at_tp1"
        elif distance_to_tp1_pct > config.near_tp1_pct:
            tp1_status = "below_tp1"
        elif distance_to_tp1_pct > config.at_tp1_pct:
            tp1_status = "near_tp1"
        elif distance_to_tp1_pct >= -config.tp1_exceeded_pct:
            tp1_status = "above_tp1"
        else:
            tp1_status = "tp1_exceeded"

    stop_status = None
    if distance_to_stop_pct is not None:
        if distance_to_stop_pct < 0:
            stop_status = "below_stop"
        elif distance_to_stop_pct <= config.stop_risk_pct:
            stop_status = "at_risk_of_invalidation"
        elif distance_to_stop_pct <= config.near_stop_pct:
            stop_status = "near_stop"
        else:
            stop_status = "far_from_stop"

    if live_price is None:
        return {
            "live_price": row.get("live_price"),
            "live_price_asof": live_price_asof,
            "distance_to_entry_pct": distance_to_entry_pct,
            "distance_to_stop_pct": distance_to_stop_pct,
            "distance_to_tp1_pct": distance_to_tp1_pct,
            "distance_to_tp2_pct": distance_to_tp2_pct,
            "entry_status": entry_status,
            "tp1_status": tp1_status,
            "stop_status": stop_status,
            "plan_freshness_status": None,
            "live_vs_plan_alignment": None,
            "replan_needed": False,
            "live_scenario_status": live_scenario["live_scenario_status"],
            "preferred_scenario_changed": live_scenario["preferred_scenario_changed"],
            "live_consistency_summary": "Live price is unavailable, so plan freshness cannot be evaluated yet.",
        }

    plan_freshness_status = "fresh"
    live_vs_plan_alignment = "aligned"
    replan_needed = False

    if stop_status == "below_stop":
        plan_freshness_status = "invalidated"
        live_vs_plan_alignment = "needs_refresh"
        replan_needed = True
    elif tp1_status == "tp1_exceeded":
        plan_freshness_status = "stale_for_live_price"
        live_vs_plan_alignment = "target_already_hit" if not rebound_or_repair_setup else "rebound_already_moved"
        replan_needed = True
    elif tp1_status == "above_tp1":
        plan_freshness_status = "partially_stale"
        live_vs_plan_alignment = "target_already_hit" if not rebound_or_repair_setup else "rebound_already_moved"
        replan_needed = True
    elif stop_status == "at_risk_of_invalidation":
        plan_freshness_status = "partially_stale"
        live_vs_plan_alignment = "near_invalidation"
    elif entry_status == "extended_beyond_entry":
        if continuation_setup and tp1_status in {"below_tp1", "near_tp1", "at_tp1", None}:
            plan_freshness_status = "live_but_extended"
            live_vs_plan_alignment = "continuation_extended"
        elif rebound_or_repair_setup:
            plan_freshness_status = "stale_for_live_price"
            live_vs_plan_alignment = "rebound_already_moved"
            replan_needed = True
        else:
            plan_freshness_status = "stale_for_live_price"
            live_vs_plan_alignment = "entry_missed"
            replan_needed = True
    elif entry_status == "above_entry_zone":
        if continuation_setup:
            plan_freshness_status = "live_but_extended"
            live_vs_plan_alignment = "continuation_extended"
        elif rebound_or_repair_setup:
            plan_freshness_status = "partially_stale"
            live_vs_plan_alignment = "rebound_already_moved"
        else:
            plan_freshness_status = "partially_stale"
            live_vs_plan_alignment = "entry_missed"
    elif entry_status == "below_entry_zone" and stop_status == "near_stop":
        plan_freshness_status = "partially_stale"
        live_vs_plan_alignment = "near_invalidation"

    if distance_to_entry_pct is not None and distance_to_entry_pct >= config.hard_extension_pct and not continuation_setup:
        plan_freshness_status = "stale_for_live_price"
        live_vs_plan_alignment = "entry_missed" if not rebound_or_repair_setup else "rebound_already_moved"
        replan_needed = True

    if live_scenario.get("replan_needed"):
        replan_needed = True
        if live_scenario.get("live_scenario_status") == "scenario_invalidated":
            plan_freshness_status = "invalidated"
            live_vs_plan_alignment = "needs_refresh"
        elif live_scenario.get("live_scenario_status") in {"tp1_hit_replan", "preferred_entry_missed"}:
            plan_freshness_status = "stale_for_live_price"
        elif live_scenario.get("live_scenario_status") == "breakout_activated":
            plan_freshness_status = "partially_stale"
            live_vs_plan_alignment = "needs_refresh"

    if plan_freshness_status == "invalidated":
        summary = "Live price is at or below the saved stop area, so the original plan is invalidated and needs a refresh."
    elif live_vs_plan_alignment == "target_already_hit":
        summary = "Live price has already exceeded TP1, so the original plan is stale for live price and should be refreshed."
    elif live_vs_plan_alignment == "rebound_already_moved":
        summary = "The rebound or repair move is already well advanced versus the saved plan, so the original setup likely needs a refresh."
    elif live_vs_plan_alignment == "continuation_extended":
        summary = "Live price is above the preferred entry area. The setup still aligns with continuation, but the original entry is now extended."
    elif live_vs_plan_alignment == "entry_missed":
        summary = "Live price is well above the preferred entry zone, so the original pullback-style entry looks missed and likely needs a replan."
    elif live_vs_plan_alignment == "near_invalidation":
        summary = "Live price is pressing the saved stop area, so the original plan is close to invalidation."
    elif entry_status == "below_entry_zone":
        summary = "Live price is still below the preferred entry area, so the saved plan remains pending rather than active."
    else:
        summary = "Live price is still near the preferred entry zone, so the original plan remains fresh."

    return {
        "live_price": row.get("live_price"),
        "live_price_asof": live_price_asof,
        "distance_to_entry_pct": distance_to_entry_pct,
        "distance_to_stop_pct": distance_to_stop_pct,
        "distance_to_tp1_pct": distance_to_tp1_pct,
        "distance_to_tp2_pct": distance_to_tp2_pct,
        "entry_status": entry_status,
        "tp1_status": tp1_status,
        "stop_status": stop_status,
        "plan_freshness_status": plan_freshness_status,
        "live_vs_plan_alignment": live_vs_plan_alignment,
        "replan_needed": bool(replan_needed),
        "live_scenario_status": live_scenario["live_scenario_status"],
        "preferred_scenario_changed": live_scenario["preferred_scenario_changed"],
        "live_consistency_summary": summary,
    }


def enrich_live_plan_consistency_df(
    df: pd.DataFrame,
    *,
    config: LivePlanConsistencyConfig = DEFAULT_LIVE_PLAN_CONSISTENCY_CONFIG,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    records = [evaluate_live_plan_consistency(row, config=config) for row in out.to_dict(orient="records")]
    consistency_df = pd.DataFrame(records, index=out.index)
    for column in consistency_df.columns:
        out[column] = consistency_df[column]
    return out
