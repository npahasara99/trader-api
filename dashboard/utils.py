"""Utility helpers for the Streamlit reporting dashboard."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

try:
    from app.live_plan_consistency import enrich_live_plan_consistency_df
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.live_plan_consistency import enrich_live_plan_consistency_df


WATCH_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ACTIONABILITY_ORDER = {"ready_soon": 0, "monitor": 1, "background": 2}
PLAN_FRESHNESS_ORDER = {
    "fresh": 0,
    "live_but_extended": 1,
    "partially_stale": 2,
    "stale_for_live_price": 3,
    "invalidated": 4,
}


def safe_json(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def sort_watchlist_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["watch_priority_sort"] = out["watch_priority"].map(WATCH_PRIORITY_ORDER).fillna(3)
    out["actionability_sort"] = out["actionability_label"].map(ACTIONABILITY_ORDER).fillna(3)
    plan_freshness = out["plan_freshness_status"] if "plan_freshness_status" in out.columns else pd.Series([None] * len(out), index=out.index)
    live_alignment = out["live_vs_plan_alignment"] if "live_vs_plan_alignment" in out.columns else pd.Series([None] * len(out), index=out.index)
    replan_needed = out["replan_needed"] if "replan_needed" in out.columns else pd.Series([False] * len(out), index=out.index)
    watchlist_tier = out["watchlist_tier"] if "watchlist_tier" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["plan_freshness_sort"] = plan_freshness.map(PLAN_FRESHNESS_ORDER).fillna(5)
    out["actionability_score"] = pd.to_numeric(out["actionability_score"], errors="coerce")
    distance_to_entry = out["distance_to_entry_pct"] if "distance_to_entry_pct" in out.columns else pd.Series([None] * len(out), index=out.index)
    distance_to_stop = out["distance_to_stop_pct"] if "distance_to_stop_pct" in out.columns else pd.Series([None] * len(out), index=out.index)
    live_available = out["live_price_available"] if "live_price_available" in out.columns else pd.Series([False] * len(out), index=out.index)
    out["distance_to_entry_abs"] = pd.to_numeric(distance_to_entry, errors="coerce").abs()
    out["distance_to_stop_pct_sort"] = pd.to_numeric(distance_to_stop, errors="coerce")
    out["live_price_available_sort"] = live_available.fillna(False).astype(int)

    proximity_bonus = (12.0 - out["distance_to_entry_abs"].clip(upper=12.0)).fillna(0.0)
    extension_penalty = pd.to_numeric(distance_to_entry, errors="coerce").clip(lower=0).fillna(0.0)
    stop_pressure_penalty = out["distance_to_stop_pct_sort"].apply(
        lambda value: 10.0 if pd.notna(value) and value <= 1.5 else (4.0 if pd.notna(value) and value <= 4.0 else 0.0)
    )
    freshness_bonus = plan_freshness.map(
        {
            "fresh": 9.0,
            "live_but_extended": 4.5,
            "partially_stale": -4.0,
            "stale_for_live_price": -14.0,
            "invalidated": -22.0,
        }
    ).fillna(0.0)
    alignment_penalty = live_alignment.map(
        {
            "continuation_extended": 1.0,
            "entry_missed": 8.0,
            "near_invalidation": 12.0,
            "target_already_hit": 13.0,
            "rebound_already_moved": 11.0,
            "needs_refresh": 18.0,
        }
    ).fillna(0.0)
    replan_penalty = replan_needed.fillna(False).astype(int) * 8.0
    tier_bonus = watchlist_tier.map({"primary": 3.0, "secondary": 1.0}).fillna(0.0)
    live_bonus = out["live_price_available_sort"] * 2.5
    out["active_rank_score"] = (
        out["actionability_score"].fillna(0.0) * 100.0
        + proximity_bonus
        + freshness_bonus
        + tier_bonus
        + live_bonus
        - extension_penalty
        - stop_pressure_penalty
        - alignment_penalty
        - replan_penalty
    )

    out = out.sort_values(
        by=["plan_freshness_sort", "actionability_sort", "active_rank_score", "watch_priority_sort", "updated_at", "ticker"],
        ascending=[True, True, False, True, False, True],
        na_position="last",
    )
    return out.drop(
        columns=[
            "watch_priority_sort",
            "actionability_sort",
            "plan_freshness_sort",
            "distance_to_entry_abs",
            "distance_to_stop_pct_sort",
            "live_price_available_sort",
        ],
        errors="ignore",
    )

def filter_watchlist_df(
    df: pd.DataFrame,
    *,
    final_action: str,
    watchlist_tier: str,
    actionability_label: str,
    ticker_search: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if final_action != "All":
        out = out[out["final_action"] == final_action]
    if watchlist_tier != "All":
        out = out[out["watchlist_tier"] == watchlist_tier]
    if actionability_label != "All":
        out = out[out["actionability_label"] == actionability_label]
    if ticker_search.strip():
        query = ticker_search.strip().upper()
        out = out[out["ticker"].astype(str).str.upper().str.contains(query, na=False)]
    return out


def format_ts(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def format_short_date(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def format_price(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def format_pct(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return str(value)

def first_non_empty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def parse_ticker_text(raw: str) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        ticker = part.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def format_runner_plan_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [
        "ticker",
        "final_action",
        "watchlist_tier",
        "watch_priority",
        "actionability_soon",
        "swing_trade_suitability",
        "trend_state",
        "preferred_entry",
        "stop_loss",
        "take_profit_1",
        "max_hold_date",
    ]
    available = [column for column in keep if column in df.columns]
    out = df[available].copy()
    if "actionability_soon" in out.columns:
        out["actionability_label"] = out["actionability_soon"].apply(
            lambda value: (value or {}).get("actionability_label") if isinstance(value, dict) else None
        )
        out = out.drop(columns=["actionability_soon"])
    if "swing_trade_suitability" in out.columns:
        out["suitability_label"] = out["swing_trade_suitability"].apply(
            lambda value: (value or {}).get("suitability_label") if isinstance(value, dict) else None
        )
        out = out.drop(columns=["swing_trade_suitability"])
    ordered_columns = [
        "ticker",
        "final_action",
        "watchlist_tier",
        "watch_priority",
        "actionability_label",
        "suitability_label",
        "trend_state",
        "preferred_entry",
        "stop_loss",
        "take_profit_1",
        "max_hold_date",
    ]
    ordered_columns = [column for column in ordered_columns if column in out.columns]
    out = out[ordered_columns]
    for column in ["preferred_entry", "stop_loss", "take_profit_1"]:
        if column in out.columns:
            out[column] = out[column].apply(format_price)
    if "max_hold_date" in out.columns:
        out["max_hold_date"] = out["max_hold_date"].apply(format_short_date)
    return out



def _safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _pct_from_level(price, level):
    price_val = _safe_float(price)
    level_val = _safe_float(level)
    if price_val is None or level_val is None or level_val == 0:
        return None
    return round(((price_val - level_val) / level_val) * 100.0, 2)


def _pct_to_target(price, target):
    price_val = _safe_float(price)
    target_val = _safe_float(target)
    if price_val is None or target_val is None or price_val == 0:
        return None
    return round(((target_val - price_val) / price_val) * 100.0, 2)


def build_active_market_view(snapshot_df: pd.DataFrame, live_quotes_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df

    out = snapshot_df.copy()
    out["snapshot_price"] = out.get("current_price")
    out["snapshot_price_asof"] = out.get("current_price_asof")
    out["live_price"] = None
    out["live_price_asof"] = None
    out["live_price_available"] = False
    out["live_quote_status"] = "unavailable"
    out["live_price_source"] = "unavailable"

    if live_quotes_df is not None and not live_quotes_df.empty:
        merged = out.merge(live_quotes_df, on="ticker", how="left", suffixes=("", "_live"))
        out = merged
        if "live_price" not in out.columns and "live_price_live" in out.columns:
            out["live_price"] = out["live_price_live"]
        if "live_price_asof" not in out.columns and "live_price_asof_live" in out.columns:
            out["live_price_asof"] = out["live_price_asof_live"]
        if "available" in out.columns:
            out["live_price_available"] = out["available"].fillna(False)
        if "status" in out.columns:
            out["live_quote_status"] = out["status"].fillna("unavailable")
        if "price_source" in out.columns:
            out["live_price_source"] = out["price_source"].fillna("unavailable")

    out["distance_to_entry_pct"] = out.apply(lambda row: _pct_from_level(row.get("live_price"), row.get("preferred_entry")), axis=1)
    out["distance_to_stop_pct"] = out.apply(lambda row: _pct_from_level(row.get("live_price"), row.get("stop_loss")), axis=1)
    out["distance_to_tp1_pct"] = out.apply(lambda row: _pct_to_target(row.get("live_price"), row.get("take_profit_1")), axis=1)
    return enrich_live_plan_consistency_df(out)
