"""Utility helpers for the Streamlit reporting dashboard."""

from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


WATCH_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


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
    out["actionability_score"] = pd.to_numeric(out["actionability_score"], errors="coerce")
    out = out.sort_values(
        by=["actionability_score", "watch_priority_sort", "ticker"],
        ascending=[False, True, True],
        na_position="last",
    )
    return out.drop(columns=["watch_priority_sort"])


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


def first_non_empty(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None

