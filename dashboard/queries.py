"""Cached read-only queries for the Streamlit reporting dashboard."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

try:
    from .db import get_engine
except ImportError:
    from db import get_engine


LATEST_RUN_SQL = """
with latest_run as (
    select *
    from public.scan_runs
    order by created_at desc
    limit 1
)
select
    lr.id,
    lr.created_at,
    lr.workflow_type,
    lr.market_regime,
    lr.top_scan,
    lr.top_plan,
    lr.pre_scan_shortlist,
    lr.pre_scanned_count,
    lr.pre_scan_shortlist_count,
    lr.selected_count,
    lr.rows_logged,
    lr.selection_message,
    coalesce(sum(case when tr.actionability_label = 'ready_soon' then 1 else 0 end), 0) as ready_soon_count,
    coalesce(sum(case when tr.actionability_label = 'monitor' then 1 else 0 end), 0) as monitor_count,
    coalesce(sum(case when tr.actionability_label = 'background' then 1 else 0 end), 0) as background_count,
    coalesce(sum(case when tr.watchlist_tier = 'primary' then 1 else 0 end), 0) as primary_watchlist_count,
    coalesce(sum(case when tr.watchlist_tier = 'secondary' then 1 else 0 end), 0) as secondary_watchlist_count
from latest_run lr
left join public.scan_ticker_results tr
    on tr.run_id = lr.id
group by
    lr.id,
    lr.created_at,
    lr.workflow_type,
    lr.market_regime,
    lr.top_scan,
    lr.top_plan,
    lr.pre_scan_shortlist,
    lr.pre_scanned_count,
    lr.pre_scan_shortlist_count,
    lr.selected_count,
    lr.rows_logged,
    lr.selection_message
"""


RUN_HISTORY_SQL = """
select
    id,
    created_at,
    workflow_type,
    market_regime,
    pre_scanned_count,
    pre_scan_shortlist_count,
    selected_count,
    rows_logged,
    selection_message
from public.scan_runs
order by created_at desc
limit %(limit)s
"""


LATEST_SNAPSHOTS_SQL = """
select
    ticker,
    updated_at,
    source_run_id,
    final_action,
    watchlist_tier,
    watch_priority,
    actionability_label,
    actionability_score,
    suitability_label,
    suitability_score,
    trend_state,
    preferred_entry,
    stop_loss,
    take_profit_1,
    max_hold_date,
    short_summary,
    raw_result_json
from public.watchlist_snapshots
"""


RUN_RESULTS_SQL = """
select
    id,
    created_at,
    run_id,
    ticker,
    rank,
    final_action,
    quant_action,
    llm_action,
    watchlist_tier,
    watch_priority,
    actionability_label,
    actionability_score,
    suitability_label,
    suitability_score,
    trend_state,
    preferred_entry,
    stop_loss,
    take_profit_1,
    max_hold_date,
    pre_scan_score,
    scanner_rank_score,
    immediate_rank_score,
    watchlist_rank_score,
    sector_relative_strength,
    expected_return,
    prob_tp,
    prob_sl,
    chart_execution_view_json,
    what_to_watch_json,
    actionability_soon_json,
    raw_result_json
from public.scan_ticker_results
where run_id = %(run_id)s
order by rank asc, ticker asc
"""


LATEST_TICKER_RESULT_SQL = """
select
    created_at,
    run_id,
    ticker,
    rank,
    final_action,
    quant_action,
    llm_action,
    watchlist_tier,
    watch_priority,
    actionability_label,
    actionability_score,
    suitability_label,
    suitability_score,
    trend_state,
    preferred_entry,
    stop_loss,
    take_profit_1,
    max_hold_date,
    chart_execution_view_json,
    what_to_watch_json,
    actionability_soon_json,
    raw_result_json
from public.scan_ticker_results
where ticker = %(ticker)s
order by created_at desc
limit 1
"""


RUN_TICKER_RESULT_SQL = """
select
    created_at,
    run_id,
    ticker,
    rank,
    final_action,
    quant_action,
    llm_action,
    watchlist_tier,
    watch_priority,
    actionability_label,
    actionability_score,
    suitability_label,
    suitability_score,
    trend_state,
    preferred_entry,
    stop_loss,
    take_profit_1,
    max_hold_date,
    chart_execution_view_json,
    what_to_watch_json,
    actionability_soon_json,
    raw_result_json
from public.scan_ticker_results
where run_id = %(run_id)s and ticker = %(ticker)s
order by created_at desc
limit 1
"""


TOP_WATCH_SQL = """
select
    ticker,
    final_action,
    watchlist_tier,
    watch_priority,
    actionability_label,
    actionability_score,
    suitability_label,
    suitability_score,
    trend_state,
    preferred_entry,
    stop_loss,
    take_profit_1,
    max_hold_date,
    short_summary,
    raw_result_json
from public.watchlist_snapshots
where final_action in ('WAIT', 'BUY')
order by
    case actionability_label
        when 'ready_soon' then 0
        when 'monitor' then 1
        when 'background' then 2
        else 3
    end asc,
    actionability_score desc nulls last,
    case watchlist_tier
        when 'primary' then 0
        when 'secondary' then 1
        else 2
    end asc,
    suitability_score desc nulls last,
    ticker asc
limit %(limit)s
"""


def _normalize_json_column(value):
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_df_json(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].apply(_normalize_json_column)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_run_summary() -> pd.DataFrame:
    return pd.read_sql_query(LATEST_RUN_SQL, get_engine())


@st.cache_data(ttl=60, show_spinner=False)
def fetch_run_history(limit: int = 25) -> pd.DataFrame:
    return pd.read_sql_query(RUN_HISTORY_SQL, get_engine(), params={"limit": limit})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_snapshots() -> pd.DataFrame:
    df = pd.read_sql_query(LATEST_SNAPSHOTS_SQL, get_engine())
    return _normalize_df_json(df, ["raw_result_json"])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_run_results(run_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(RUN_RESULTS_SQL, get_engine(), params={"run_id": run_id})
    return _normalize_df_json(
        df,
        ["chart_execution_view_json", "what_to_watch_json", "actionability_soon_json", "raw_result_json"],
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_ticker_result(ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(LATEST_TICKER_RESULT_SQL, get_engine(), params={"ticker": ticker})
    return _normalize_df_json(
        df,
        ["chart_execution_view_json", "what_to_watch_json", "actionability_soon_json", "raw_result_json"],
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_run_ticker_result(run_id: str, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(RUN_TICKER_RESULT_SQL, get_engine(), params={"run_id": run_id, "ticker": ticker})
    return _normalize_df_json(
        df,
        ["chart_execution_view_json", "what_to_watch_json", "actionability_soon_json", "raw_result_json"],
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_top_watch(limit: int = 5) -> pd.DataFrame:
    df = pd.read_sql_query(TOP_WATCH_SQL, get_engine(), params={"limit": limit})
    return _normalize_df_json(df, ["raw_result_json"])
