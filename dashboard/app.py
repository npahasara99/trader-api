"""Read-only Streamlit dashboard for Supabase scan/watchlist reporting."""

from __future__ import annotations

import streamlit as st

from dashboard.queries import (
    fetch_latest_run_summary,
    fetch_latest_snapshots,
    fetch_latest_ticker_result,
    fetch_run_history,
    fetch_run_results,
    fetch_run_ticker_result,
    fetch_top_watch,
)
from dashboard.utils import (
    filter_watchlist_df,
    first_non_empty,
    format_price,
    format_ts,
    safe_json,
    sort_watchlist_table,
)


st.set_page_config(
    page_title="Trader Watch Dashboard",
    layout="wide",
)


def _load_dashboard_data():
    latest_run = fetch_latest_run_summary()
    snapshots = fetch_latest_snapshots()
    run_history = fetch_run_history()
    top_watch = fetch_top_watch()
    return latest_run, snapshots, run_history, top_watch


def _summary_from_row(row) -> str:
    raw = safe_json(row.get("raw_result_json"))
    what_to_watch = (raw or {}).get("what_to_watch") or {}
    actionability = (raw or {}).get("actionability_soon") or {}
    return first_non_empty(
        row.get("short_summary"),
        what_to_watch.get("watch_summary_short"),
        actionability.get("actionability_summary"),
        (raw or {}).get("watchlist_summary"),
    ) or "No short summary available."


def _render_json_block(title: str, payload):
    with st.expander(title, expanded=False):
        if payload in (None, "", {}):
            st.caption("No data available.")
        else:
            st.json(payload)


st.title("Trader Watch Dashboard")
st.caption("Read-only view of the latest scan, watchlist, and run history from the Supabase reporting database.")

try:
    latest_run_df, snapshots_df, run_history_df, top_watch_df = _load_dashboard_data()
except Exception as exc:
    st.error("Could not load the Supabase reporting database.")
    with st.expander("Error details", expanded=True):
        st.code(str(exc))
    st.stop()

if latest_run_df.empty:
    st.warning("No scan runs have been written to the reporting database yet.")
    st.stop()

latest_run = latest_run_df.iloc[0]
sorted_snapshots_df = sort_watchlist_table(snapshots_df)

st.subheader("Overview")
metric_cols = st.columns(8)
metric_cols[0].metric("Market Regime", str(latest_run.get("market_regime") or "-"))
metric_cols[1].metric("Selected", int(latest_run.get("selected_count") or 0))
metric_cols[2].metric("Rows Logged", int(latest_run.get("rows_logged") or 0))
metric_cols[3].metric("Ready Soon", int(latest_run.get("ready_soon_count") or 0))
metric_cols[4].metric("Monitor", int(latest_run.get("monitor_count") or 0))
metric_cols[5].metric("Background", int(latest_run.get("background_count") or 0))
metric_cols[6].metric("Primary", int(latest_run.get("primary_watchlist_count") or 0))
metric_cols[7].metric("Secondary", int(latest_run.get("secondary_watchlist_count") or 0))
st.caption(
    f"Latest run: {format_ts(latest_run.get('created_at'))} | "
    f"{latest_run.get('workflow_type') or 'workflow'} | "
    f"{latest_run.get('selection_message') or 'No selection message'}"
)

st.subheader("Latest Top 5 Watch")
top_watch_cols = st.columns(5)
for idx, (_, row) in enumerate(top_watch_df.iterrows()):
    with top_watch_cols[idx % 5]:
        st.markdown(f"**{row['ticker']}**")
        st.caption(
            f"{row.get('final_action') or '-'} | "
            f"{row.get('actionability_label') or '-'} | "
            f"{row.get('trend_state') or '-'}"
        )
        st.write(f"Entry: `{format_price(row.get('preferred_entry'))}`")
        st.write(f"Stop: `{format_price(row.get('stop_loss'))}`")
        st.write(f"TP1: `{format_price(row.get('take_profit_1'))}`")
        st.caption(_summary_from_row(row))

st.subheader("Latest Watchlist")
filter_cols = st.columns([1, 1, 1, 2])
selected_final_action = filter_cols[0].selectbox(
    "Final Action",
    ["All"] + sorted([x for x in sorted_snapshots_df.get("final_action", []).dropna().unique().tolist()]),
)
selected_watchlist_tier = filter_cols[1].selectbox(
    "Watchlist Tier",
    ["All"] + sorted([x for x in sorted_snapshots_df.get("watchlist_tier", []).dropna().unique().tolist()]),
)
selected_actionability = filter_cols[2].selectbox(
    "Actionability",
    ["All"] + sorted([x for x in sorted_snapshots_df.get("actionability_label", []).dropna().unique().tolist()]),
)
ticker_search = filter_cols[3].text_input("Ticker Search", placeholder="Search ticker")

filtered_snapshots_df = filter_watchlist_df(
    sorted_snapshots_df,
    final_action=selected_final_action,
    watchlist_tier=selected_watchlist_tier,
    actionability_label=selected_actionability,
    ticker_search=ticker_search,
)

watchlist_table_cols = [
    "ticker",
    "final_action",
    "watchlist_tier",
    "watch_priority",
    "actionability_label",
    "actionability_score",
    "suitability_label",
    "suitability_score",
    "trend_state",
    "preferred_entry",
    "stop_loss",
    "take_profit_1",
    "max_hold_date",
]
st.dataframe(filtered_snapshots_df[watchlist_table_cols], use_container_width=True, hide_index=True)

st.subheader("Ready Soon")
ready_df = filtered_snapshots_df[
    (filtered_snapshots_df["final_action"] == "WAIT")
    & (filtered_snapshots_df["actionability_label"] == "ready_soon")
]
if ready_df.empty:
    st.caption("No ready-soon WAIT names in the current filtered view.")
else:
    ready_cols = st.columns(3)
    for idx, (_, row) in enumerate(ready_df.iterrows()):
        with ready_cols[idx % 3]:
            raw = safe_json(row.get("raw_result_json")) or {}
            actionability = raw.get("actionability_soon") or {}
            st.markdown(f"**{row['ticker']}**")
            st.caption(f"{row.get('trend_state') or '-'} | {row.get('watchlist_tier') or '-'}")
            st.write(f"Entry: `{format_price(row.get('preferred_entry'))}`")
            st.write(f"Stop: `{format_price(row.get('stop_loss'))}`")
            st.write(f"TP1: `{format_price(row.get('take_profit_1'))}`")
            st.caption(first_non_empty(row.get("short_summary"), actionability.get("actionability_summary")) or "No watch note.")

st.subheader("Ticker Detail")
detail_cols = st.columns([2, 1])
available_tickers = filtered_snapshots_df["ticker"].dropna().astype(str).tolist() or sorted_snapshots_df["ticker"].dropna().astype(str).tolist()
if not available_tickers:
    st.caption("No ticker snapshots available yet.")
else:
    selected_ticker = detail_cols[0].selectbox("Ticker", options=available_tickers, index=0)
    selected_run_label = detail_cols[1].selectbox(
        "Use Run Context",
        options=["Latest snapshot"] + [
            f"{format_ts(row.created_at)} | {row.workflow_type} | {row.id}"
            for row in run_history_df.itertuples(index=False)
        ],
    )

    selected_run_id = None
    if selected_run_label != "Latest snapshot":
        selected_run_id = selected_run_label.split("|")[-1].strip()

    detail_df = (
        fetch_run_ticker_result(selected_run_id, selected_ticker)
        if selected_run_id and selected_ticker
        else fetch_latest_ticker_result(selected_ticker)
    )
    snapshot_detail = sorted_snapshots_df[sorted_snapshots_df["ticker"] == selected_ticker]
    snapshot_row = snapshot_detail.iloc[0] if not snapshot_detail.empty else None
    detail_row = detail_df.iloc[0] if not detail_df.empty else None

    if snapshot_row is not None:
        info_cols = st.columns(4)
        info_cols[0].metric("Final Action", snapshot_row.get("final_action") or "-")
        info_cols[1].metric("Watchlist Tier", snapshot_row.get("watchlist_tier") or "-")
        info_cols[2].metric("Actionability", snapshot_row.get("actionability_label") or "-")
        info_cols[3].metric("Suitability", snapshot_row.get("suitability_label") or "-")

        st.write(
            f"Trend: `{snapshot_row.get('trend_state') or '-'}`  \n"
            f"Preferred Entry: `{format_price(snapshot_row.get('preferred_entry'))}`  \n"
            f"Stop Loss: `{format_price(snapshot_row.get('stop_loss'))}`  \n"
            f"Take Profit 1: `{format_price(snapshot_row.get('take_profit_1'))}`  \n"
            f"Max Hold Date: `{format_ts(snapshot_row.get('max_hold_date'))}`"
        )
        st.caption(_summary_from_row(snapshot_row))

    if detail_row is not None:
        _render_json_block("Chart Execution View", detail_row.get("chart_execution_view_json"))
        _render_json_block("What To Watch", detail_row.get("what_to_watch_json"))
        _render_json_block("Actionability Soon", detail_row.get("actionability_soon_json"))
        _render_json_block("Raw Result JSON", detail_row.get("raw_result_json"))
    else:
        st.caption("No ticker result row was found for the selected context.")

st.subheader("Scan Run History")
history_display_cols = [
    "created_at",
    "workflow_type",
    "market_regime",
    "pre_scanned_count",
    "pre_scan_shortlist_count",
    "selected_count",
    "rows_logged",
    "selection_message",
]
st.dataframe(run_history_df[history_display_cols], use_container_width=True, hide_index=True)

run_options = [
    (str(row.id), f"{format_ts(row.created_at)} | {row.workflow_type} | {row.market_regime} | selected={row.selected_count}")
    for row in run_history_df.itertuples(index=False)
]
if run_options:
    selected_history_run_id = st.selectbox(
        "Select Run",
        options=[run_id for run_id, _ in run_options],
        format_func=lambda run_id: next(label for value, label in run_options if value == run_id),
    )
    run_results_df = fetch_run_results(selected_history_run_id)
    st.dataframe(
        run_results_df[
            [
                "rank",
                "ticker",
                "final_action",
                "watchlist_tier",
                "watch_priority",
                "actionability_label",
                "actionability_score",
                "suitability_label",
                "trend_state",
                "preferred_entry",
                "stop_loss",
                "take_profit_1",
                "max_hold_date",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

