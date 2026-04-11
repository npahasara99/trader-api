"""Read-only Streamlit dashboard for Supabase scan/watchlist reporting."""

from __future__ import annotations

import streamlit as st

from components import (
    format_run_history_display,
    format_watchlist_display,
    render_actionability,
    render_badge_row,
    render_chart_execution_view,
    render_header,
    render_key_value_grid,
    render_kpi_card,
    render_raw_json_block,
    render_top_watch_card,
    render_what_to_watch,
    summary_from_row,
)
from queries import (
    fetch_latest_run_summary,
    fetch_latest_snapshots,
    fetch_latest_ticker_snapshot,
    fetch_run_history,
    fetch_run_results,
    fetch_top_watch,
)
from styles import inject_styles
from utils import (
    filter_watchlist_df,
    format_price,
    format_short_date,
    format_ts,
    sort_watchlist_table,
)


st.set_page_config(
    page_title="Trader Watch Dashboard",
    layout="wide",
)
inject_styles()


def _snapshot_payload(snapshot_row, key: str):
    if snapshot_row is None:
        return None
    raw_payload = snapshot_row.get("raw_result_json") or {}
    if isinstance(raw_payload, dict):
        return raw_payload.get(key)
    return None


def _load_dashboard_data():
    latest_run = fetch_latest_run_summary()
    snapshots = fetch_latest_snapshots()
    run_history = fetch_run_history()
    top_watch = fetch_top_watch()
    return latest_run, snapshots, run_history, top_watch


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
latest_data_ts = None if sorted_snapshots_df.empty else format_ts(sorted_snapshots_df["updated_at"].max())

render_header(
    latest_run_ts=format_ts(latest_run.get("created_at")),
    latest_data_ts=latest_data_ts,
)

active_tab, history_tab = st.tabs(["Active Dashboard", "History"])

with active_tab:
    st.markdown("### Overview")
    st.caption("Current-state metrics are driven only by non-expired rows in watchlist snapshots.")
    primary_metrics = st.columns(5)
    with primary_metrics[0]:
        render_kpi_card("Market Regime", str(latest_run.get("market_regime") or "-"))
    with primary_metrics[1]:
        render_kpi_card("Active Names", int(latest_run.get("selected_count") or 0))
    with primary_metrics[2]:
        render_kpi_card("Ready Soon", int(latest_run.get("ready_soon_count") or 0))
    with primary_metrics[3]:
        render_kpi_card("Monitor", int(latest_run.get("monitor_count") or 0))
    with primary_metrics[4]:
        render_kpi_card("Background", int(latest_run.get("background_count") or 0))

    secondary_metrics = st.columns(2)
    with secondary_metrics[0]:
        render_kpi_card("Primary Watchlist", int(latest_run.get("primary_watchlist_count") or 0), small=True)
    with secondary_metrics[1]:
        render_kpi_card("Secondary Watchlist", int(latest_run.get("secondary_watchlist_count") or 0), small=True)

    st.markdown("### Top 5 Active Watch")
    st.caption("The highest-priority names from the latest active snapshot set, ranked to surface what deserves attention first.")
    top_watch_cols = st.columns(5)
    for idx, (_, row) in enumerate(top_watch_df.iterrows()):
        with top_watch_cols[idx % 5]:
            render_top_watch_card(row)

    st.markdown("### Latest Watchlist")
    st.caption("These rows come only from current non-expired ticker snapshots.")
    filter_container = st.container(border=True)
    with filter_container:
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
        "suitability_label",
        "trend_state",
        "preferred_entry",
        "stop_loss",
        "take_profit_1",
        "max_hold_date",
    ]
    st.dataframe(
        format_watchlist_display(filtered_snapshots_df[watchlist_table_cols]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Ready Soon")
    st.caption("The most actionable WAIT setups from the active snapshot set.")
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
                render_top_watch_card(row)

    st.markdown("### Selected Ticker")
    st.caption("Ticker detail here always comes from the latest active snapshot. Historical per-run results are available in the History tab.")
    available_tickers = filtered_snapshots_df["ticker"].dropna().astype(str).tolist() or sorted_snapshots_df["ticker"].dropna().astype(str).tolist()
    if not available_tickers:
        st.caption("No ticker snapshots available yet.")
    else:
        selected_ticker = st.selectbox("Ticker", options=available_tickers, index=0)
        detail_df = fetch_latest_ticker_snapshot(selected_ticker)
        snapshot_detail = sorted_snapshots_df[sorted_snapshots_df["ticker"] == selected_ticker]
        snapshot_row = snapshot_detail.iloc[0] if not snapshot_detail.empty else None
        detail_row = detail_df.iloc[0] if not detail_df.empty else None
        chart_execution_payload = _snapshot_payload(detail_row, "chart_execution_view")
        what_to_watch_payload = _snapshot_payload(detail_row, "what_to_watch")
        actionability_payload = _snapshot_payload(detail_row, "actionability_soon")

        if snapshot_row is not None:
            render_badge_row(snapshot_row)

        overview_tab, execution_tab, watch_tab, actionability_tab, debug_tab = st.tabs(
            ["Overview", "Execution", "What to Watch", "Actionability", "Debug"]
        )

        with overview_tab:
            if snapshot_row is None:
                st.caption("No overview data available.")
            else:
                render_key_value_grid(
                    [
                        ("Final Action", snapshot_row.get("final_action") or "-"),
                        ("Watchlist Tier", snapshot_row.get("watchlist_tier") or "-"),
                        ("Actionability", snapshot_row.get("actionability_label") or "-"),
                        ("Suitability", snapshot_row.get("suitability_label") or "-"),
                        ("Trend State", snapshot_row.get("trend_state") or "-"),
                        ("Preferred Entry", format_price(snapshot_row.get("preferred_entry"))),
                        ("Stop Loss", format_price(snapshot_row.get("stop_loss"))),
                        ("Take Profit 1", format_price(snapshot_row.get("take_profit_1"))),
                        ("Max Hold Date", format_short_date(snapshot_row.get("max_hold_date"))),
                    ],
                    columns=3,
                )
                st.caption(summary_from_row(snapshot_row))

        with execution_tab:
            if detail_row is None:
                st.caption("No execution data available.")
            else:
                render_chart_execution_view(chart_execution_payload)

        with watch_tab:
            if detail_row is None:
                st.caption("No watch data available.")
            else:
                render_what_to_watch(what_to_watch_payload)

        with actionability_tab:
            if detail_row is None:
                st.caption("No actionability data available.")
            else:
                render_actionability(actionability_payload)

        with debug_tab:
            if detail_row is None:
                st.caption("No debug data available.")
            else:
                render_raw_json_block("Raw Result JSON", detail_row.get("raw_result_json"))
                render_raw_json_block("Raw JSON: Chart Execution View", chart_execution_payload)
                render_raw_json_block("Raw JSON: What To Watch", what_to_watch_payload)
                render_raw_json_block("Raw JSON: Actionability Soon", actionability_payload)

with history_tab:
    st.markdown("### History")
    st.caption("Past runs and archived per-run selections remain available here. This section reads from scan_runs and scan_ticker_results.")
    st.dataframe(
        format_run_history_display(
            run_history_df[
                [
                    "created_at",
                    "workflow_type",
                    "market_regime",
                    "selected_count",
                    "rows_logged",
                    "selection_message",
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    run_options = [
        (str(row.id), f"{format_ts(row.created_at)} | {row.workflow_type} | {row.market_regime} | selected={row.selected_count}")
        for row in run_history_df.itertuples(index=False)
    ]
    if run_options:
        selected_history_run_id = st.selectbox(
            "Inspect Historical Run",
            options=[run_id for run_id, _ in run_options],
            format_func=lambda run_id: next(label for value, label in run_options if value == run_id),
        )
        run_results_df = fetch_run_results(selected_history_run_id)
        st.dataframe(
            format_watchlist_display(
                run_results_df[
                    [
                        "rank",
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
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
