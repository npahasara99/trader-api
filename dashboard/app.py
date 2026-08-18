"""Read-only Streamlit dashboard for Supabase scan/watchlist reporting."""

from __future__ import annotations

from pathlib import Path
import sys


# Streamlit adds the script directory to sys.path. Put the repository root
# first so backend imports such as app.live_plan_consistency cannot resolve to
# dashboard/app.py and create a circular import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
if project_root_text in sys.path:
    sys.path.remove(project_root_text)
sys.path.insert(0, project_root_text)

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    TraderAPIError,
    api_config_status,
    bot_action,
    fetch_earnings_calendar,
    fetch_earnings_detail,
    fetch_live_quotes,
    fetch_bot_config,
    fetch_bot_payload,
    fetch_bot_rows,
    fetch_bot_status,
    run_manual_basket,
    run_single_stock_workflow,
    run_sp100_workflow,
    run_sp500_daily_opportunities,
    update_bot_config,
)
from dashboard.components import (
    format_run_history_display,
    format_watchlist_display,
    pretty_label,
    render_actionability,
    render_badge_row,
    render_chart_execution_view,
    render_execution_scenario_panel,
    render_header,
    render_chip_list,
    render_key_value_grid,
    render_kpi_card,
    render_raw_json_block,
    render_runner_bucket_panel,
    render_daily_trade_card,
    render_next_trigger_card,
    render_runner_note,
    render_status_bar,
    render_top_watch_card,
    render_what_to_watch,
    summary_from_row,
)
from dashboard.queries import (
    fetch_latest_run_summary,
    fetch_latest_snapshots,
    fetch_latest_ticker_snapshot,
    fetch_run_history,
    fetch_run_results,
)
from dashboard.styles import inject_styles
from dashboard.tradingview_chart import render_tradingview_chart
from dashboard.utils import (
    build_active_market_view,
    filter_watchlist_df,
    first_non_empty,
    format_pct,
    format_price,
    format_runner_plan_rows,
    format_short_date,
    format_ts,
    parse_ticker_text,
    sort_watchlist_table,
)


st.set_page_config(
    page_title="Trader Watch Dashboard",
    layout="wide",
)
inject_styles()


RUNNER_DEFAULTS = {
    "runner_type": "SP500 Daily Opportunities",
    "single_ticker": "",
    "single_lookback_days": 30,
    "single_learning_limit": 200,
    "single_mode": "manual",
    "single_llm_provider": "chatgpt-actions",
    "single_llm_model": "gpt-5",
    "single_llm_style": "swing_v2_structured",
    "sp100_top_scan": 100,
    "sp100_top_plan": 10,
    "sp100_pre_scan_shortlist": 25,
    "sp100_lookback_days": 180,
    "sp100_min_history_samples": 3,
    "sp100_sector": "",
    "sp100_industry": "",
    "sp100_mode": "sp100_auto",
    "sp100_llm_provider": "chatgpt-actions",
    "sp100_llm_model": "gpt-5",
    "sp100_llm_style": "sp100_ranker_v2_structured",
    "sp100_compact_response": False,
    "sp500_prescan_limit": 75,
    "sp500_deep_analysis_limit": 30,
    "sp500_best_setups_count": 10,
    "sp500_best_trades_max": 2,
    "sp500_next_to_trigger_count": 5,
    "sp500_lookback_days": 180,
    "sp500_min_history_samples": 3,
    "sp500_sector": "",
    "sp500_industry": "",
    "sp500_mode": "sp500_daily_opportunities",
    "sp500_llm_provider": "chatgpt-actions",
    "sp500_llm_model": "gpt-5",
    "sp500_llm_style": "sp500_daily_ranker_v1",
    "sp500_compact_response": True,
    "basket_tickers": "",
    "basket_mode": "manual",
    "basket_llm_provider": "chatgpt-actions",
    "basket_llm_model": "gpt-5",
    "basket_llm_style": "swing_v2_structured",
}


def _init_runner_state() -> None:
    for key, value in RUNNER_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("runner_last_result", None)
    st.session_state.setdefault("runner_last_error", None)


def _apply_runner_preset(preset: str) -> None:
    mapping = {
        "sp500_default": {
            "runner_type": "SP500 Daily Opportunities",
            "sp500_sector": "",
            "sp500_industry": "",
        },
        "sp500_tech": {
            "runner_type": "SP500 Sector / Industry",
            "sp500_sector": "Information Technology",
            "sp500_industry": "",
        },
        "sp500_semis": {
            "runner_type": "SP500 Sector / Industry",
            "sp500_sector": "",
            "sp500_industry": "Semiconductor",
        },
        "sp500_energy": {
            "runner_type": "SP500 Sector / Industry",
            "sp500_sector": "Energy",
            "sp500_industry": "",
        },
        "sp100_default": {
            "runner_type": "SP100 Top 10",
            "sp100_sector": "",
            "sp100_industry": "",
            "sp100_top_scan": 100,
            "sp100_top_plan": 10,
            "sp100_pre_scan_shortlist": 25,
        },
        "sp100_tech": {
            "runner_type": "SP100 Sector / Industry",
            "sp100_sector": "tech",
            "sp100_industry": "",
            "sp100_top_scan": 100,
            "sp100_top_plan": 10,
            "sp100_pre_scan_shortlist": 25,
        },
        "sp100_semis": {
            "runner_type": "SP100 Sector / Industry",
            "sp100_sector": "",
            "sp100_industry": "semiconductors",
            "sp100_top_scan": 100,
            "sp100_top_plan": 10,
            "sp100_pre_scan_shortlist": 25,
        },
        "sp100_energy": {
            "runner_type": "SP100 Sector / Industry",
            "sp100_sector": "energy",
            "sp100_industry": "",
            "sp100_top_scan": 100,
            "sp100_top_plan": 10,
            "sp100_pre_scan_shortlist": 25,
        },
        "single_stock": {
            "runner_type": "Single Stock",
        },
        "manual_basket": {
            "runner_type": "Manual Basket",
        },
    }
    for key, value in mapping.get(preset, {}).items():
        st.session_state[key] = value


def _runner_row_summary(row: dict) -> str:
    return first_non_empty(
        ((row.get("what_to_watch") or {}).get("watch_summary_short")),
        ((row.get("actionability_soon") or {}).get("actionability_summary")),
        row.get("watchlist_summary"),
        ((row.get("chart_execution_view") or {}).get("chart_execution_summary")),
    ) or "No short summary available."


def _runner_badge_row(row: dict) -> dict:
    return {
        **row,
        "actionability_label": ((row.get("actionability_soon") or {}).get("actionability_label")),
        "suitability_label": ((row.get("swing_trade_suitability") or {}).get("suitability_label")),
    }


def _refresh_dashboard_data() -> None:
    st.cache_data.clear()
    st.rerun()


def _workflow_conclusion(result: dict) -> str:
    selection_message = (result.get("selection_message") or "").strip()
    if selection_message:
        return selection_message

    regime = str(result.get("market_regime") or "neutral")
    immediate = result.get("best_immediate_tickers") or []
    watchlist = result.get("best_watchlist_tickers") or []
    rejected = result.get("rejected_or_low_priority_tickers") or []
    selected = int(result.get("selected_count") or 0)

    if selected == 0:
        return f"{regime.title()} regime with no selected names from this run."
    if not immediate and watchlist:
        return f"{regime.title()} regime with no immediate setups; focus on the best watchlist names."
    if immediate and not watchlist:
        return f"{regime.title()} regime with immediate setups leading this run."
    if immediate and watchlist:
        return f"{regime.title()} regime with both immediate and watchlist names in play."
    if rejected and not watchlist and not immediate:
        return f"{regime.title()} regime with low-priority output and no clear active setups."
    return f"{regime.title()} regime with mostly WAIT-style output from this run."


def _runner_result_conclusion(run_type: str, response: dict) -> str:
    if run_type == "sp500_workflow":
        return response.get("selection_message") or "S&P 500 daily opportunity scan completed."
    if run_type == "sp100_workflow":
        return _workflow_conclusion(response)

    rows = response.get("rows") or ([response.get("row")] if isinstance(response.get("row"), dict) else [])
    if not rows:
        return "No detailed rows were returned for this run."

    buy_count = sum(1 for row in rows if (row or {}).get("final_action") == "BUY")
    wait_count = sum(1 for row in rows if (row or {}).get("final_action") == "WAIT")
    top_row = rows[0]
    top_summary = _runner_row_summary(top_row)

    if buy_count:
        return f"This run returned {buy_count} BUY-ready setup{'s' if buy_count != 1 else ''}. {top_summary}"
    if wait_count:
        return f"This run produced mostly WAIT setups. {top_summary}"
    return top_summary


def _store_runner_result(run_type: str, response: dict, *, title: str) -> None:
    st.session_state["runner_last_result"] = {
        "run_type": run_type,
        "title": title,
        "response": response,
    }
    st.session_state["runner_last_error"] = None
    st.cache_data.clear()
    st.rerun()


def _render_runner_workflow_result(result: dict) -> None:
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_kpi_card("Market Regime", result.get("market_regime") or "-")
    with metric_cols[1]:
        render_kpi_card("Pre-Scanned", int(result.get("pre_scanned_count") or 0), small=True)
    with metric_cols[2]:
        render_kpi_card("Shortlist", int(result.get("pre_scan_shortlist_count") or 0), small=True)
    with metric_cols[3]:
        render_kpi_card("Selected", int(result.get("selected_count") or 0), small=True)
    with metric_cols[4]:
        render_kpi_card("Rows Logged", int(result.get("rows_logged") or 0), small=True)

    supabase_persisted = bool(result.get("supabase_persisted"))
    supabase_run_id = result.get("supabase_scan_run_id")
    supabase_error = result.get("supabase_persistence_error")
    if supabase_persisted:
        toast_message = "Supabase active-watchlist persistence succeeded."
        if supabase_run_id:
            toast_message += f" Scan run id: {supabase_run_id}"
        st.toast(toast_message, icon=":material/check_circle:")
    elif supabase_error:
        st.error(
            "Supabase active-watchlist persistence failed, so the Active Dashboard will not refresh from this run. "
            f"{supabase_error}"
        )
    else:
        st.warning(
            "Supabase active-watchlist persistence status was not returned. "
            "If the Active Dashboard did not update, check the API service deployment and logs."
        )

    st.markdown('<div class="runner-result-gap"></div>', unsafe_allow_html=True)
    bucket_cols = st.columns(3)
    with bucket_cols[0]:
        with st.container(border=True):
            render_runner_bucket_panel("Best Immediate", result.get("best_immediate_tickers") or [], empty_text="No immediate names")
    with bucket_cols[1]:
        with st.container(border=True):
            render_runner_bucket_panel("Best Watchlist", result.get("best_watchlist_tickers") or [], empty_text="No watchlist names")
    with bucket_cols[2]:
        with st.container(border=True):
            render_runner_bucket_panel(
                "Rejected / Low Priority",
                result.get("rejected_or_low_priority_tickers") or [],
                empty_text="No rejected names",
            )

    st.markdown("**Selected Tickers**")
    render_chip_list(result.get("selected_tickers") or [], empty_text="No selected tickers", variant="muted")

    rows = result.get("rows") or []
    if rows:
        st.dataframe(format_runner_plan_rows(rows), use_container_width=True, hide_index=True)


def _render_sp500_workflow_result(result: dict) -> None:
    summary = result.get("scan_summary") or {}
    portfolio = result.get("portfolio_summary") or {}
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_kpi_card("Market", result.get("market_regime") or "-")
    with metric_cols[1]:
        render_kpi_card("Universe", int(result.get("universe_size") or 0), small=True)
    with metric_cols[2]:
        render_kpi_card("Deep Analyzed", int(summary.get("deep_analyzed") or 0), small=True)
    with metric_cols[3]:
        render_kpi_card("Actionable", int(summary.get("actionable_count") or 0), small=True)
    with metric_cols[4]:
        render_kpi_card("Open Slots", int(portfolio.get("available_position_slots") or 0), small=True)

    if result.get("universe_used_fallback"):
        st.caption(
            f"Universe fallback snapshot in use ({result.get('universe_as_of')}). "
            f"Live source warning: {result.get('universe_warning') or 'unavailable'}"
        )
    else:
        st.caption(f"Universe as of {result.get('universe_as_of')} | Source: {result.get('universe_source')}")

    if result.get("supabase_persisted"):
        st.toast("S&P 500 scan persisted to the reporting database.", icon=":material/check_circle:")
    elif result.get("supabase_persistence_error"):
        st.error(f"Reporting persistence failed: {result.get('supabase_persistence_error')}")

    st.markdown("### Best Trades Today")
    trades = result.get("best_trades_today") or []
    if not trades:
        st.info("NO HIGH-QUALITY TRADE CURRENTLY CONFIRMED")
    else:
        trade_cols = st.columns(min(len(trades), 2), gap="large")
        for index, item in enumerate(trades):
            with trade_cols[index % len(trade_cols)]:
                render_daily_trade_card(item)

    st.markdown("### Best Setups")
    st.caption("Objective individual setup quality. Portfolio concentration does not change this ranking.")
    setups = result.get("best_setups") or []
    if setups:
        setup_rows = [
            {
                "Rank": item.get("rank"),
                "Ticker": item.get("ticker"),
                "Grade": item.get("grade"),
                "Raw Setup": item.get("raw_setup_score"),
                "Setup Type": pretty_label(item.get("setup_type")),
                "Actionability": item.get("actionability_score"),
                "Entry State": pretty_label(item.get("actionability_state")),
                "Trigger": item.get("confirmation_trigger"),
                "Sector": item.get("sector"),
            }
            for item in setups
        ]
        st.dataframe(pd.DataFrame(setup_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No setups completed the deep-analysis stage.")

    st.markdown("### Next To Trigger")
    next_items = result.get("next_to_trigger") or []
    if next_items:
        trigger_cols = st.columns(min(len(next_items), 3), gap="medium")
        for index, item in enumerate(next_items):
            with trigger_cols[index % len(trigger_cols)]:
                render_next_trigger_card(item)
    else:
        st.caption("No high-quality setup is currently close to a defined trigger.")

    with st.expander("Scanner Diagnostics", expanded=False):
        st.json(result.get("diagnostics") or {})


def _render_runner_plan_result(rows: list[dict], *, planned_at: str | None = None, market_regime: str | None = None) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        render_kpi_card("Rows", len(rows), small=True)
    with summary_cols[1]:
        render_kpi_card("Market Regime", market_regime or "-", small=True)
    with summary_cols[2]:
        render_kpi_card("Planned At", format_ts(planned_at), small=True)

    if not rows:
        return

    st.dataframe(format_runner_plan_rows(rows), use_container_width=True, hide_index=True)

    detail_ticker = st.selectbox(
        "Result Detail",
        options=[str((row or {}).get("ticker") or "-") for row in rows],
        key=f"runner_detail_{planned_at or 'rows'}",
    )
    detail_row = next((row for row in rows if str(row.get("ticker")) == detail_ticker), rows[0])
    render_badge_row(_runner_badge_row(detail_row))
    st.caption(_runner_row_summary(detail_row))

    overview_tab, execution_tab, watch_tab, actionability_tab, debug_tab = st.tabs(
        ["Overview", "Execution", "What to Watch", "Actionability", "Debug"]
    )

    with overview_tab:
        render_key_value_grid(
            [
                ("Final Action", detail_row.get("final_action") or "-"),
                ("Watchlist Tier", detail_row.get("watchlist_tier") or "-"),
                ("Actionability", ((detail_row.get("actionability_soon") or {}).get("actionability_label")) or "-"),
                ("Suitability", ((detail_row.get("swing_trade_suitability") or {}).get("suitability_label")) or "-"),
                ("Trend State", detail_row.get("trend_state") or "-"),
                ("Preferred Entry", format_price(detail_row.get("preferred_entry"))),
                ("Stop Loss", format_price(detail_row.get("stop_loss"))),
                ("Take Profit 1", format_price(detail_row.get("take_profit_1"))),
                ("Max Hold Date", format_short_date(detail_row.get("max_hold_date"))),
            ],
            columns=3,
        )

    with execution_tab:
        render_chart_execution_view(detail_row.get("chart_execution_view"))

    with watch_tab:
        render_what_to_watch(detail_row.get("what_to_watch"))

    with actionability_tab:
        render_actionability(detail_row.get("actionability_soon"))

    with debug_tab:
        render_raw_json_block("Raw Row JSON", detail_row)
        render_raw_json_block("Raw JSON: Chart Execution View", detail_row.get("chart_execution_view"))
        render_raw_json_block("Raw JSON: What To Watch", detail_row.get("what_to_watch"))
        render_raw_json_block("Raw JSON: Actionability Soon", detail_row.get("actionability_soon"))


def _snapshot_payload(snapshot_row, key: str):
    if snapshot_row is None:
        return None
    raw_payload = snapshot_row.get("raw_result_json") or {}
    if isinstance(raw_payload, dict):
        return raw_payload.get(key)
    return None


def _filter_earnings_df(
    df: pd.DataFrame,
    *,
    max_days: int,
    ticker_search: str,
    sector_filter: str,
    industry_filter: str,
    watchlist_only: bool,
    watchlist_tickers: set[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out[out["days_to_earnings"].fillna(9999) <= max_days]
    if sector_filter != "All":
        out = out[out["sector"].fillna("") == sector_filter]
    if industry_filter != "All":
        out = out[out["industry"].fillna("") == industry_filter]
    if ticker_search.strip():
        query = ticker_search.strip().upper()
        out = out[out["ticker"].astype(str).str.upper().str.contains(query, na=False)]
    if watchlist_only:
        out = out[out["ticker"].astype(str).isin(watchlist_tickers)]
    return out.sort_values(by=["days_to_earnings", "ticker"], ascending=[True, True], na_position="last")


def _load_dashboard_data():
    latest_run = fetch_latest_run_summary()
    snapshots = fetch_latest_snapshots()
    run_history = fetch_run_history()
    return latest_run, snapshots, run_history


@st.cache_data(ttl=900, show_spinner=False)
def _load_earnings_calendar_data(*, days_ahead: int, sp100_only: bool) -> dict:
    return fetch_earnings_calendar(days_ahead=days_ahead, sp100_only=sp100_only)


@st.cache_data(ttl=900, show_spinner=False)
def _load_earnings_detail_data(ticker: str, *, days_ahead: int) -> dict:
    return fetch_earnings_detail(ticker, days_ahead=days_ahead)


@st.cache_data(ttl=60, show_spinner=False)
def _load_live_quotes_data(tickers: tuple[str, ...]) -> dict:
    return fetch_live_quotes(list(tickers))


_init_runner_state()


try:
    latest_run_df, snapshots_df, run_history_df = _load_dashboard_data()
except Exception as exc:
    st.error("Could not load the Supabase reporting database.")
    with st.expander("Error details", expanded=True):
        st.code(str(exc))
    st.stop()

latest_run = latest_run_df.iloc[0] if not latest_run_df.empty else {}
sorted_snapshots_df = sort_watchlist_table(snapshots_df)
latest_data_ts = format_ts(latest_run.get("latest_snapshot_updated_at")) if latest_run.get("latest_snapshot_updated_at") else (
    None if sorted_snapshots_df.empty else format_ts(sorted_snapshots_df["updated_at"].max())
)
latest_snapshot_updated_at = latest_run.get("latest_snapshot_updated_at")
live_quote_error = None
live_quote_payload = {}
active_market_df = build_active_market_view(sorted_snapshots_df, pd.DataFrame())
active_tickers = tuple(sorted_snapshots_df["ticker"].dropna().astype(str).tolist())
if active_tickers:
    try:
        live_quote_payload = _load_live_quotes_data(active_tickers)
        live_quotes_df = pd.DataFrame(live_quote_payload.get("rows") or [])
        active_market_df = build_active_market_view(sorted_snapshots_df, live_quotes_df)
    except TraderAPIError as exc:
        live_quote_error = str(exc)
        active_market_df = build_active_market_view(sorted_snapshots_df, pd.DataFrame())
active_market_df = sort_watchlist_table(active_market_df)

render_header(
    latest_run_ts=format_ts(latest_run.get("created_at")),
    latest_data_ts=latest_data_ts,
)

scanner_tab, bot_tab, active_tab, earnings_tab, history_tab = st.tabs(["Run Scanner", "Trading Bot", "Active Dashboard", "Earnings", "History"])

with scanner_tab:
    st.markdown("### Scanner / Runner")
    st.caption("Run supported trader API workflows from the dashboard. The API remains the source of truth, and Supabase-backed dashboard views refresh after successful runs.")

    api_status = api_config_status()
    if not api_status.get("base_url"):
        st.warning("TRADER_API_BASE_URL is not configured yet. You can still use the viewer, but API runs from this tab will fail until that environment variable is set.")
    status_cols = st.columns([5, 1], gap="medium")
    with status_cols[0]:
        render_status_bar(
            [
                ("API Base URL", api_status.get("base_url") or "Not configured"),
                ("Auth Token", "Configured" if api_status.get("has_bearer_token") else "Missing"),
            ]
        )
    with status_cols[1]:
        if st.button("Refresh Dashboard Data", key="runner_refresh_button", use_container_width=True, type="secondary"):
            _refresh_dashboard_data()

    with st.container(border=True):
        st.markdown("**Presets**")
        preset_cols = st.columns(5)
        if preset_cols[0].button("SP500 Daily", use_container_width=True, type="secondary"):
            _apply_runner_preset("sp500_default")
            st.rerun()
        if preset_cols[1].button("Tech", use_container_width=True, type="secondary"):
            _apply_runner_preset("sp500_tech")
            st.rerun()
        if preset_cols[2].button("Semis", use_container_width=True, type="secondary"):
            _apply_runner_preset("sp500_semis")
            st.rerun()
        if preset_cols[3].button("Energy", use_container_width=True, type="secondary"):
            _apply_runner_preset("sp500_energy")
            st.rerun()
        if preset_cols[4].button("Manual Basket", use_container_width=True, type="secondary"):
            _apply_runner_preset("manual_basket")
            st.rerun()

    control_col, result_col = st.columns([1.05, 1.35], gap="large")
    with control_col:
        with st.container(border=True):
            st.markdown("**Runner Controls**")
            st.caption("Choose a supported run type, adjust only what matters, and launch from here.")
            st.session_state["runner_type"] = st.selectbox(
                "Run Type",
                options=[
                    "SP500 Daily Opportunities",
                    "SP500 Sector / Industry",
                    "Single Stock",
                    "SP100 Top 10",
                    "SP100 Sector / Industry",
                    "Manual Basket",
                ],
                index=[
                    "SP500 Daily Opportunities",
                    "SP500 Sector / Industry",
                    "Single Stock",
                    "SP100 Top 10",
                    "SP100 Sector / Industry",
                    "Manual Basket",
                ].index(st.session_state["runner_type"]),
            )

        run_type = st.session_state["runner_type"]
        if run_type == "Single Stock":
            with st.form("single_stock_runner_form", clear_on_submit=False):
                st.markdown("**Ticker**")
                st.text_input("Ticker", key="single_ticker", placeholder="AAPL")
                st.markdown("**Lookback / History**")
                history_cols = st.columns(2)
                history_cols[0].number_input("Lookback Days", min_value=7, max_value=720, key="single_lookback_days")
                history_cols[1].number_input("Learning Limit", min_value=20, max_value=500, key="single_learning_limit")
                st.markdown("**Model Settings**")
                st.text_input("Mode", key="single_mode")
                llm_cols = st.columns(3)
                llm_cols[0].text_input("LLM Provider", key="single_llm_provider")
                llm_cols[1].text_input("LLM Model", key="single_llm_model")
                llm_cols[2].text_input("LLM Style", key="single_llm_style")
                single_submit = st.form_submit_button("Run Single Stock Workflow", use_container_width=True, type="primary")

            if single_submit:
                try:
                    ticker = (st.session_state.get("single_ticker") or "").strip().upper()
                    if not ticker:
                        raise TraderAPIError("Enter a ticker before running the single-stock workflow.")
                    payload = {
                        "ticker": ticker,
                        "lookback_days": int(st.session_state["single_lookback_days"]),
                        "learning_limit": int(st.session_state["single_learning_limit"]),
                        "mode": st.session_state["single_mode"],
                        "llm_provider": st.session_state["single_llm_provider"],
                        "llm_model": st.session_state["single_llm_model"],
                        "llm_style": st.session_state["single_llm_style"],
                    }
                    response = run_single_stock_workflow(payload)
                    _store_runner_result("single_stock", response, title=f"Single Stock: {ticker}")
                except TraderAPIError as exc:
                    st.session_state["runner_last_error"] = str(exc)

        elif run_type in {"SP500 Daily Opportunities", "SP500 Sector / Industry"}:
            with st.form("sp500_runner_form", clear_on_submit=False):
                st.markdown("**Scan Pipeline**")
                scan_cols = st.columns(3)
                scan_cols[0].number_input("Pre-Scan Limit", min_value=10, max_value=200, key="sp500_prescan_limit")
                scan_cols[1].number_input("Deep Analysis", min_value=1, max_value=75, key="sp500_deep_analysis_limit")
                scan_cols[2].number_input("Best Setups", min_value=1, max_value=25, key="sp500_best_setups_count")
                output_cols = st.columns(2)
                output_cols[0].number_input("Max Trades Today", min_value=0, max_value=5, key="sp500_best_trades_max")
                output_cols[1].number_input("Next To Trigger", min_value=1, max_value=10, key="sp500_next_to_trigger_count")
                st.markdown("**Scope**")
                scope_cols = st.columns(2)
                scope_cols[0].text_input("Sector", key="sp500_sector", placeholder="Information Technology")
                scope_cols[1].text_input("Industry", key="sp500_industry", placeholder="Semiconductor")
                st.markdown("**Lookback / History**")
                history_cols = st.columns(2)
                history_cols[0].number_input("Lookback Days", min_value=30, max_value=720, key="sp500_lookback_days")
                history_cols[1].number_input("Min History Samples", min_value=1, max_value=20, key="sp500_min_history_samples")
                st.markdown("**Model Settings**")
                llm_cols = st.columns(4)
                llm_cols[0].text_input("Mode", key="sp500_mode")
                llm_cols[1].text_input("LLM Provider", key="sp500_llm_provider")
                llm_cols[2].text_input("LLM Model", key="sp500_llm_model")
                llm_cols[3].text_input("LLM Style", key="sp500_llm_style")
                st.checkbox("Compact Response", key="sp500_compact_response")
                sp500_submit = st.form_submit_button("Run S&P 500 Daily Scan", use_container_width=True, type="primary")

            if sp500_submit:
                try:
                    payload = {
                        "prescan_limit": int(st.session_state["sp500_prescan_limit"]),
                        "deep_analysis_limit": int(st.session_state["sp500_deep_analysis_limit"]),
                        "best_setups_count": int(st.session_state["sp500_best_setups_count"]),
                        "best_trades_today_max": int(st.session_state["sp500_best_trades_max"]),
                        "next_to_trigger_count": int(st.session_state["sp500_next_to_trigger_count"]),
                        "lookback_days": int(st.session_state["sp500_lookback_days"]),
                        "min_history_samples": int(st.session_state["sp500_min_history_samples"]),
                        "sector": (st.session_state.get("sp500_sector") or "").strip() or None,
                        "industry": (st.session_state.get("sp500_industry") or "").strip() or None,
                        "mode": st.session_state["sp500_mode"],
                        "llm_provider": st.session_state["sp500_llm_provider"],
                        "llm_model": st.session_state["sp500_llm_model"],
                        "llm_style": st.session_state["sp500_llm_style"],
                        "compact_response": bool(st.session_state["sp500_compact_response"]),
                    }
                    response = run_sp500_daily_opportunities(payload)
                    title = "S&P 500 Daily Opportunities"
                    if payload.get("industry"):
                        title = f"S&P 500 {payload['industry']}"
                    elif payload.get("sector"):
                        title = f"S&P 500 {payload['sector']}"
                    _store_runner_result("sp500_workflow", response, title=title)
                except TraderAPIError as exc:
                    st.session_state["runner_last_error"] = str(exc)

        elif run_type in {"SP100 Top 10", "SP100 Sector / Industry"}:
            with st.form("sp100_runner_form", clear_on_submit=False):
                st.markdown("**Scan Size**")
                scan_cols = st.columns(3)
                scan_cols[0].number_input("Top Scan", min_value=10, max_value=100, key="sp100_top_scan")
                scan_cols[1].number_input("Top Plan", min_value=1, max_value=20, key="sp100_top_plan")
                scan_cols[2].number_input("Pre-Scan Shortlist", min_value=1, max_value=60, key="sp100_pre_scan_shortlist")
                st.markdown("**Lookback / History**")
                analysis_cols = st.columns(2)
                analysis_cols[0].number_input("Lookback Days", min_value=30, max_value=720, key="sp100_lookback_days")
                analysis_cols[1].number_input("Min History Samples", min_value=1, max_value=20, key="sp100_min_history_samples")
                if run_type == "SP100 Sector / Industry":
                    st.markdown("**Scope**")
                    scope_cols = st.columns(2)
                    scope_cols[0].text_input("Sector", key="sp100_sector", placeholder="tech")
                    scope_cols[1].text_input("Industry", key="sp100_industry", placeholder="semiconductors")
                else:
                    st.session_state["sp100_sector"] = ""
                    st.session_state["sp100_industry"] = ""
                st.markdown("**Model Settings**")
                llm_cols = st.columns(4)
                llm_cols[0].text_input("Mode", key="sp100_mode")
                llm_cols[1].text_input("LLM Provider", key="sp100_llm_provider")
                llm_cols[2].text_input("LLM Model", key="sp100_llm_model")
                llm_cols[3].text_input("LLM Style", key="sp100_llm_style")
                st.checkbox("Compact Response", key="sp100_compact_response")
                sp100_submit = st.form_submit_button("Run SP100 Workflow", use_container_width=True, type="primary")

            if sp100_submit:
                try:
                    payload = {
                        "top_scan": int(st.session_state["sp100_top_scan"]),
                        "top_plan": int(st.session_state["sp100_top_plan"]),
                        "pre_scan_shortlist": int(st.session_state["sp100_pre_scan_shortlist"]),
                        "lookback_days": int(st.session_state["sp100_lookback_days"]),
                        "min_history_samples": int(st.session_state["sp100_min_history_samples"]),
                        "sector": (st.session_state.get("sp100_sector") or "").strip() or None,
                        "industry": (st.session_state.get("sp100_industry") or "").strip() or None,
                        "mode": st.session_state["sp100_mode"],
                        "llm_provider": st.session_state["sp100_llm_provider"],
                        "llm_model": st.session_state["sp100_llm_model"],
                        "llm_style": st.session_state["sp100_llm_style"],
                        "compact_response": bool(st.session_state["sp100_compact_response"]),
                    }
                    response = run_sp100_workflow(payload)
                    title = "SP100 Workflow"
                    if payload.get("industry"):
                        title = f"SP100 {payload['industry'].title()}"
                    elif payload.get("sector"):
                        title = f"SP100 {payload['sector'].title()}"
                    _store_runner_result("sp100_workflow", response, title=title)
                except TraderAPIError as exc:
                    st.session_state["runner_last_error"] = str(exc)

        elif run_type == "Manual Basket":
            with st.form("manual_basket_runner_form", clear_on_submit=False):
                st.markdown("**Basket**")
                st.text_area("Tickers", key="basket_tickers", placeholder="AAPL, MSFT, NVDA")
                st.markdown("**Model Settings**")
                st.text_input("Mode", key="basket_mode")
                llm_cols = st.columns(3)
                llm_cols[0].text_input("LLM Provider", key="basket_llm_provider")
                llm_cols[1].text_input("LLM Model", key="basket_llm_model")
                llm_cols[2].text_input("LLM Style", key="basket_llm_style")
                basket_submit = st.form_submit_button("Run Manual Basket", use_container_width=True, type="primary")

            if basket_submit:
                try:
                    tickers = parse_ticker_text(st.session_state.get("basket_tickers", ""))
                    if not tickers:
                        raise TraderAPIError("Enter at least one ticker for the manual basket run.")
                    payload = {
                        "tickers": tickers,
                        "mode": st.session_state["basket_mode"],
                        "llm_used": True,
                        "llm_provider": st.session_state["basket_llm_provider"],
                        "llm_model": st.session_state["basket_llm_model"],
                        "llm_style": st.session_state["basket_llm_style"],
                    }
                    response = run_manual_basket(payload)
                    _store_runner_result("manual_basket", response, title=f"Manual Basket: {', '.join(tickers)}")
                except TraderAPIError as exc:
                    st.session_state["runner_last_error"] = str(exc)

        st.caption("S&P 500 is the primary daily universe. SP100 routes remain available for backward compatibility.")

    with result_col:
        with st.container(border=True):
            st.markdown("### Latest Runner Result")
            st.caption("Structured response from the live trader API. The viewer updates from Supabase after successful runs.")
            runner_error = st.session_state.get("runner_last_error")
            if runner_error:
                st.error(runner_error)

            last_result = st.session_state.get("runner_last_result")
            if not last_result:
                render_runner_note("Run a workflow from the left panel to see the API response here.")
            else:
                workflow_label = str(last_result.get("run_type") or "run").replace("_", " ").title()
                st.markdown(f"**{last_result.get('title') or 'Run Result'}**")
                st.caption(workflow_label)
                response = last_result.get("response") or {}
                st.markdown(
                    f'<div class="runner-conclusion">{_runner_result_conclusion(last_result.get("run_type") or "", response)}</div>',
                    unsafe_allow_html=True,
                )
                if last_result.get("run_type") == "sp500_workflow":
                    _render_sp500_workflow_result(response)
                elif last_result.get("run_type") == "sp100_workflow":
                    _render_runner_workflow_result(response)
                elif last_result.get("run_type") == "single_stock":
                    row = response.get("row")
                    _render_runner_plan_result(
                        [row] if isinstance(row, dict) else [],
                        planned_at=response.get("planned_at"),
                        market_regime=response.get("market_regime"),
                    )
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        render_kpi_card("Ticker", response.get("ticker") or "-", small=True)
                    with info_cols[1]:
                        render_kpi_card("Rows Logged", int(response.get("rows_logged") or 0), small=True)
                    with info_cols[2]:
                        render_kpi_card("Learning Samples", int(response.get("learning_samples") or 0), small=True)
                    if response.get("logging_skipped_reason"):
                        st.caption(response.get("logging_skipped_reason"))
                    st.info("Single-stock runs do not currently update the active watchlist in Supabase. The Active Dashboard reflects the latest persisted SP500 or SP100 watchlist workflow.")
                elif last_result.get("run_type") == "manual_basket":
                    _render_runner_plan_result(
                        response.get("rows") or [],
                        planned_at=response.get("planned_at"),
                        market_regime=response.get("market_regime"),
                    )
                    st.info("Manual basket runs do not currently update the active watchlist in Supabase. The Active Dashboard reflects the latest persisted SP500 or SP100 watchlist workflow.")

with bot_tab:
    st.markdown("### Trading Bot")
    st.caption("Paper-safe bot controls and visibility for broker health, watchlist candidates, trade proposals, orders, positions, journal, memory and risk controls.")

    bot_status = {}
    bot_config = {}
    bot_status_error = None
    try:
        bot_status = fetch_bot_status()
        bot_config = fetch_bot_config().get("config") or {}
    except TraderAPIError as exc:
        bot_status_error = str(exc)

    if bot_status_error:
        st.error(bot_status_error)
    else:
        status_cols = st.columns(6)
        with status_cols[0]:
            render_kpi_card("Bot State", str(bot_status.get("state") or "-"), small=True)
        with status_cols[1]:
            render_kpi_card("Trading Mode", str(bot_status.get("trading_mode") or "-"), small=True)
        with status_cols[2]:
            render_kpi_card("Broker State", str(bot_status.get("broker_state") or "-"), small=True)
        with status_cols[3]:
            render_kpi_card("Auto Execution", "On" if bot_status.get("auto_execution") else "Off", small=True)
        with status_cols[4]:
            render_kpi_card("Kill Switch", "Active" if bot_status.get("kill_switch_active") else "Off", small=True)
        with status_cols[5]:
            render_kpi_card("Reconcile Lock", "Yes" if bot_status.get("reconciliation_required") else "No", small=True)

        control_cols = st.columns(7)
        if control_cols[0].button("Start Bot", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/bot/start")
            st.rerun()
        if control_cols[1].button("Pause", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/bot/pause")
            st.rerun()
        if control_cols[2].button("Resume", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/bot/resume")
            st.rerun()
        if control_cols[3].button("Stop", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/bot/stop")
            st.rerun()
        if control_cols[4].button("Reconnect Broker", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/broker/reconnect")
            st.rerun()
        if control_cols[5].button("Reconcile", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/broker/reconcile")
            st.rerun()
        if control_cols[6].button("Refresh Watchlist", use_container_width=True):
            st.session_state["bot_action_result"] = bot_action("/watchlist/refresh")
            st.rerun()

        if st.session_state.get("bot_action_result"):
            result = st.session_state.get("bot_action_result")
            if result.get("ok", True):
                st.success(result.get("message") or "Action completed.")
            else:
                st.error(result.get("message") or "Action failed.")
                rejection_reasons = result.get("rejection_reasons") or (result.get("preview") or {}).get("rejection_reasons") or []
                if rejection_reasons:
                    st.caption("Blocked by: " + " | ".join(str(reason) for reason in rejection_reasons))

        with st.expander("Bot Configuration", expanded=False):
            editable_keys = [
                "trading_mode",
                "auto_execution",
                "trading_budget",
                "risk_per_trade_pct",
                "max_position_pct",
                "max_open_positions",
                "max_portfolio_risk_pct",
                "max_daily_loss_pct",
                "max_weekly_loss_pct",
                "max_spread_pct",
                "stale_quote_seconds",
            ]
            cfg_payload = {}
            cfg_cols = st.columns(2)
            for idx, key in enumerate(editable_keys):
                value = bot_config.get(key)
                with cfg_cols[idx % 2]:
                    if isinstance(value, bool):
                        cfg_payload[key] = st.checkbox(key.replace("_", " ").title(), value=bool(value), key=f"bot_cfg_{key}")
                    else:
                        cfg_payload[key] = st.text_input(key.replace("_", " ").title(), value="" if value is None else str(value), key=f"bot_cfg_{key}")
            if st.button("Save Bot Config", type="secondary"):
                normalized = {}
                for key, value in cfg_payload.items():
                    if isinstance(bot_config.get(key), bool):
                        normalized[key] = bool(value)
                    elif isinstance(bot_config.get(key), int):
                        normalized[key] = int(value)
                    elif isinstance(bot_config.get(key), float):
                        normalized[key] = float(value)
                    else:
                        normalized[key] = value
                st.session_state["bot_action_result"] = update_bot_config(normalized)
                st.rerun()

        bot_overview_tab, candidates_tab, proposals_tab, orders_tab, positions_tab, journal_tab, memory_tab, safety_tab = st.tabs(
            ["Overview", "Candidates", "Proposals", "Orders", "Positions", "Journal", "Memory", "Safety"]
        )

        with bot_overview_tab:
            try:
                broker_account = fetch_bot_payload("/broker/account")
            except Exception:
                broker_account = {}
            st.caption(
                f"Last heartbeat: {format_ts(bot_status.get('last_heartbeat'))} | "
                f"Last scan: {format_ts(bot_status.get('last_scan_time'))} | "
                f"Last broker sync: {format_ts(bot_status.get('last_broker_sync'))}"
            )
            overview_cols = st.columns(4)
            with overview_cols[0]:
                render_kpi_card("Account Type", str(broker_account.get("account_type") or "-"), small=True)
            with overview_cols[1]:
                render_kpi_card("Buying Power", format_price(broker_account.get("buying_power")), small=True)
            with overview_cols[2]:
                render_kpi_card("Cash", format_price(broker_account.get("cash_balance")), small=True)
            with overview_cols[3]:
                render_kpi_card("Net Liq", format_price(broker_account.get("net_liquidation_value")), small=True)
            st.json(bot_status)

        with candidates_tab:
            try:
                candidate_rows = fetch_bot_rows("/candidates")
            except TraderAPIError as exc:
                st.error(str(exc))
                candidate_rows = []
            if candidate_rows:
                candidate_df = pd.DataFrame(candidate_rows)
                st.dataframe(candidate_df, use_container_width=True, hide_index=True)
                selected_candidate = st.selectbox("Candidate Detail", options=[row["candidate_id"] for row in candidate_rows], key="bot_candidate_detail")
                candidate_row = next((row for row in candidate_rows if row["candidate_id"] == selected_candidate), None)
                if candidate_row:
                    st.json(candidate_row)
                    action_cols = st.columns(3)
                    if action_cols[0].button("Preview Execution", key="bot_preview_candidate"):
                        st.session_state["bot_preview_result"] = bot_action("/execution/preview", {"candidate_id": selected_candidate})
                        st.rerun()
                    if action_cols[1].button("Reject Candidate", key="bot_reject_candidate"):
                        st.session_state["bot_action_result"] = bot_action(f"/candidates/{selected_candidate}/reject")
                        st.rerun()
            else:
                st.caption("No bot candidates yet. Refresh the watchlist or start the bot.")

        with proposals_tab:
            st.caption("Execution previews are persisted as trade proposals.")
            preview_result = st.session_state.get("bot_preview_result")
            if preview_result:
                eligible = bool(preview_result.get("eligible"))
                rejection_reasons = preview_result.get("rejection_reasons") or []
                if eligible:
                    st.success("This proposal passed the configured execution checks.")
                else:
                    st.error("This proposal cannot be submitted because it failed one or more execution checks.")
                    for reason in rejection_reasons:
                        st.markdown(f"- {reason}")

                target_prices = preview_result.get("target_prices") or []
                preview_cols = st.columns(6)
                with preview_cols[0]:
                    render_kpi_card("Entry", format_price(preview_result.get("entry_price")), small=True)
                with preview_cols[1]:
                    render_kpi_card("Stop", format_price(preview_result.get("stop_price")), small=True)
                with preview_cols[2]:
                    render_kpi_card("TP1", format_price(target_prices[0] if target_prices else None), small=True)
                with preview_cols[3]:
                    render_kpi_card("Quantity", preview_result.get("quantity") or 0, small=True)
                with preview_cols[4]:
                    render_kpi_card("Reward / Risk", preview_result.get("reward_to_risk") or "-", small=True)
                with preview_cols[5]:
                    render_kpi_card("Required R / R", preview_result.get("minimum_reward_to_risk") or "-", small=True)

                with st.expander("Execution Preview Details", expanded=False):
                    st.json(preview_result)
                proposal_id = preview_result.get("proposal_id")
                if proposal_id and eligible:
                    proposal_cols = st.columns(2)
                    if proposal_cols[0].button("Approve / Submit Proposal", key="bot_submit_proposal"):
                        st.session_state["bot_action_result"] = bot_action("/execution/submit", {"proposal_id": proposal_id})
                        st.rerun()
                    if proposal_cols[1].button("Approve Manual Proposal", key="bot_approve_proposal"):
                        st.session_state["bot_action_result"] = bot_action(f"/execution/{proposal_id}/approve")
                        st.rerun()
                elif proposal_id:
                    st.caption("Submission controls are hidden until a new preview passes every execution check.")
            else:
                st.caption("No previewed proposal in this session yet.")

        with orders_tab:
            try:
                order_rows = fetch_bot_rows("/broker/orders")
            except TraderAPIError as exc:
                st.error(str(exc))
                order_rows = []
            if order_rows:
                st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No open broker orders.")

        with positions_tab:
            try:
                position_rows = fetch_bot_rows("/broker/positions")
            except TraderAPIError as exc:
                st.error(str(exc))
                position_rows = []
            if position_rows:
                st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No open broker positions.")

        with journal_tab:
            try:
                journal_rows = fetch_bot_rows("/journal/trades")
            except TraderAPIError as exc:
                st.error(str(exc))
                journal_rows = []
            if journal_rows:
                st.dataframe(pd.DataFrame(journal_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No journaled trades yet.")

        with memory_tab:
            try:
                memory_rows = fetch_bot_rows("/memory/statistics")
            except TraderAPIError as exc:
                st.error(str(exc))
                memory_rows = []
            if st.button("Rebuild Memory", key="bot_rebuild_memory"):
                st.session_state["bot_action_result"] = bot_action("/memory/rebuild")
                st.rerun()
            if memory_rows:
                st.dataframe(pd.DataFrame(memory_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No memory statistics yet.")

        with safety_tab:
            try:
                risk_status = fetch_bot_payload("/risk/status")
                risk_exposure = fetch_bot_payload("/risk/exposure")
            except TraderAPIError as exc:
                st.error(str(exc))
                risk_status = {}
                risk_exposure = {}
            st.json(risk_status)
            st.json(risk_exposure)
            kill_cols = st.columns(2)
            if kill_cols[0].button("Activate Kill Switch", type="secondary", key="bot_kill_switch"):
                st.session_state["bot_action_result"] = bot_action("/risk/kill-switch", {"reason": "manual_dashboard_trigger"})
                st.rerun()
            if kill_cols[1].button("Reset Kill Switch", type="secondary", key="bot_kill_reset"):
                st.session_state["bot_action_result"] = bot_action("/risk/kill-switch/reset")
                st.rerun()
            if st.button("Flatten All Positions", type="primary", key="bot_flatten_all"):
                st.session_state["bot_action_result"] = bot_action("/execution/flatten")
                st.rerun()

with active_tab:
    st.markdown("### Overview")
    st.caption("Planner state comes from the latest non-expired watchlist snapshots. Live Price comes from runtime quote fetches through the trader API and falls back to the most recent close when a live quote is unavailable, such as when markets are closed. A separate runtime consistency layer checks whether the saved plan is still fresh, extended, stale, or invalidated at the current price.")
    if latest_snapshot_updated_at:
        now_ts = pd.Timestamp.now(tz="UTC")
        snapshot_ts = pd.to_datetime(latest_snapshot_updated_at, errors="coerce", utc=True)
        if pd.notna(snapshot_ts):
            age = now_ts - snapshot_ts
            if age.days >= 1:
                st.warning(
                    f"Active watchlist data is stale. Latest persisted watchlist snapshot is from {format_ts(latest_snapshot_updated_at)}. "
                    "SP500 daily-opportunity and SP100 workflow runs refresh the active dashboard. "
                    "If you ran either today, check the API service logs for Supabase persistence warnings."
                )
    if live_quote_error:
        st.warning(f"Live quotes are currently unavailable. Active views keep planner levels visible, but Live Price stays N/A. {live_quote_error}")
    elif not active_market_df.empty and not bool(active_market_df["live_price_available"].fillna(False).any()):
        st.info("Live quotes were requested, but none were available for the active ticker set. Active views are showing planner levels with Live Price marked N/A.")
    elif not active_market_df.empty and bool((active_market_df.get("live_price_source") == "recent_close_fallback").any()):
        st.info("Some active prices are using the most recent close because live quotes were unavailable for those tickers.")
    if latest_run_df.empty:
        st.info("No historical scan runs are stored yet. You can still use the Scanner tab to run workflows and populate the dashboard.")
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

    secondary_metrics = st.columns(3)
    with secondary_metrics[0]:
        render_kpi_card("Primary Watchlist", int(latest_run.get("primary_watchlist_count") or 0), small=True)
    with secondary_metrics[1]:
        render_kpi_card("Secondary Watchlist", int(latest_run.get("secondary_watchlist_count") or 0), small=True)
    with secondary_metrics[2]:
        render_kpi_card("Snapshot Updated", latest_data_ts or "-", small=True)

    st.markdown("### Top 5 Active Watch")
    st.caption("The highest-priority names from the active snapshot set, now reordered with live proximity and plan freshness in mind. Live Price comes from the API at dashboard runtime and may use the most recent close fallback when markets are closed.")
    top_watch_cols = st.columns(5)
    for idx, (_, row) in enumerate(active_market_df.head(5).iterrows()):
        with top_watch_cols[idx % 5]:
            render_top_watch_card(row)

    st.markdown("### Latest Watchlist")
    st.caption("These rows use the latest applicable non-expired planner snapshot per ticker, merged with live quotes at runtime. The saved plan stays fixed; the dashboard adds a separate live-consistency layer so you can see whether the original entry is still fresh, extended, stale, or invalidated.")
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

    filtered_active_df = filter_watchlist_df(
        active_market_df,
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
        "live_price",
        "live_price_asof",
        "entry_status",
        "tp1_status",
        "plan_freshness_status",
        "live_vs_plan_alignment",
        "replan_needed",
        "distance_to_entry_pct",
        "distance_to_stop_pct",
        "distance_to_tp1_pct",
        "preferred_entry",
        "stop_loss",
        "take_profit_1",
        "max_hold_date",
        "updated_at",
    ]
    display_watchlist_cols = [column for column in watchlist_table_cols if column in filtered_active_df.columns]
    st.dataframe(
        format_watchlist_display(filtered_active_df[display_watchlist_cols]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Ready Soon")
    st.caption("The most actionable WAIT setups from the active snapshot set, with live proximity to entry factored into the active ordering.")
    ready_df = filtered_active_df[
        (filtered_active_df["final_action"] == "WAIT")
        & (filtered_active_df["actionability_label"] == "ready_soon")
    ]
    if ready_df.empty:
        st.caption("No ready-soon WAIT names in the current filtered view.")
    else:
        ready_cols = st.columns(3)
        for idx, (_, row) in enumerate(ready_df.iterrows()):
            with ready_cols[idx % 3]:
                render_top_watch_card(row)

    st.markdown("### Selected Ticker")
    st.caption("Ticker detail merges the latest active planner snapshot with runtime live quotes. If a live quote is unavailable, it uses the most recent close fallback. Historical per-run results remain in the History tab.")
    available_tickers = filtered_active_df["ticker"].dropna().astype(str).tolist() or active_market_df["ticker"].dropna().astype(str).tolist()
    if not available_tickers:
        st.caption("No ticker snapshots available yet.")
    else:
        selected_ticker = st.selectbox("Ticker", options=available_tickers, index=0)
        detail_df = fetch_latest_ticker_snapshot(selected_ticker)
        snapshot_detail = active_market_df[active_market_df["ticker"] == selected_ticker]
        snapshot_row = snapshot_detail.iloc[0] if not snapshot_detail.empty else None
        detail_row = detail_df.iloc[0] if not detail_df.empty else None
        chart_execution_payload = _snapshot_payload(detail_row, "chart_execution_view")
        what_to_watch_payload = _snapshot_payload(detail_row, "what_to_watch")
        actionability_payload = _snapshot_payload(detail_row, "actionability_soon")
        raw_plan_payload = (detail_row.get("raw_result_json") if detail_row is not None else None) or {}

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
                        ("Live Price", format_price(snapshot_row.get("live_price")) if snapshot_row.get("live_price_available") else "N/A"),
                        ("Live Price As Of", format_ts(snapshot_row.get("live_price_asof")) if snapshot_row.get("live_price_available") else "Unavailable"),
                        ("Live Price Source", str(snapshot_row.get("live_price_source") or "unavailable").replace("_", " ").title()),
                        ("Entry Status", pretty_label(snapshot_row.get("entry_status"))),
                        ("TP1 Status", pretty_label(snapshot_row.get("tp1_status"))),
                        ("Plan Freshness", pretty_label(snapshot_row.get("plan_freshness_status"))),
                        ("Live Alignment", pretty_label(snapshot_row.get("live_vs_plan_alignment"))),
                        ("Replan Needed", "Yes" if snapshot_row.get("replan_needed") else "No"),
                        ("Distance to Entry", format_pct(snapshot_row.get("distance_to_entry_pct"))),
                        ("Distance to Stop", format_pct(snapshot_row.get("distance_to_stop_pct"))),
                        ("Distance to TP1", format_pct(snapshot_row.get("distance_to_tp1_pct"))),
                        ("Planner Snapshot Updated", format_ts(snapshot_row.get("updated_at"))),
                        ("Preferred Entry", format_price(snapshot_row.get("preferred_entry"))),
                        ("Stop Loss", format_price(snapshot_row.get("stop_loss"))),
                        ("Take Profit 1", format_price(snapshot_row.get("take_profit_1"))),
                        ("Max Hold Date", format_short_date(snapshot_row.get("max_hold_date"))),
                    ],
                    columns=3,
                )
                st.caption("Live Price above comes from the trader API at dashboard runtime and may fall back to the most recent close when live quotes are unavailable. Planner levels below remain fixed to the saved active snapshot and are not overwritten by live quotes.")
                st.caption(snapshot_row.get("live_consistency_summary") or "No live consistency summary available.")
                st.caption(summary_from_row(snapshot_row))

        with execution_tab:
            if detail_row is None:
                st.caption("No execution data available.")
            else:
                interval = st.radio(
                    "Chart Interval",
                    options=["D", "60", "30"],
                    format_func=lambda value: {"D": "Daily", "60": "1 Hour", "30": "30 Minute"}[value],
                    horizontal=True,
                    key=f"tradingview_interval_{selected_ticker}",
                )
                chart_col, analysis_col = st.columns([3, 2], gap="large")
                with chart_col:
                    render_tradingview_chart(
                        selected_ticker,
                        exchange=raw_plan_payload.get("exchange"),
                        interval=interval,
                        height=650,
                    )
                with analysis_col:
                    render_execution_scenario_panel(raw_plan_payload, snapshot_row)
                with st.expander("Legacy Execution View", expanded=False):
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
                chart_context_payload = raw_plan_payload.get("chart_context") or {}
                timeframe_payload = chart_context_payload.get("timeframes") or raw_plan_payload.get("timeframe_context") or {}
                render_raw_json_block("Raw Daily Context", timeframe_payload.get("daily"))
                render_raw_json_block("Raw 1H Context", timeframe_payload.get("hourly"))
                render_raw_json_block("Raw 30m Context", timeframe_payload.get("thirty_minute"))
                render_raw_json_block("Generated Execution Scenarios", raw_plan_payload.get("execution_scenarios"))
                render_raw_json_block(
                    "Scenario Selection",
                    {
                        "preferred_scenario": raw_plan_payload.get("preferred_scenario"),
                        "execution_action": raw_plan_payload.get("execution_action"),
                        "scenario_confidence": raw_plan_payload.get("execution_scenario_confidence"),
                        "why": (raw_plan_payload.get("llm_review") or {}).get("scenario_why"),
                    },
                )

with earnings_tab:
    st.markdown("### Earnings")
    st.caption("Upcoming earnings events from the trader API for the next 30 days, with local filters for watchlist monitoring.")

    earnings_top = st.columns([4, 1], gap="medium")
    with earnings_top[0]:
        st.caption("This tab stays read-only and uses the API calendar as the source of truth for dates and sessions.")
    with earnings_top[1]:
        earnings_sp100_only = st.toggle("SP100 Only", value=True, key="earnings_sp100_only")

    try:
        earnings_payload = _load_earnings_calendar_data(days_ahead=30, sp100_only=earnings_sp100_only)
    except TraderAPIError as exc:
        st.error(str(exc))
        earnings_payload = {}

    earnings_rows = earnings_payload.get("rows") or []
    earnings_df = pd.DataFrame(earnings_rows)
    if not earnings_df.empty:
        earnings_df["days_to_earnings"] = pd.to_numeric(earnings_df["days_to_earnings"], errors="coerce")
        earnings_df["earnings_date"] = earnings_df["earnings_date"].astype(str)

    active_watchlist_tickers = set(sorted_snapshots_df["ticker"].dropna().astype(str).tolist())
    earnings_filter_box = st.container(border=True)
    with earnings_filter_box:
        filter_cols = st.columns([1, 2, 2, 2, 1], gap="medium")
        earnings_horizon = filter_cols[0].selectbox(
            "Window",
            [7, 14, 30],
            index=2,
            format_func=lambda value: f"Next {value}d",
            key="earnings_horizon",
        )
        earnings_search = filter_cols[1].text_input(
            "Ticker Search",
            placeholder="Search ticker",
            key="earnings_search",
        )
        sector_options = ["All"]
        industry_options = ["All"]
        if not earnings_df.empty:
            sector_options += sorted([value for value in earnings_df["sector"].dropna().astype(str).unique().tolist() if value])
            industry_options += sorted([value for value in earnings_df["industry"].dropna().astype(str).unique().tolist() if value])
        earnings_sector = filter_cols[2].selectbox("Sector", sector_options, key="earnings_sector")
        earnings_industry = filter_cols[3].selectbox("Industry", industry_options, key="earnings_industry")
        earnings_watchlist_only = filter_cols[4].toggle("Watchlist Only", value=False, key="earnings_watchlist_only")

    filtered_earnings_df = _filter_earnings_df(
        earnings_df,
        max_days=int(earnings_horizon),
        ticker_search=earnings_search,
        sector_filter=earnings_sector,
        industry_filter=earnings_industry,
        watchlist_only=earnings_watchlist_only,
        watchlist_tickers=active_watchlist_tickers,
    )

    summary_cols = st.columns(4)
    next_7_count = 0 if filtered_earnings_df.empty else int((filtered_earnings_df["days_to_earnings"] <= 7).sum())
    next_window_count = len(filtered_earnings_df.index)
    before_open_count = 0 if filtered_earnings_df.empty else int((filtered_earnings_df["earnings_session"] == "before_open").sum())
    after_close_count = 0 if filtered_earnings_df.empty else int((filtered_earnings_df["earnings_session"] == "after_close").sum())
    with summary_cols[0]:
        render_kpi_card("Next 7 Days", next_7_count, small=True)
    with summary_cols[1]:
        render_kpi_card(f"Next {earnings_horizon} Days", next_window_count, small=True)
    with summary_cols[2]:
        render_kpi_card("Before Open", before_open_count, small=True)
    with summary_cols[3]:
        render_kpi_card("After Close", after_close_count, small=True)

    st.markdown("### Upcoming Earnings Calendar")
    if filtered_earnings_df.empty:
        st.caption("No earnings events match the current filters.")
    else:
        calendar_columns = [
            "ticker",
            "company_name",
            "earnings_date",
            "earnings_session",
            "days_to_earnings",
            "sector",
            "industry",
        ]
        calendar_df = filtered_earnings_df[[column for column in calendar_columns if column in filtered_earnings_df.columns]].copy()
        calendar_df["company_name"] = calendar_df["company_name"].fillna("-")
        calendar_df["earnings_session"] = calendar_df["earnings_session"].astype(str).str.replace("_", " ").str.title()
        st.dataframe(calendar_df, use_container_width=True, hide_index=True)

        st.markdown("### Earnings Detail")
        selected_earnings_ticker = st.selectbox(
            "Ticker Detail",
            options=filtered_earnings_df["ticker"].astype(str).tolist(),
            key="earnings_detail_ticker",
        )
        earnings_detail = filtered_earnings_df[filtered_earnings_df["ticker"].astype(str) == selected_earnings_ticker].iloc[0]
        detail_payload = {}
        detail_error = None
        try:
            detail_payload = _load_earnings_detail_data(selected_earnings_ticker, days_ahead=30)
        except TraderAPIError as exc:
            detail_error = str(exc)

        detail_source = detail_payload or earnings_detail.to_dict()

        render_key_value_grid(
            [
                ("Ticker", detail_source.get("ticker") or "-"),
                ("Company", detail_source.get("company_name") or "-"),
                ("Earnings Date", detail_source.get("earnings_date") or "-"),
                ("Session", str(detail_source.get("earnings_session") or "-").replace("_", " ").title()),
                ("Days To Earnings", detail_source.get("days_to_earnings")),
                ("Sector", detail_source.get("sector") or "-"),
                ("Industry", detail_source.get("industry") or "-"),
            ],
            columns=4,
        )
        if detail_error:
            st.caption(f"Historical reaction detail is temporarily unavailable: {detail_error}")
        st.markdown("**Earnings Reaction Context**")
        render_key_value_grid(
            [
                ("Avg Post-Earnings Move %", detail_source.get("avg_post_earnings_move_pct") if pd.notna(detail_source.get("avg_post_earnings_move_pct")) else "-"),
                ("Post-Earnings Up Rate", detail_source.get("post_earnings_up_rate") if pd.notna(detail_source.get("post_earnings_up_rate")) else "-"),
                ("Reaction Samples", detail_source.get("reaction_samples") if pd.notna(detail_source.get("reaction_samples")) else "-"),
                ("Avg Surprise %", detail_source.get("avg_surprise_percent") if pd.notna(detail_source.get("avg_surprise_percent")) else "-"),
                ("Earnings Risk Flag", "Yes" if detail_source.get("earnings_risk_flag") else "No"),
            ],
            columns=3,
        )

with history_tab:
    st.markdown("### History")
    st.caption("Past runs and archived per-run selections remain available here. This section reads from scan_runs and scan_ticker_results.")
    if run_history_df.empty:
        st.info("No historical workflow runs have been written to the reporting tables yet.")
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


