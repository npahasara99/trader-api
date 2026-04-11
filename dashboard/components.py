"""Reusable UI components for the Streamlit trader dashboard."""

from __future__ import annotations

import html

import streamlit as st

from .utils import first_non_empty, format_price, format_short_date, format_ts, safe_json


def _badge_class(kind: str, value) -> str:
    value_slug = str(value or "").strip().lower().replace("_", "-")
    if kind == "final_action":
        return {"buy": "buy", "wait": "wait", "avoid": "avoid"}.get(value_slug, "muted")
    if kind == "watchlist_tier":
        return {"primary": "primary", "secondary": "secondary"}.get(value_slug, "muted")
    if kind == "actionability":
        return {"ready-soon": "ready-soon", "monitor": "monitor", "background": "background"}.get(value_slug, "muted")
    if kind == "suitability":
        return {"high": "high", "medium": "medium", "low": "low", "unsuitable": "unsuitable"}.get(value_slug, "muted")
    return "muted"


def pretty_label(value) -> str:
    if value in (None, ""):
        return "-"
    return str(value).replace("_", " ").title()


def badge_html(kind: str, value) -> str:
    label = pretty_label(value)
    css_class = _badge_class(kind, value)
    return f'<span class="badge {css_class}">{html.escape(label)}</span>'


def render_header(*, latest_run_ts: str, latest_data_ts: str | None) -> None:
    st.title("Trader Watch Dashboard")
    st.markdown(
        f'<div class="dash-subtitle">Read-only view of the latest scan, watchlist, and run history from the Supabase reporting database.</div>'
        f'<div class="section-caption">Latest run: {html.escape(latest_run_ts)}'
        + (f" | Latest data: {html.escape(latest_data_ts)}" if latest_data_ts else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value, *, small: bool = False) -> None:
    size_class = " small" if small else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value{size_class}">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_from_row(row) -> str:
    raw = safe_json(row.get("raw_result_json"))
    what_to_watch = (raw or {}).get("what_to_watch") or {}
    actionability = (raw or {}).get("actionability_soon") or {}
    return first_non_empty(
        row.get("short_summary"),
        what_to_watch.get("watch_summary_short"),
        actionability.get("actionability_summary"),
        (raw or {}).get("watchlist_summary"),
    ) or "No short summary available."


def render_top_watch_card(row) -> None:
    badges = "".join(
        [
            badge_html("final_action", row.get("final_action")),
            badge_html("watchlist_tier", row.get("watchlist_tier")),
            badge_html("actionability", row.get("actionability_label")),
        ]
    )
    summary = summary_from_row(row)
    st.markdown(
        f"""
        <div class="watch-card">
            <h4>{html.escape(str(row.get("ticker") or "-"))}</h4>
            <div class="badge-row">{badges}</div>
            <div class="mini-grid">
                <div><div class="mini-label">Trend</div><div class="mini-value">{html.escape(pretty_label(row.get("trend_state")))}</div></div>
                <div><div class="mini-label">Entry</div><div class="mini-value">{html.escape(format_price(row.get("preferred_entry")))}</div></div>
                <div><div class="mini-label">Stop</div><div class="mini-value">{html.escape(format_price(row.get("stop_loss")))}</div></div>
                <div><div class="mini-label">TP1</div><div class="mini-value">{html.escape(format_price(row.get("take_profit_1")))}</div></div>
            </div>
            <div class="summary-note">{html.escape(summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge_row(snapshot_row) -> None:
    badges = "".join(
        [
            badge_html("final_action", snapshot_row.get("final_action")),
            badge_html("watchlist_tier", snapshot_row.get("watchlist_tier")),
            badge_html("actionability", snapshot_row.get("actionability_label")),
            badge_html("suitability", snapshot_row.get("suitability_label")),
        ]
    )
    st.markdown(f'<div class="detail-chip-grid">{badges}</div>', unsafe_allow_html=True)


def render_key_value_grid(items: list[tuple[str, object]], *, columns: int = 4) -> None:
    cols = st.columns(columns)
    for idx, (label, value) in enumerate(items):
        with cols[idx % columns]:
            st.caption(label)
            st.markdown(f"**{value if value not in (None, '') else '-'}**")


def render_bullets(title: str, lines: list[str]) -> None:
    st.markdown(f"**{title}**")
    useful = [line for line in lines if line]
    if not useful:
        st.caption("No data available.")
        return
    for line in useful:
        st.write(f"- {line}")


def zone_display(zone) -> str:
    zone = safe_json(zone)
    if not isinstance(zone, dict):
        return "-"
    return str(zone.get("display") or "-")


def render_chart_execution_view(payload) -> None:
    data = safe_json(payload) or {}
    render_key_value_grid(
        [
            ("Trade Shape", pretty_label(data.get("trade_shape"))),
            ("Enter Now", pretty_label(data.get("enter_now"))),
            ("Price Location", pretty_label(data.get("current_price_location"))),
            ("Execution Bias", pretty_label(data.get("execution_bias"))),
        ],
        columns=4,
    )
    render_key_value_grid(
        [
            ("Current Execution Anchor", zone_display(data.get("current_execution_anchor"))),
            ("Breakout Trigger", zone_display(data.get("breakout_point"))),
            ("Pullback Zone", zone_display(data.get("pullback_entry_zone"))),
            ("Deeper Pullback", zone_display(data.get("deeper_pullback_zone"))),
        ],
        columns=4,
    )
    if data.get("chart_execution_summary"):
        st.caption(data.get("chart_execution_summary"))


def render_what_to_watch(payload) -> None:
    data = safe_json(payload) or {}
    render_key_value_grid(
        [
            ("Bullish Hold", zone_display(data.get("bullish_hold_zone"))),
            ("Deeper Reset Trigger", zone_display(data.get("deeper_reset_trigger_zone"))),
            ("Deeper Reset Target", zone_display(data.get("deeper_reset_target_zone"))),
            ("Continuation Trigger", zone_display(data.get("continuation_trigger_zone"))),
        ],
        columns=4,
    )
    render_bullets(
        "Watch Conditions",
        [
            data.get("bullish_hold_reason"),
            data.get("deeper_reset_reason"),
            data.get("continuation_reason"),
        ],
    )
    if data.get("watch_summary_short"):
        st.caption(data.get("watch_summary_short"))


def render_actionability(payload) -> None:
    data = safe_json(payload) or {}
    render_key_value_grid(
        [
            ("Label", pretty_label(data.get("actionability_label"))),
            ("Score", data.get("actionability_score")),
            ("Active Watch", "Yes" if data.get("active_watch") else "No"),
            ("Urgency", pretty_label(data.get("watch_urgency"))),
            ("Days To Action", data.get("days_to_action_estimate")),
            ("Closest Trigger", pretty_label(data.get("closest_trigger_type"))),
        ],
        columns=3,
    )
    render_bullets("Why It Matters", list(data.get("key_reasons") or []))
    if data.get("not_ready_reasons"):
        render_bullets("What Is Still Missing", list(data.get("not_ready_reasons") or []))
    if data.get("actionability_summary"):
        st.caption(data.get("actionability_summary"))


def render_raw_json_block(title: str, payload) -> None:
    with st.expander(title, expanded=False):
        if payload in (None, "", {}):
            st.caption("No data available.")
        else:
            st.json(payload)


def format_watchlist_display(df):
    if df.empty:
        return df
    out = df.copy()
    for column in ["preferred_entry", "stop_loss", "take_profit_1"]:
        if column in out.columns:
            out[column] = out[column].apply(format_price)
    if "max_hold_date" in out.columns:
        out["max_hold_date"] = out["max_hold_date"].apply(format_short_date)
    return out


def format_run_history_display(df):
    if df.empty:
        return df
    out = df.copy()
    if "created_at" in out.columns:
        out["created_at"] = out["created_at"].apply(format_ts)
    return out
