"""Reusable UI components for the Streamlit trader dashboard."""

from __future__ import annotations

import html

import streamlit as st

from .utils import first_non_empty, format_pct, format_price, format_short_date, format_ts, safe_json


def _badge_class(kind: str, value) -> str:
    value_slug = str(value or "").strip().lower().replace("_", "-")
    if kind == "final_action":
        return {"buy": "buy", "wait": "wait", "avoid": "avoid"}.get(value_slug, "muted")
    if kind == "watchlist_tier":
        return {"primary": "primary", "secondary": "secondary"}.get(value_slug, "muted")
    if kind == "actionability":
        return {
            "ready-soon": "ready-soon",
            "actionable": "buy",
            "confirmed": "ready-soon",
            "awaiting-confirmation": "wait",
            "in-price-zone": "wait",
            "too-early": "background",
            "monitor": "monitor",
            "background": "background",
            "extended": "low",
            "missed": "avoid",
            "invalidated": "avoid",
        }.get(value_slug, "muted")
    if kind == "suitability":
        return {"high": "high", "medium": "medium", "low": "low", "unsuitable": "unsuitable"}.get(value_slug, "muted")
    if kind == "consistency":
        return {
            "fresh": "fresh",
            "live-but-extended": "live-but-extended",
            "partially-stale": "partially-stale",
            "stale-for-live-price": "stale-for-live-price",
            "invalidated": "invalidated",
            "aligned": "aligned",
            "continuation-extended": "continuation-extended",
            "rebound-already-moved": "rebound-already-moved",
            "entry-missed": "entry-missed",
            "near-invalidation": "near-invalidation",
            "target-already-hit": "target-already-hit",
            "needs-refresh": "needs-refresh",
            "below-entry-zone": "muted",
            "in-entry-zone": "fresh",
            "above-entry-zone": "continuation-extended",
            "extended-beyond-entry": "entry-missed",
            "below-tp1": "muted",
            "near-tp1": "partially-stale",
            "at-tp1": "target-already-hit",
            "above-tp1": "target-already-hit",
            "tp1-exceeded": "stale-for-live-price",
            "far-from-stop": "muted",
            "near-stop": "partially-stale",
            "at-risk-of-invalidation": "near-invalidation",
            "below-stop": "invalidated",
        }.get(value_slug, "muted")
    return "muted"


def pretty_label(value) -> str:
    if value in (None, ""):
        return "-"
    return str(value).replace("_", " ").title()


def badge_html(kind: str, value) -> str:
    label = pretty_label(value)
    css_class = _badge_class(kind, value)
    return f'<span class="badge {css_class}">{html.escape(label)}</span>'


def render_live_consistency_badges(row) -> str:
    values = []
    if row.get("plan_freshness_status"):
        values.append(badge_html("consistency", row.get("plan_freshness_status")))
    if row.get("entry_status"):
        values.append(badge_html("consistency", row.get("entry_status")))
    if row.get("tp1_status"):
        values.append(badge_html("consistency", row.get("tp1_status")))
    if row.get("replan_needed"):
        values.append(badge_html("consistency", "needs_refresh"))
    return "".join(values)


def render_header(*, latest_run_ts: str, latest_data_ts: str | None) -> None:
    st.title("Trader Watch Dashboard")
    st.markdown(
        f'<div class="dash-subtitle">Read-only view of the latest scan, watchlist, and run history from the Supabase reporting database.</div>'
        f'<div class="section-caption">Latest persisted watchlist run: {html.escape(latest_run_ts)}'
        + (f" | Active snapshots updated: {html.escape(latest_data_ts)}" if latest_data_ts else "")
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


def render_status_bar(items: list[tuple[str, object]]) -> None:
    parts = []
    for label, value in items:
        parts.append(
            f'<div class="runner-status-item"><span class="runner-status-label">{html.escape(str(label))}</span>'
            f'<span class="runner-status-value">{html.escape(str(value if value not in (None, "") else "-"))}</span></div>'
        )
    st.markdown(f'<div class="runner-status-bar">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_chip_list(values: list[str], *, empty_text: str = "None", variant: str = "default") -> None:
    clean_values = [str(value).strip() for value in values if str(value).strip()]
    if not clean_values:
        st.markdown(f'<div class="runner-empty-note">{html.escape(empty_text)}</div>', unsafe_allow_html=True)
        return
    css_class = "ticker-chip muted" if variant == "muted" else "ticker-chip"
    chips = "".join(f'<span class="{css_class}">{html.escape(value)}</span>' for value in clean_values)
    st.markdown(f'<div class="ticker-chip-grid">{chips}</div>', unsafe_allow_html=True)


def render_runner_bucket_panel(title: str, values: list[str], *, empty_text: str = "None") -> None:
    count = len([str(value).strip() for value in values if str(value).strip()])
    clean_values = [str(value).strip() for value in values if str(value).strip()]
    if clean_values:
        content = "".join(f'<span class="ticker-chip">{html.escape(value)}</span>' for value in clean_values)
        content = f'<div class="ticker-chip-grid">{content}</div>'
    else:
        content = f'<div class="runner-empty-note">{html.escape(empty_text)}</div>'

    st.markdown(
        f"""
        <div class="runner-bucket-inner">
            <div class="runner-bucket-title">{html.escape(title)} <span class="runner-bucket-count">{count}</span></div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_runner_note(text: str) -> None:
    st.markdown(f'<div class="runner-empty-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def _waiting_for_text(values: list[dict]) -> str:
    parts: list[str] = []
    for item in values or []:
        trigger_type = str(item.get("type") or "confirmation").replace("_", " ")
        value = item.get("value")
        if value is not None:
            parts.append(f"{trigger_type} {format_price(value)}")
        else:
            parts.append(trigger_type)
    return ", ".join(parts) if parts else "No pending trigger"


def render_daily_trade_card(item: dict) -> None:
    badges = "".join(
        [
            badge_html("final_action", item.get("action")),
            badge_html("actionability", item.get("actionability_state")),
        ]
    )
    st.markdown(
        f"""
        <div class="watch-card daily-opportunity-card">
            <div class="opportunity-rank">#{int(item.get('rank') or 0)}</div>
            <h4>{html.escape(str(item.get('ticker') or '-'))}</h4>
            <div class="badge-row">{badges}<span class="badge primary">{html.escape(str(item.get('grade') or '-'))}</span></div>
            <div class="mini-grid">
                <div><div class="mini-label">Trade Today</div><div class="mini-value">{_safe_display(item.get('trade_today_score'))}</div></div>
                <div><div class="mini-label">Raw Setup</div><div class="mini-value">{_safe_display(item.get('raw_setup_score'))}</div></div>
                <div><div class="mini-label">Portfolio Fit</div><div class="mini-value">{_safe_display(item.get('portfolio_fit_score'))}</div></div>
                <div><div class="mini-label">Trigger</div><div class="mini-value">{html.escape(format_price(item.get('confirmation_trigger')))}</div></div>
                <div><div class="mini-label">Stop</div><div class="mini-value">{html.escape(format_price(item.get('stop_loss')))}</div></div>
                <div><div class="mini-label">TP1 / TP2</div><div class="mini-value">{html.escape(format_price(item.get('take_profit_1')))} / {html.escape(format_price(item.get('take_profit_2')))}</div></div>
            </div>
            <div class="summary-note">{html.escape(pretty_label(item.get('setup_type')))} | {html.escape(str(item.get('sector') or 'Unknown'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_display(value) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def render_next_trigger_card(item: dict) -> None:
    waiting = _waiting_for_text(item.get("waiting_for") or [])
    st.markdown(
        f"""
        <div class="runner-bucket-inner next-trigger-card">
            <div class="runner-bucket-title">#{int(item.get('rank') or 0)} {html.escape(str(item.get('ticker') or '-'))}
                <span class="runner-bucket-count">{html.escape(str(item.get('grade') or '-'))}</span>
            </div>
            <div class="summary-note">Wait for {html.escape(waiting)}.</div>
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
    consistency_badges = render_live_consistency_badges(row)
    summary = summary_from_row(row)
    live_price = format_price(row.get("live_price")) if row.get("live_price_available") else "N/A"
    live_asof = format_ts(row.get("live_price_asof")) if row.get("live_price_available") else "Unavailable"
    live_summary = row.get("live_consistency_summary") or summary
    st.markdown(
        f"""
        <div class="watch-card">
            <h4>{html.escape(str(row.get("ticker") or "-"))}</h4>
            <div class="badge-row">{badges}</div>
            <div class="badge-row">{consistency_badges}</div>
            <div class="mini-grid">
                <div><div class="mini-label">Trend</div><div class="mini-value">{html.escape(pretty_label(row.get("trend_state")))}</div></div>
                <div><div class="mini-label">Live Price</div><div class="mini-value">{html.escape(live_price)}</div></div>
                <div><div class="mini-label">Preferred Entry</div><div class="mini-value">{html.escape(format_price(row.get("preferred_entry")))}</div></div>
                <div><div class="mini-label">Stop Loss</div><div class="mini-value">{html.escape(format_price(row.get("stop_loss")))}</div></div>
                <div><div class="mini-label">TP1 Target</div><div class="mini-value">{html.escape(format_price(row.get("take_profit_1")))}</div></div>
                <div><div class="mini-label">Dist to Entry</div><div class="mini-value">{html.escape(format_pct(row.get("distance_to_entry_pct")))}</div></div>
            </div>
            <div class="section-caption">Live price as of: {html.escape(live_asof)} | Planner snapshot: {html.escape(format_ts(row.get("updated_at")))}</div>
            <div class="summary-note">{html.escape(str(live_summary))}</div>
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
    consistency_badges = render_live_consistency_badges(snapshot_row)
    st.markdown(f'<div class="detail-chip-grid">{badges}</div>', unsafe_allow_html=True)
    if consistency_badges:
        st.markdown(f'<div class="detail-chip-grid">{consistency_badges}</div>', unsafe_allow_html=True)


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


def render_execution_scenario_panel(payload, snapshot_row=None) -> None:
    """Render the selected quant scenario without deriving any new levels."""
    data = safe_json(payload) or {}
    snapshot = safe_json(snapshot_row)
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    scenarios = safe_json(data.get("execution_scenarios")) or {}
    preferred = str(data.get("preferred_scenario") or "none")
    selected = scenarios.get(preferred) if isinstance(scenarios, dict) else None
    selected = selected if isinstance(selected, dict) else {}
    review = safe_json(data.get("llm_review")) or {}
    live_price = snapshot.get("live_price") if snapshot.get("live_price_available") else data.get("current_price")

    st.markdown(f"#### {html.escape(str(data.get('ticker') or snapshot.get('ticker') or '-'))} - {html.escape(pretty_label(data.get('execution_action')))}")
    render_key_value_grid(
        [
            ("Live / Latest Price", format_price(live_price)),
            ("Final Action", pretty_label(snapshot.get("final_action") or data.get("final_action"))),
            ("Execution Action", pretty_label(data.get("execution_action"))),
            ("Trend", pretty_label(data.get("trend_state"))),
            ("Price Location", pretty_label(data.get("price_location_context"))),
            ("Preferred Scenario", pretty_label(preferred)),
            ("Enter Now", pretty_label(review.get("enter_now_assessment"))),
            ("Scenario Confidence", format_pct((data.get("execution_scenario_confidence") or 0) * 100.0)),
        ],
        columns=2,
    )
    render_key_value_grid(
        [
            ("Scenario Entry Zone", zone_display(selected.get("entry_zone"))),
            ("Pullback Zone", zone_display(data.get("pullback_entry_zone"))),
            ("Breakout Trigger", zone_display(data.get("breakout_trigger_zone"))),
            ("Repair Trigger", zone_display(data.get("repair_trigger_zone"))),
            ("Stop Loss", format_price(selected.get("stop_loss") or data.get("stop_loss"))),
            ("TP1", format_price(selected.get("take_profit_1") or data.get("take_profit_1"))),
            ("TP2", format_price(selected.get("take_profit_2") or data.get("take_profit_2"))),
            ("Plan Freshness", pretty_label(snapshot.get("plan_freshness_status") or data.get("live_scenario_status"))),
        ],
        columns=2,
    )
    if data.get("scenario_selection_reason"):
        st.caption(data.get("scenario_selection_reason"))
    reasons = list(review.get("scenario_why") or selected.get("reasons") or [])
    if reasons:
        render_bullets("Why This Scenario", reasons)
    watch = safe_json(data.get("what_to_watch")) or {}
    if watch.get("watch_summary_short"):
        st.info(watch.get("watch_summary_short"))


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
    for column in ["live_price", "preferred_entry", "stop_loss", "take_profit_1", "snapshot_price"]:
        if column in out.columns:
            out[column] = out[column].apply(format_price)
    if "max_hold_date" in out.columns:
        out["max_hold_date"] = out["max_hold_date"].apply(format_short_date)
    if "updated_at" in out.columns:
        out["updated_at"] = out["updated_at"].apply(format_ts)
    if "live_price_asof" in out.columns:
        out["live_price_asof"] = out["live_price_asof"].apply(format_ts)
    if "snapshot_price_asof" in out.columns:
        out["snapshot_price_asof"] = out["snapshot_price_asof"].apply(format_ts)
    for column in ["distance_to_entry_pct", "distance_to_stop_pct", "distance_to_tp1_pct"]:
        if column in out.columns:
            out[column] = out[column].apply(format_pct)
    for column in ["entry_status", "tp1_status", "plan_freshness_status", "live_vs_plan_alignment"]:
        if column in out.columns:
            out[column] = out[column].apply(pretty_label)
    if "replan_needed" in out.columns:
        out["replan_needed"] = out["replan_needed"].apply(lambda value: "Yes" if value else "No")
    out = out.rename(
        columns={
            "ticker": "Ticker",
            "final_action": "Final Action",
            "watchlist_tier": "Watchlist Tier",
            "watch_priority": "Watch Priority",
            "actionability_label": "Actionability",
            "suitability_label": "Suitability",
            "trend_state": "Trend State",
            "live_price": "Live Price",
            "live_price_asof": "Live Price As Of",
            "entry_status": "Entry Status",
            "tp1_status": "TP1 Status",
            "plan_freshness_status": "Plan Freshness",
            "live_vs_plan_alignment": "Live Alignment",
            "replan_needed": "Replan Needed",
            "distance_to_entry_pct": "Dist to Entry %",
            "distance_to_stop_pct": "Dist to Stop %",
            "distance_to_tp1_pct": "Dist to TP1 %",
            "preferred_entry": "Preferred Entry",
            "stop_loss": "Stop Loss",
            "take_profit_1": "TP1",
            "max_hold_date": "Max Hold Date",
            "updated_at": "Planner Snapshot Updated",
            "snapshot_price": "Snapshot Price",
            "snapshot_price_asof": "Snapshot Price As Of",
        }
    )
    return out


def format_run_history_display(df):
    if df.empty:
        return df
    out = df.copy()
    if "created_at" in out.columns:
        out["created_at"] = out["created_at"].apply(format_ts)
    return out

