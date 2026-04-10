"""Best-effort reporting writes for workflow scan results into Supabase."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.orm import sessionmaker


logger = logging.getLogger(__name__)


def _normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


@lru_cache(maxsize=1)
def _get_supabase_components():
    db_url = os.getenv("SUPABASE_DATABASE_URL")
    if not db_url:
        return None

    engine = create_engine(_normalize_db_url(db_url), pool_pre_ping=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    metadata = MetaData(schema="public")
    scan_runs = Table("scan_runs", metadata, autoload_with=engine)
    scan_ticker_results = Table("scan_ticker_results", metadata, autoload_with=engine)
    watchlist_snapshots = Table("watchlist_snapshots", metadata, autoload_with=engine)
    return {
        "engine": engine,
        "session_factory": session_factory,
        "scan_runs": scan_runs,
        "scan_ticker_results": scan_ticker_results,
        "watchlist_snapshots": watchlist_snapshots,
    }


def _column_names(table: Table) -> set[str]:
    return {column.name for column in table.columns}


def _json_ready(value: Any) -> str:
    return json.dumps(value, default=str)


def _coerce_value(table: Table, column_name: str, value: Any) -> Any:
    if column_name not in _column_names(table):
        return None
    column = table.columns[column_name]
    type_name = column.type.__class__.__name__.lower()
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value if "json" in type_name else _json_ready(value)
    return value


def _filtered_values(table: Table, payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    cols = _column_names(table)
    for key, value in payload.items():
        if key in cols:
            values[key] = _coerce_value(table, key, value)
    return values


def _set_first_present(payload: dict[str, Any], table: Table, column_names: list[str], value: Any) -> None:
    cols = _column_names(table)
    for name in column_names:
        if name in cols:
            payload[name] = value
            return


def _extract_actionability(row) -> tuple[str | None, float | None]:
    actionability = getattr(row, "actionability_soon", None) or {}
    label = actionability.get("actionability_label")
    score = actionability.get("actionability_score")
    return (str(label) if label is not None else None, float(score) if score is not None else None)


def _extract_suitability(row) -> tuple[str | None, float | None]:
    suitability = getattr(row, "swing_trade_suitability", None) or {}
    label = suitability.get("suitability_label")
    score = suitability.get("suitability_score")
    return (str(label) if label is not None else None, float(score) if score is not None else None)


def save_supabase_scan_run(
    session,
    *,
    workflow_type: str,
    planned_at,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    workflow_response,
):
    components = _get_supabase_components()
    if not components:
        return None

    table = components["scan_runs"]
    payload = _filtered_values(
        table,
        {
            "workflow_type": workflow_type,
            "planned_at": planned_at,
            "market_regime": getattr(workflow_response, "market_regime", None),
            "top_scan": request_payload.get("top_scan"),
            "top_plan": request_payload.get("top_plan"),
            "pre_scan_shortlist": request_payload.get("pre_scan_shortlist"),
            "pre_scanned_count": getattr(workflow_response, "pre_scanned_count", None),
            "pre_scan_shortlist_count": getattr(workflow_response, "pre_scan_shortlist_count", None),
            "selected_count": getattr(workflow_response, "selected_count", None),
            "rows_logged": getattr(workflow_response, "rows_logged", None),
            "selection_message": getattr(workflow_response, "selection_message", None),
            "request_payload_json": request_payload,
            "response_json": response_payload,
        },
    )
    pk_columns = list(table.primary_key.columns)
    returning_column = pk_columns[0] if pk_columns else table.c.id
    result = session.execute(table.insert().returning(returning_column), payload)
    return result.scalar_one()


def save_supabase_ticker_results(session, *, scan_run_id, ranked_rows: list) -> None:
    components = _get_supabase_components()
    if not components or not ranked_rows:
        return

    table = components["scan_ticker_results"]
    values: list[dict[str, Any]] = []
    for ranked in ranked_rows:
        row = ranked.row
        actionability_label, actionability_score = _extract_actionability(row)
        suitability_label, suitability_score = _extract_suitability(row)
        raw_payload = {
            "ticker": row.ticker,
            "rank": ranked.rank,
            "final_action": row.final_action,
            "quant_action": row.quant_action,
            "llm_action": row.llm_action,
            "watchlist_tier": row.watchlist_tier,
            "watch_priority": row.watch_priority,
            "actionability_label": actionability_label,
            "actionability_score": actionability_score,
            "suitability_label": suitability_label,
            "suitability_score": suitability_score,
            "trend_state": row.trend_state,
            "preferred_entry": row.preferred_entry,
            "stop_loss": row.stop_loss,
            "take_profit_1": row.take_profit_1,
            "max_hold_date": row.max_hold_date,
            "pre_scan_score": row.pre_scan_score,
            "scanner_rank_score": row.scanner_rank_score,
            "immediate_rank_score": row.immediate_rank_score,
            "watchlist_rank_score": row.watchlist_rank_score,
            "sector_relative_strength": row.sector_relative_strength,
            "expected_return": row.expected_return,
            "prob_tp": row.prob_tp,
            "prob_sl": row.prob_sl,
            "chart_execution_view_json": row.chart_execution_view,
            "what_to_watch_json": getattr(row, "what_to_watch", None),
            "actionability_soon_json": getattr(row, "actionability_soon", None),
            "raw_result_json": row.model_dump(mode="json"),
        }
        _set_first_present(raw_payload, table, ["run_id", "scan_run_id", "source_run_id"], scan_run_id)
        payload = _filtered_values(table, raw_payload)
        values.append(payload)

    if values:
        session.execute(table.insert(), values)


def upsert_supabase_watchlist_snapshots(session, *, scan_run_id, ranked_rows: list, planned_at) -> None:
    components = _get_supabase_components()
    if not components or not ranked_rows:
        return

    table = components["watchlist_snapshots"]
    ticker_column = table.c.ticker
    rows_by_ticker = {ranked.row.ticker: ranked.row for ranked in ranked_rows}

    for ticker, row in rows_by_ticker.items():
        actionability_label, actionability_score = _extract_actionability(row)
        suitability_label, suitability_score = _extract_suitability(row)
        short_summary = None
        what_to_watch = getattr(row, "what_to_watch", None) or {}
        if what_to_watch.get("watch_summary_short"):
            short_summary = what_to_watch.get("watch_summary_short")
        elif getattr(row, "watchlist_summary", None):
            short_summary = getattr(row, "watchlist_summary", None)
        elif actionability_label:
            short_summary = (getattr(row, "actionability_soon", None) or {}).get("actionability_summary")

        payload = _filtered_values(
            table,
            {
                "ticker": ticker,
                "updated_at": planned_at,
                "source_run_id": scan_run_id,
                "final_action": row.final_action,
                "watchlist_tier": row.watchlist_tier,
                "watch_priority": row.watch_priority,
                "actionability_label": actionability_label,
                "actionability_score": actionability_score,
                "suitability_label": suitability_label,
                "suitability_score": suitability_score,
                "trend_state": row.trend_state,
                "preferred_entry": row.preferred_entry,
                "stop_loss": row.stop_loss,
                "take_profit_1": row.take_profit_1,
                "max_hold_date": row.max_hold_date,
                "short_summary": short_summary,
                "raw_result_json": row.model_dump(mode="json"),
            },
        )

        existing = session.execute(select(ticker_column).where(ticker_column == ticker)).first()
        if existing:
            session.execute(table.update().where(ticker_column == ticker).values(**payload))
        else:
            session.execute(table.insert().values(**payload))


def persist_sp100_workflow_to_supabase(
    *,
    workflow_request,
    workflow_response,
    selected_rows: list,
) -> None:
    try:
        components = _get_supabase_components()
        if not components:
            logger.warning("Supabase scan persistence skipped: SUPABASE_DATABASE_URL is not configured.")
            return

        session = components["session_factory"]()
        request_payload = workflow_request.model_dump(mode="json")
        response_payload = workflow_response.model_dump(mode="json")
        scan_run_id = save_supabase_scan_run(
            session,
            workflow_type="sp100_top10_log",
            planned_at=workflow_response.planned_at,
            request_payload=request_payload,
            response_payload=response_payload,
            workflow_response=workflow_response,
        )
        save_supabase_ticker_results(session, scan_run_id=scan_run_id, ranked_rows=selected_rows)
        upsert_supabase_watchlist_snapshots(
            session,
            scan_run_id=scan_run_id,
            ranked_rows=selected_rows,
            planned_at=workflow_response.planned_at,
        )
        session.commit()
    except Exception as exc:
        if "session" in locals():
            session.rollback()
        logger.warning("Supabase scan persistence failed for sp100 workflow: %s", exc)
    finally:
        if "session" in locals():
            session.close()
