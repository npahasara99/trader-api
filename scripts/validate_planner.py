from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from app.db import SessionLocal
from app.logic import build_swing_plan, detect_market_regime, get_sp100_universe
from app.market_data import get_bars as get_timeframe_bars
from app.main import (
    _apply_prob_and_action,
    _build_daily_bars_loader,
    _build_daily_closes_loader,
    _compute_dynamic_thresholds,
    _history_stats_by_ticker,
    _rank_pre_scan_universe,
    _rolling_performance_snapshot,
)

try:
    from dashboard.db import get_engine as get_supabase_engine
except Exception:  # pragma: no cover - optional for local baskets without Supabase
    get_supabase_engine = None


DETAIL_COLUMNS = [
    "ticker",
    "basket_name",
    "planned_at",
    "market_regime",
    "current_price",
    "price_location_context",
    "setup_type",
    "setup_scenario",
    "preferred_trade_shape",
    "preferred_scenario",
    "execution_action",
    "available_timeframes",
    "continuation_vs_reversion_bias",
    "news_regime_alignment",
    "preferred_entry",
    "stop_loss",
    "take_profit_1",
    "final_action",
    "actionability_label",
    "suitability_label",
    "stop_width_pct",
    "tp1_distance_pct",
    "swing_realism_flag",
    "hold_window_reachability_score",
    "macro_alignment_score",
    "chart_news_alignment",
    "tp_aggressiveness",
    "sl_tolerance",
    "expected_move_profile",
    "scenario_confidence",
    "planner_status",
    "scan_rejection_reason",
    "strategy_reason",
]

COMPARE_COLUMNS = [
    "price_location_context",
    "setup_type",
    "setup_scenario",
    "preferred_trade_shape",
    "preferred_scenario",
    "execution_action",
    "continuation_vs_reversion_bias",
    "news_regime_alignment",
    "preferred_entry",
    "stop_loss",
    "take_profit_1",
    "final_action",
    "actionability_label",
    "suitability_label",
    "stop_width_pct",
    "tp1_distance_pct",
    "swing_realism_flag",
    "hold_window_reachability_score",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the upgraded trading planner over a validation basket and export "
            "detail/summary CSVs for manual review."
        )
    )
    parser.add_argument("--tickers", help="Comma-separated ticker list, e.g. LIN,AMD,NVDA")
    parser.add_argument("--tickers-file", help="Text file containing tickers separated by commas, spaces, or newlines")
    parser.add_argument("--sector", help="SP100 sector filter, e.g. technology")
    parser.add_argument("--industry", help="SP100 industry filter, e.g. semiconductors")
    parser.add_argument("--top-n", type=int, default=15, help="Limit for sector/industry baskets")
    parser.add_argument(
        "--top-watchlist",
        type=int,
        help="Pull the latest active watchlist names from Supabase watchlist_snapshots",
    )
    parser.add_argument(
        "--watchlist-actions",
        default="BUY,WAIT",
        help="Comma-separated final_action filter for --top-watchlist (default: BUY,WAIT)",
    )
    parser.add_argument("--llm-provider", default="chatgpt-actions")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-style", default="validation_harness_v1")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "validation_outputs"),
        help="Directory for generated CSV/JSON outputs",
    )
    parser.add_argument(
        "--output-prefix",
        default="planner_validation",
        help="Filename prefix for generated outputs",
    )
    parser.add_argument(
        "--compare-csv",
        help="Optional prior detail CSV to compare against current output for before/after review",
    )
    return parser.parse_args()


def _split_tickers(raw: str) -> list[str]:
    cleaned = raw.replace("\n", ",").replace("\r", ",").replace("\t", ",").replace(" ", ",")
    return [token.strip().upper() for token in cleaned.split(",") if token.strip()]


def _load_tickers_from_file(path: str) -> list[str]:
    return _split_tickers(Path(path).read_text(encoding="utf-8"))


def _fetch_top_watchlist(limit: int, *, allowed_actions: list[str]) -> list[str]:
    if get_supabase_engine is None:
        raise RuntimeError("Supabase dashboard DB helper is unavailable in this environment.")

    sql = text(
        """
        with ranked_snapshots as (
            select
                ticker,
                final_action,
                actionability_label,
                actionability_score,
                watchlist_tier,
                watch_priority,
                updated_at,
                row_number() over (
                    partition by ticker
                    order by updated_at desc nulls last, source_run_id desc nulls last
                ) as snapshot_rank
            from public.watchlist_snapshots
            where max_hold_date is null or max_hold_date >= now()
        )
        select
            ticker
        from ranked_snapshots
        where snapshot_rank = 1
          and final_action = any(:actions)
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
            case watch_priority
                when 'high' then 0
                when 'medium' then 1
                when 'low' then 2
                else 3
            end asc,
            ticker asc
        limit :limit
        """
    )

    engine = get_supabase_engine()
    with engine.connect() as connection:
        df = pd.read_sql_query(sql, connection, params={"limit": int(limit), "actions": allowed_actions})
    return [str(value).strip().upper() for value in df["ticker"].tolist() if str(value).strip()]


def _resolve_basket(args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.tickers:
        tickers = _split_tickers(args.tickers)
        return "manual", tickers
    if args.tickers_file:
        tickers = _load_tickers_from_file(args.tickers_file)
        return "manual_file", tickers
    if args.top_watchlist:
        actions = [item.strip().upper() for item in str(args.watchlist_actions).split(",") if item.strip()]
        tickers = _fetch_top_watchlist(int(args.top_watchlist), allowed_actions=actions)
        return "supabase_active_watchlist", tickers
    if args.sector or args.industry:
        tickers = get_sp100_universe(None, sector=args.sector, industry=args.industry)
        if args.top_n:
            tickers = tickers[: max(1, int(args.top_n))]
        return "sp100_filtered", tickers
    raise SystemExit(
        "No validation basket selected. Use --tickers, --tickers-file, --top-watchlist, "
        "or --sector/--industry."
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    if isinstance(row, dict):
        return dict(row)
    return {key: getattr(row, key) for key in dir(row) if not key.startswith("_")}


def _planner_status(row_data: dict[str, Any]) -> str:
    if row_data.get("scan_rejection_reason") == "planner_crashed":
        return "planner_crashed"
    if row_data.get("preferred_entry") is None or row_data.get("stop_loss") is None or row_data.get("take_profit_1") is None:
        return "incomplete_levels"
    return "ok"


def _extract_detail_record(
    *,
    basket_name: str,
    planned_at: str,
    market_regime: str | None,
    row: Any,
) -> dict[str, Any]:
    row_data = _row_to_dict(row)
    actionability = row_data.get("actionability_soon") or {}
    suitability = row_data.get("swing_trade_suitability") or {}
    record = {
        "ticker": row_data.get("ticker"),
        "basket_name": basket_name,
        "planned_at": planned_at,
        "market_regime": market_regime,
        "current_price": row_data.get("current_price"),
        "price_location_context": row_data.get("price_location_context"),
        "setup_type": row_data.get("setup_type"),
        "setup_scenario": row_data.get("setup_scenario"),
        "preferred_trade_shape": row_data.get("preferred_trade_shape"),
        "preferred_scenario": row_data.get("preferred_scenario"),
        "execution_action": row_data.get("execution_action"),
        "available_timeframes": ",".join(((row_data.get("chart_context") or {}).get("available_timeframes") or [])),
        "continuation_vs_reversion_bias": row_data.get("continuation_vs_reversion_bias"),
        "news_regime_alignment": row_data.get("news_regime_alignment"),
        "preferred_entry": row_data.get("preferred_entry"),
        "stop_loss": row_data.get("stop_loss"),
        "take_profit_1": row_data.get("take_profit_1"),
        "final_action": row_data.get("final_action"),
        "actionability_label": actionability.get("actionability_label"),
        "suitability_label": suitability.get("suitability_label"),
        "stop_width_pct": row_data.get("stop_width_pct"),
        "tp1_distance_pct": row_data.get("tp1_distance_pct"),
        "swing_realism_flag": row_data.get("swing_realism_flag"),
        "hold_window_reachability_score": row_data.get("hold_window_reachability_score"),
        "macro_alignment_score": row_data.get("macro_alignment_score"),
        "chart_news_alignment": row_data.get("chart_news_alignment"),
        "tp_aggressiveness": row_data.get("tp_aggressiveness"),
        "sl_tolerance": row_data.get("sl_tolerance"),
        "expected_move_profile": row_data.get("expected_move_profile"),
        "scenario_confidence": row_data.get("scenario_confidence"),
        "planner_status": _planner_status(row_data),
        "scan_rejection_reason": row_data.get("scan_rejection_reason"),
        "strategy_reason": row_data.get("strategy_reason"),
    }
    return record


def _build_summary(detail_df: pd.DataFrame, *, basket_name: str, planned_at: str, market_regime: str | None) -> dict[str, Any]:
    usable = detail_df.copy()
    numeric_stop = pd.to_numeric(usable["stop_width_pct"], errors="coerce")
    numeric_tp = pd.to_numeric(usable["tp1_distance_pct"], errors="coerce")
    numeric_reachability = pd.to_numeric(usable["hold_window_reachability_score"], errors="coerce")

    realism_counts = usable["swing_realism_flag"].fillna("missing").value_counts().to_dict()
    action_counts = usable["final_action"].fillna("missing").value_counts().to_dict()
    scenario_counts = usable["setup_scenario"].fillna("missing").value_counts().to_dict()
    planner_status_counts = usable["planner_status"].fillna("missing").value_counts().to_dict()

    summary = {
        "basket_name": basket_name,
        "planned_at": planned_at,
        "market_regime": market_regime,
        "ticker_count": int(len(usable)),
        "continuation_favored_count": int((usable["continuation_vs_reversion_bias"] == "continuation_favored").sum()),
        "rebound_candidate_count": int((usable["continuation_vs_reversion_bias"] == "rebound_candidate").sum()),
        "repair_count": int(usable["setup_type"].isin(["repair_after_breakdown", "deep_rebound_attempt"]).sum()),
        "buy_count": int((usable["final_action"] == "BUY").sum()),
        "wait_count": int((usable["final_action"] == "WAIT").sum()),
        "avoid_count": int((usable["final_action"] == "AVOID").sum()),
        "planner_crashed_count": int((usable["planner_status"] == "planner_crashed").sum()),
        "avg_stop_width_pct": None if numeric_stop.dropna().empty else round(float(numeric_stop.dropna().mean()), 4),
        "avg_tp1_distance_pct": None if numeric_tp.dropna().empty else round(float(numeric_tp.dropna().mean()), 4),
        "avg_hold_window_reachability_score": None if numeric_reachability.dropna().empty else round(float(numeric_reachability.dropna().mean()), 4),
        "realism_flag_counts": realism_counts,
        "action_counts": action_counts,
        "setup_scenario_counts": scenario_counts,
        "planner_status_counts": planner_status_counts,
    }
    return summary


def _summary_to_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in [
        "basket_name",
        "planned_at",
        "market_regime",
        "ticker_count",
        "continuation_favored_count",
        "rebound_candidate_count",
        "repair_count",
        "buy_count",
        "wait_count",
        "avoid_count",
        "planner_crashed_count",
        "avg_stop_width_pct",
        "avg_tp1_distance_pct",
        "avg_hold_window_reachability_score",
    ]:
        rows.append({"metric": key, "value": summary.get(key)})

    for label, count in sorted((summary.get("realism_flag_counts") or {}).items()):
        rows.append({"metric": f"realism_flag::{label}", "value": count})
    for label, count in sorted((summary.get("action_counts") or {}).items()):
        rows.append({"metric": f"action::{label}", "value": count})
    for label, count in sorted((summary.get("setup_scenario_counts") or {}).items()):
        rows.append({"metric": f"setup_scenario::{label}", "value": count})
    for label, count in sorted((summary.get("planner_status_counts") or {}).items()):
        rows.append({"metric": f"planner_status::{label}", "value": count})
    return pd.DataFrame(rows)


def _build_comparison(current_df: pd.DataFrame, baseline_csv: str) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline_csv)
    current_subset = current_df[["ticker"] + COMPARE_COLUMNS].copy()
    baseline_subset = baseline_df[["ticker"] + [col for col in COMPARE_COLUMNS if col in baseline_df.columns]].copy()
    baseline_subset = baseline_subset.rename(columns={col: f"{col}_baseline" for col in baseline_subset.columns if col != "ticker"})
    comparison = current_subset.merge(baseline_subset, on="ticker", how="left")

    for column in COMPARE_COLUMNS:
        baseline_col = f"{column}_baseline"
        if baseline_col not in comparison.columns:
            continue
        current_col = comparison[column]
        baseline_vals = comparison[baseline_col]
        changed_col = f"{column}_changed"
        comparison[changed_col] = current_col.fillna("").astype(str) != baseline_vals.fillna("").astype(str)
    return comparison


def _run_validation(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    basket_name, tickers = _resolve_basket(args)
    if not tickers:
        raise SystemExit("Resolved basket is empty. Adjust the filters or source and try again.")

    session = SessionLocal()
    try:
        daily_closes_loader = _build_daily_closes_loader(session)
        daily_bars_loader = _build_daily_bars_loader(session)
        regime_snapshot = detect_market_regime(tickers, daily_closes_loader=daily_closes_loader)

        try:
            perf = _rolling_performance_snapshot(session, lookback_days=180)
        except SQLAlchemyError:
            perf = {
                "overall_samples": 0,
                "overall_avg_return": 0.0,
                "overall_abs_return": 0.0,
                "overall_win_rate": 0.0,
                "buy_samples": 0,
                "buy_avg_return": 0.0,
                "buy_win_rate": 0.0,
            }
            session.rollback()

        thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)

        try:
            ticker_hist = _history_stats_by_ticker(session, lookback_days=180)
        except SQLAlchemyError:
            ticker_hist = {}
            session.rollback()

        ranked_prescan = _rank_pre_scan_universe(
            tickers,
            daily_closes_loader=daily_closes_loader,
            daily_bars_loader=daily_bars_loader,
            timeframe_bars_loader=get_timeframe_bars,
        )
        pre_scan_by_ticker = {
            item["ticker"]: {
                **item,
                "scan_shortlisted": True,
                "scan_rejection_reason": None,
            }
            for item in ranked_prescan
        }

        rows = build_swing_plan(
            tickers,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            daily_closes_loader=daily_closes_loader,
            daily_bars_loader=daily_bars_loader,
            history_stats_by_ticker=ticker_hist,
            pre_scan_by_ticker=pre_scan_by_ticker,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_style=args.llm_style,
        )

        for row in rows:
            history = ticker_hist.get(row.ticker, {})
            _apply_prob_and_action(
                row,
                regime=regime_snapshot["regime"],
                buy_threshold=thresholds["buy_threshold"],
                avoid_threshold=thresholds["avoid_threshold"],
                history_win_rate=(float(history["win_rate"]) if "win_rate" in history else None),
                history_samples=int(history.get("samples", 0)),
            )

        response = {
            "planned_at": pd.Timestamp.utcnow().to_pydatetime(),
            "market_regime": regime_snapshot["regime"],
            "regime_score": regime_snapshot["score"],
            "buy_threshold": thresholds["buy_threshold"],
            "avoid_threshold": thresholds["avoid_threshold"],
            "rows": rows,
        }
    finally:
        session.close()

    planned_at = response["planned_at"].isoformat() if hasattr(response["planned_at"], "isoformat") else str(response["planned_at"])
    market_regime = response.get("market_regime")
    rows = response.get("rows") or []
    detail_records = [
        _extract_detail_record(
            basket_name=basket_name,
            planned_at=planned_at,
            market_regime=market_regime,
            row=row,
        )
        for row in rows
    ]
    detail_df = pd.DataFrame(detail_records)
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    else:
        ordered = [col for col in DETAIL_COLUMNS if col in detail_df.columns]
        detail_df = detail_df[ordered + [col for col in detail_df.columns if col not in ordered]]

    summary = _build_summary(detail_df, basket_name=basket_name, planned_at=planned_at, market_regime=market_regime)
    return detail_df, summary, {
        "basket_name": basket_name,
        "tickers": tickers,
        "planned_at": planned_at,
        "market_regime": market_regime,
        "raw_response": response,
    }


def main() -> None:
    args = _parse_args()
    detail_df, summary, meta = _run_validation(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.output_prefix}_{meta['basket_name']}_{stamp}"

    detail_path = output_dir / f"{prefix}_details.csv"
    summary_csv_path = output_dir / f"{prefix}_summary.csv"
    summary_json_path = output_dir / f"{prefix}_summary.json"
    detail_df.to_csv(detail_path, index=False)
    _summary_to_frame(summary).to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comparison_path = None
    if args.compare_csv:
        comparison_df = _build_comparison(detail_df, args.compare_csv)
        comparison_path = output_dir / f"{prefix}_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

    print(f"Validation basket: {meta['basket_name']}")
    print(f"Tickers run: {len(meta['tickers'])}")
    print(f"Market regime: {meta['market_regime']}")
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    if comparison_path is not None:
        print(f"Comparison CSV: {comparison_path}")

    print("\nSummary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
