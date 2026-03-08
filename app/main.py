
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta, date
from .logic import bucket_news, classify_assumption
import json
import os

from sqlalchemy.orm import Session
from sqlalchemy import text, func

from .db import Base, engine, get_db
from .models import SwingDecision, DailyBar
from .logic import (
    scan_swing_candidates_largecaps,
    build_swing_plan,
    get_last_price,
    evaluate_plan_row,
    get_sp100_universe,
    detect_market_regime,
    estimate_trade_probabilities,
)
from .market_data import (
    ensure_cached_daily_closes,
    backfill_universe_daily_bars,
    fetch_finnhub_daily_bars_with_meta,
)


def _ensure_runtime_columns() -> None:
    required_cols = {
        "news_score": "INTEGER",
        "news_json": "TEXT",
        "earnings_score": "INTEGER",
        "earnings_context_json": "TEXT",
    }
    try:
        with engine.begin() as conn:
            dialect = conn.dialect.name
            if dialect == "sqlite":
                existing = {
                    row[1]
                    for row in conn.execute(text("PRAGMA table_info(swing_decisions)")).fetchall()
                }
                for col, col_type in required_cols.items():
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE swing_decisions ADD COLUMN {col} {col_type}"))
                return

            for col, col_type in required_cols.items():
                conn.execute(text(f"ALTER TABLE swing_decisions ADD COLUMN IF NOT EXISTS {col} {col_type}"))
    except Exception:
        # Do not block startup if migration cannot be applied here.
        pass


# Create tables + best-effort additive columns
Base.metadata.create_all(bind=engine)
_ensure_runtime_columns()

app = FastAPI(
    title="Trader Backend (Stocks Only)",
    version="0.1.3",
    servers=[
        {"url": "https://trader-api-production-7875.up.railway.app", "description": "Production"}
    ],
)


def require_bearer_token(authorization: Optional[str] = Header(default=None)):
    expected = os.getenv("API_BEARER_TOKEN")
    # If you haven't set a token, don't block (useful for local dev).
    if not expected:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


# --- Requests/Responses ---
class NewsItem(BaseModel):
    headline: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    datetime: Optional[str] = None
    url: Optional[str] = None


class ScanRequest(BaseModel):
    universe: List[str]
    top_n: int = 8


class ScanResponse(BaseModel):
    tickers: List[str]


class PlanRequest(BaseModel):
    tickers: List[str]
    mode: str = "manual"  # manual/scan
    llm_used: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_style: Optional[str] = None


class PlanRowOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    last: Optional[float] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_date: Optional[datetime] = None

    strategy_action: Optional[str] = None
    strategy_reason: Optional[str] = None

    news: Optional[List[NewsItem]] = None
    news_score: int = 0
    earnings_score: int = 0
    earnings_context: Optional[dict] = None

    signal_score: int = 0
    market_regime: Optional[str] = None
    prob_tp: Optional[float] = None
    prob_sl: Optional[float] = None
    prob_open: Optional[float] = None
    expected_return: Optional[float] = None
    confidence: Optional[float] = None
    buy_threshold: Optional[int] = None
    avoid_threshold: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    hold_days: Optional[int] = None
    risk_tuning_reason: Optional[str] = None

    llm_action: Optional[str] = None
    llm_rationale: Optional[str] = None


class PlanResponse(BaseModel):
    planned_at: datetime
    market_regime: Optional[str] = None
    regime_score: Optional[float] = None
    buy_threshold: Optional[int] = None
    avoid_threshold: Optional[int] = None
    rows: List[PlanRowOut]


class LogRequest(BaseModel):
    planned_at: datetime
    mode: str = "manual"
    rows: List[PlanRowOut]
    meta: dict = Field(default_factory=dict)


class Sp100WorkflowRequest(BaseModel):
    top_scan: int = 100
    top_plan: int = 10
    lookback_days: int = 180
    min_history_samples: int = 3
    mode: str = "sp100_auto"
    llm_provider: Optional[str] = "chatgpt-actions"
    llm_model: Optional[str] = None
    llm_style: Optional[str] = "sp100_ranker_v1"


class RankedPlanOut(BaseModel):
    rank: int
    score: float
    signal_score: int
    history_boost: float = 0.0
    history_samples: int = 0
    history_win_rate: Optional[float] = None
    history_avg_return: Optional[float] = None
    row: PlanRowOut


class Sp100WorkflowResponse(BaseModel):
    planned_at: datetime
    market_regime: str
    regime_score: float
    buy_threshold: int
    avoid_threshold: int
    scanned_universe_size: int
    candidates_with_price: int
    selected_count: int
    rows_logged: int
    rows: List[RankedPlanOut]

class DailyBarsBackfillRequest(BaseModel):
    symbols: Optional[List[str]] = None
    use_sp100: bool = True
    top_n: int = 100
    years: int = 10
    refresh: bool = False
    commit_every: int = 5


class DailyBarsBackfillResponse(BaseModel):
    as_of: datetime
    universe_size: int
    total: int
    updated: int
    skipped_cached: int
    failed: int
    no_data: int = 0
    results: List[dict]


class DailyBarsStatusRow(BaseModel):
    symbol: str
    count: int = 0
    min_date: Optional[date] = None
    max_date: Optional[date] = None


class DailyBarsStatusResponse(BaseModel):
    as_of: datetime
    requested_symbols: int
    symbols_with_data: int
    total_rows: int
    rows: List[DailyBarsStatusRow]


def _to_plan_row_out(r) -> PlanRowOut:
    return PlanRowOut(
        ticker=r.ticker,
        last=r.last,
        entry=r.entry,
        stop=r.stop,
        take_profit=r.take_profit,
        max_hold_date=r.max_hold_date,
        strategy_action=r.strategy_action,
        strategy_reason=r.strategy_reason,
        llm_action=r.llm_action,
        llm_rationale=r.llm_rationale,
        news_score=getattr(r, "news_score", 0),
        earnings_score=getattr(r, "earnings_score", 0),
        earnings_context=getattr(r, "earnings_context", None),
        signal_score=getattr(r, "signal_score", 0),
        market_regime=getattr(r, "market_regime", None),
        prob_tp=getattr(r, "prob_tp", None),
        prob_sl=getattr(r, "prob_sl", None),
        prob_open=getattr(r, "prob_open", None),
        expected_return=getattr(r, "expected_return", None),
        confidence=getattr(r, "confidence", None),
        buy_threshold=getattr(r, "buy_threshold", None),
        avoid_threshold=getattr(r, "avoid_threshold", None),
        stop_loss_pct=getattr(r, "stop_loss_pct", None),
        take_profit_pct=getattr(r, "take_profit_pct", None),
        hold_days=getattr(r, "hold_days", None),
        risk_tuning_reason=getattr(r, "risk_tuning_reason", None),
        news=[NewsItem(**n) for n in (getattr(r, "news", None) or [])],
    )


def _queue_rows_for_logging(db: Session, *, planned_at: datetime, mode: str, rows: List[PlanRowOut], meta: dict) -> int:
    rows_logged = 0
    for r in rows:
        if r.entry is None or r.stop is None or r.take_profit is None:
            continue

        entry_val = float(r.entry)
        stop_val = float(r.stop)
        tp_val = float(r.take_profit)

        news_items = []
        for n in (r.news or []):
            if isinstance(n, dict):
                news_items.append(n)
            else:
                news_items.append(n.model_dump())

        db.add(
            SwingDecision(
                ticker=r.ticker,
                planned_at=planned_at,
                mode=mode,
                entry=entry_val,
                stop=stop_val,
                take_profit=tp_val,
                max_hold_date=r.max_hold_date,
                strategy_action=r.strategy_action,
                strategy_reason=r.strategy_reason,
                llm_used=bool(meta.get("llm_used", False)),
                llm_provider=meta.get("llm_provider"),
                llm_model=meta.get("llm_model"),
                llm_style=meta.get("llm_style"),
                llm_action=r.llm_action,
                llm_rationale=r.llm_rationale,
                news_score=int(r.news_score) if r.news_score is not None else None,
                earnings_score=int(r.earnings_score) if r.earnings_score is not None else None,
                earnings_context_json=(json.dumps(r.earnings_context) if r.earnings_context is not None else None),
                news_json=json.dumps(news_items),
            )
        )
        rows_logged += 1

    return rows_logged

def _normalize_symbols(symbols: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for s in symbols:
        sym = (s or "").strip().upper()
        if not sym or sym in seen:
            continue
        out.append(sym)
        seen.add(sym)
    return out


def _resolve_universe(symbols: Optional[List[str]], *, use_sp100: bool, top_n: int) -> List[str]:
    if symbols:
        return _normalize_symbols(symbols)
    if use_sp100:
        n = max(1, min(int(top_n), 100))
        return get_sp100_universe(n)
    return []


def _build_daily_closes_loader(db: Session):
    memo: dict[tuple[str, date, date], dict[date, float]] = {}

    def _loader(symbol: str, frm: date, to: date) -> dict[date, float]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return {}

        key = (sym, frm, to)
        if key in memo:
            return memo[key]

        closes = ensure_cached_daily_closes(db, sym, frm, to, auto_fetch=True, commit=False)
        memo[key] = closes
        return closes

    return _loader


def _daily_bars_status_rows(db: Session, symbols: List[str]) -> List[DailyBarsStatusRow]:
    if not symbols:
        return []

    agg = (
        db.query(
            DailyBar.symbol.label("symbol"),
            func.min(DailyBar.bar_date).label("min_date"),
            func.max(DailyBar.bar_date).label("max_date"),
            func.count(DailyBar.symbol).label("count"),
        )
        .filter(DailyBar.symbol.in_(symbols))
        .group_by(DailyBar.symbol)
        .all()
    )

    by_symbol = {str(r.symbol): r for r in agg}
    out: List[DailyBarsStatusRow] = []
    for sym in symbols:
        row = by_symbol.get(sym)
        if row is None:
            out.append(DailyBarsStatusRow(symbol=sym, count=0, min_date=None, max_date=None))
            continue

        out.append(
            DailyBarsStatusRow(
                symbol=sym,
                count=int(row.count or 0),
                min_date=row.min_date,
                max_date=row.max_date,
            )
        )
    return out


def _history_stats_by_ticker(db: Session, lookback_days: int) -> dict[str, dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.last_eval_return.isnot(None))
        .all()
    )

    raw: dict[str, dict] = {}
    for d in rows:
        if d.last_eval_return is None:
            continue
        r = float(d.last_eval_return)
        t = d.ticker
        s = raw.setdefault(t, {"samples": 0, "wins": 0, "ret_sum": 0.0, "abs_ret_sum": 0.0})
        s["samples"] += 1
        s["ret_sum"] += r
        s["abs_ret_sum"] += abs(r)
        if r > 0:
            s["wins"] += 1

    out: dict[str, dict] = {}
    for t, s in raw.items():
        n = s["samples"]
        out[t] = {
            "samples": n,
            "avg_return": (s["ret_sum"] / max(n, 1)),
            "avg_abs_return": (s["abs_ret_sum"] / max(n, 1)),
            "win_rate": (s["wins"] / max(n, 1)),
        }
    return out


def _rolling_performance_snapshot(db: Session, lookback_days: int = 180) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.last_eval_return.isnot(None))
        .all()
    )

    overall_samples = 0
    overall_wins = 0
    overall_sum = 0.0
    overall_abs_sum = 0.0

    buy_samples = 0
    buy_wins = 0
    buy_sum = 0.0

    for d in rows:
        if d.last_eval_return is None:
            continue
        ret = float(d.last_eval_return)
        overall_samples += 1
        overall_sum += ret
        overall_abs_sum += abs(ret)
        if ret > 0:
            overall_wins += 1

        action = (d.llm_action or d.strategy_action or "").strip().upper()
        if action.startswith("BUY"):
            buy_samples += 1
            buy_sum += ret
            if ret > 0:
                buy_wins += 1

    return {
        "overall_samples": overall_samples,
        "overall_avg_return": (overall_sum / max(overall_samples, 1)) if overall_samples else 0.0,
        "overall_abs_return": (overall_abs_sum / max(overall_samples, 1)) if overall_samples else 0.0,
        "overall_win_rate": (overall_wins / max(overall_samples, 1)) if overall_samples else 0.0,
        "buy_samples": buy_samples,
        "buy_avg_return": (buy_sum / max(buy_samples, 1)) if buy_samples else 0.0,
        "buy_win_rate": (buy_wins / max(buy_samples, 1)) if buy_samples else 0.0,
    }


def _compute_dynamic_thresholds(regime: str, perf: dict) -> dict:
    base_buy = 4
    base_avoid = -4
    if regime == "risk_on":
        base_buy = 3
        base_avoid = -5
    elif regime == "risk_off":
        base_buy = 6
        base_avoid = -3

    buy_adj = 0
    avoid_adj = 0

    if perf.get("buy_samples", 0) >= 12:
        buy_win = float(perf.get("buy_win_rate", 0.0))
        buy_avg = float(perf.get("buy_avg_return", 0.0))

        if buy_win < 0.45 or buy_avg < 0.0:
            buy_adj += 1
            avoid_adj += 1
        elif buy_win > 0.58 and buy_avg > 0.01:
            buy_adj -= 1
            avoid_adj -= 1

    overall_avg = float(perf.get("overall_avg_return", 0.0))
    if perf.get("overall_samples", 0) >= 20:
        if overall_avg < -0.004:
            buy_adj += 1
        elif overall_avg > 0.008:
            buy_adj -= 1

    buy_threshold = max(3, min(7, base_buy + buy_adj))
    avoid_threshold = max(-7, min(-2, base_avoid + avoid_adj))

    return {
        "buy_threshold": int(buy_threshold),
        "avoid_threshold": int(avoid_threshold),
    }



def _compute_adaptive_trade_levels(
    *,
    entry: float,
    ticker: str,
    regime: str,
    planned_at: datetime,
    ticker_stats: dict,
    perf: dict,
) -> dict:
    base_stop_pct = 0.03
    base_tp_pct = 0.06
    base_hold_days = 20

    samples = int(ticker_stats.get("samples", 0)) if ticker_stats else 0
    ticker_win = float(ticker_stats.get("win_rate", perf.get("overall_win_rate", 0.5))) if ticker_stats else float(perf.get("overall_win_rate", 0.5))
    ticker_avg = float(ticker_stats.get("avg_return", perf.get("overall_avg_return", 0.0))) if ticker_stats else float(perf.get("overall_avg_return", 0.0))
    ticker_abs = float(ticker_stats.get("avg_abs_return", perf.get("overall_abs_return", 0.025))) if ticker_stats else float(perf.get("overall_abs_return", 0.025))

    confidence = min(1.0, samples / 12.0)

    stop_pct = max(0.018, min(0.06, 0.012 + 0.6 * max(ticker_abs, 0.005)))
    tp_pct = base_tp_pct
    hold_days = base_hold_days

    edge = (ticker_win - 0.5) * 2.2 + ticker_avg * 24.0
    edge *= (0.55 + 0.45 * confidence)

    if edge > 0:
        tp_pct += min(0.03, 0.012 * edge)
        stop_pct += min(0.01, 0.005 * edge)
        hold_days += int(round(min(8.0, 3.0 * edge)))
    else:
        tp_pct -= min(0.02, 0.012 * abs(edge))
        stop_pct -= min(0.01, 0.004 * abs(edge))
        hold_days -= int(round(min(8.0, 3.0 * abs(edge))))

    if regime == "risk_on":
        tp_pct += 0.006
        stop_pct += 0.002
        hold_days += 2
    elif regime == "risk_off":
        tp_pct -= 0.010
        stop_pct -= 0.004
        hold_days -= 4

    stop_pct = max(0.015, min(0.065, stop_pct))
    tp_pct = max(0.03, min(0.12, tp_pct))
    hold_days = max(7, min(35, hold_days))

    stop = entry * (1.0 - stop_pct)
    take_profit = entry * (1.0 + tp_pct)
    max_hold_date = planned_at + timedelta(days=hold_days)

    reason = (
        f"adaptive-risk ticker={ticker}; samples={samples}; win={ticker_win:.2f}; "
        f"avg_ret={ticker_avg:.3f}; regime={regime}; sl%={stop_pct:.3f}; "
        f"tp%={tp_pct:.3f}; hold={hold_days}"
    )

    return {
        "stop": float(stop),
        "take_profit": float(take_profit),
        "max_hold_date": max_hold_date,
        "stop_loss_pct": float(round(stop_pct, 6)),
        "take_profit_pct": float(round(tp_pct, 6)),
        "hold_days": int(hold_days),
        "risk_tuning_reason": reason,
    }


def _apply_adaptive_risk_controls(
    row,
    *,
    planned_at: datetime,
    regime: str,
    ticker_stats: dict,
    perf: dict,
) -> None:
    if row.entry is None:
        return

    levels = _compute_adaptive_trade_levels(
        entry=float(row.entry),
        ticker=row.ticker,
        regime=regime,
        planned_at=planned_at,
        ticker_stats=ticker_stats,
        perf=perf,
    )

    row.stop = levels["stop"]
    row.take_profit = levels["take_profit"]
    row.max_hold_date = levels["max_hold_date"]
    row.stop_loss_pct = levels["stop_loss_pct"]
    row.take_profit_pct = levels["take_profit_pct"]
    row.hold_days = levels["hold_days"]
    row.risk_tuning_reason = levels["risk_tuning_reason"]

    prior = row.strategy_reason or ""
    row.strategy_reason = (prior + " | " + levels["risk_tuning_reason"]).strip(" |")


def _apply_prob_and_action(
    row,
    *,
    regime: str,
    buy_threshold: int,
    avoid_threshold: int,
    history_win_rate: float | None,
    history_samples: int,
) -> dict:
    if row.entry is None or row.stop is None or row.take_profit is None:
        return {
            "p_tp": None,
            "p_sl": None,
            "p_open": None,
            "expected_return": None,
            "confidence": None,
            "action": "WAIT",
        }

    signal_score = int(getattr(row, "signal_score", getattr(row, "news_score", 0) + getattr(row, "earnings_score", 0)))
    probs = estimate_trade_probabilities(
        signal_score=signal_score,
        entry=float(row.entry),
        stop=float(row.stop),
        take_profit=float(row.take_profit),
        regime=regime,
        history_win_rate=history_win_rate,
        history_samples=history_samples,
    )

    buy_ok = signal_score >= buy_threshold and probs["p_tp"] >= 0.5 and probs["expected_return"] > 0
    avoid_ok = signal_score <= avoid_threshold or probs["p_sl"] >= 0.47

    if buy_ok:
        action = "BUY"
        strategy_action = "BUY"
    elif avoid_ok:
        action = "AVOID"
        strategy_action = "WAIT / AVOID"
    else:
        action = "WAIT"
        strategy_action = "HOLD / WAIT"

    row.signal_score = signal_score
    row.market_regime = regime
    row.prob_tp = probs["p_tp"]
    row.prob_sl = probs["p_sl"]
    row.prob_open = probs["p_open"]
    row.expected_return = probs["expected_return"]
    row.confidence = probs["confidence"]
    row.buy_threshold = buy_threshold
    row.avoid_threshold = avoid_threshold
    row.strategy_action = strategy_action
    row.llm_action = action
    row.llm_rationale = (
        f"regime={regime}; signal={signal_score}; p_tp={probs['p_tp']:.2f}; "
        f"p_sl={probs['p_sl']:.2f}; exp_ret={probs['expected_return']:.3f}; "
        f"confidence={probs['confidence']:.2f}; history_samples={history_samples}"
    )

    return {
        "p_tp": probs["p_tp"],
        "p_sl": probs["p_sl"],
        "p_open": probs["p_open"],
        "expected_return": probs["expected_return"],
        "confidence": probs["confidence"],
        "action": action,
    }


@app.get("/debug/model")
def debug_model(_=Depends(require_bearer_token)):
    cols = list(SwingDecision.__table__.columns.keys())
    return {"columns": cols}


@app.get("/debug/finnhub")
def debug_finnhub(_=Depends(require_bearer_token)):
    today = datetime.now(timezone.utc).date()
    bars, fetch_status = fetch_finnhub_daily_bars_with_meta("AAPL", today - timedelta(days=14), today)
    last = get_last_price("AAPL")
    return {
        "finnhub_key_configured": bool(os.getenv("FINNHUB_API_KEY")),
        "candle_fetch_status": fetch_status,
        "candle_bars": len(bars),
        "aapl_last_price": last,
    }


@app.get("/scan/sp100", response_model=ScanResponse)
def scan_sp100(top_n: int = 100, _=Depends(require_bearer_token)):
    return {"tickers": get_sp100_universe(top_n)}


@app.post("/scan/swing", response_model=ScanResponse)
def scan_swing(req: ScanRequest, _=Depends(require_bearer_token)):
    picks = scan_swing_candidates_largecaps(req.universe, top_n=req.top_n)
    return {"tickers": picks}


@app.post("/plan/swing", response_model=PlanResponse)
def plan_swing(req: PlanRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)

    daily_closes_loader = _build_daily_closes_loader(db)
    regime_snapshot = detect_market_regime(req.tickers, daily_closes_loader=daily_closes_loader)
    perf = _rolling_performance_snapshot(db, lookback_days=180)
    thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)
    ticker_hist = _history_stats_by_ticker(db, lookback_days=180)

    try:
        rows = build_swing_plan(
            req.tickers,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            daily_closes_loader=daily_closes_loader,
        )
    except Exception as e:
        out = [
            PlanRowOut(
                ticker=t,
                last=None,
                entry=None,
                stop=None,
                take_profit=None,
                max_hold_date=datetime.now(timezone.utc),
                strategy_action="NO DATA",
                strategy_reason=f"Planner crashed: {e}",
                news=[],
                news_score=0,
                earnings_score=0,
                earnings_context=None,
                signal_score=0,
                market_regime=regime_snapshot["regime"],
                buy_threshold=thresholds["buy_threshold"],
                avoid_threshold=thresholds["avoid_threshold"],
                llm_action="WAIT",
                llm_rationale="no-data",
            )
            for t in req.tickers
        ]
        return {
            "planned_at": planned_at,
            "market_regime": regime_snapshot["regime"],
            "regime_score": regime_snapshot["score"],
            "buy_threshold": thresholds["buy_threshold"],
            "avoid_threshold": thresholds["avoid_threshold"],
            "rows": out,
        }

    for r in rows:
        h = ticker_hist.get(r.ticker, {})
        _apply_adaptive_risk_controls(
            r,
            planned_at=planned_at,
            regime=regime_snapshot["regime"],
            ticker_stats=h,
            perf=perf,
        )
        _apply_prob_and_action(
            r,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            history_win_rate=(float(h["win_rate"]) if "win_rate" in h else None),
            history_samples=int(h.get("samples", 0)),
        )

    out = [_to_plan_row_out(r) for r in rows]
    return {
        "planned_at": planned_at,
        "market_regime": regime_snapshot["regime"],
        "regime_score": regime_snapshot["score"],
        "buy_threshold": thresholds["buy_threshold"],
        "avoid_threshold": thresholds["avoid_threshold"],
        "rows": out,
    }


@app.post("/workflow/sp100/top10-log", response_model=Sp100WorkflowResponse)
def workflow_sp100_top10_log(req: Sp100WorkflowRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)
    top_scan = max(10, min(int(req.top_scan), 100))
    top_plan = max(1, min(int(req.top_plan), 20))
    lookback_days = max(30, min(int(req.lookback_days), 720))
    min_history_samples = max(1, min(int(req.min_history_samples), 20))

    universe = get_sp100_universe(top_scan)
    daily_closes_loader = _build_daily_closes_loader(db)
    regime_snapshot = detect_market_regime(universe[:20], daily_closes_loader=daily_closes_loader)
    perf = _rolling_performance_snapshot(db, lookback_days=lookback_days)
    thresholds = _compute_dynamic_thresholds(regime_snapshot["regime"], perf)

    rows = build_swing_plan(
        universe,
        regime=regime_snapshot["regime"],
        buy_threshold=thresholds["buy_threshold"],
        avoid_threshold=thresholds["avoid_threshold"],
        daily_closes_loader=daily_closes_loader,
    )
    history_stats = _history_stats_by_ticker(db, lookback_days=lookback_days)

    ranked: list[dict] = []
    for r in rows:
        if r.entry is None or r.stop is None or r.take_profit is None:
            continue

        h = history_stats.get(r.ticker)
        history_samples = 0
        history_win_rate = None
        history_avg_return = None
        history_boost = 0.0

        if h:
            history_samples = int(h.get("samples", 0))
            history_win_rate = float(h.get("win_rate"))
            history_avg_return = float(h.get("avg_return"))
            if history_samples >= min_history_samples:
                confidence = min(1.0, history_samples / 8.0)
                hist_raw = (history_avg_return * 100.0) * 0.35 + (history_win_rate - 0.5) * 4.0
                history_boost = max(-3.0, min(3.0, hist_raw * confidence))

        _apply_adaptive_risk_controls(
            r,
            planned_at=planned_at,
            regime=regime_snapshot["regime"],
            ticker_stats=(h or {}),
            perf=perf,
        )

        decision = _apply_prob_and_action(
            r,
            regime=regime_snapshot["regime"],
            buy_threshold=thresholds["buy_threshold"],
            avoid_threshold=thresholds["avoid_threshold"],
            history_win_rate=history_win_rate,
            history_samples=history_samples,
        )

        signal_score = int(getattr(r, "signal_score", 0))
        exp_ret = float(decision["expected_return"] or 0.0)
        confidence = float(decision["confidence"] or 0.0)
        p_edge = float((decision["p_tp"] or 0.0) - (decision["p_sl"] or 0.0))

        score = (
            float(signal_score)
            + float(history_boost)
            + exp_ret * 120.0
            + p_edge * 2.5
            + confidence * 1.5
        )

        ranked.append(
            {
                "score": score,
                "signal_score": signal_score,
                "history_boost": history_boost,
                "history_samples": history_samples,
                "history_win_rate": history_win_rate,
                "history_avg_return": history_avg_return,
                "row": r,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    selected = ranked[:top_plan]

    out_rows: list[RankedPlanOut] = []
    for idx, item in enumerate(selected, start=1):
        row_out = _to_plan_row_out(item["row"])
        out_rows.append(
            RankedPlanOut(
                rank=idx,
                score=float(round(item["score"], 4)),
                signal_score=int(item["signal_score"]),
                history_boost=float(round(item["history_boost"], 4)),
                history_samples=int(item["history_samples"]),
                history_win_rate=item["history_win_rate"],
                history_avg_return=item["history_avg_return"],
                row=row_out,
            )
        )

    meta = {
        "llm_used": True,
        "llm_provider": req.llm_provider,
        "llm_model": req.llm_model,
        "llm_style": req.llm_style,
    }

    rows_logged = 0
    try:
        rows_logged = _queue_rows_for_logging(
            db,
            planned_at=planned_at,
            mode=req.mode,
            rows=[x.row for x in out_rows],
            meta=meta,
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"SP100 workflow logging failed: {e}")

    return Sp100WorkflowResponse(
        planned_at=planned_at,
        market_regime=regime_snapshot["regime"],
        regime_score=float(regime_snapshot["score"]),
        buy_threshold=thresholds["buy_threshold"],
        avoid_threshold=thresholds["avoid_threshold"],
        scanned_universe_size=len(universe),
        candidates_with_price=len(ranked),
        selected_count=len(out_rows),
        rows_logged=rows_logged,
        rows=out_rows,
    )



@app.post("/data/daily-bars/backfill", response_model=DailyBarsBackfillResponse)
def daily_bars_backfill(req: DailyBarsBackfillRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    symbols = _resolve_universe(req.symbols, use_sp100=req.use_sp100, top_n=req.top_n)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided. Pass symbols or set use_sp100=true.")

    years = max(1, min(int(req.years), 15))
    commit_every = max(1, min(int(req.commit_every), 50))

    try:
        result = backfill_universe_daily_bars(
            db,
            symbols,
            years=years,
            refresh=bool(req.refresh),
            commit_every=commit_every,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Daily bars backfill failed: {e}")

    return DailyBarsBackfillResponse(
        as_of=datetime.now(timezone.utc),
        universe_size=len(symbols),
        total=int(result.get("total", 0)),
        updated=int(result.get("updated", 0)),
        skipped_cached=int(result.get("skipped_cached", 0)),
        failed=int(result.get("failed", 0)),
        no_data=int(result.get("no_data", 0)),
        results=list(result.get("results", [])),
    )


@app.get("/data/daily-bars/status", response_model=DailyBarsStatusResponse)
def daily_bars_status(
    symbols: Optional[List[str]] = Query(default=None),
    use_sp100: bool = True,
    top_n: int = 100,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    universe = _resolve_universe(symbols, use_sp100=use_sp100, top_n=top_n)
    if not universe:
        raise HTTPException(status_code=400, detail="No symbols provided. Pass symbols or set use_sp100=true.")

    rows = _daily_bars_status_rows(db, universe)
    symbols_with_data = sum(1 for r in rows if int(r.count) > 0)
    total_rows = sum(int(r.count) for r in rows)

    return DailyBarsStatusResponse(
        as_of=datetime.now(timezone.utc),
        requested_symbols=len(universe),
        symbols_with_data=symbols_with_data,
        total_rows=total_rows,
        rows=rows,
    )

@app.post("/history/log")
def log_history(req: LogRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    try:
        rows_logged = _queue_rows_for_logging(
            db,
            planned_at=req.planned_at,
            mode=req.mode,
            rows=req.rows,
            meta=req.meta,
        )
        db.commit()
        return {"ok": True, "rows_logged": rows_logged}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Logging failed: {e}")


@app.get("/history/evaluate")
def evaluate_history(limit: int = 200, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    q = (
        db.query(SwingDecision)
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )
    results = []
    for d in q:
        last = get_last_price(d.ticker)
        if last is None:
            continue

        outcome, ret = evaluate_plan_row(d.entry, d.stop, d.take_profit, last, d.max_hold_date)
        d.last_eval_ts = datetime.now(timezone.utc)
        d.last_eval_price = float(last)
        d.last_eval_outcome = outcome
        d.last_eval_return = float(ret)
        results.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "entry": d.entry,
                "stop": d.stop,
                "tp": d.take_profit,
                "max_hold_date": d.max_hold_date,
                "last_price": last,
                "outcome": outcome,
                "return_since_entry": ret,
                "strategy_action": d.strategy_action,
                "llm_action": d.llm_action,
                "news_score": getattr(d, "news_score", None),
                "earnings_score": getattr(d, "earnings_score", None),
            }
        )
    db.commit()
    return {"rows": results, "evaluated": len(results)}


@app.get("/analysis/earnings-score")
def earnings_score_analysis(
    lookback_days: int = 180,
    limit: int = 500,
    refresh_prices: bool = True,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    q = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .filter(SwingDecision.earnings_score.isnot(None))
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )

    samples = []
    for d in q:
        if d.entry is None or d.stop is None or d.take_profit is None or d.earnings_score is None:
            continue

        outcome = d.last_eval_outcome
        ret = d.last_eval_return
        last_price = d.last_eval_price

        if refresh_prices:
            live_last = get_last_price(d.ticker)
            if live_last is not None:
                outcome, ret = evaluate_plan_row(d.entry, d.stop, d.take_profit, live_last, d.max_hold_date)
                d.last_eval_ts = now
                d.last_eval_price = float(live_last)
                d.last_eval_outcome = outcome
                d.last_eval_return = float(ret)
                last_price = float(live_last)

        if ret is None:
            continue

        score = int(d.earnings_score)
        if score <= -4:
            bucket = "negative"
        elif score >= 4:
            bucket = "positive"
        else:
            bucket = "neutral"

        samples.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "earnings_score": score,
                "bucket": bucket,
                "outcome": outcome,
                "return_since_entry": float(ret),
                "last_price": last_price,
            }
        )

    db.commit()

    def rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {
                "samples": 0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "tp_rate": 0.0,
                "sl_rate": 0.0,
                "expired_rate": 0.0,
                "open_rate": 0.0,
            }

        avg_return = sum(r["return_since_entry"] for r in rows) / n
        wins = sum(1 for r in rows if r["return_since_entry"] > 0)
        tp = sum(1 for r in rows if r["outcome"] == "TP hit")
        sl = sum(1 for r in rows if r["outcome"] == "SL hit")
        expired = sum(1 for r in rows if r["outcome"] == "Expired")
        open_ = sum(1 for r in rows if r["outcome"] == "Open / In range")

        return {
            "samples": n,
            "avg_return": avg_return,
            "win_rate": rate(wins, n),
            "tp_rate": rate(tp, n),
            "sl_rate": rate(sl, n),
            "expired_rate": rate(expired, n),
            "open_rate": rate(open_, n),
        }

    by_bucket = {
        "negative": summarize([r for r in samples if r["bucket"] == "negative"]),
        "neutral": summarize([r for r in samples if r["bucket"] == "neutral"]),
        "positive": summarize([r for r in samples if r["bucket"] == "positive"]),
    }

    def pearson(rows: list[dict]) -> float | None:
        n = len(rows)
        if n < 2:
            return None
        xs = [float(r["earnings_score"]) for r in rows]
        ys = [float(r["return_since_entry"]) for r in rows]
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 1e-12 or vy <= 1e-12:
            return None
        return cov / ((vx ** 0.5) * (vy ** 0.5))

    return {
        "as_of": now,
        "lookback_days": lookback_days,
        "samples": len(samples),
        "refresh_prices": refresh_prices,
        "overall": summarize(samples),
        "score_return_correlation": pearson(samples),
        "by_bucket": by_bucket,
        "rows_preview": samples[:25],
    }


@app.get("/learning/patterns")
def learning_patterns(
    lookback_days: int = 120,
    limit: int = 500,
    db: Session = Depends(get_db),
    _=Depends(require_bearer_token),
):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    q = (
        db.query(SwingDecision)
        .filter(SwingDecision.planned_at >= cutoff)
        .order_by(SwingDecision.planned_at.desc())
        .limit(limit)
        .all()
    )

    rows = []
    for d in q:
        # only learn from rows with valid levels
        if d.entry is None or d.stop is None or d.take_profit is None:
            continue

        last = get_last_price(d.ticker)
        if last is None:
            continue

        assumed_executed, label, ret = classify_assumption(
            llm_action=d.llm_action,
            entry=float(d.entry),
            stop=float(d.stop),
            take_profit=float(d.take_profit),
            last_price=float(last),
            max_hold_date=d.max_hold_date,
            now=now,
        )

        rows.append(
            {
                "id": d.id,
                "ticker": d.ticker,
                "planned_at": d.planned_at,
                "max_hold_date": d.max_hold_date,
                "llm_action": d.llm_action,
                "news_score": getattr(d, "news_score", None),
                "news_bucket": bucket_news(getattr(d, "news_score", None)),
                "entry": float(d.entry),
                "stop": float(d.stop),
                "take_profit": float(d.take_profit),
                "last_price": float(last),
                "assumed_executed": assumed_executed,
                "label": label,
                "return_since_entry": float(ret),
            }
        )

    # --- aggregate ---
    total = len(rows)
    by_label = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    def rate(n: int, d: int) -> float:
        return (n / d) if d else 0.0

    buy_total = sum(1 for r in rows if r["assumed_executed"])
    buy_success = sum(1 for r in rows if r["label"] in ("buy_success_tp", "buy_expired_win"))
    buy_fail = sum(1 for r in rows if r["label"] in ("buy_fail_sl", "buy_expired_loss"))

    wait_total = total - buy_total
    wait_good_avoid = sum(1 for r in rows if r["label"] == "wait_good_avoid")
    wait_missed = sum(1 for r in rows if r["label"] in ("wait_missed_tp", "wait_missed_tp_expired"))

    # by news bucket (BUY success rate, WAIT missed rate)
    buckets = ["negative", "neutral", "positive", "unknown"]
    by_bucket = {}
    for b in buckets:
        br = [x for x in rows if x["news_bucket"] == b]
        b_buy = [x for x in br if x["assumed_executed"]]
        b_wait = [x for x in br if not x["assumed_executed"]]
        by_bucket[b] = {
            "samples": len(br),
            "buy_samples": len(b_buy),
            "buy_success_rate": rate(sum(1 for x in b_buy if x["label"] in ("buy_success_tp", "buy_expired_win")), len(b_buy)),
            "wait_samples": len(b_wait),
            "wait_missed_rate": rate(sum(1 for x in b_wait if x["label"] in ("wait_missed_tp", "wait_missed_tp_expired")), len(b_wait)),
            "wait_good_avoid_rate": rate(sum(1 for x in b_wait if x["label"] == "wait_good_avoid"), len(b_wait)),
        }

    # by ticker quick stats
    by_ticker = {}
    for r in rows:
        t = r["ticker"]
        d = by_ticker.setdefault(t, {"samples": 0, "buy": 0, "buy_success": 0, "wait": 0, "wait_missed": 0, "avg_ret": 0.0})
        d["samples"] += 1
        d["avg_ret"] += r["return_since_entry"]
        if r["assumed_executed"]:
            d["buy"] += 1
            if r["label"] in ("buy_success_tp", "buy_expired_win"):
                d["buy_success"] += 1
        else:
            d["wait"] += 1
            if r["label"] in ("wait_missed_tp", "wait_missed_tp_expired"):
                d["wait_missed"] += 1

    for t, d in by_ticker.items():
        d["avg_ret"] = d["avg_ret"] / max(d["samples"], 1)
        d["buy_success_rate"] = rate(d["buy_success"], d["buy"])
        d["wait_missed_rate"] = rate(d["wait_missed"], d["wait"])

    # --- prompt context (what you inject into next plans) ---
    prompt_context = (
        "Learning snapshot (assumptions: BUY executed; WAIT not executed).\n"
        f"Lookback: {lookback_days}d, samples: {total}\n"
        f"BUY success rate: {buy_success}/{buy_total} = {rate(buy_success,buy_total):.0%}; "
        f"BUY fail rate: {buy_fail}/{buy_total} = {rate(buy_fail,buy_total):.0%}\n"
        f"WAIT missed rate: {wait_missed}/{wait_total} = {rate(wait_missed,wait_total):.0%}; "
        f"WAIT good-avoid rate: {wait_good_avoid}/{wait_total} = {rate(wait_good_avoid,wait_total):.0%}\n"
        "News buckets impact:\n"
        + "\n".join(
            [
                f"- {b}: buy_success_rate={by_bucket[b]['buy_success_rate']:.0%} "
                f"(n={by_bucket[b]['buy_samples']}), "
                f"wait_missed_rate={by_bucket[b]['wait_missed_rate']:.0%} "
                f"(n={by_bucket[b]['wait_samples']})"
                for b in buckets
            ]
        )
    )

    return {
        "as_of": now,
        "lookback_days": lookback_days,
        "samples": total,
        "by_label": by_label,
        "rates": {
            "buy_success_rate": rate(buy_success, buy_total),
            "buy_fail_rate": rate(buy_fail, buy_total),
            "wait_missed_rate": rate(wait_missed, wait_total),
            "wait_good_avoid_rate": rate(wait_good_avoid, wait_total),
        },
        "by_bucket": by_bucket,
        "by_ticker": by_ticker,
        "prompt_context": prompt_context,
    }
