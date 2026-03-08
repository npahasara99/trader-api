from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from .logic import bucket_news, classify_assumption
import json
import os

from sqlalchemy.orm import Session
from sqlalchemy import text

from .db import Base, engine, get_db
from .models import SwingDecision
from .logic import (
    scan_swing_candidates_largecaps,
    build_swing_plan,
    get_last_price,
    evaluate_plan_row,
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
    version="0.1.1",
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

    llm_action: Optional[str] = None
    llm_rationale: Optional[str] = None


class PlanResponse(BaseModel):
    planned_at: datetime
    rows: List[PlanRowOut]


@app.get("/debug/model")
def debug_model(_=Depends(require_bearer_token)):
    cols = list(SwingDecision.__table__.columns.keys())
    return {"columns": cols}


@app.post("/scan/swing", response_model=ScanResponse)
def scan_swing(req: ScanRequest, _=Depends(require_bearer_token)):
    picks = scan_swing_candidates_largecaps(req.universe, top_n=req.top_n)
    return {"tickers": picks}


@app.post("/plan/swing", response_model=PlanResponse)
def plan_swing(req: PlanRequest, _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)

    try:
        rows = build_swing_plan(req.tickers)
    except Exception as e:
        # Never return 500 for planner bugs; return a NO DATA row per ticker
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
                llm_action=None,
                llm_rationale=None,
            )
            for t in req.tickers
        ]
        return {"planned_at": planned_at, "rows": out}

    out: list[PlanRowOut] = []
    for r in rows:
        out.append(
            PlanRowOut(
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
                news=[NewsItem(**n) for n in (getattr(r, "news", None) or [])],
            )
        )

    return {"planned_at": planned_at, "rows": out}


class LogRequest(BaseModel):
    planned_at: datetime
    mode: str = "manual"
    rows: List[PlanRowOut]
    meta: dict = Field(default_factory=dict)


@app.post("/history/log")
def log_history(req: LogRequest, db: Session = Depends(get_db), _=Depends(require_bearer_token)):
    rows_logged = 0

    try:
        for r in req.rows:
            # Skip rows that have no price data (DB columns are NOT NULL)
            if r.entry is None or r.stop is None or r.take_profit is None:
                continue

            # Convert numeric fields (safe now because not None)
            entry_val = float(r.entry)
            stop_val = float(r.stop)
            tp_val = float(r.take_profit)

            # Convert news items safely (works whether items are dicts or Pydantic models)
            news_items = []
            for n in (r.news or []):
                if isinstance(n, dict):
                    news_items.append(n)
                else:
                    news_items.append(n.model_dump())

            db.add(
                SwingDecision(
                    ticker=r.ticker,
                    planned_at=req.planned_at,
                    mode=req.mode,
                    entry=entry_val,
                    stop=stop_val,
                    take_profit=tp_val,
                    max_hold_date=r.max_hold_date,
                    strategy_action=r.strategy_action,
                    strategy_reason=r.strategy_reason,
                    llm_used=bool(req.meta.get("llm_used", False)),
                    llm_provider=req.meta.get("llm_provider"),
                    llm_model=req.meta.get("llm_model"),
                    llm_style=req.meta.get("llm_style"),
                    llm_action=r.llm_action,
                    llm_rationale=r.llm_rationale,
                    news_score=int(r.news_score) if r.news_score is not None else None,
                    earnings_score=int(r.earnings_score) if r.earnings_score is not None else None,
                    earnings_context_json=(json.dumps(r.earnings_context) if r.earnings_context is not None else None),
                    news_json=json.dumps(news_items),
                )
            )
            rows_logged += 1

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
