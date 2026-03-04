from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from .logic import bucket_news, classify_assumption
import json
import os

from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import SwingDecision
from .logic import (
    scan_swing_candidates_largecaps,
    build_swing_plan,
    get_last_price,
    evaluate_plan_row,
)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Trader Backend (Stocks Only)",
    version="0.1.0",
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
                news=[NewsItem(**n) for n in (getattr(r, "news", None) or [])],
            )
        )

    return {"planned_at": planned_at, "rows": out}

class LogRequest(BaseModel):
    planned_at: datetime
    mode: str = "manual"
    rows: List[PlanRowOut]
    meta: dict = Field(default_factory=dict)


from fastapi import HTTPException  # make sure this import exists

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
                    # pydantic BaseModel
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
                    news_json=json.dumps(news_items),
                )
            )
            rows_logged += 1

        db.commit()
        return {"ok": True, "rows_logged": rows_logged}

    except Exception as e:
        db.rollback()
        # Return the real error so you can see it in GPT / curl instead of a generic 500
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
            }
        )
    db.commit()
    return {"rows": results, "evaluated": len(results)}

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