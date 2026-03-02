from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone
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

@app.post("/scan/swing", response_model=ScanResponse)
def scan_swing(req: ScanRequest, _=Depends(require_bearer_token)):
    picks = scan_swing_candidates_largecaps(req.universe, top_n=req.top_n)
    return {"tickers": picks}

@app.post("/plan/swing", response_model=PlanResponse)
def plan_swing(req: PlanRequest, _=Depends(require_bearer_token)):
    planned_at = datetime.now(timezone.utc)

    rows = build_swing_plan(req.tickers)
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
                news=[NewsItem(**n) for n in (r.news or [])],
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
    for r in req.rows:
        # Safely convert numeric fields only when present
        entry_val = float(r.entry) if (r.entry is not None) else None
        stop_val = float(r.stop) if (r.stop is not None) else None
        tp_val = float(r.take_profit) if (r.take_profit is not None) else None

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
            )
        )
    db.commit()
    return {"ok": True, "rows_logged": len(req.rows)}

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
            }
        )
    db.commit()
    return {"rows": results, "evaluated": len(results)}