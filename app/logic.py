from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import yfinance as yf
import os
import requests
from dateutil.parser import isoparse
@dataclass
class PlanRow:
    ticker: str
    last: float | None
    entry: float | None
    stop: float | None
    take_profit: float | None
    strategy_action: str
    strategy_reason: str
    max_hold_date: datetime | None
    news: list[dict] | None = None
    llm_action: str | None = None
    llm_rationale: str | None = None

import yfinance as yf

def get_last_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        # fast_info often works when download fails
        fi = getattr(t, "fast_info", None)
        if fi:
            lp = fi.get("last_price") or fi.get("lastPrice")
            if lp:
                return float(lp)

        hist = t.history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

def scan_swing_candidates_largecaps(universe: list[str], top_n: int = 8) -> list[str]:
    # TODO: replace with your existing scan logic
    # keep it simple for now: return first N
    return universe[:top_n]

def build_swing_plan(tickers: list[str]) -> list[PlanRow]:
    rows: list[PlanRow] = []

    for t in tickers:
        last = get_last_price(t)
        news = get_company_news_summary(t, days=7, limit=5)

        if last is None:
            rows.append(
                PlanRow(
                    ticker=t,
                    last=None,
                    entry=None,
                    stop=None,
                    take_profit=None,
                    strategy_action="NO DATA",
                    strategy_reason="Price unavailable (Finnhub quote failed or key missing)",
                    max_hold_date=datetime.now(timezone.utc) + timedelta(days=20),
                    news=news,
                )
            )
            continue

        entry = float(last)
        stop = entry * 0.97
        take_profit = entry * 1.06

        rows.append(
            PlanRow(
                ticker=t,
                last=entry,
                entry=entry,
                stop=stop,
                take_profit=take_profit,
                strategy_action="HOLD / WAIT",
                strategy_reason="placeholder",
                max_hold_date=datetime.now(timezone.utc) + timedelta(days=20),
                news=news,
            )
        )

    return rows

def evaluate_plan_row(entry: float, stop: float, take_profit: float, last_price: float, max_hold_date: datetime | None):
    outcome = "Open / In range"
    if last_price <= stop:
        outcome = "SL hit"
    elif last_price >= take_profit:
        outcome = "TP hit"
    if max_hold_date and datetime.now(timezone.utc) > max_hold_date:
        outcome = "Expired"
    ret = (last_price - entry) / max(entry, 1e-9)
    return outcome, ret

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"

def finnhub_get(path: str, params: dict) -> dict | None:
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get(
            f"{FINNHUB_BASE}{path}",
            params={**params, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_last_price(ticker: str) -> float | None:
    data = finnhub_get("/quote", {"symbol": ticker})
    # Finnhub quote fields: c=current price
    if not data or data.get("c") in (None, 0):
        return None
    return float(data["c"])

def get_company_news_summary(ticker: str, days: int = 7, limit: int = 5) -> list[dict]:
    """
    Returns list of: {headline, summary, source, datetime, url}
    """
    now = datetime.now(timezone.utc)
    frm = (now - timedelta(days=days)).date().isoformat()
    to = now.date().isoformat()

    data = finnhub_get("/company-news", {"symbol": ticker, "from": frm, "to": to})
    if not data or not isinstance(data, list):
        return []

    items = []
    for x in data[: max(limit, 0)]:
        items.append(
            {
                "headline": x.get("headline"),
                "summary": x.get("summary"),
                "source": x.get("source"),
                "datetime": datetime.fromtimestamp(int(x.get("datetime", 0)), tz=timezone.utc).isoformat(),
                "url": x.get("url"),
            }
        )
    return items