from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import yfinance as yf
import os
import requests
import math
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
    news_score: int = 0

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

from datetime import datetime, timezone
import re

POSITIVE_KWS = [
    "beat", "beats", "upgrade", "raises", "raise", "record", "surge", "partnership",
    "launch", "expands", "expansion", "buyback", "strong", "bullish", "wins", "milestone"
]
NEGATIVE_KWS = [
    "miss", "misses", "downgrade", "cuts", "cut", "lawsuit", "probe", "investigation",
    "layoff", "layoffs", "weak", "bearish", "recall", "fall", "plunge", "risk"
]

def _count_hits(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)

def compute_news_score(news_items: list[dict]) -> int:
    """
    Returns int in [-10, +10].
    Uses per-article (pos-neg) then weighted average by recency.
    """
    if not news_items:
        return 0

    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    weight_total = 0.0

    for item in news_items:
        headline = (item.get("headline") or "").strip()
        summary = (item.get("summary") or "").strip()
        text = f"{headline}. {summary}"

        pos = _count_hits(text, POSITIVE_KWS)
        neg = _count_hits(text, NEGATIVE_KWS)
        raw = float(pos - neg)  # per-article balance

        # Recency weight
        w = 1.0
        dt_str = item.get("datetime")
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")) if dt_str else None
        except Exception:
            dt = None

        if dt:
            age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            if age_hours <= 24:
                w = 1.0
            elif age_hours <= 72:
                w = 0.6
            else:
                w = 0.3

        weighted_sum += raw * w
        weight_total += w

    avg = weighted_sum / max(weight_total, 1e-9)

    # Scale average balance to [-10, +10] using tanh for stability
    # avg ~ +/-3 already becomes strong; adjust multiplier if you want
    scaled = 10.0 * math.tanh(avg / 2.0)

    # Clamp + round
    return int(round(max(-10.0, min(10.0, scaled))))

def build_swing_plan(tickers: list[str]) -> list[PlanRow]:
    rows: list[PlanRow] = []

    for t in tickers:
        last = get_last_price(t)
        news = get_company_news_summary(t, days=7, limit=5)

        # 🔥 Compute score HERE
        score = compute_news_score(news)

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
                    news_score=score,   # ← use score here
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
                news_score=score,   # ← and here
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

@dataclass
class LearningRow:
    id: int
    ticker: str
    planned_at: datetime
    max_hold_date: datetime | None
    llm_action: str | None
    news_score: int | None
    entry: float
    stop: float
    take_profit: float
    last_price: float
    assumed_executed: bool
    label: str
    ret: float

def bucket_news(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= -5:
        return "negative"
    if score >= 5:
        return "positive"
    return "neutral"

def classify_assumption(
    *,
    llm_action: str | None,
    entry: float,
    stop: float,
    take_profit: float,
    last_price: float,
    max_hold_date: datetime | None,
    now: datetime,
) -> tuple[bool, str, float]:
    """
    Snapshot classifier:
    - BUY => assumed executed
    - WAIT/HOLD/None => assumed not executed
    """
    action = (llm_action or "").strip().upper()
    assumed_executed = action == "BUY"

    ret = (last_price - entry) / max(entry, 1e-9)

    expired = (max_hold_date is not None) and (now > max_hold_date)

    if assumed_executed:
        if last_price <= stop:
            return True, "buy_fail_sl", ret
        if last_price >= take_profit:
            return True, "buy_success_tp", ret
        if expired:
            return True, ("buy_expired_win" if last_price >= entry else "buy_expired_loss"), ret
        return True, "buy_open", ret

    # WAIT / HOLD / anything else => not executed
    if last_price <= stop:
        return False, "wait_good_avoid", ret
    if last_price >= take_profit:
        # If still within the holding window, it's “currently missed opportunity”
        # If expired, it's “missed by expiry” (still missed)
        return False, ("wait_missed_tp" if not expired else "wait_missed_tp_expired"), ret
    return False, "wait_neutral", ret