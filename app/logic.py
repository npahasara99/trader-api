from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
import os
import math
import requests


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
    earnings_score: int = 0
    earnings_context: dict | None = None


# Static S&P 100-like liquid large-cap universe for API-side scanning.
SP100_UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMT", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVX", "DHR",
    "DIS", "DOW", "DUK", "EMR", "F", "GD", "GE", "GILD", "GM", "GOOG",
    "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI",
    "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDT", "MET", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL",
    "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO",
    "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
    "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "AIG", "DE", "NOW",
]


def get_sp100_universe(top_n: int | None = None) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for t in SP100_UNIVERSE:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    if top_n is None:
        return uniq
    n = max(1, min(int(top_n), len(uniq)))
    return uniq[:n]


def scan_swing_candidates_largecaps(universe: list[str], top_n: int = 8) -> list[str]:
    # TODO: replace with your existing scan logic
    return universe[:top_n]


POSITIVE_KWS = [
    "beat", "beats", "upgrade", "raises", "raise", "record", "surge", "partnership",
    "launch", "expands", "expansion", "buyback", "strong", "bullish", "wins", "milestone",
]
NEGATIVE_KWS = [
    "miss", "misses", "downgrade", "cuts", "cut", "lawsuit", "probe", "investigation",
    "layoff", "layoffs", "weak", "bearish", "recall", "fall", "plunge", "risk",
]


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"


def finnhub_get(path: str, params: dict) -> dict | list | None:
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get(
            f"{FINNHUB_BASE}{path}",
            params={**params, "token": FINNHUB_API_KEY},
            timeout=12,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_last_price(ticker: str) -> float | None:
    data = finnhub_get("/quote", {"symbol": ticker})
    if not isinstance(data, dict) or data.get("c") in (None, 0):
        return None
    return float(data["c"])


def get_company_news_summary(ticker: str, days: int = 7, limit: int = 5) -> list[dict]:
    now = datetime.now(timezone.utc)
    frm = (now - timedelta(days=days)).date().isoformat()
    to = now.date().isoformat()

    data = finnhub_get("/company-news", {"symbol": ticker, "from": frm, "to": to})
    if not isinstance(data, list):
        return []

    items: list[dict] = []
    for x in data[: max(limit, 0)]:
        try:
            dt_val = datetime.fromtimestamp(int(x.get("datetime", 0)), tz=timezone.utc).isoformat()
        except Exception:
            dt_val = None
        items.append(
            {
                "headline": x.get("headline"),
                "summary": x.get("summary"),
                "source": x.get("source"),
                "datetime": dt_val,
                "url": x.get("url"),
            }
        )
    return items


def _count_hits(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def compute_news_score(news_items: list[dict]) -> int:
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
        raw = float(pos - neg)

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
    scaled = 10.0 * math.tanh(avg / 2.0)
    return int(round(max(-10.0, min(10.0, scaled))))


def _safe_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None


def _get_earnings_calendar(ticker: str, days_ahead: int = 45) -> dict | None:
    today = datetime.now(timezone.utc).date()
    to = today + timedelta(days=days_ahead)
    data = finnhub_get(
        "/calendar/earnings",
        {
            "from": today.isoformat(),
            "to": to.isoformat(),
            "symbol": ticker,
        },
    )
    if not isinstance(data, dict):
        return None

    rows = data.get("earningsCalendar") or []
    if not rows:
        return None

    upcoming = rows[0]
    return {
        "date": upcoming.get("date"),
        "eps_estimate": upcoming.get("epsEstimate"),
        "revenue_estimate": upcoming.get("revenueEstimate"),
        "hour": upcoming.get("hour"),
    }


def _get_earnings_history(ticker: str, limit: int = 8) -> list[dict]:
    data = finnhub_get("/stock/earnings", {"symbol": ticker, "limit": max(4, limit)})
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for row in data[:limit]:
        out.append(
            {
                "period": row.get("period"),
                "actual": row.get("actual"),
                "estimate": row.get("estimate"),
                "surprise_percent": row.get("surprisePercent"),
            }
        )
    return out


def _get_daily_closes(ticker: str, frm: date, to: date) -> dict[date, float]:
    data = finnhub_get(
        "/stock/candle",
        {
            "symbol": ticker,
            "resolution": "D",
            "from": int(datetime(frm.year, frm.month, frm.day, tzinfo=timezone.utc).timestamp()),
            "to": int(datetime(to.year, to.month, to.day, tzinfo=timezone.utc).timestamp()),
        },
    )
    if not isinstance(data, dict):
        return {}
    if data.get("s") != "ok":
        return {}

    closes = data.get("c") or []
    times = data.get("t") or []
    out: dict[date, float] = {}
    for ts, close in zip(times, closes):
        try:
            d = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            out[d] = float(close)
        except Exception:
            continue
    return out


def _price_change_after_event(closes: dict[date, float], event_day: date) -> float | None:
    # Compare first close after event with previous close before event.
    prev_candidates = [d for d in closes.keys() if d < event_day]
    next_candidates = [d for d in closes.keys() if d > event_day]
    if not prev_candidates or not next_candidates:
        return None

    prev_day = max(prev_candidates)
    next_day = min(next_candidates)

    prev_close = closes.get(prev_day)
    next_close = closes.get(next_day)
    if prev_close in (None, 0) or next_close is None:
        return None

    return ((next_close - prev_close) / prev_close) * 100.0


def _compute_52w_position(last_price: float | None, ticker: str) -> float | None:
    if last_price is None:
        return None

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=370)
    closes = _get_daily_closes(ticker, start, end)
    if not closes:
        return None

    vals = list(closes.values())
    low = min(vals)
    high = max(vals)
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (last_price - low) / (high - low)))


def compute_earnings_signal(ticker: str, last_price: float | None) -> tuple[int, dict]:
    upcoming = _get_earnings_calendar(ticker, days_ahead=45)
    history = _get_earnings_history(ticker, limit=8)

    periods = [_safe_date(h.get("period")) for h in history]
    periods = [p for p in periods if p is not None]

    reaction_avg = None
    reaction_up_rate = None
    reaction_samples = 0

    if periods:
        start = min(periods) - timedelta(days=7)
        end = datetime.now(timezone.utc).date()
        closes = _get_daily_closes(ticker, start, end)
        reactions: list[float] = []
        for p in periods:
            chg = _price_change_after_event(closes, p)
            if chg is not None:
                reactions.append(chg)
        if reactions:
            reaction_samples = len(reactions)
            reaction_avg = sum(reactions) / len(reactions)
            reaction_up_rate = sum(1 for x in reactions if x > 0) / len(reactions)

    surprise_vals = [h.get("surprise_percent") for h in history if isinstance(h.get("surprise_percent"), (int, float))]
    avg_surprise = (sum(surprise_vals) / len(surprise_vals)) if surprise_vals else None

    pos = _compute_52w_position(last_price, ticker)

    score_raw = 0.0
    if reaction_avg is not None:
        score_raw += max(-1.5, min(1.5, reaction_avg / 3.0))
    if reaction_up_rate is not None:
        score_raw += (reaction_up_rate - 0.5) * 1.4
    if avg_surprise is not None:
        score_raw += max(-1.2, min(1.2, avg_surprise / 8.0))

    # If price is near 52w high, cap optimistic earnings bias.
    if pos is not None:
        if pos >= 0.85 and score_raw > 0:
            score_raw *= 0.55
        elif pos <= 0.15 and score_raw > 0:
            score_raw *= 1.15
        elif pos >= 0.85 and score_raw < 0:
            score_raw *= 1.1

    days_to_earnings = None
    if upcoming and upcoming.get("date"):
        d = _safe_date(upcoming.get("date"))
        if d:
            days_to_earnings = (d - datetime.now(timezone.utc).date()).days

    # Increase weight when announcement is near.
    horizon_mult = 1.0
    if days_to_earnings is not None:
        if days_to_earnings <= 7:
            horizon_mult = 1.4
        elif days_to_earnings <= 14:
            horizon_mult = 1.2

    earnings_score = int(round(max(-10.0, min(10.0, 4.0 * score_raw * horizon_mult))))

    context = {
        "upcoming": upcoming,
        "days_to_earnings": days_to_earnings,
        "avg_post_earnings_move_pct": None if reaction_avg is None else round(reaction_avg, 2),
        "post_earnings_up_rate": None if reaction_up_rate is None else round(reaction_up_rate, 2),
        "reaction_samples": reaction_samples,
        "avg_surprise_percent": None if avg_surprise is None else round(avg_surprise, 2),
        "price_position_52w": None if pos is None else round(pos, 3),
    }
    return earnings_score, context


def build_swing_plan(tickers: list[str]) -> list[PlanRow]:
    rows: list[PlanRow] = []

    for t in tickers:
        last = get_last_price(t)
        news = get_company_news_summary(t, days=7, limit=5)
        news_score = compute_news_score(news)

        earnings_score, earnings_context = compute_earnings_signal(t, last)
        combined_signal = news_score + earnings_score

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
                    news_score=news_score,
                    earnings_score=earnings_score,
                    earnings_context=earnings_context,
                )
            )
            continue

        entry = float(last)
        stop = entry * 0.97
        take_profit = entry * 1.06

        if combined_signal >= 4:
            strategy_action = "BUY"
        elif combined_signal <= -4:
            strategy_action = "WAIT"
        else:
            strategy_action = "HOLD / WAIT"

        reason = (
            f"signal={combined_signal} (news={news_score}, earnings={earnings_score}); "
            f"earnings move avg={earnings_context.get('avg_post_earnings_move_pct')}, "
            f"up-rate={earnings_context.get('post_earnings_up_rate')}, "
            f"52w-pos={earnings_context.get('price_position_52w')}, "
            f"surprise% avg={earnings_context.get('avg_surprise_percent')}"
        )

        rows.append(
            PlanRow(
                ticker=t,
                last=entry,
                entry=entry,
                stop=stop,
                take_profit=take_profit,
                strategy_action=strategy_action,
                strategy_reason=reason,
                max_hold_date=datetime.now(timezone.utc) + timedelta(days=20),
                news=news,
                news_score=news_score,
                earnings_score=earnings_score,
                earnings_context=earnings_context,
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

    if last_price <= stop:
        return False, "wait_good_avoid", ret
    if last_price >= take_profit:
        return False, ("wait_missed_tp" if not expired else "wait_missed_tp_expired"), ret
    return False, "wait_neutral", ret
