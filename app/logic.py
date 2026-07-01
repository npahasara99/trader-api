from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
import os
import math
import requests
from typing import Callable

from .config import DEFAULT_PLANNING_CONFIG
from .planner import generate_structured_plan


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
    signal_score: int = 0
    market_regime: str | None = None
    prob_tp: float | None = None
    prob_sl: float | None = None
    prob_open: float | None = None
    expected_return: float | None = None
    confidence: float | None = None
    buy_threshold: int | None = None
    avoid_threshold: int | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    hold_days: int | None = None
    risk_tuning_reason: str | None = None
    current_price: float | None = None
    trend_state: str | None = None
    range_position_1m: float | None = None
    range_position_3m: float | None = None
    range_position_12m: float | None = None
    local_range_position: float | None = None
    distance_to_1m_high_pct: float | None = None
    distance_to_1m_low_pct: float | None = None
    distance_to_3m_high_pct: float | None = None
    distance_to_3m_low_pct: float | None = None
    distance_to_12m_high_pct: float | None = None
    distance_to_12m_low_pct: float | None = None
    distance_from_ema20_pct: float | None = None
    distance_from_sma50_pct: float | None = None
    distance_from_sma100_pct: float | None = None
    distance_from_sma200_pct: float | None = None
    recent_expansion_state: str | None = None
    recent_compression_state: str | None = None
    breakout_extension_state: str | None = None
    historical_range_context: str | None = None
    price_location_context: str | None = None
    setup_type: str | None = None
    catalyst_signals: list[str] = field(default_factory=list)
    news_directional_bias: str | None = None
    catalyst_strength_score: float | None = None
    catalyst_recency_score: float | None = None
    chart_news_alignment: str | None = None
    news_supports_continuation: bool | None = None
    news_supports_rebound: bool | None = None
    news_conflicts_with_chart: bool | None = None
    news_neutral: bool | None = None
    sector_regime: str | None = None
    macro_sensitivity_tag: str | None = None
    macro_alignment_score: float | None = None
    macro_context_label: str | None = None
    setup_scenario: str | None = None
    continuation_vs_reversion_bias: str | None = None
    news_regime_alignment: str | None = None
    tp_aggressiveness: str | None = None
    sl_tolerance: str | None = None
    expected_move_profile: str | None = None
    scenario_confidence: float | None = None
    scenario_rationale: str | None = None
    setup_context_summary: str | None = None
    location_context_summary: str | None = None
    support_zone_1: dict | None = None
    support_zone_2: dict | None = None
    resistance_zone_1: dict | None = None
    resistance_zone_2: dict | None = None
    atr: float | None = None
    atr_pct: float | None = None
    fib_levels: dict | None = None
    moving_averages: dict | None = None
    volume_context: dict | None = None
    relative_strength: dict | None = None
    earnings: dict | None = None
    entry_candidates: list[dict] = field(default_factory=list)
    preferred_entry: float | None = None
    preferred_entry_type: str | None = None
    entry_quality_score: float | None = None
    entry_distance_from_current_price_pct: float | None = None
    entry_confluence_score: float | None = None
    entry_requires_confirmation: bool | None = None
    confirmation_trigger: str | None = None
    stop_loss: float | None = None
    stop_basis: str | None = None
    stop_distance_pct: float | None = None
    stop_width_pct: float | None = None
    stop_width_atr: float | None = None
    stop_too_tight_flag: bool | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    take_profit_final: float | None = None
    tp1_distance_pct: float | None = None
    tp1_distance_atr: float | None = None
    tp_basis: str | None = None
    reward_risk: dict | None = None
    tp_too_optimistic_flag: bool | None = None
    hold_window_reachability_score: float | None = None
    swing_realism_flag: str | None = None
    risk_width_flag: str | None = None
    target_reachability_flag: str | None = None
    level_geometry_flag: str | None = None
    stop_generation_reason: str | None = None
    tp1_generation_reason: str | None = None
    max_hold_days: int | None = None
    trend_quality_score: float | None = None
    pullback_quality_score: float | None = None
    support_quality_score: float | None = None
    volatility_quality_score: float | None = None
    relative_strength_score: float | None = None
    volume_confirmation_score: float | None = None
    earnings_risk_score: float | None = None
    reward_risk_score: float | None = None
    historical_analogue_score: float | None = None
    llm_quality_score: float | None = None
    context_score: float | None = None
    catalyst_score: float | None = None
    macro_score: float | None = None
    scenario_score: float | None = None
    composite_score: float | None = None
    llm_review: dict | None = None
    quant_action: str | None = None
    reconciled_action: str | None = None
    final_action: str | None = None
    action_alignment: str | None = None
    action_reason_bucket: str | None = None
    monitorable_setup: bool | None = None
    avoid_severity_score: float | None = None
    wait_reason: str | None = None
    avoid_reason: str | None = None
    buy_blockers: list[str] = field(default_factory=list)
    constructive_traits: list[str] = field(default_factory=list)
    wait_type: str | None = None
    monitor_window_days: int | None = None
    monitor_until_date: datetime | None = None
    stale_after_date: datetime | None = None
    watch_priority: str | None = None
    days_to_trigger_estimate: float | None = None
    support_zone_1_display: str | None = None
    support_zone_2_display: str | None = None
    resistance_zone_1_display: str | None = None
    resistance_zone_2_display: str | None = None
    support_zone_1_midpoint: float | None = None
    support_zone_2_midpoint: float | None = None
    support_zone_1_width_pct: float | None = None
    support_zone_2_width_pct: float | None = None
    support_zone_1_note: str | None = None
    support_zone_2_note: str | None = None
    support_zone_summary: list[str] = field(default_factory=list)
    resistance_zone_summary: list[str] = field(default_factory=list)
    upgrade_triggers: list[str] = field(default_factory=list)
    failure_triggers: list[str] = field(default_factory=list)
    next_check_focus: list[str] = field(default_factory=list)
    setup_monitoring_summary: str | None = None
    chart_execution_view: dict | None = None
    what_to_watch: dict | None = None
    swing_trade_suitability: dict | None = None
    actionability_soon: dict | None = None
    watchlist_tier: str | None = None
    watchlist_bucket: str | None = None
    watchlist_summary: str | None = None
    watchlist_reason: str | None = None
    is_primary_watchlist_candidate: bool | None = None
    is_secondary_watchlist_candidate: bool | None = None
    pre_scan_score: float | None = None
    pre_scan_reason_tags: list[str] = field(default_factory=list)
    sector_relative_strength: float | None = None
    scanner_rank_score: float | None = None
    immediate_rank_score: float | None = None
    watchlist_rank_score: float | None = None
    ranking_bucket: str | None = None
    scan_shortlisted: bool | None = None
    scan_rejection_reason: str | None = None
    structure_flags: list[str] = field(default_factory=list)
    breakout_level: float | None = None
    prior_breakout_retest_zone: dict | None = None
    consolidation_range: dict | None = None
    gap_zone: dict | None = None
    recent_swing_highs: list[dict] = field(default_factory=list)
    recent_swing_lows: list[dict] = field(default_factory=list)


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


SP100_CLASSIFICATION = {
    "AAPL": {"sector": "technology", "industry": "consumer electronics"},
    "ABBV": {"sector": "health care", "industry": "biopharma"},
    "ABT": {"sector": "health care", "industry": "medical devices"},
    "ACN": {"sector": "technology", "industry": "it services"},
    "ADBE": {"sector": "technology", "industry": "software"},
    "AMD": {"sector": "technology", "industry": "semiconductors"},
    "AMGN": {"sector": "health care", "industry": "biopharma"},
    "AMT": {"sector": "real estate", "industry": "telecom towers"},
    "AMZN": {"sector": "consumer discretionary", "industry": "internet retail"},
    "AVGO": {"sector": "technology", "industry": "semiconductors"},
    "AXP": {"sector": "financials", "industry": "payments"},
    "BA": {"sector": "industrials", "industry": "aerospace"},
    "BAC": {"sector": "financials", "industry": "banks"},
    "BK": {"sector": "financials", "industry": "asset servicing"},
    "BKNG": {"sector": "consumer discretionary", "industry": "travel services"},
    "BLK": {"sector": "financials", "industry": "asset management"},
    "BMY": {"sector": "health care", "industry": "biopharma"},
    "BRK.B": {"sector": "financials", "industry": "insurance"},
    "C": {"sector": "financials", "industry": "banks"},
    "CAT": {"sector": "industrials", "industry": "machinery"},
    "CHTR": {"sector": "communication services", "industry": "cable"},
    "CL": {"sector": "consumer staples", "industry": "household products"},
    "CMCSA": {"sector": "communication services", "industry": "media"},
    "COF": {"sector": "financials", "industry": "consumer finance"},
    "COP": {"sector": "energy", "industry": "oil and gas"},
    "COST": {"sector": "consumer staples", "industry": "retail"},
    "CRM": {"sector": "technology", "industry": "software"},
    "CSCO": {"sector": "technology", "industry": "networking"},
    "CVX": {"sector": "energy", "industry": "oil and gas"},
    "DHR": {"sector": "health care", "industry": "life sciences"},
    "DIS": {"sector": "communication services", "industry": "entertainment"},
    "DOW": {"sector": "materials", "industry": "chemicals"},
    "DUK": {"sector": "utilities", "industry": "electric utilities"},
    "EMR": {"sector": "industrials", "industry": "electrical equipment"},
    "F": {"sector": "consumer discretionary", "industry": "autos"},
    "GD": {"sector": "industrials", "industry": "aerospace"},
    "GE": {"sector": "industrials", "industry": "industrial conglomerates"},
    "GILD": {"sector": "health care", "industry": "biotech"},
    "GM": {"sector": "consumer discretionary", "industry": "autos"},
    "GOOG": {"sector": "communication services", "industry": "internet platforms"},
    "GOOGL": {"sector": "communication services", "industry": "internet platforms"},
    "GS": {"sector": "financials", "industry": "capital markets"},
    "HD": {"sector": "consumer discretionary", "industry": "home improvement retail"},
    "HON": {"sector": "industrials", "industry": "industrial conglomerates"},
    "IBM": {"sector": "technology", "industry": "it services"},
    "INTC": {"sector": "technology", "industry": "semiconductors"},
    "JNJ": {"sector": "health care", "industry": "pharma"},
    "JPM": {"sector": "financials", "industry": "banks"},
    "KHC": {"sector": "consumer staples", "industry": "food products"},
    "KMI": {"sector": "energy", "industry": "midstream"},
    "KO": {"sector": "consumer staples", "industry": "beverages"},
    "LIN": {"sector": "materials", "industry": "industrial gases"},
    "LLY": {"sector": "health care", "industry": "biopharma"},
    "LMT": {"sector": "industrials", "industry": "aerospace"},
    "LOW": {"sector": "consumer discretionary", "industry": "home improvement retail"},
    "MA": {"sector": "financials", "industry": "payments"},
    "MCD": {"sector": "consumer discretionary", "industry": "restaurants"},
    "MDT": {"sector": "health care", "industry": "medical devices"},
    "MET": {"sector": "financials", "industry": "insurance"},
    "META": {"sector": "communication services", "industry": "internet platforms"},
    "MMM": {"sector": "industrials", "industry": "industrial conglomerates"},
    "MO": {"sector": "consumer staples", "industry": "tobacco"},
    "MRK": {"sector": "health care", "industry": "biopharma"},
    "MS": {"sector": "financials", "industry": "capital markets"},
    "MSFT": {"sector": "technology", "industry": "software"},
    "NEE": {"sector": "utilities", "industry": "electric utilities"},
    "NFLX": {"sector": "communication services", "industry": "streaming"},
    "NKE": {"sector": "consumer discretionary", "industry": "apparel"},
    "NVDA": {"sector": "technology", "industry": "semiconductors"},
    "ORCL": {"sector": "technology", "industry": "software"},
    "PEP": {"sector": "consumer staples", "industry": "beverages"},
    "PFE": {"sector": "health care", "industry": "biopharma"},
    "PG": {"sector": "consumer staples", "industry": "household products"},
    "PM": {"sector": "consumer staples", "industry": "tobacco"},
    "PYPL": {"sector": "financials", "industry": "payments"},
    "QCOM": {"sector": "technology", "industry": "semiconductors"},
    "RTX": {"sector": "industrials", "industry": "aerospace"},
    "SBUX": {"sector": "consumer discretionary", "industry": "restaurants"},
    "SCHW": {"sector": "financials", "industry": "brokerage"},
    "SO": {"sector": "utilities", "industry": "electric utilities"},
    "SPG": {"sector": "real estate", "industry": "retail reit"},
    "T": {"sector": "communication services", "industry": "telecom"},
    "TGT": {"sector": "consumer staples", "industry": "retail"},
    "TMO": {"sector": "health care", "industry": "life sciences"},
    "TMUS": {"sector": "communication services", "industry": "telecom"},
    "TSLA": {"sector": "consumer discretionary", "industry": "autos"},
    "TXN": {"sector": "technology", "industry": "semiconductors"},
    "UNH": {"sector": "health care", "industry": "managed care"},
    "UNP": {"sector": "industrials", "industry": "railroads"},
    "UPS": {"sector": "industrials", "industry": "logistics"},
    "USB": {"sector": "financials", "industry": "banks"},
    "V": {"sector": "financials", "industry": "payments"},
    "VZ": {"sector": "communication services", "industry": "telecom"},
    "WBA": {"sector": "consumer staples", "industry": "pharmacy retail"},
    "WFC": {"sector": "financials", "industry": "banks"},
    "WMT": {"sector": "consumer staples", "industry": "retail"},
    "XOM": {"sector": "energy", "industry": "oil and gas"},
    "AIG": {"sector": "financials", "industry": "insurance"},
    "DE": {"sector": "industrials", "industry": "machinery"},
    "NOW": {"sector": "technology", "industry": "software"},
}

FILTER_ALIASES = {
    "tech": "technology",
    "technology": "technology",
    "it": "technology",
    "software": "software",
    "semis": "semiconductors",
    "semiconductor": "semiconductors",
    "semiconductors": "semiconductors",
    "healthcare": "health care",
    "health care": "health care",
    "pharma": "biopharma",
    "biotech": "biotech",
    "finance": "financials",
    "financial": "financials",
    "financials": "financials",
    "bank": "banks",
    "banks": "banks",
    "comms": "communication services",
    "communication": "communication services",
    "communication services": "communication services",
    "media": "media",
    "telecom": "telecom",
    "consumer": "consumer discretionary",
    "consumer discretionary": "consumer discretionary",
    "consumer staples": "consumer staples",
    "retail": "retail",
    "energy": "energy",
    "oil": "oil and gas",
    "industrials": "industrials",
    "industrial": "industrials",
    "aerospace": "aerospace",
    "defense": "aerospace",
    "utilities": "utilities",
    "materials": "materials",
    "real estate": "real estate",
    "reit": "retail reit",
}


def _normalize_filter_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    return FILTER_ALIASES.get(cleaned, cleaned)


def _ticker_matches_filter(ticker: str, *, sector: str | None = None, industry: str | None = None) -> bool:
    if sector is None and industry is None:
        return True

    meta = SP100_CLASSIFICATION.get(ticker, {})
    sector_val = _normalize_filter_value(meta.get("sector"))
    industry_val = _normalize_filter_value(meta.get("industry"))

    if sector is not None and sector not in {sector_val, industry_val}:
        return False
    if industry is not None and industry not in {industry_val, sector_val}:
        return False
    return True


def get_sp100_universe(
    top_n: int | None = None,
    *,
    sector: str | None = None,
    industry: str | None = None,
) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    sector_filter = _normalize_filter_value(sector)
    industry_filter = _normalize_filter_value(industry)
    for t in SP100_UNIVERSE:
        if t not in seen:
            seen.add(t)
            if _ticker_matches_filter(t, sector=sector_filter, industry=industry_filter):
                uniq.append(t)

    if top_n is None:
        return uniq
    n = max(1, min(int(top_n), len(uniq)))
    return uniq[:n]


DailyClosesLoader = Callable[[str, date, date], dict[date, float]]
DailyBarsLoader = Callable[[str], list[dict]]

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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def get_last_price(ticker: str) -> float | None:
    data = finnhub_get("/quote", {"symbol": ticker})
    if not isinstance(data, dict) or data.get("c") in (None, 0):
        return None
    return float(data["c"])


def get_last_price_or_recent_close(
    ticker: str,
    *,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> float | None:
    last = get_last_price(ticker)
    if last is not None:
        return last

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=14)
    close_map = _get_daily_closes(ticker, start, end, daily_closes_loader=daily_closes_loader)
    if not close_map:
        return None

    latest_day = max(close_map.keys())
    return float(close_map[latest_day])


def _moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    tail = values[-window:]
    return sum(tail) / max(len(tail), 1)


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


def _normalize_earnings_session(hour: str | None) -> str:
    normalized = str(hour or "").strip().lower()
    if normalized in {"bmo", "before market", "before open", "pre-market", "premarket"}:
        return "before_open"
    if normalized in {"amc", "after market", "after close", "post-market", "postmarket"}:
        return "after_close"
    return "unknown"


def get_upcoming_earnings_calendar(
    *,
    days_ahead: int = 30,
    sector: str | None = None,
    industry: str | None = None,
    sp100_only: bool = False,
    tickers: list[str] | None = None,
) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    to = today + timedelta(days=max(1, int(days_ahead)))
    data = finnhub_get(
        "/calendar/earnings",
        {
            "from": today.isoformat(),
            "to": to.isoformat(),
        },
    )
    if not isinstance(data, dict):
        return []

    rows = data.get("earningsCalendar") or []
    allowed_symbols: set[str] | None = None
    if tickers:
        allowed_symbols = {str(t).strip().upper() for t in tickers if str(t).strip()}

    sector_filter = _normalize_filter_value(sector)
    industry_filter = _normalize_filter_value(industry)

    results: list[dict] = []
    for row in rows:
        ticker = str(row.get("symbol") or "").strip().upper()
        if not ticker:
            continue
        if allowed_symbols is not None and ticker not in allowed_symbols:
            continue
        if sp100_only and ticker not in SP100_CLASSIFICATION:
            continue
        if not _ticker_matches_filter(ticker, sector=sector_filter, industry=industry_filter):
            continue

        earnings_date = _safe_date(row.get("date"))
        if earnings_date is None:
            continue
        days_to_earnings = (earnings_date - today).days
        if days_to_earnings < 0:
            continue

        meta = SP100_CLASSIFICATION.get(ticker, {})
        session = _normalize_earnings_session(row.get("hour"))
        results.append(
            {
                "ticker": ticker,
                "company_name": row.get("name") or row.get("company") or None,
                "earnings_date": earnings_date.isoformat(),
                "earnings_session": session,
                "earnings_time": row.get("hour"),
                "days_to_earnings": days_to_earnings,
                "sector": meta.get("sector"),
                "industry": meta.get("industry"),
                "eps_estimate": row.get("epsEstimate"),
                "eps_actual": row.get("epsActual"),
                "revenue_estimate": row.get("revenueEstimate"),
                "revenue_actual": row.get("revenueActual"),
            }
        )

    results.sort(key=lambda item: (item.get("days_to_earnings", 9999), item.get("ticker", "")))
    return results


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


def _get_daily_closes(
    ticker: str,
    frm: date,
    to: date,
    *,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> dict[date, float]:
    if daily_closes_loader is not None:
        try:
            cached = daily_closes_loader(ticker, frm, to)
            if cached:
                return cached
        except Exception:
            pass

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


def _get_daily_bars(
    ticker: str,
    *,
    daily_bars_loader: DailyBarsLoader | None = None,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> list[dict]:
    """Load daily OHLCV bars, synthesizing from close-only history when needed."""
    if daily_bars_loader is not None:
        try:
            rows = daily_bars_loader(ticker)
            if rows:
                return rows
        except Exception:
            pass

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DEFAULT_PLANNING_CONFIG.history_lookback_days)
    closes = _get_daily_closes(ticker, start, end, daily_closes_loader=daily_closes_loader)
    if not closes:
        return []

    out: list[dict] = []
    for bar_day in sorted(closes.keys()):
        close_val = float(closes[bar_day])
        out.append(
            {
                "symbol": ticker,
                "bar_date": bar_day,
                "open": close_val,
                "high": close_val,
                "low": close_val,
                "close": close_val,
                "volume": None,
                "adjusted_close": close_val,
                "source": "close_only_fallback",
            }
        )
    return out


def detect_market_regime(
    breadth_universe: list[str] | None = None,
    *,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    end = now.date()
    start = end - timedelta(days=130)

    spy_closes_map = _get_daily_closes("SPY", start, end, daily_closes_loader=daily_closes_loader)
    spy_closes = [spy_closes_map[d] for d in sorted(spy_closes_map.keys())]

    spy_price = spy_closes[-1] if spy_closes else None
    spy_ma20 = _moving_average(spy_closes, 20)
    spy_ma50 = _moving_average(spy_closes, 50)

    trend_score = 0.0
    if spy_price is not None and spy_ma20 is not None:
        trend_score += 0.7 if spy_price >= spy_ma20 else -0.7
    if spy_price is not None and spy_ma50 is not None:
        trend_score += 1.1 if spy_price >= spy_ma50 else -1.1
    if spy_ma20 is not None and spy_ma50 is not None:
        trend_score += 0.7 if spy_ma20 >= spy_ma50 else -0.7

    breadth = breadth_universe or [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA", "JPM", "XOM", "UNH", "HD", "LLY"
    ]
    breadth = breadth[:12]
    breadth_ok = 0
    breadth_total = 0

    for t in breadth:
        c_map = _get_daily_closes(t, end - timedelta(days=70), end, daily_closes_loader=daily_closes_loader)
        if not c_map:
            continue
        vals = [c_map[d] for d in sorted(c_map.keys())]
        if not vals:
            continue
        ma20 = _moving_average(vals, 20)
        if ma20 is None:
            continue
        breadth_total += 1
        if vals[-1] >= ma20:
            breadth_ok += 1

    breadth_ratio = (breadth_ok / breadth_total) if breadth_total else None
    if breadth_ratio is not None:
        trend_score += (breadth_ratio - 0.5) * 2.0

    if trend_score >= 0.85:
        regime = "risk_on"
    elif trend_score <= -0.85:
        regime = "risk_off"
    else:
        regime = "neutral"

    return {
        "as_of": now,
        "regime": regime,
        "score": round(trend_score, 4),
        "spy_price": spy_price,
        "spy_ma20": spy_ma20,
        "spy_ma50": spy_ma50,
        "breadth_ratio": breadth_ratio,
        "breadth_samples": breadth_total,
    }


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


def _compute_52w_position(
    last_price: float | None,
    ticker: str,
    *,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> float | None:
    if last_price is None:
        return None

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=370)
    closes = _get_daily_closes(ticker, start, end, daily_closes_loader=daily_closes_loader)
    if not closes:
        return None

    vals = list(closes.values())
    low = min(vals)
    high = max(vals)
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (last_price - low) / (high - low)))


def compute_earnings_signal(
    ticker: str,
    last_price: float | None,
    *,
    daily_closes_loader: DailyClosesLoader | None = None,
) -> tuple[int, dict]:
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
        closes = _get_daily_closes(ticker, start, end, daily_closes_loader=daily_closes_loader)
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

    pos = _compute_52w_position(last_price, ticker, daily_closes_loader=daily_closes_loader)

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


def estimate_trade_probabilities(
    *,
    signal_score: int,
    entry: float,
    stop: float,
    take_profit: float,
    regime: str,
    history_win_rate: float | None,
    history_samples: int,
) -> dict:
    # Baseline logits from signal strength.
    score_tp = -0.2 + 0.24 * float(signal_score)
    score_sl = -0.35 - 0.19 * float(signal_score)

    if regime == "risk_on":
        score_tp += 0.35
        score_sl -= 0.22
    elif regime == "risk_off":
        score_tp -= 0.45
        score_sl += 0.35

    confidence_hist = min(1.0, max(0.0, history_samples / 10.0))
    if history_win_rate is not None:
        score_tp += (history_win_rate - 0.5) * 1.3 * confidence_hist
        score_sl += (0.5 - history_win_rate) * 0.9 * confidence_hist

    p_tp = _sigmoid(score_tp)
    p_sl = _sigmoid(score_sl)

    # Keep room for open/undecided state.
    total = p_tp + p_sl
    if total > 0.92:
        k = 0.92 / total
        p_tp *= k
        p_sl *= k

    p_open = max(0.0, 1.0 - p_tp - p_sl)

    reward = (take_profit - entry) / max(entry, 1e-9)
    loss = (entry - stop) / max(entry, 1e-9)
    open_drift = (p_tp - p_sl) * 0.25 * reward

    expected_return = p_tp * reward - p_sl * loss + p_open * open_drift

    confidence = 0.35 + 0.3 * min(1.0, abs(signal_score) / 8.0) + 0.25 * confidence_hist
    if regime != "neutral":
        confidence += 0.1
    confidence = max(0.0, min(0.95, confidence))

    return {
        "p_tp": float(round(p_tp, 6)),
        "p_sl": float(round(p_sl, 6)),
        "p_open": float(round(p_open, 6)),
        "expected_return": float(round(expected_return, 6)),
        "confidence": float(round(confidence, 6)),
    }


def build_swing_plan(
    tickers: list[str],
    *,
    regime: str | None = None,
    buy_threshold: int = 4,
    avoid_threshold: int = -4,
    daily_closes_loader: DailyClosesLoader | None = None,
    daily_bars_loader: DailyBarsLoader | None = None,
    history_stats_by_ticker: dict[str, dict] | None = None,
    pre_scan_by_ticker: dict[str, dict] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_style: str | None = None,
) -> list[PlanRow]:
    rows: list[PlanRow] = []
    regime_val = regime or "neutral"
    benchmark_bars: dict[str, list[dict]] = {}

    for benchmark in DEFAULT_PLANNING_CONFIG.benchmark_symbols:
        benchmark_bars[benchmark] = _get_daily_bars(
            benchmark,
            daily_bars_loader=daily_bars_loader,
            daily_closes_loader=daily_closes_loader,
        )

    for t in tickers:
        pre_scan = (pre_scan_by_ticker or {}).get(t, {}) or {}
        last = get_last_price_or_recent_close(t, daily_closes_loader=daily_closes_loader)
        news = get_company_news_summary(t, days=7, limit=5)
        news_score = compute_news_score(news)

        earnings_score, earnings_context = compute_earnings_signal(
            t,
            last,
            daily_closes_loader=daily_closes_loader,
        )
        signal_score = int(news_score + earnings_score)

        if last is None:
            rows.append(
                PlanRow(
                    ticker=t,
                    last=None,
                    entry=None,
                    stop=None,
                    take_profit=None,
                    strategy_action="WAIT",
                    strategy_reason="Price unavailable (Finnhub quote failed or key missing)",
                    max_hold_date=datetime.now(timezone.utc) + timedelta(days=20),
                    news=news,
                    news_score=news_score,
                    earnings_score=earnings_score,
                    earnings_context=earnings_context,
                    signal_score=signal_score,
                    market_regime=regime_val,
                    buy_threshold=buy_threshold,
                    avoid_threshold=avoid_threshold,
                    current_price=None,
                    pre_scan_score=pre_scan.get("pre_scan_score"),
                    pre_scan_reason_tags=list(pre_scan.get("pre_scan_reason_tags") or []),
                    sector_relative_strength=pre_scan.get("sector_relative_strength"),
                    scan_shortlisted=pre_scan.get("scan_shortlisted"),
                    scan_rejection_reason=pre_scan.get("scan_rejection_reason"),
                )
            )
            continue

        bars = _get_daily_bars(
            t,
            daily_bars_loader=daily_bars_loader,
            daily_closes_loader=daily_closes_loader,
        )

        if not bars:
            rows.append(
                PlanRow(
                    ticker=t,
                    last=float(last),
                    entry=None,
                    stop=None,
                    take_profit=None,
                    strategy_action="WAIT",
                    strategy_reason="Historical bars unavailable for structured planning",
                    max_hold_date=datetime.now(timezone.utc) + timedelta(days=20),
                    news=news,
                    news_score=news_score,
                    earnings_score=earnings_score,
                    earnings_context=earnings_context,
                    signal_score=signal_score,
                    market_regime=regime_val,
                    current_price=float(last),
                    buy_threshold=buy_threshold,
                    avoid_threshold=avoid_threshold,
                    pre_scan_score=pre_scan.get("pre_scan_score"),
                    pre_scan_reason_tags=list(pre_scan.get("pre_scan_reason_tags") or []),
                    sector_relative_strength=pre_scan.get("sector_relative_strength"),
                    scan_shortlisted=pre_scan.get("scan_shortlisted"),
                    scan_rejection_reason=pre_scan.get("scan_rejection_reason"),
                )
            )
            continue

        structured = generate_structured_plan(
            ticker=t,
            current_price=float(last),
            bars=bars,
            news_items=news,
            news_score=news_score,
            earnings_score=earnings_score,
            earnings_context=earnings_context,
            market_regime=regime_val,
            buy_threshold=buy_threshold,
            avoid_threshold=avoid_threshold,
            history_stats=(history_stats_by_ticker or {}).get(t),
            benchmark_bars=benchmark_bars,
            ticker_meta=SP100_CLASSIFICATION.get(t),
            sector_relative_strength=pre_scan.get("sector_relative_strength"),
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_style=llm_style,
        )

        probs = estimate_trade_probabilities(
            signal_score=signal_score,
            entry=float(structured["preferred_entry"]),
            stop=float(structured["stop_loss"]),
            take_profit=float(structured["take_profit_1"]),
            regime=regime_val,
            history_win_rate=((history_stats_by_ticker or {}).get(t, {}) or {}).get("win_rate"),
            history_samples=int(((history_stats_by_ticker or {}).get(t, {}) or {}).get("samples", 0)),
        )

        llm_review = structured.get("llm_review") or {}
        reason = (
            f"{structured['strategy_reason']}; news={news_score}; earnings={earnings_score}; "
            f"entry_type={structured.get('preferred_entry_type')}; "
            f"rr1={(structured.get('reward_risk') or {}).get('tp1')}; "
            f"risk={llm_review.get('key_risk')}"
        )

        rows.append(
            PlanRow(
                ticker=t,
                last=float(last),
                entry=float(structured["preferred_entry"]),
                stop=float(structured["stop_loss"]),
                take_profit=float(structured["take_profit_1"]),
                strategy_action=str(structured["strategy_action"]),
                strategy_reason=reason,
                max_hold_date=structured["max_hold_date"],
                news=news,
                llm_action=llm_review.get("llm_action"),
                llm_rationale=" | ".join(llm_review.get("rationale") or []),
                news_score=news_score,
                earnings_score=earnings_score,
                earnings_context=earnings_context,
                signal_score=signal_score,
                market_regime=regime_val,
                prob_tp=probs["p_tp"],
                prob_sl=probs["p_sl"],
                prob_open=probs["p_open"],
                expected_return=probs["expected_return"],
                confidence=probs["confidence"],
                buy_threshold=buy_threshold,
                avoid_threshold=avoid_threshold,
                stop_loss_pct=(float(structured["preferred_entry"] - structured["stop_loss"]) / max(float(structured["preferred_entry"]), 1e-9)),
                take_profit_pct=(float(structured["take_profit_1"] - structured["preferred_entry"]) / max(float(structured["preferred_entry"]), 1e-9)),
                hold_days=structured["max_hold_days"],
                risk_tuning_reason=structured["risk_tuning_reason"],
                current_price=structured["current_price"],
                trend_state=structured["trend_state"],
                range_position_1m=structured.get("range_position_1m"),
                range_position_3m=structured.get("range_position_3m"),
                range_position_12m=structured.get("range_position_12m"),
                local_range_position=structured.get("local_range_position"),
                distance_to_1m_high_pct=structured.get("distance_to_1m_high_pct"),
                distance_to_1m_low_pct=structured.get("distance_to_1m_low_pct"),
                distance_to_3m_high_pct=structured.get("distance_to_3m_high_pct"),
                distance_to_3m_low_pct=structured.get("distance_to_3m_low_pct"),
                distance_to_12m_high_pct=structured.get("distance_to_12m_high_pct"),
                distance_to_12m_low_pct=structured.get("distance_to_12m_low_pct"),
                distance_from_ema20_pct=structured.get("distance_from_ema20_pct"),
                distance_from_sma50_pct=structured.get("distance_from_sma50_pct"),
                distance_from_sma100_pct=structured.get("distance_from_sma100_pct"),
                distance_from_sma200_pct=structured.get("distance_from_sma200_pct"),
                recent_expansion_state=structured.get("recent_expansion_state"),
                recent_compression_state=structured.get("recent_compression_state"),
                breakout_extension_state=structured.get("breakout_extension_state"),
                historical_range_context=structured.get("historical_range_context"),
                price_location_context=structured.get("price_location_context"),
                setup_type=structured.get("setup_type"),
                catalyst_signals=list(structured.get("catalyst_signals") or []),
                news_directional_bias=structured.get("news_directional_bias"),
                catalyst_strength_score=structured.get("catalyst_strength_score"),
                catalyst_recency_score=structured.get("catalyst_recency_score"),
                chart_news_alignment=structured.get("chart_news_alignment"),
                news_supports_continuation=structured.get("news_supports_continuation"),
                news_supports_rebound=structured.get("news_supports_rebound"),
                news_conflicts_with_chart=structured.get("news_conflicts_with_chart"),
                news_neutral=structured.get("news_neutral"),
                sector_regime=structured.get("sector_regime"),
                macro_sensitivity_tag=structured.get("macro_sensitivity_tag"),
                macro_alignment_score=structured.get("macro_alignment_score"),
                macro_context_label=structured.get("macro_context_label"),
                setup_scenario=structured.get("setup_scenario"),
                continuation_vs_reversion_bias=structured.get("continuation_vs_reversion_bias"),
                news_regime_alignment=structured.get("news_regime_alignment"),
                tp_aggressiveness=structured.get("tp_aggressiveness"),
                sl_tolerance=structured.get("sl_tolerance"),
                expected_move_profile=structured.get("expected_move_profile"),
                scenario_confidence=structured.get("scenario_confidence"),
                scenario_rationale=structured.get("scenario_rationale"),
                setup_context_summary=structured.get("setup_context_summary"),
                location_context_summary=structured.get("location_context_summary"),
                support_zone_1=structured["support_zone_1"],
                support_zone_2=structured["support_zone_2"],
                resistance_zone_1=structured["resistance_zone_1"],
                resistance_zone_2=structured["resistance_zone_2"],
                atr=structured["atr"],
                atr_pct=structured["atr_pct"],
                fib_levels=structured["fib_levels"],
                moving_averages=structured["moving_averages"],
                volume_context=structured["volume_context"],
                relative_strength=structured["relative_strength"],
                earnings=structured["earnings"],
                entry_candidates=structured["entry_candidates"],
                preferred_entry=structured["preferred_entry"],
                preferred_entry_type=structured["preferred_entry_type"],
                entry_quality_score=structured["entry_quality_score"],
                entry_distance_from_current_price_pct=structured["entry_distance_from_current_price_pct"],
                entry_confluence_score=structured["entry_confluence_score"],
                entry_requires_confirmation=structured["entry_requires_confirmation"],
                confirmation_trigger=structured["confirmation_trigger"],
                stop_loss=structured["stop_loss"],
                stop_basis=structured["stop_basis"],
                stop_distance_pct=structured["stop_distance_pct"],
                stop_width_pct=structured["stop_width_pct"],
                stop_width_atr=structured["stop_width_atr"],
                stop_too_tight_flag=structured["stop_too_tight_flag"],
                take_profit_1=structured["take_profit_1"],
                take_profit_2=structured["take_profit_2"],
                take_profit_final=structured["take_profit_final"],
                tp1_distance_pct=structured["tp1_distance_pct"],
                tp1_distance_atr=structured["tp1_distance_atr"],
                tp_basis=structured["tp_basis"],
                reward_risk=structured["reward_risk"],
                tp_too_optimistic_flag=structured["tp_too_optimistic_flag"],
                hold_window_reachability_score=structured["hold_window_reachability_score"],
                swing_realism_flag=structured["swing_realism_flag"],
                risk_width_flag=structured["risk_width_flag"],
                target_reachability_flag=structured["target_reachability_flag"],
                level_geometry_flag=structured["level_geometry_flag"],
                stop_generation_reason=structured["stop_generation_reason"],
                tp1_generation_reason=structured["tp1_generation_reason"],
                max_hold_days=structured["max_hold_days"],
                trend_quality_score=structured["trend_quality_score"],
                pullback_quality_score=structured["pullback_quality_score"],
                support_quality_score=structured["support_quality_score"],
                volatility_quality_score=structured["volatility_quality_score"],
                relative_strength_score=structured["relative_strength_score"],
                volume_confirmation_score=structured["volume_confirmation_score"],
                earnings_risk_score=structured["earnings_risk_score"],
                reward_risk_score=structured["reward_risk_score"],
                historical_analogue_score=structured["historical_analogue_score"],
                llm_quality_score=structured["llm_quality_score"],
                context_score=structured.get("context_score"),
                catalyst_score=structured.get("catalyst_score"),
                macro_score=structured.get("macro_score"),
                scenario_score=structured.get("scenario_score"),
                composite_score=structured["composite_score"],
                llm_review=structured["llm_review"],
                pre_scan_score=pre_scan.get("pre_scan_score"),
                pre_scan_reason_tags=list(pre_scan.get("pre_scan_reason_tags") or []),
                sector_relative_strength=pre_scan.get("sector_relative_strength"),
                scan_shortlisted=pre_scan.get("scan_shortlisted"),
                scan_rejection_reason=pre_scan.get("scan_rejection_reason"),
                structure_flags=structured["structure_flags"],
                breakout_level=structured["breakout_level"],
                prior_breakout_retest_zone=structured["prior_breakout_retest_zone"],
                consolidation_range=structured["consolidation_range"],
                gap_zone=structured["gap_zone"],
                recent_swing_highs=structured["recent_swing_highs"],
                recent_swing_lows=structured["recent_swing_lows"],
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
