from types import SimpleNamespace

from app.config import DEFAULT_PLANNING_CONFIG
from app.opportunity_ranking import (
    build_daily_actionability_profile,
    build_portfolio_snapshot,
    build_raw_setup_profile,
    rank_daily_opportunities,
)
from app.universe import _load_fallback


def _row(
    ticker: str,
    *,
    current_price: float = 100.0,
    trigger: float = 99.0,
    final_action: str = "BUY",
    confirmed: bool = True,
    component_score: float = 9.0,
    stop: float = 95.0,
    tp1: float = 112.0,
):
    confirmation_score = 9.0 if confirmed else 5.0
    return SimpleNamespace(
        ticker=ticker,
        current_price=current_price,
        last=current_price,
        preferred_entry=100.0,
        preferred_entry_low=99.0,
        preferred_entry_high=101.0,
        confirmation_trigger_price=trigger,
        stop_loss=stop,
        invalidation_level=stop,
        take_profit_1=tp1,
        take_profit_2=118.0,
        atr=2.0,
        enhanced_trend_state="healthy_pullback",
        setup_type="constructive_pullback",
        final_action=final_action,
        entry_status="confirmed" if confirmed else "in_price_zone",
        confirmation_state="confirmed" if confirmed else "waiting_for_price",
        price_confirmed=confirmed,
        volume_confirmed=confirmed,
        executable_stop_technically_valid=True,
        tp_too_optimistic_flag=False,
        universe_eligible=True,
        universe_rejection_reasons=[],
        trend_score=component_score,
        price_location_score=component_score,
        support_confluence_score=component_score,
        multi_timeframe_alignment_score=component_score,
        relative_strength_score=component_score,
        volatility_suitability_score=component_score,
        liquidity_score=component_score,
        target_realism_score=component_score,
        reward_risk_score=component_score,
        catalyst_score=component_score,
        macro_score=component_score,
        confirmation_score=confirmation_score,
        hold_window_reachability_score=component_score,
        volume_confirmation_score=component_score if confirmed else 4.0,
        sector_relative_strength=0.03,
        reward_risk={"tp1": 2.4, "tp2": 3.6},
        signal_score=5,
    )


def _metadata(*tickers: str, sector: str = "Information Technology", industry: str = "Semiconductors"):
    return {
        ticker: {
            "ticker": ticker,
            "company_name": ticker,
            "sector": sector,
            "industry": industry,
        }
        for ticker in tickers
    }


def _portfolio(metadata, positions=(), max_positions=8):
    return build_portfolio_snapshot(
        positions,
        metadata_by_ticker=metadata,
        max_positions=max_positions,
        trading_budget=10_000.0,
    )


def _rank(rows, metadata, portfolio=None, best_trades=2):
    return rank_daily_opportunities(
        rows,
        metadata_by_ticker=metadata,
        market_regime="risk_on",
        portfolio=portfolio or _portfolio(metadata),
        best_setups_count=10,
        best_trades_max=best_trades,
        next_to_trigger_count=5,
    )


def test_sp500_fallback_universe_does_not_assume_exactly_500_symbols():
    snapshot = _load_fallback()
    assert snapshot.name == "SP500"
    assert len(snapshot.tickers) >= 500
    assert len(snapshot.tickers) != 500
    assert len(snapshot.tickers) == len(set(snapshot.tickers))


def test_excellent_setup_can_be_best_setup_and_next_to_trigger_but_not_trade_today():
    row = _row("AVGO", trigger=103.0, final_action="WAIT", confirmed=False)
    metadata = _metadata("AVGO")
    result = _rank([row], metadata)

    assert result["best_setups"][0]["raw_setup_score"] >= 8.0
    assert result["best_trades_today"] == []
    assert result["next_to_trigger"][0]["ticker"] == "AVGO"
    assert result["next_to_trigger"][0]["actionability_state"] == "awaiting_confirmation"
    assert {item["type"] for item in result["next_to_trigger"][0]["waiting_for"]} >= {
        "price_above",
        "volume_confirmation",
    }


def test_confirmation_raises_actionability_and_allows_trade_eligibility():
    pending = _row("JPM", trigger=103.0, final_action="WAIT", confirmed=False)
    confirmed = _row("JPM", current_price=104.0, trigger=103.0, final_action="BUY", confirmed=True, tp1=116.0)

    pending_profile = build_daily_actionability_profile(pending, market_regime="risk_on")
    confirmed_profile = build_daily_actionability_profile(confirmed, market_regime="risk_on")
    metadata = _metadata("JPM", sector="Financials", industry="Diversified Banks")
    result = _rank([confirmed], metadata)

    assert confirmed_profile["actionability_score"] > pending_profile["actionability_score"]
    assert confirmed_profile["actionability_state"] == "actionable"
    assert [item["ticker"] for item in result["best_trades_today"]] == ["JPM"]


def test_semiconductors_can_dominate_raw_setups_while_daily_trades_diversify():
    rows = [
        _row("AVGO", component_score=9.2),
        _row("AMD", component_score=9.0),
        _row("NVDA", component_score=8.9),
        _row("JPM", component_score=8.8),
    ]
    metadata = {
        **_metadata("AVGO", "AMD", "NVDA"),
        **_metadata("JPM", sector="Financials", industry="Diversified Banks"),
    }
    result = _rank(rows, metadata)

    assert [item["ticker"] for item in result["best_setups"][:3]] == ["AVGO", "AMD", "NVDA"]
    assert [item["ticker"] for item in result["best_trades_today"]] == ["AVGO", "JPM"]


def test_existing_semiconductor_exposure_reduces_portfolio_fit_not_raw_setup():
    metadata = _metadata("AVGO", "AMD", "NVDA")
    row = _row("AVGO")
    raw_before = build_raw_setup_profile(row)["raw_setup_score"]
    portfolio = _portfolio(
        metadata,
        positions=[
            {"ticker": "AMD", "quantity": 5, "average_entry_price": 100.0},
            {"ticker": "NVDA", "quantity": 5, "average_entry_price": 100.0},
        ],
    )
    result = _rank([row], metadata, portfolio=portfolio)
    candidate = result["best_setups"][0]

    assert candidate["raw_setup_score"] == raw_before
    assert candidate["portfolio_fit_score"] < DEFAULT_PLANNING_CONFIG.min_portfolio_fit_score
    assert "sector_overexposure" in candidate["exclusion_reasons"]


def test_no_qualified_trade_does_not_lower_thresholds():
    rows = [_row("LOWQ", final_action="WAIT", confirmed=False, trigger=104.0, component_score=7.0)]
    metadata = _metadata("LOWQ", sector="Consumer Discretionary", industry="Specialty Retail")
    result = _rank(rows, metadata)

    assert result["best_setups"]
    assert result["best_trades_today"] == []


def test_missed_trade_is_not_actionable_at_poor_current_rr():
    row = _row("MISS", current_price=111.0, trigger=100.0, final_action="BUY", confirmed=True, tp1=112.0)
    profile = build_daily_actionability_profile(row, market_regime="risk_on")

    assert profile["actionability_state"] in {"missed", "extended"}
    assert profile["actionability_score"] < DEFAULT_PLANNING_CONFIG.min_actionability_score


def test_non_technology_candidate_can_rank_first():
    rows = [_row("TECH", component_score=8.3), _row("CAT", component_score=9.1)]
    metadata = {
        **_metadata("TECH"),
        **_metadata("CAT", sector="Industrials", industry="Construction Machinery"),
    }
    result = _rank(rows, metadata, best_trades=1)

    assert result["best_setups"][0]["ticker"] == "CAT"
    assert result["best_trades_today"][0]["ticker"] == "CAT"


def test_full_position_slots_leave_best_setups_but_no_daily_trade():
    metadata = {
        **_metadata("JPM", sector="Financials", industry="Diversified Banks"),
        **_metadata("AAPL"),
    }
    portfolio = _portfolio(
        metadata,
        positions=[{"ticker": "AAPL", "quantity": 10, "average_entry_price": 100.0}],
        max_positions=1,
    )
    result = _rank([_row("JPM")], metadata, portfolio=portfolio)

    assert result["best_setups"][0]["ticker"] == "JPM"
    assert result["best_trades_today"] == []


def test_missing_ticker_plan_is_isolated_in_failure_diagnostics():
    failed = SimpleNamespace(
        ticker="FAIL",
        preferred_entry=None,
        stop_loss=None,
        scan_rejection_reason="missing_required_data",
        strategy_reason="bars unavailable",
    )
    metadata = {
        **_metadata("GOOD"),
        **_metadata("FAIL", sector="Industrials", industry="Machinery"),
    }
    result = _rank([_row("GOOD"), failed], metadata)

    assert result["best_setups"][0]["ticker"] == "GOOD"
    assert result["failures"] == [
        {"ticker": "FAIL", "reason": "missing_required_data", "details": "bars unavailable"}
    ]
