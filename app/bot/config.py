from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from app.settings import settings

from .enums import TradingMode


def _parse_time(value: str, default: time) -> time:
    try:
        hour, minute = str(value or "").split(":")
        return time(hour=int(hour), minute=int(minute))
    except Exception:
        return default


@dataclass(frozen=True)
class BotConfig:
    trading_mode: TradingMode
    auto_execution: bool
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    ibkr_account_id: str | None
    ibkr_require_paper_account: bool
    ibkr_read_only: bool
    ibkr_reconnect_enabled: bool
    trading_budget: float
    max_capital_utilization_pct: float
    risk_per_trade_pct: float
    max_position_pct: float
    max_open_positions: int
    max_portfolio_risk_pct: float
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    min_reward_risk: float
    max_sector_exposure_pct: float
    allow_shorts: bool
    allow_extended_hours: bool
    new_entry_start_time: time
    new_entry_end_time: time
    max_holding_days: int
    auto_close_before_earnings: bool
    earnings_exit_days: int
    cancel_unfilled_after_minutes: int
    max_slippage_pct: float
    max_spread_pct: float
    stale_quote_seconds: int
    bot_scan_interval_seconds: int
    bot_trigger_check_interval_seconds: int
    bot_position_monitor_interval_seconds: int
    bot_reconcile_interval_seconds: int
    live_trading_unlocked: bool
    live_trading_confirmation: str | None
    live_max_budget: float
    live_max_open_positions: int


def load_bot_config() -> BotConfig:
    mode_raw = str(settings.TRADING_MODE or "disabled").strip().lower()
    try:
        mode = TradingMode(mode_raw)
    except Exception:
        mode = TradingMode.DISABLED
    return BotConfig(
        trading_mode=mode,
        auto_execution=bool(settings.AUTO_EXECUTION),
        ibkr_host=settings.IBKR_HOST,
        ibkr_port=int(settings.IBKR_PORT),
        ibkr_client_id=int(settings.IBKR_CLIENT_ID),
        ibkr_account_id=settings.IBKR_ACCOUNT_ID,
        ibkr_require_paper_account=bool(settings.IBKR_REQUIRE_PAPER_ACCOUNT),
        ibkr_read_only=bool(settings.IBKR_READ_ONLY),
        ibkr_reconnect_enabled=bool(settings.IBKR_RECONNECT_ENABLED),
        trading_budget=float(settings.TRADING_BUDGET),
        max_capital_utilization_pct=float(settings.MAX_CAPITAL_UTILIZATION_PCT),
        risk_per_trade_pct=float(settings.RISK_PER_TRADE_PCT),
        max_position_pct=float(settings.MAX_POSITION_PCT),
        max_open_positions=int(settings.MAX_OPEN_POSITIONS),
        max_portfolio_risk_pct=float(settings.MAX_PORTFOLIO_RISK_PCT),
        max_daily_loss_pct=float(settings.MAX_DAILY_LOSS_PCT),
        max_weekly_loss_pct=float(settings.MAX_WEEKLY_LOSS_PCT),
        min_reward_risk=float(settings.MIN_REWARD_RISK),
        max_sector_exposure_pct=float(settings.MAX_SECTOR_EXPOSURE_PCT),
        allow_shorts=bool(settings.ALLOW_SHORTS),
        allow_extended_hours=bool(settings.ALLOW_EXTENDED_HOURS),
        new_entry_start_time=_parse_time(settings.NEW_ENTRY_START_TIME, time(hour=10, minute=0)),
        new_entry_end_time=_parse_time(settings.NEW_ENTRY_END_TIME, time(hour=15, minute=30)),
        max_holding_days=int(settings.MAX_HOLDING_DAYS),
        auto_close_before_earnings=bool(settings.AUTO_CLOSE_BEFORE_EARNINGS),
        earnings_exit_days=int(settings.EARNINGS_EXIT_DAYS),
        cancel_unfilled_after_minutes=int(settings.CANCEL_UNFILLED_AFTER_MINUTES),
        max_slippage_pct=float(settings.MAX_SLIPPAGE_PCT),
        max_spread_pct=float(settings.MAX_SPREAD_PCT),
        stale_quote_seconds=int(settings.STALE_QUOTE_SECONDS),
        bot_scan_interval_seconds=int(settings.BOT_SCAN_INTERVAL_SECONDS),
        bot_trigger_check_interval_seconds=int(settings.BOT_TRIGGER_CHECK_INTERVAL_SECONDS),
        bot_position_monitor_interval_seconds=int(settings.BOT_POSITION_MONITOR_INTERVAL_SECONDS),
        bot_reconcile_interval_seconds=int(settings.BOT_RECONCILE_INTERVAL_SECONDS),
        live_trading_unlocked=bool(settings.LIVE_TRADING_UNLOCKED),
        live_trading_confirmation=settings.LIVE_TRADING_CONFIRMATION,
        live_max_budget=float(settings.LIVE_MAX_BUDGET),
        live_max_open_positions=int(settings.LIVE_MAX_OPEN_POSITIONS),
    )
