from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Keep core secrets in .env while allowing local broker settings to live in
    # an independently ignored file. Later files override earlier ones.
    model_config = SettingsConfigDict(env_file=(".env", ".env.bot"), extra="ignore")
    DATABASE_URL: str
    SUPABASE_DATABASE_URL: str | None = None
    OPENAI_API_KEY: str | None = None
    API_BEARER_TOKEN: str | None = None
    TRADINGVIEW_WEBHOOK_SECRET: str | None = None
    ENV: str = "prod"

    TRADING_MODE: str = "disabled"
    AUTO_EXECUTION: bool = False

    IBKR_HOST: str = "127.0.0.1"
    IBKR_PORT: int = 7497
    IBKR_CLIENT_ID: int = 21
    IBKR_ACCOUNT_ID: str | None = None
    IBKR_REQUIRE_PAPER_ACCOUNT: bool = True
    IBKR_READ_ONLY: bool = False
    IBKR_RECONNECT_ENABLED: bool = True

    TRADING_BUDGET: float = 10_000.0
    MAX_CAPITAL_UTILIZATION_PCT: float = 70.0
    RISK_PER_TRADE_PCT: float = 0.75
    MAX_POSITION_PCT: float = 20.0
    MAX_OPEN_POSITIONS: int = 8
    MAX_PORTFOLIO_RISK_PCT: float = 4.0
    MAX_DAILY_LOSS_PCT: float = 2.0
    MAX_WEEKLY_LOSS_PCT: float = 5.0
    MIN_REWARD_RISK: float = 2.0
    MAX_SECTOR_EXPOSURE_PCT: float = 35.0

    ALLOW_SHORTS: bool = False
    ALLOW_EXTENDED_HOURS: bool = False
    NEW_ENTRY_START_TIME: str = "10:00"
    NEW_ENTRY_END_TIME: str = "15:30"
    MAX_HOLDING_DAYS: int = 15
    AUTO_CLOSE_BEFORE_EARNINGS: bool = True
    EARNINGS_EXIT_DAYS: int = 1

    CANCEL_UNFILLED_AFTER_MINUTES: int = 30
    MAX_SLIPPAGE_PCT: float = 0.5
    MAX_SPREAD_PCT: float = 0.75
    STALE_QUOTE_SECONDS: int = 60

    BOT_SCAN_INTERVAL_SECONDS: int = 900
    BOT_TRIGGER_CHECK_INTERVAL_SECONDS: int = 300
    BOT_POSITION_MONITOR_INTERVAL_SECONDS: int = 60
    BOT_RECONCILE_INTERVAL_SECONDS: int = 60

    LIVE_TRADING_UNLOCKED: bool = False
    LIVE_TRADING_CONFIRMATION: str | None = None
    LIVE_MAX_BUDGET: float = 0.0
    LIVE_MAX_OPEN_POSITIONS: int = 1

settings = Settings()
