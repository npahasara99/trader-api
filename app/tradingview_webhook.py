"""Validated TradingView monitoring events with an explicit no-execution boundary.

This module deliberately has no dependency on the broker, proposal, or order
subsystems. Applications may persist accepted events through ``event_sink`` and
may separately enqueue a planner re-evaluation after persistence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import inspect
import math
import os
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator


TRADINGVIEW_WEBHOOK_SECRET_ENV = "TRADINGVIEW_WEBHOOK_SECRET"
MINIMUM_SECRET_LENGTH = 16
_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{0,31}$")
_TIMEFRAME_ALIASES = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "45": "45m",
    "60": "1h",
    "120": "2h",
    "180": "3h",
    "240": "4h",
    "D": "1d",
    "1D": "1d",
    "W": "1w",
    "1W": "1w",
    "M": "1mo",
    "1M": "1mo",
}


class TradingViewEventType(str, Enum):
    """Only monitoring events understood by the planner refresh pipeline."""

    SUPPORT_HOLD = "SUPPORT_HOLD"
    SUPPORT_BREAK = "SUPPORT_BREAK"
    EMA20_RECLAIM = "EMA20_RECLAIM"
    EMA50_RECLAIM = "EMA50_RECLAIM"
    BREAKOUT = "BREAKOUT"
    BREAKOUT_FAILURE = "BREAKOUT_FAILURE"
    MOMENTUM_IMPROVING = "MOMENTUM_IMPROVING"
    MOMENTUM_WEAKENING = "MOMENTUM_WEAKENING"
    RSI_RECOVERY = "RSI_RECOVERY"
    RSI_OVEREXTENDED = "RSI_OVEREXTENDED"


def _finite_number(value: float | None, *, field_name: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class TradingViewWebhookPayload(BaseModel):
    """Strict external payload accepted from the TradingView alert monitor."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    secret: SecretStr = Field(min_length=MINIMUM_SECRET_LENGTH, max_length=512)
    ticker: str = Field(min_length=1, max_length=32)
    timeframe: str = Field(min_length=1, max_length=8)
    event_type: TradingViewEventType = Field(
        validation_alias=AliasChoices("event_type", "signal")
    )
    price: float = Field(
        gt=0,
        validation_alias=AliasChoices("price", "close"),
    )
    event_time: datetime
    ema20: float | None = Field(default=None, gt=0)
    ema50: float | None = Field(default=None, gt=0)
    ema100: float | None = Field(default=None, gt=0)
    ema200: float | None = Field(default=None, gt=0)
    rsi: float | None = Field(default=None, ge=0, le=100)
    atr: float | None = Field(default=None, ge=0)
    volume: float | None = Field(default=None, ge=0)
    relative_volume: float | None = Field(default=None, ge=0)
    local_high: float | None = Field(default=None, gt=0)
    local_low: float | None = Field(default=None, gt=0)

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("ticker must be a string")
        normalized = value.strip().upper()
        if not _TICKER_PATTERN.fullmatch(normalized):
            raise ValueError("ticker contains unsupported characters")
        return normalized.rsplit(":", 1)[-1]

    @field_validator("timeframe", mode="before")
    @classmethod
    def normalize_timeframe(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("timeframe must be a string")
        raw = value.strip()
        normalized = _TIMEFRAME_ALIASES.get(raw.upper())
        if normalized is None:
            raise ValueError("unsupported TradingView timeframe")
        return normalized

    @field_validator("event_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator(
        "price",
        "ema20",
        "ema50",
        "ema100",
        "ema200",
        "rsi",
        "atr",
        "volume",
        "relative_volume",
        "local_high",
        "local_low",
    )
    @classmethod
    def require_finite_numbers(cls, value: float | None, info: Any) -> float | None:
        return _finite_number(value, field_name=info.field_name)


class NormalizedTradingViewEvent(BaseModel):
    """Persistence-friendly event with credentials removed and no trade action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    source: Literal["tradingview"] = "tradingview"
    ticker: str
    timeframe: str
    event_type: TradingViewEventType
    price: float
    occurred_at: datetime
    received_at: datetime
    indicators: dict[str, float | None]
    payload: dict[str, Any]
    processed: bool = False
    processing_status: Literal["pending"] = "pending"
    re_evaluation_requested: bool = True
    execution_requested: Literal[False] = False

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping suitable for a JSON/JSONB column."""

        return self.model_dump(mode="json")


class TradingViewWebhookResponse(BaseModel):
    accepted: Literal[True] = True
    event_id: str
    ticker: str
    timeframe: str
    event_type: TradingViewEventType
    processing_status: Literal["pending"] = "pending"
    re_evaluation_requested: Literal[True] = True
    execution_requested: Literal[False] = False


TradingViewEventSink = Callable[
    [NormalizedTradingViewEvent], None | Awaitable[None]
]


def configured_webhook_secret(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read the dedicated webhook secret without consulting application config."""

    source = os.environ if environ is None else environ
    value = source.get(TRADINGVIEW_WEBHOOK_SECRET_ENV, "").strip()
    return value or None


def verify_webhook_secret(
    provided: str | SecretStr | None,
    expected: str | None = None,
) -> bool:
    """Compare webhook credentials in constant time and fail closed."""

    configured = expected if expected is not None else configured_webhook_secret()
    if isinstance(provided, SecretStr):
        candidate = provided.get_secret_value()
    else:
        candidate = provided
    if not configured or len(configured) < MINIMUM_SECRET_LENGTH or not candidate:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))


def require_valid_webhook_secret(
    provided: str | SecretStr | None,
    expected: str | None = None,
) -> None:
    """Raise generic HTTP errors without exposing either credential."""

    configured = expected if expected is not None else configured_webhook_secret()
    if not configured or len(configured) < MINIMUM_SECRET_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook is not configured",
        )
    if not verify_webhook_secret(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook credential",
        )


def normalize_tradingview_event(
    payload: TradingViewWebhookPayload,
    *,
    received_at: datetime | None = None,
) -> NormalizedTradingViewEvent:
    """Normalize a validated alert and irreversibly remove its secret."""

    received = received_at or datetime.now(timezone.utc)
    if received.tzinfo is None or received.utcoffset() is None:
        raise ValueError("received_at must include a timezone")
    received = received.astimezone(timezone.utc)

    fingerprint = "|".join(
        (
            payload.ticker,
            payload.timeframe,
            payload.event_type.value,
            payload.event_time.isoformat(),
            format(payload.price, ".10g"),
        )
    )
    event_id = "tv_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
    indicators = {
        name: getattr(payload, name)
        for name in (
            "ema20",
            "ema50",
            "ema100",
            "ema200",
            "rsi",
            "atr",
            "volume",
            "relative_volume",
            "local_high",
            "local_low",
        )
    }
    sanitized_payload = payload.model_dump(mode="json", exclude={"secret"})

    return NormalizedTradingViewEvent(
        event_id=event_id,
        ticker=payload.ticker,
        timeframe=payload.timeframe,
        event_type=payload.event_type,
        price=payload.price,
        occurred_at=payload.event_time,
        received_at=received,
        indicators=indicators,
        payload=sanitized_payload,
    )


def create_tradingview_router(
    *,
    expected_secret: str | None = None,
    event_sink: TradingViewEventSink | None = None,
) -> APIRouter:
    """Build an opt-in webhook router with an optional persistence-only sink."""

    webhook_router = APIRouter(tags=["tradingview-webhook"])

    @webhook_router.post(
        "/webhooks/tradingview",
        response_model=TradingViewWebhookResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def receive_tradingview_event(
        payload: TradingViewWebhookPayload,
    ) -> TradingViewWebhookResponse:
        require_valid_webhook_secret(payload.secret, expected_secret)
        event = normalize_tradingview_event(payload)
        if event_sink is not None:
            sink_result = event_sink(event)
            if inspect.isawaitable(sink_result):
                await sink_result
        return TradingViewWebhookResponse(
            event_id=event.event_id,
            ticker=event.ticker,
            timeframe=event.timeframe,
            event_type=event.event_type,
        )

    return webhook_router


# Registration is intentionally opt-in: app.main must include this router in a
# separate integration change. Importing this module cannot activate execution.
router = create_tradingview_router()


__all__ = [
    "NormalizedTradingViewEvent",
    "TradingViewEventSink",
    "TradingViewEventType",
    "TradingViewWebhookPayload",
    "TradingViewWebhookResponse",
    "configured_webhook_secret",
    "create_tradingview_router",
    "normalize_tradingview_event",
    "require_valid_webhook_secret",
    "router",
    "verify_webhook_secret",
]
