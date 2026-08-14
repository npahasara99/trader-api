from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.tradingview_webhook import (
    TradingViewEventType,
    TradingViewWebhookPayload,
    create_tradingview_router,
    normalize_tradingview_event,
    verify_webhook_secret,
)


SECRET = "test-webhook-secret-32-characters"


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "secret": SECRET,
        "ticker": "NASDAQ:AMD",
        "timeframe": "60",
        "event_type": "BREAKOUT",
        "price": 182.45,
        "event_time": "2026-08-13T14:30:00Z",
        "ema20": 180.1,
        "ema50": 176.2,
        "ema100": 169.8,
        "ema200": 155.4,
        "rsi": 61.2,
        "atr": 4.3,
        "volume": 1250000,
        "relative_volume": 1.4,
        "local_high": 181.9,
        "local_low": 171.2,
    }
    payload.update(overrides)
    return payload


def test_secret_verification_is_fail_closed() -> None:
    assert verify_webhook_secret(SECRET, SECRET)
    assert not verify_webhook_secret("wrong-webhook-secret", SECRET)
    assert not verify_webhook_secret(None, SECRET)
    assert not verify_webhook_secret(SECRET, "short")


def test_payload_normalizes_ticker_timeframe_and_aliases() -> None:
    payload = valid_payload()
    payload["signal"] = payload.pop("event_type")
    payload["close"] = payload.pop("price")

    parsed = TradingViewWebhookPayload.model_validate(payload)

    assert parsed.ticker == "AMD"
    assert parsed.timeframe == "1h"
    assert parsed.event_type is TradingViewEventType.BREAKOUT
    assert parsed.price == 182.45


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_type", "BUY"),
        ("event_type", "breakout"),
        ("timeframe", "90"),
        ("ticker", "AMD;DROP TABLE"),
        ("price", -1),
        ("rsi", 101),
        ("event_time", "2026-08-13T14:30:00"),
    ],
)
def test_payload_rejects_unsupported_or_unsafe_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TradingViewWebhookPayload.model_validate(valid_payload(**{field: value}))


def test_payload_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TradingViewWebhookPayload.model_validate(valid_payload(order_action="BUY"))


def test_normalized_event_is_idempotent_serializable_and_secret_free() -> None:
    parsed = TradingViewWebhookPayload.model_validate(valid_payload())
    received_at = datetime(2026, 8, 13, 14, 30, 2, tzinfo=timezone.utc)

    first = normalize_tradingview_event(parsed, received_at=received_at)
    second = normalize_tradingview_event(parsed, received_at=received_at)
    record = first.to_record()

    assert first.event_id == second.event_id
    assert first.ticker == "AMD"
    assert first.timeframe == "1h"
    assert record["occurred_at"] == "2026-08-13T14:30:00Z"
    assert "secret" not in record["payload"]
    assert SECRET not in str(record)
    assert record["processed"] is False
    assert record["re_evaluation_requested"] is True
    assert record["execution_requested"] is False


def test_route_rejects_invalid_secret_without_calling_sink() -> None:
    stored = []
    app = FastAPI()
    app.include_router(create_tradingview_router(expected_secret=SECRET, event_sink=stored.append))
    client = TestClient(app)

    response = client.post(
        "/webhooks/tradingview",
        json=valid_payload(secret="wrong-webhook-secret-value"),
    )

    assert response.status_code == 401
    assert stored == []
    assert SECRET not in response.text


def test_valid_route_persists_monitoring_event_without_execution() -> None:
    stored = []
    app = FastAPI()
    app.include_router(create_tradingview_router(expected_secret=SECRET, event_sink=stored.append))
    client = TestClient(app)

    response = client.post("/webhooks/tradingview", json=valid_payload())

    assert response.status_code == 202
    assert response.json()["execution_requested"] is False
    assert response.json()["re_evaluation_requested"] is True
    assert len(stored) == 1
    assert stored[0].execution_requested is False
    assert "secret" not in stored[0].payload
