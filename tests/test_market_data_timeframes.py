from datetime import datetime, timezone

import pytest

from app import market_data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1d", "daily"), ("D", "daily"), ("60m", "hourly"), ("1h", "hourly"), ("30m", "thirty_minute")],
)
def test_timeframe_aliases_are_provider_neutral(raw, expected):
    assert market_data.normalize_timeframe(raw) == expected


def test_unsupported_timeframe_fails_before_vendor_request():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        market_data.normalize_timeframe("2m")


def test_yahoo_payload_is_normalized_to_shared_ohlcv_shape():
    timestamp = int(datetime(2026, 8, 13, 14, 30, tzinfo=timezone.utc).timestamp())
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [timestamp],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [102.0],
                                "low": [99.0],
                                "close": [101.5],
                                "volume": [123456],
                            }
                        ]
                    },
                }
            ]
        }
    }

    bars = market_data._normalized_bars_from_yahoo_payload("AMD", "30m", payload)

    assert bars == [
        {
            "symbol": "AMD",
            "date": datetime(2026, 8, 13, 14, 30, tzinfo=timezone.utc),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.5,
            "volume": 123456.0,
            "timeframe": "thirty_minute",
            "source": "yahoo",
        }
    ]
