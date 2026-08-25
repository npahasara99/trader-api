from dashboard import api_client


def test_sp500_backfill_uses_bounded_explicit_universe_payload(monkeypatch):
    captured = {}

    def fake_post(path, payload, *, timeout):
        captured.update(path=path, payload=payload, timeout=timeout)
        return {"processed_count": payload["batch_size"]}

    monkeypatch.setattr(api_client, "post_json", fake_post)
    monkeypatch.setattr(api_client, "get_api_timeout_seconds", lambda default: default)

    result = api_client.backfill_sp500_daily_bars(start_index=20, batch_size=10, years=2)

    assert result == {"processed_count": 10}
    assert captured["path"] == "/data/daily-bars/backfill"
    assert captured["timeout"] == 300
    assert captured["payload"] == {
        "symbols": None,
        "use_sp100": False,
        "use_sp500": True,
        "top_n": 600,
        "years": 2,
        "refresh": False,
        "commit_every": 5,
        "start_index": 20,
        "batch_size": 10,
        "include_results": False,
    }


def test_sp500_status_requests_full_universe(monkeypatch):
    captured = {}

    def fake_get(path, params, *, timeout):
        captured.update(path=path, params=params, timeout=timeout)
        return {"requested_symbols": 503, "symbols_with_data": 25}

    monkeypatch.setattr(api_client, "get_json", fake_get)
    monkeypatch.setattr(api_client, "get_api_timeout_seconds", lambda default: default)

    result = api_client.fetch_sp500_daily_bars_status()

    assert result["symbols_with_data"] == 25
    assert captured == {
        "path": "/data/daily-bars/status",
        "params": {"use_sp100": False, "use_sp500": True, "top_n": 600},
        "timeout": 120,
    }
