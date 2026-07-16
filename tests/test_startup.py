from __future__ import annotations

from dashboard.start import _resolve_port as resolve_dashboard_port
from scripts.start_api import _resolve_port as resolve_api_port


def test_railway_port_wins_for_both_services(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "$PORT")
    monkeypatch.setenv("UVICORN_PORT", "$PORT")

    assert resolve_dashboard_port() == 9123
    assert resolve_api_port() == 9123


def test_invalid_literal_port_values_use_service_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "$PORT")
    monkeypatch.setenv("UVICORN_PORT", "$PORT")

    assert resolve_dashboard_port() == 8501
    assert resolve_api_port() == 8080
