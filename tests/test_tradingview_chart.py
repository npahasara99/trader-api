from __future__ import annotations

import json
import re

import pytest

from dashboard import tradingview_chart


@pytest.mark.parametrize(
    ("ticker", "exchange", "expected"),
    [
        ("aapl", None, "AAPL"),
        ("$msft", "nasd", "NASDAQ:MSFT"),
        ("brk.b", "nyse", "NYSE:BRK.B"),
        ("NYSE:brk.b", "NASDAQ", "NYSE:BRK.B"),
        ("spy", "NYSE Arca", "AMEX:SPY"),
        ("otcmkts:abcxy", None, "OTC:ABCXY"),
    ],
)
def test_normalize_tradingview_symbol_for_us_exchanges(
    ticker: str,
    exchange: str | None,
    expected: str,
) -> None:
    assert tradingview_chart.normalize_tradingview_symbol(ticker, exchange=exchange) == expected


@pytest.mark.parametrize(
    "ticker",
    [None, "", "NYSE:", "LSE:VOD", "NASDAQ:AAPL:EXTRA", "AAPL</script>", "AAPL/US"],
)
def test_normalize_tradingview_symbol_rejects_unsafe_or_non_us_input(ticker: str | None) -> None:
    assert tradingview_chart.normalize_tradingview_symbol(ticker) is None


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("30m", "30"), ("1h", "60"), ("4H", "240"), ("1d", "D"), ("weekly", "W"), ("bad", "D")],
)
def test_normalize_tradingview_interval(requested: str, expected: str) -> None:
    assert tradingview_chart.normalize_tradingview_interval(requested) == expected


def test_build_config_is_dark_and_uses_normalized_values() -> None:
    config = tradingview_chart.build_tradingview_config(
        "brk.b",
        exchange="nyse",
        interval="1h",
    )

    assert config["symbol"] == "NYSE:BRK.B"
    assert config["interval"] == "60"
    assert config["theme"] == "dark"
    assert config["autosize"] is True
    assert config["allow_symbol_change"] is False
    assert config["support_host"] == "https://www.tradingview.com"


def test_build_config_rejects_invalid_symbol() -> None:
    with pytest.raises(ValueError, match="valid US ticker"):
        tradingview_chart.build_tradingview_config("AAPL</script>")


def test_embed_html_contains_script_safe_config_and_disclosure() -> None:
    embed_html = tradingview_chart.build_tradingview_embed_html(
        "NYSE:BRK.B",
        interval="30m",
    )

    assert "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" in embed_html
    assert "Visualization only." in embed_html
    assert "trader API" in embed_html
    assert "</script>" in embed_html

    match = re.search(r"async>(\{.*?\})</script>", embed_html, flags=re.DOTALL)
    assert match is not None
    embedded_config = json.loads(match.group(1))
    assert embedded_config["symbol"] == "NYSE:BRK.B"
    assert embedded_config["interval"] == "30"
    assert embedded_config["theme"] == "dark"


def test_render_uses_components_html(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered: dict[str, object] = {}

    def fake_components_html(body: str, *, height: int, scrolling: bool) -> None:
        rendered.update(body=body, height=height, scrolling=scrolling)

    monkeypatch.setattr(tradingview_chart.components, "html", fake_components_html)

    assert tradingview_chart.render_tradingview_chart("AAPL", interval="D", height=700) is True
    assert '"symbol":"AAPL"' in str(rendered["body"])
    assert rendered["height"] == 700
    assert rendered["scrolling"] is False


def test_render_degrades_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_called = False

    def fail_render(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("component unavailable")

    def fake_fallback() -> None:
        nonlocal fallback_called
        fallback_called = True

    monkeypatch.setattr(tradingview_chart.components, "html", fail_render)
    monkeypatch.setattr(tradingview_chart, "_show_render_fallback", fake_fallback)

    assert tradingview_chart.render_tradingview_chart("AAPL") is False
    assert fallback_called is True


def test_invalid_input_degrades_without_calling_component(monkeypatch: pytest.MonkeyPatch) -> None:
    component_called = False

    def fake_components_html(*_args: object, **_kwargs: object) -> None:
        nonlocal component_called
        component_called = True

    monkeypatch.setattr(tradingview_chart.components, "html", fake_components_html)
    monkeypatch.setattr(tradingview_chart, "_show_render_fallback", lambda: None)

    assert tradingview_chart.render_tradingview_chart("<script>") is False
    assert component_called is False
