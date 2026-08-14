"""Safe, standalone TradingView chart embedding for the Streamlit dashboard.

TradingView is used only for human visualization. Market data, planner levels,
and execution decisions must continue to come from the trader API.
"""

from __future__ import annotations

import json
import re
from typing import Final

import streamlit as st
import streamlit.components.v1 as components


VISUALIZATION_NOTICE: Final[str] = (
    "Visualization only. Planner levels, live data, and trading decisions come "
    "from the trader API, not this TradingView widget."
)

_DEFAULT_EXCHANGE: Final[str | None] = None
_DEFAULT_INTERVAL: Final[str] = "D"
_TICKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,19}$")

# TradingView uses a smaller set of canonical exchange prefixes than many data
# providers. Aliases are normalized before a symbol reaches the embed document.
_EXCHANGE_ALIASES: Final[dict[str, str]] = {
    "AMEX": "AMEX",
    "ARCA": "AMEX",
    "BATS": "BATS",
    "CBOE": "CBOE",
    "CBOEBZX": "CBOE",
    "NASD": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NASDAQCM": "NASDAQ",
    "NASDAQGM": "NASDAQ",
    "NASDAQGS": "NASDAQ",
    "NYSE": "NYSE",
    "NYSEAMERICAN": "AMEX",
    "NYSEARCA": "AMEX",
    "NYSEMKT": "AMEX",
    "OTC": "OTC",
    "OTCBB": "OTC",
    "OTCMKTS": "OTC",
    "OTCQB": "OTC",
    "OTCQX": "OTC",
    "PINK": "OTC",
}

_INTERVAL_ALIASES: Final[dict[str, str]] = {
    "1": "1",
    "1M": "1",
    "3": "3",
    "3M": "3",
    "5": "5",
    "5M": "5",
    "15": "15",
    "15M": "15",
    "30": "30",
    "30M": "30",
    "45": "45",
    "45M": "45",
    "60": "60",
    "60M": "60",
    "1H": "60",
    "120": "120",
    "2H": "120",
    "180": "180",
    "3H": "180",
    "240": "240",
    "4H": "240",
    "D": "D",
    "1D": "D",
    "DAY": "D",
    "DAILY": "D",
    "W": "W",
    "1W": "W",
    "WEEK": "W",
    "WEEKLY": "W",
    "MO": "M",
    "1MO": "M",
    "MONTH": "M",
    "MONTHLY": "M",
}


def _normalize_exchange(value: str | None) -> str | None:
    if value is None:
        return None
    exchange_key = re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())
    return _EXCHANGE_ALIASES.get(exchange_key)


def normalize_tradingview_symbol(
    ticker: str | None,
    *,
    exchange: str | None = None,
    default_exchange: str | None = _DEFAULT_EXCHANGE,
) -> str | None:
    """Return a safe canonical ``EXCHANGE:TICKER`` symbol for a US security.

    A TradingView-qualified ticker (for example, ``NYSE:BRK.B``) takes
    precedence over the separate ``exchange`` argument. If the backend does
    not know the listing exchange, the safe ticker is left unqualified rather
    than incorrectly assuming NASDAQ.
    """

    raw_ticker = str(ticker or "").strip().upper()
    if raw_ticker.startswith("$"):
        raw_ticker = raw_ticker[1:]
    if not raw_ticker:
        return None

    symbol_exchange: str | None = None
    if ":" in raw_ticker:
        if raw_ticker.count(":") != 1:
            return None
        exchange_text, raw_ticker = raw_ticker.split(":", 1)
        symbol_exchange = _normalize_exchange(exchange_text)
        if symbol_exchange is None:
            return None

    if not _TICKER_PATTERN.fullmatch(raw_ticker):
        return None

    if symbol_exchange is None:
        if exchange is not None:
            symbol_exchange = _normalize_exchange(exchange)
        elif default_exchange is not None:
            symbol_exchange = _normalize_exchange(default_exchange)
        if exchange is not None and symbol_exchange is None:
            return None

    return f"{symbol_exchange}:{raw_ticker}" if symbol_exchange else raw_ticker


def normalize_tradingview_interval(interval: str | int | None) -> str:
    """Return a supported TradingView interval, defaulting safely to daily."""

    raw_interval = str(interval or _DEFAULT_INTERVAL).strip().upper()
    return _INTERVAL_ALIASES.get(raw_interval, _DEFAULT_INTERVAL)


def build_tradingview_config(
    ticker: str | None,
    *,
    exchange: str | None = None,
    interval: str | int | None = _DEFAULT_INTERVAL,
) -> dict[str, object]:
    """Build the deterministic dark-theme TradingView widget configuration."""

    symbol = normalize_tradingview_symbol(ticker, exchange=exchange)
    if symbol is None:
        raise ValueError("A valid US ticker and supported exchange are required.")

    return {
        "autosize": True,
        "symbol": symbol,
        "interval": normalize_tradingview_interval(interval),
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "backgroundColor": "rgba(9, 14, 25, 1)",
        "gridColor": "rgba(42, 55, 78, 0.35)",
        "hide_top_toolbar": False,
        "hide_side_toolbar": False,
        "allow_symbol_change": False,
        "save_image": False,
        "calendar": False,
        "withdateranges": True,
        "details": True,
        "hotlist": False,
        "support_host": "https://www.tradingview.com",
    }


def _json_for_script(value: object) -> str:
    """Serialize JSON while neutralizing characters significant in HTML."""

    serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return (
        serialized.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def build_tradingview_embed_html(
    ticker: str | None,
    *,
    exchange: str | None = None,
    interval: str | int | None = _DEFAULT_INTERVAL,
) -> str:
    """Generate a self-contained TradingView advanced-chart embed document."""

    config_json = _json_for_script(
        build_tradingview_config(ticker, exchange=exchange, interval=interval)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body {{ margin: 0; height: 100%; background: #090e19; color: #aebbd0; }}
    .chart-shell {{ display: flex; flex-direction: column; height: 100%; overflow: hidden; }}
    .tradingview-widget-container {{ flex: 1 1 auto; min-height: 0; }}
    .tradingview-widget-container__widget {{ height: 100%; width: 100%; }}
    .visualization-notice {{
      flex: 0 0 auto;
      padding: 7px 12px;
      border-top: 1px solid rgba(124, 146, 180, 0.22);
      background: rgba(13, 20, 34, 0.96);
      color: #9eabc0;
      font: 12px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .visualization-notice strong {{ color: #dce6f6; }}
  </style>
</head>
<body>
  <div class="chart-shell">
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
        async>{config_json}</script>
    </div>
    <div class="visualization-notice">
      <strong>Visualization only.</strong> Planner levels, live data, and trading decisions come from the trader API, not this TradingView widget.
    </div>
  </div>
</body>
</html>"""


def _show_render_fallback() -> None:
    try:
        st.warning(
            "TradingView chart is temporarily unavailable. "
            "Planner data remains available from the trader API."
        )
        st.caption(VISUALIZATION_NOTICE)
    except Exception:
        # Rendering a fallback must never turn an optional chart into an app error.
        return


def render_tradingview_chart(
    ticker: str | None,
    *,
    exchange: str | None = None,
    interval: str | int | None = _DEFAULT_INTERVAL,
    height: int = 620,
) -> bool:
    """Render the chart through ``components.html`` and degrade safely.

    Returns ``True`` when the component call succeeds and ``False`` when input
    validation or Streamlit component rendering fails.
    """

    try:
        embed_html = build_tradingview_embed_html(
            ticker,
            exchange=exchange,
            interval=interval,
        )
        safe_height = max(320, min(int(height), 1200))
        components.html(embed_html, height=safe_height, scrolling=False)
    except Exception:
        _show_render_fallback()
        return False
    return True


__all__ = [
    "VISUALIZATION_NOTICE",
    "build_tradingview_config",
    "build_tradingview_embed_html",
    "normalize_tradingview_interval",
    "normalize_tradingview_symbol",
    "render_tradingview_chart",
]
