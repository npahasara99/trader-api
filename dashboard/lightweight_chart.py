"""TradingView Lightweight Charts rendering from trader-API canonical OHLCV."""

from __future__ import annotations

import html
import json
from typing import Any

import streamlit.components.v1 as components


LIGHTWEIGHT_CHARTS_VERSION = "5.2.0"


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )


def build_lightweight_chart_html(
    *,
    ticker: str,
    title: str,
    timeframe_payload: dict[str, Any],
    levels: list[dict[str, Any]],
    markers: list[dict[str, Any]] | None = None,
) -> str:
    bars = timeframe_payload.get("bars") or []
    candles = [
        {
            "time": int(bar["timestamp"]),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
        }
        for bar in bars
    ]
    volumes = [
        {
            "time": int(bar["timestamp"]),
            "value": float(bar.get("volume") or 0.0),
            "color": "rgba(33,212,168,.38)" if float(bar["close"]) >= float(bar["open"]) else "rgba(255,101,119,.38)",
        }
        for bar in bars
    ]
    indicators = timeframe_payload.get("indicators") or {}
    payload = {
        "candles": candles,
        "volumes": volumes,
        "indicators": indicators,
        "levels": levels,
        "markers": markers or [],
    }
    safe_title = html.escape(f"{ticker} | {title}")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@{LIGHTWEIGHT_CHARTS_VERSION}/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    html,body{{margin:0;height:100%;background:#09111e;color:#dbe7f6;font:12px/1.4 ui-sans-serif,system-ui,sans-serif}}
    .shell{{height:100%;display:grid;grid-template-rows:auto 1fr auto;border:1px solid #24344c;border-radius:12px;overflow:hidden;background:linear-gradient(145deg,#0d1728,#0a1220)}}
    .head{{display:flex;justify-content:space-between;gap:8px;padding:9px 12px;border-bottom:1px solid #21314a}}
    .head strong{{letter-spacing:.02em}} .head span{{color:#8fa4bf}}
    #chart{{min-height:0}}
    .foot{{padding:4px 10px;border-top:1px solid #21314a;color:#7186a2;text-align:right}}
    .foot a{{color:#8fb8ee;text-decoration:none}}
  </style>
</head>
<body>
<div class="shell">
  <div class="head"><strong>{safe_title}</strong><span>Canonical API OHLCV</span></div>
  <div id="chart"></div>
  <div class="foot">Charts by <a href="https://www.tradingview.com/" target="_blank" rel="noopener">TradingView</a></div>
</div>
<script>
const data={_script_json(payload)};
const host=document.getElementById('chart');
if(!window.LightweightCharts){{
  host.innerHTML='<div style="padding:18px;color:#9fb0c8">Interactive chart library unavailable. Canonical levels and monitor data remain available above.</div>';
  throw new Error('Lightweight Charts failed to load');
}}
const chart=LightweightCharts.createChart(host,{{
  autoSize:true,
  layout:{{background:{{type:'solid',color:'#0a1322'}},textColor:'#a9bad0',attributionLogo:true}},
  grid:{{vertLines:{{color:'rgba(42,59,84,.26)'}},horzLines:{{color:'rgba(42,59,84,.26)'}}}},
  rightPriceScale:{{borderColor:'#2a3b55'}},
  timeScale:{{borderColor:'#2a3b55',timeVisible:true,secondsVisible:false}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}}
}});
const candles=chart.addSeries(LightweightCharts.CandlestickSeries,{{upColor:'#21d4a8',downColor:'#ff6577',wickUpColor:'#21d4a8',wickDownColor:'#ff6577',borderVisible:false}});
candles.setData(data.candles);
const volume=chart.addSeries(LightweightCharts.HistogramSeries,{{priceFormat:{{type:'volume'}},priceScaleId:'volume',lastValueVisible:false,priceLineVisible:false}});
volume.priceScale().applyOptions({{scaleMargins:{{top:.82,bottom:0}}}});
volume.setData(data.volumes);
const colors={{ema20:'#f0b35a',ema50:'#6ea8fe',ema100:'#b084ff',ema200:'#e0e6ef',vwap:'#4cc9f0'}};
for(const [name,points] of Object.entries(data.indicators||{{}})){{
  if(!points.length||!colors[name]) continue;
  const line=chart.addSeries(LightweightCharts.LineSeries,{{color:colors[name],lineWidth:name==='vwap'?2:1,lastValueVisible:false,priceLineVisible:false,title:name.toUpperCase()}});
  line.setData(points);
}}
for(const level of data.levels||[]){{
  candles.createPriceLine({{price:Number(level.price),color:level.color||'#fff',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:`${{level.label}} [${{level.source}}]`}});
}}
if(data.markers&&data.markers.length&&LightweightCharts.createSeriesMarkers){{LightweightCharts.createSeriesMarkers(candles,data.markers);}}
chart.timeScale().fitContent();
</script>
</body>
</html>"""


def render_lightweight_chart(
    *,
    ticker: str,
    title: str,
    timeframe_payload: dict[str, Any],
    levels: list[dict[str, Any]],
    markers: list[dict[str, Any]] | None = None,
    height: int = 430,
) -> None:
    components.html(
        build_lightweight_chart_html(
            ticker=ticker,
            title=title,
            timeframe_payload=timeframe_payload,
            levels=levels,
            markers=markers,
        ),
        height=max(320, min(int(height), 700)),
        scrolling=False,
    )


__all__ = ["LIGHTWEIGHT_CHARTS_VERSION", "build_lightweight_chart_html", "render_lightweight_chart"]
