"""Server-side chart PNG rendering for immutable monitor decision evidence."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import ChartSnapshot, LiveWatch, MonitorSetup

from .config import LiveMonitorConfig


IMPORTANT_EVENTS = {
    "monitor_added",
    "setup_reanalyzed",
    "NEAR_TRIGGER",
    "REJECTED_BREAKOUT",
    "APPROVED",
    "STRONGLY_CONFIRMED",
    "INVALIDATED",
    "MISSED",
    "manual_trade_entered",
    "manual_trade_exited",
    "post_trade",
}


def cleanup_chart_snapshot_retention(db: Session, *, config: LiveMonitorConfig) -> int:
    """Remove only non-decision-critical images beyond the configured window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.chart_retention_days)
    rows = (
        db.query(ChartSnapshot)
        .filter(
            ChartSnapshot.retain_permanently.is_(False),
            ChartSnapshot.generated_at < cutoff,
        )
        .all()
    )
    for row in rows:
        try:
            Path(row.image_path).unlink(missing_ok=True)
        except OSError:
            pass
        db.delete(row)
    return len(rows)


def _payload_hash(bundle: dict, timeframe: str, event_type: str) -> str:
    material = {
        "ticker": bundle.get("ticker"),
        "boundary": bundle.get("decision_time_boundary"),
        "event": event_type,
        "timeframe": timeframe,
        "bars": ((bundle.get("timeframes") or {}).get(timeframe) or {}).get("bars") or [],
        "levels": bundle.get("levels") or [],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _render_png(
    path: Path,
    *,
    ticker: str,
    timeframe: str,
    timeframe_payload: dict,
    levels: list[dict],
    markers: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    bars = timeframe_payload.get("bars") or []
    if not bars:
        raise ValueError(f"No {timeframe} bars are available to render")
    figure, (price_axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        dpi=120,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04},
    )
    figure.patch.set_facecolor("#090e19")
    for axis in (price_axis, volume_axis):
        axis.set_facecolor("#0d1524")
        axis.tick_params(colors="#9fb0c8", labelsize=8)
        axis.grid(color="#26364e", alpha=0.32, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color("#26364e")
    width = 0.62
    for index, bar in enumerate(bars):
        opened = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        color = "#21d4a8" if close >= opened else "#ff6577"
        price_axis.vlines(index, low, high, color=color, linewidth=0.9)
        price_axis.add_patch(
            Rectangle(
                (index - width / 2, min(opened, close)),
                width,
                max(abs(close - opened), 1e-6),
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
            )
        )
        volume_axis.bar(index, float(bar.get("volume") or 0.0), width=width, color=color, alpha=0.55)
    indicator_colors = {"ema20": "#f0b35a", "ema50": "#6ea8fe", "ema100": "#b084ff", "ema200": "#e0e6ef", "vwap": "#4cc9f0"}
    indicators = timeframe_payload.get("indicators") or {}
    timestamp_index = {int(bar["timestamp"]): index for index, bar in enumerate(bars)}
    for name, color in indicator_colors.items():
        points = indicators.get(name) or []
        xy = [(timestamp_index.get(int(point["time"])), point["value"]) for point in points]
        xy = [(x, y) for x, y in xy if x is not None]
        if xy:
            price_axis.plot([x for x, _ in xy], [y for _, y in xy], color=color, linewidth=0.9, alpha=0.88, label=name.upper())
    for marker in markers:
        marker_time = int(marker.get("time") or 0)
        if not timestamp_index:
            continue
        nearest_time = min(timestamp_index, key=lambda value: abs(value - marker_time))
        index = timestamp_index[nearest_time]
        bar = bars[index]
        rejected = marker.get("position") == "aboveBar"
        marker_price = float(bar["high"] if rejected else bar["low"])
        price_axis.scatter(
            [index],
            [marker_price],
            marker="v" if rejected else "^",
            color=marker.get("color") or ("#ff6577" if rejected else "#21d4a8"),
            s=28,
            zorder=5,
        )
    for level in levels:
        price = level.get("price")
        if price is None:
            continue
        price_axis.axhline(float(price), color=level.get("color") or "#ffffff", linewidth=0.9, linestyle="--", alpha=0.82)
        price_axis.text(
            max(len(bars) - 1, 0),
            float(price),
            f" {level.get('label')} {float(price):.2f}",
            color=level.get("color") or "#ffffff",
            fontsize=7,
            va="bottom",
            ha="right",
            bbox={"facecolor": "#0d1524", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
    price_axis.set_title(f"{ticker} | {timeframe.title()} | Canonical OHLCV", color="#e6edf7", fontsize=12, loc="left")
    if any(indicators.get(name) for name in indicator_colors):
        price_axis.legend(loc="upper left", fontsize=7, framealpha=0.2)
    tick_count = min(8, len(bars))
    if tick_count:
        step = max(1, len(bars) // tick_count)
        ticks = list(range(0, len(bars), step))
        volume_axis.set_xticks(ticks)
        volume_axis.set_xticklabels([str(bars[index]["time"])[5:16].replace("T", " ") for index in ticks], rotation=25, ha="right")
    volume_axis.set_ylabel("Volume", color="#9fb0c8", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def render_chart_snapshots(
    db: Session,
    *,
    watch: LiveWatch,
    setup: MonitorSetup,
    bundle: dict[str, Any],
    event_type: str,
    config: LiveMonitorConfig,
    decision_event_id: str | None = None,
    force: bool = False,
) -> list[ChartSnapshot]:
    """Render and persist only decision-relevant, time-bounded chart images."""
    if not force and event_type not in IMPORTANT_EVENTS:
        return []
    boundary = datetime.fromisoformat(str(bundle["decision_time_boundary"]).replace("Z", "+00:00"))
    output: list[ChartSnapshot] = []
    for timeframe in ("daily", "structure", "execution"):
        timeframe_payload = (bundle.get("timeframes") or {}).get(timeframe) or {}
        if not timeframe_payload.get("bars"):
            continue
        content_hash = _payload_hash(bundle, timeframe, event_type)
        existing = db.query(ChartSnapshot).filter(ChartSnapshot.content_hash == content_hash).one_or_none()
        if existing:
            output.append(existing)
            continue
        safe_at = boundary.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(config.chart_snapshot_dir) / watch.ticker / setup.id / f"{safe_at}-{event_type.lower()}-{timeframe}.png"
        _render_png(
            path,
            ticker=watch.ticker,
            timeframe=timeframe,
            timeframe_payload=timeframe_payload,
            levels=bundle.get("levels") or [],
            markers=bundle.get("attempt_markers") or [],
        )
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        last_at = timeframe_payload.get("last_bar_timestamp")
        row = ChartSnapshot(
            id=str(__import__("uuid").uuid4()),
            watch_id=watch.id,
            setup_id=setup.id,
            decision_event_id=decision_event_id,
            ticker=watch.ticker,
            timeframe=timeframe,
            event_type=event_type,
            image_path=str(path),
            image_data_base64=encoded,
            content_hash=content_hash,
            data_source=bundle.get("data_source"),
            data_last_bar_at=datetime.fromisoformat(str(last_at).replace("Z", "+00:00")) if last_at else None,
            decision_time_boundary=boundary,
            metadata_json=json.dumps(
                {
                    "decision_time_boundary": bundle.get("decision_time_boundary"),
                    "data_source": bundle.get("data_source"),
                    "data_freshness_seconds": bundle.get("data_freshness_seconds"),
                    "bar_count": timeframe_payload.get("bar_count"),
                },
                separators=(",", ":"),
            ),
            retain_permanently=True,
        )
        db.add(row)
        output.append(row)
    return output


__all__ = ["IMPORTANT_EVENTS", "cleanup_chart_snapshot_retention", "render_chart_snapshots"]
