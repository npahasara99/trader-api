from __future__ import annotations

"""Deterministic multi-timeframe chart context from normalized OHLCV bars.

The module is intentionally independent of data providers and planner state.  It
accepts mappings/lists, emits plain JSON-compatible dictionaries, and records
missing timeframes rather than manufacturing bars or levels.
"""

from collections.abc import Mapping, Sequence
from typing import Any
import math

import numpy as np
import pandas as pd

from .config import DEFAULT_PLANNING_CONFIG, PlanningConfig
from .indicators import add_indicator_columns
from .structure import classify_structure_state, find_pivots


TIMEFRAME_ALIASES = {
    "daily": ("daily", "day", "1d", "d"),
    "four_hour": ("four_hour", "4h", "240m", "240min"),
    "hourly": ("hourly", "hour", "1h", "60m", "60min"),
    "thirty_minute": ("thirty_minute", "30m", "30min", "30_minute"),
}


def derive_four_hour_bars(hourly: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    """Aggregate valid hourly data into four-hour OHLCV bars."""

    if hourly.empty or len(hourly) < config.four_hour_min_hourly_bars or hourly["date"].isna().all():
        return pd.DataFrame(columns=hourly.columns)
    ordered = hourly.dropna(subset=["date"]).sort_values("date").copy()
    gaps = ordered["date"].diff().dropna().dt.total_seconds().div(60.0)
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.empty or float(positive_gaps.median()) > config.four_hour_max_median_gap_minutes:
        return pd.DataFrame(columns=hourly.columns)
    derived = (
        ordered.set_index("date")
        .resample("4h", origin="start_day")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    return derived if len(derived) >= 8 else pd.DataFrame(columns=hourly.columns)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _round(value: Any, digits: int = 6) -> float | None:
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 6) if math.isfinite(number) else None
    return value


def _rows_from_input(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, pd.DataFrame):
        return raw.to_dict(orient="records")
    if isinstance(raw, Mapping):
        for container_key in ("bars", "data", "results", "candles"):
            nested = raw.get(container_key)
            if isinstance(nested, (list, tuple, pd.DataFrame)):
                return _rows_from_input(nested)

        close_values = raw.get("close", raw.get("c"))
        if isinstance(close_values, Sequence) and not isinstance(close_values, (str, bytes)):
            length = len(close_values)
            rows: list[dict[str, Any]] = []
            for index in range(length):
                row: dict[str, Any] = {}
                for key, values in raw.items():
                    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                        row[key] = values[index] if index < len(values) else None
                    else:
                        row[key] = values
                rows.append(row)
            return rows
        return [dict(raw)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def normalize_ohlcv_bars(raw: Any) -> pd.DataFrame:
    """Normalize common OHLCV aliases into a chronological DataFrame.

    Timestamps are optional.  When absent, input order is retained so synthetic
    fixtures and provider-normalized arrays remain valid inputs.
    """

    rows = _rows_from_input(raw)
    columns = ["date", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=columns)

    aliases = {
        "date": ("date", "bar_date", "datetime", "timestamp", "time", "t"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c", "adjusted_close"),
        "volume": ("volume", "v"),
    }
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row: dict[str, Any] = {"_order": index}
        for canonical, names in aliases.items():
            row[canonical] = next((source.get(name) for name in names if source.get(name) is not None), None)
        normalized.append(row)

    frame = pd.DataFrame(normalized)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["open"] = frame["open"].fillna(frame["close"])
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1))
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1))
    frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
    frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
    frame["volume"] = frame["volume"].fillna(0.0).clip(lower=0.0)

    parsed = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    # Numeric provider timestamps may be milliseconds rather than seconds.
    numeric_dates = pd.to_numeric(frame["date"], errors="coerce")
    if parsed.isna().all() and numeric_dates.notna().any():
        unit = "ms" if float(numeric_dates.dropna().abs().median()) > 10_000_000_000 else "s"
        parsed = pd.to_datetime(numeric_dates, unit=unit, errors="coerce", utc=True)
    frame["date"] = parsed
    if frame["date"].notna().any():
        frame = frame.sort_values(["date", "_order"], na_position="last")
    else:
        # Existing structure helpers expect a datetime column.
        frame["date"] = pd.date_range("2000-01-01", periods=len(frame), freq="min", tz="UTC")
        frame = frame.sort_values("_order")
    return frame[columns].reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    relative = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative))
    result = result.where(avg_loss != 0.0, 100.0)
    return result.where(avg_gain != 0.0, 0.0)


def _enrich(frame: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    enriched = add_indicator_columns(frame, atr_window=config.atr_window, volume_window=config.volume_window)
    enriched["rsi14"] = _rsi(enriched["close"], 14)
    true_range = pd.concat(
        [
            enriched["high"] - enriched["low"],
            (enriched["high"] - enriched["close"].shift(1)).abs(),
            (enriched["low"] - enriched["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    fallback_atr = true_range.tail(min(14, len(true_range))).median()
    fallback_atr = _finite(fallback_atr, max(float(enriched["close"].iloc[-1]) * 0.01, 0.01))
    enriched["atr"] = enriched["atr"].fillna(float(fallback_atr or 0.01))
    enriched["atr_pct"] = enriched["atr"] / enriched["close"].replace(0.0, np.nan)
    return enriched


def _range_metrics(frame: pd.DataFrame, window: int) -> dict[str, float | None]:
    recent = frame.tail(min(window, len(frame)))
    if recent.empty:
        return {"low": None, "high": None, "position": None, "distance_to_high_pct": None, "distance_to_low_pct": None}
    current = float(frame["close"].iloc[-1])
    low = float(recent["low"].min())
    high = float(recent["high"].max())
    position = 0.5 if high <= low else _clip((current - low) / (high - low), 0.0, 1.0)
    return {
        "low": _round(low),
        "high": _round(high),
        "position": _round(position, 4),
        "distance_to_high_pct": _round(((high - current) / max(current, 1e-9)) * 100.0, 4),
        "distance_to_low_pct": _round(((current - low) / max(current, 1e-9)) * 100.0, 4),
    }


def _structure_label(frame: pd.DataFrame, pivot_highs: list[Any], pivot_lows: list[Any]) -> tuple[str, list[str]]:
    close = float(frame["close"].iloc[-1])
    ema20 = _finite(frame["ema20"].iloc[-1])
    sma50 = _finite(frame["sma50"].iloc[-1])
    lookback = min(20, max(2, len(frame) - 1))
    prior = float(frame["close"].iloc[-lookback])
    period_return = (close - prior) / max(abs(prior), 1e-9)
    ema_slope = 0.0
    if ema20 is not None:
        earlier_ema = _finite(frame["ema20"].iloc[-min(6, len(frame))])
        if earlier_ema is not None:
            ema_slope = (ema20 - earlier_ema) / max(abs(earlier_ema), 1e-9)

    higher_highs = len(pivot_highs) >= 2 and pivot_highs[-1].price > pivot_highs[-2].price
    higher_lows = len(pivot_lows) >= 2 and pivot_lows[-1].price > pivot_lows[-2].price
    lower_highs = len(pivot_highs) >= 2 and pivot_highs[-1].price < pivot_highs[-2].price
    lower_lows = len(pivot_lows) >= 2 and pivot_lows[-1].price < pivot_lows[-2].price

    score = 0
    score += 1 if ema20 is not None and close >= ema20 else -1 if ema20 is not None else 0
    score += 1 if ema20 is not None and sma50 is not None and ema20 >= sma50 else -1 if ema20 is not None and sma50 is not None else 0
    score += 1 if period_return > 0.02 else -1 if period_return < -0.02 else 0
    score += 1 if ema_slope > 0.003 else -1 if ema_slope < -0.003 else 0
    score += 1 if higher_highs else -1 if lower_highs else 0
    score += 1 if higher_lows else -1 if lower_lows else 0

    flags: list[str] = []
    if higher_highs:
        flags.append("higher_highs")
    if higher_lows:
        flags.append("higher_lows")
    if lower_highs:
        flags.append("lower_highs")
    if lower_lows:
        flags.append("lower_lows")
    if score >= 2:
        return "uptrend", flags
    if score <= -2:
        return "downtrend", flags
    return "range", flags or ["mixed_structure"]


def _zone(levels: list[tuple[float, str]], *, atr: float, price: float, side: str) -> dict[str, Any] | None:
    if not levels:
        return None
    ordered = sorted(levels, key=lambda item: abs(price - item[0]))
    anchor = ordered[0][0]
    tolerance = max(atr * 0.55, price * 0.003)
    cluster = [(level, tag) for level, tag in ordered if abs(level - anchor) <= tolerance]
    midpoint = sum(item[0] for item in cluster) / len(cluster)
    pad = min(max(atr * 0.22, price * 0.0015), price * 0.008)
    lower = min(item[0] for item in cluster) - pad
    upper = max(item[0] for item in cluster) + pad
    max_width = max(price * 0.018, atr * 0.8)
    if upper - lower > max_width:
        lower = midpoint - max_width / 2.0
        upper = midpoint + max_width / 2.0
    return {
        "lower": _round(max(0.01, lower)),
        "upper": _round(max(0.01, upper)),
        "midpoint": _round(midpoint),
        "source_tags": sorted({item[1] for item in cluster}),
        "side": side,
    }


def _distinct_zone(candidates: list[tuple[float, str]], first: dict[str, Any] | None, *, atr: float, price: float, side: str) -> dict[str, Any] | None:
    if not first:
        return None
    first_mid = float(first["midpoint"])
    min_gap = max(atr * 0.8, price * 0.006)
    if side == "support":
        remaining = [item for item in candidates if item[0] < first_mid - min_gap]
    else:
        remaining = [item for item in candidates if item[0] > first_mid + min_gap]
    return _zone(remaining, atr=atr, price=price, side=side)


def _timeframe_context(frame: pd.DataFrame, name: str, config: PlanningConfig) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "timeframe": name, "bar_count": 0, "reason": "missing_bars"}

    data = _enrich(frame, config)
    price = float(data["close"].iloc[-1])
    atr = max(float(data["atr"].iloc[-1]), price * 0.001)
    pivot_lookback = min(config.pivot_lookback, max(2, len(data) // 20))
    highs, lows = find_pivots(data, lookback=pivot_lookback, max_points=config.pivot_max_points)
    trend, structure_flags = _structure_label(data, highs, lows)
    structure_state, rich_legacy_trend, rich_flags, ema_structure = classify_structure_state(
        data,
        highs,
        lows,
        extended_from_ema20_pct=config.structure_extended_from_ema20_pct,
        parabolic_from_ema20_pct=config.structure_parabolic_from_ema20_pct,
        base_max_atr_range=config.structure_base_max_atr_range,
    )
    trend = rich_legacy_trend or trend
    structure_flags = sorted(set(structure_flags + rich_flags))
    local_window = 20 if name != "daily" else config.context_local_range_window
    local_range = _range_metrics(data, local_window)
    one_month = _range_metrics(data, min(21, len(data)))
    three_month = _range_metrics(data, min(63, len(data)))

    prior_window = data.iloc[:-2].tail(max(12, local_window)) if len(data) > 4 else data.iloc[:-1]
    prior_high = float(prior_window["high"].max()) if not prior_window.empty else float(data["high"].iloc[-1])
    prior_low = float(prior_window["low"].min()) if not prior_window.empty else float(data["low"].iloc[-1])
    latest_volume_ratio = _finite(data["volume_ratio"].iloc[-1], 1.0) or 1.0
    last_two_above = len(data) >= 2 and bool((data["close"].tail(2) > prior_high).all())
    breakout_attempt = price > prior_high + atr * 0.05
    breakout_confirmed = breakout_attempt and (last_two_above or latest_volume_ratio >= 1.15)
    breakout_failed = float(data["high"].iloc[-1]) > prior_high and price < prior_high - atr * 0.05

    ema20 = _finite(data["ema20"].iloc[-1])
    ema50 = _finite(data["ema50"].iloc[-1])
    ema100 = _finite(data["ema100"].iloc[-1])
    ema200 = _finite(data["ema200"].iloc[-1])
    sma50 = _finite(data["sma50"].iloc[-1])
    sma100 = _finite(data["sma100"].iloc[-1])
    sma200 = _finite(data["sma200"].iloc[-1])
    rsi = _finite(data["rsi14"].iloc[-1], 50.0) or 50.0

    support_levels: list[tuple[float, str]] = [(item.price, "pivot_low") for item in lows if item.price <= price + atr * 0.4]
    resistance_levels: list[tuple[float, str]] = [(item.price, "pivot_high") for item in highs if item.price >= price - atr * 0.4]
    for value, tag in ((ema20, "ema20"), (sma50, "sma50"), (sma100, "sma100"), (sma200, "sma200")):
        if value is None:
            continue
        if value <= price + atr * 0.35:
            support_levels.append((value, tag))
        if value >= price - atr * 0.35:
            resistance_levels.append((value, tag))
    support_levels.extend([(prior_low, "active_range_low"), (float(local_range["low"]), "local_low")])
    resistance_levels.extend([(float(local_range["high"]), "local_high"), (prior_high, "prior_range_high")])
    if breakout_attempt:
        support_levels.append((prior_high, "breakout_retest"))

    nearest_support = _zone(support_levels, atr=atr, price=price, side="support")
    secondary_support = _distinct_zone(support_levels, nearest_support, atr=atr, price=price, side="support")
    nearest_resistance = _zone(resistance_levels, atr=atr, price=price, side="resistance")
    secondary_resistance = _distinct_zone(resistance_levels, nearest_resistance, atr=atr, price=price, side="resistance")
    breakout_trigger = _zone([(prior_high, "prior_range_high")], atr=atr, price=price, side="resistance")

    distance_from_ema20_atr = ((price - ema20) / atr) if ema20 is not None else 0.0
    extension_pct = ((price - ema20) / max(price, 1e-9)) if ema20 is not None else 0.0
    if distance_from_ema20_atr >= 2.2 or extension_pct >= config.context_extended_from_ema20_pct:
        extension_state = "overextended"
    elif distance_from_ema20_atr >= 1.35 and float(local_range["position"] or 0.5) >= config.context_near_high_position:
        extension_state = "extended"
    else:
        extension_state = "balanced"

    recent = data.tail(min(10, len(data)))
    recent_span_atr = (float(recent["high"].max()) - float(recent["low"].min())) / max(atr, 1e-9)
    if recent_span_atr <= config.context_compression_range_atr:
        compression_state = "compressed"
    elif recent_span_atr >= config.context_expansion_range_atr:
        compression_state = "expanded"
    else:
        compression_state = "balanced"

    prior_ema = _finite(data["ema20"].iloc[-2]) if len(data) >= 2 else None
    reclaimed_ema20 = len(data) >= 2 and ema20 is not None and prior_ema is not None and float(data["close"].iloc[-2]) < prior_ema and price > ema20
    rising_closes = len(data) >= 4 and bool((data["close"].tail(4).diff().dropna() > 0).all())
    reversal = (rising_closes and rsi >= 42.0) or (reclaimed_ema20 and latest_volume_ratio >= 0.85)

    support_distance_atr = None
    if nearest_support:
        support_distance_atr = max(0.0, price - float(nearest_support["upper"])) / atr
    near_support = support_distance_atr is not None and support_distance_atr <= 0.8
    if trend == "uptrend" and near_support and extension_state == "balanced":
        pullback_state = "constructive_pullback"
    elif trend == "downtrend" and reversal:
        pullback_state = "repair_attempt"
    elif trend == "downtrend":
        pullback_state = "weak_without_reversal"
    else:
        pullback_state = "neutral"

    if latest_volume_ratio >= 1.2:
        volume_state = "confirmed"
    elif latest_volume_ratio <= 0.75:
        volume_state = "quiet"
    else:
        volume_state = "normal"
    if rsi >= 70:
        rsi_state = "overextended"
    elif rsi >= 55:
        rsi_state = "positive"
    elif rsi <= 32:
        rsi_state = "oversold"
    elif rsi <= 45:
        rsi_state = "weak"
    else:
        rsi_state = "neutral"
    if rsi >= 55 and trend == "uptrend":
        momentum_state = "positive"
    elif reversal:
        momentum_state = "improving"
    elif rsi <= 42 or trend == "downtrend":
        momentum_state = "weak"
    else:
        momentum_state = "balanced"

    if breakout_confirmed:
        breakout_state = "confirmed_breakout"
    elif breakout_failed:
        breakout_state = "failed_breakout"
    elif breakout_attempt:
        breakout_state = "breakout_attempt"
    elif abs(price - prior_high) <= atr * 0.65:
        breakout_state = "testing_resistance"
    else:
        breakout_state = "inside_range"

    return _json_safe(
        {
            "available": True,
            "timeframe": name,
            "bar_count": len(data),
            "last_bar_time": data["date"].iloc[-1],
            "current_price": price,
            "trend": trend,
            "structure_state": structure_state,
            "structure_flags": structure_flags,
            "local_high": local_range["high"],
            "local_low": local_range["low"],
            "active_range": {"lower": local_range["low"], "upper": local_range["high"]},
            "range_position": local_range["position"],
            "range_position_1m": one_month["position"],
            "range_position_3m": three_month["position"],
            "distance_to_local_high_pct": local_range["distance_to_high_pct"],
            "distance_to_local_low_pct": local_range["distance_to_low_pct"],
            "moving_averages": {
                "ema20": ema20,
                "ema50": ema50,
                "ema100": ema100,
                "ema200": ema200,
                "sma50": sma50,
                "sma100": sma100,
                "sma200": sma200,
            },
            "ema_structure": ema_structure,
            "price_vs_moving_averages": {
                "above_ema20": ema20 is not None and price >= ema20,
                "above_sma50": sma50 is not None and price >= sma50,
                "above_sma100": sma100 is not None and price >= sma100,
                "above_sma200": sma200 is not None and price >= sma200,
            },
            "atr": atr,
            "atr_pct": (atr / max(price, 1e-9)) * 100.0,
            "rsi": rsi,
            "rsi_state": rsi_state,
            "volume_ratio": latest_volume_ratio,
            "volume_state": volume_state,
            "momentum_state": momentum_state,
            "compression_state": compression_state,
            "extension_state": extension_state,
            "breakout_state": breakout_state,
            "breakout_trigger_zone": breakout_trigger,
            "breakout_retest": breakout_attempt and abs(price - prior_high) <= atr * 0.75,
            "reclaim_state": "ema20_reclaimed" if reclaimed_ema20 else "none",
            "rejection_state": "resistance_rejection" if breakout_failed else "none",
            "pullback_state": pullback_state,
            "pullback_quality": "constructive" if pullback_state == "constructive_pullback" else "weak" if trend == "downtrend" else "neutral",
            "short_term_reversal_state": "confirmed" if reversal else "not_confirmed",
            "nearest_support_zone": nearest_support,
            "secondary_support_zone": secondary_support,
            "nearest_resistance_zone": nearest_resistance,
            "secondary_resistance_zone": secondary_resistance,
        }
    )


def _extract_timeframe(source: Mapping[str, Any], canonical: str) -> Any:
    for alias in TIMEFRAME_ALIASES[canonical]:
        if alias in source:
            return source[alias]
    return None


def _first_zone(contexts: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((context.get(key) for context in contexts if context.get("available") and context.get(key)), None)


def derive_structure_layers(timeframes: Mapping[str, dict[str, Any]]) -> dict[str, str]:
    """Separate swing trend, active setup, and execution structure."""

    daily = timeframes.get("daily") or {}
    intermediate = next(
        (
            timeframes[name]
            for name in ("four_hour", "hourly")
            if (timeframes.get(name) or {}).get("available")
        ),
        daily,
    )
    execution = next(
        (
            timeframes[name]
            for name in ("thirty_minute", "hourly", "four_hour", "daily")
            if (timeframes.get(name) or {}).get("available")
        ),
        daily,
    )
    daily_trend = str(daily.get("trend") or "range")
    daily_structure = str(daily.get("structure_state") or "range")
    intermediate_trend = str(intermediate.get("trend") or daily_trend)
    intermediate_structure = str(intermediate.get("structure_state") or "range")

    if daily_structure == "structural_breakdown" or daily_trend == "downtrend":
        broader = "structural_breakdown"
    elif daily_structure in {"trend_damage", "reversal_attempt"} or daily_trend == "weak_breakdown_risk":
        broader = "trend_damage"
    elif daily_trend in {"uptrend", "pullback_in_uptrend"}:
        broader = "uptrend"
    else:
        broader = "range"

    if daily_structure == "breakout":
        setup_type = "breakout"
    elif daily_structure == "extended":
        setup_type = "controlled_momentum_continuation"
    elif broader == "uptrend" and (
        daily_structure == "deep_pullback"
        or intermediate_trend in {"range", "weak_breakdown_risk", "downtrend"}
        or intermediate_structure in {"deep_pullback", "trend_damage"}
    ):
        setup_type = "deep_pullback"
    elif broader == "uptrend":
        setup_type = "healthy_pullback"
    elif broader in {"trend_damage", "structural_breakdown"} and (
        execution.get("short_term_reversal_state") == "confirmed"
        or intermediate_structure == "reversal_attempt"
    ):
        setup_type = "reversal_attempt"
    elif broader in {"trend_damage", "structural_breakdown"}:
        setup_type = "repair_after_breakdown"
    else:
        setup_type = "range_rebound" if execution.get("short_term_reversal_state") == "confirmed" else "range"

    execution_state = str(execution.get("structure_state") or "range")
    if execution_state == "base_building" or (
        execution.get("compression_state") == "compressed"
        and execution_state not in {"structural_breakdown", "extended"}
    ):
        execution_structure = "base_building"
    elif execution_state == "reversal_attempt":
        execution_structure = "attempting_base"
    elif execution.get("breakout_state") == "confirmed_breakout":
        execution_structure = "breakout"
    elif execution_state == "extended":
        execution_structure = "extended"
    elif execution_state in {"trend_damage", "structural_breakdown"}:
        execution_structure = "weak_structure"
    else:
        execution_structure = "range"
    return {
        "broader_structure": broader,
        "setup_type_layer": setup_type,
        "execution_structure": execution_structure,
    }


def build_chart_context(
    bars_by_timeframe: Mapping[str, Any] | None = None,
    *,
    daily_bars: Any = None,
    four_hour_bars: Any = None,
    hourly_bars: Any = None,
    thirty_minute_bars: Any = None,
    daily: Any = None,
    four_hour: Any = None,
    hourly: Any = None,
    thirty_minute: Any = None,
    current_price: float | None = None,
    config: PlanningConfig | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable daily/4h/1h/30m chart context.

    ``bars_by_timeframe`` accepts aliases such as ``1d``, ``1h`` and ``30m``.
    Explicit keyword inputs take precedence over values in that mapping.
    """

    config = config or DEFAULT_PLANNING_CONFIG
    source = dict(bars_by_timeframe or {})
    hourly_input = hourly_bars if hourly_bars is not None else hourly if hourly is not None else _extract_timeframe(source, "hourly")
    normalized_hourly = normalize_ohlcv_bars(hourly_input)
    explicit_four_hour = four_hour_bars if four_hour_bars is not None else four_hour if four_hour is not None else _extract_timeframe(source, "four_hour")
    normalized_four_hour = normalize_ohlcv_bars(explicit_four_hour)
    four_hour_source = "provided"
    if normalized_four_hour.empty:
        normalized_four_hour = derive_four_hour_bars(normalized_hourly, config)
        four_hour_source = "derived_from_hourly" if not normalized_four_hour.empty else "unavailable"
    inputs = {
        "daily": daily_bars if daily_bars is not None else daily if daily is not None else _extract_timeframe(source, "daily"),
        "four_hour": normalized_four_hour,
        "hourly": normalized_hourly,
        "thirty_minute": thirty_minute_bars if thirty_minute_bars is not None else thirty_minute if thirty_minute is not None else _extract_timeframe(source, "thirty_minute"),
    }
    timeframe_contexts = {
        name: _timeframe_context(normalize_ohlcv_bars(raw), name, config)
        for name, raw in inputs.items()
    }
    available = [name for name, context in timeframe_contexts.items() if context.get("available")]
    missing = [name for name in timeframe_contexts if name not in available]
    execution_name = next((name for name in ("thirty_minute", "hourly", "four_hour", "daily") if name in available), None)
    dominant_name = next((name for name in ("daily", "four_hour", "hourly", "thirty_minute") if name in available), None)
    execution = timeframe_contexts.get(execution_name or "daily", {})
    dominant = timeframe_contexts.get(dominant_name or "daily", {})

    if not available:
        return {
            "available": False,
            "available_timeframes": [],
            "missing_timeframes": list(timeframe_contexts),
            "timeframes": timeframe_contexts,
            "dominant_trend": "unknown",
            "current_structure": "insufficient_data",
            "broader_structure": "unknown",
            "setup_type_layer": "unknown",
            "execution_structure": "insufficient_data",
            "price_location_context": "unknown",
            "preferred_trade_shape": "no_clean_trade",
        }

    price = _finite(current_price, _finite(execution.get("current_price"))) or 0.0
    trend = str(dominant.get("trend") or execution.get("trend") or "range")
    position = _finite(execution.get("range_position"), 0.5) or 0.5
    extension = str(execution.get("extension_state") or "balanced")
    reversal = execution.get("short_term_reversal_state") == "confirmed"
    breakout = str(execution.get("breakout_state") or "inside_range")

    if position >= config.context_near_high_position:
        price_location = "extended_near_high" if extension == "overextended" else "near_high_but_supported" if trend == "uptrend" else "near_resistance"
    elif position <= config.context_near_low_position:
        price_location = "reversal_from_low" if reversal else "weak_near_low" if trend == "downtrend" else "deep_in_lower_range"
    elif trend == "downtrend":
        price_location = "damaged_mid_range"
    else:
        price_location = "mid_range_constructive"

    if trend == "downtrend":
        current_structure = "constructive_recovery" if reversal else "damaged_structure"
    elif breakout == "confirmed_breakout":
        current_structure = "confirmed_expansion"
    elif trend == "uptrend":
        current_structure = "constructive_trend"
    else:
        current_structure = "range_structure"

    nearest_support = _first_zone([execution, timeframe_contexts["hourly"], timeframe_contexts["four_hour"], timeframe_contexts["daily"]], "nearest_support_zone")
    secondary_support = _first_zone([execution, timeframe_contexts["hourly"], timeframe_contexts["four_hour"], timeframe_contexts["daily"]], "secondary_support_zone")
    nearest_resistance = _first_zone([execution, timeframe_contexts["hourly"], timeframe_contexts["four_hour"], timeframe_contexts["daily"]], "nearest_resistance_zone")
    daily_context = timeframe_contexts["daily"]
    major_resistance = daily_context.get("nearest_resistance_zone") or daily_context.get("secondary_resistance_zone") or nearest_resistance
    breakout_trigger = execution.get("breakout_trigger_zone") or nearest_resistance

    atr = _finite(execution.get("atr"), max(price * 0.01, 0.01)) or max(price * 0.01, 0.01)
    near_support = bool(nearest_support) and price <= float(nearest_support["upper"]) + atr * 0.8
    if trend == "downtrend":
        preferred_shape = "repair_trade" if reversal else "no_clean_trade"
    elif breakout == "confirmed_breakout" and extension != "overextended":
        preferred_shape = "breakout_preferred"
    elif extension in {"extended", "overextended"}:
        preferred_shape = "continuation_pullback"
    elif trend == "uptrend" and near_support:
        preferred_shape = "pullback_preferred"
    elif trend == "range" and reversal and position <= 0.35:
        preferred_shape = "rebound_trade"
    elif position >= config.context_near_high_position and trend == "uptrend":
        preferred_shape = "breakout_preferred"
    else:
        preferred_shape = "no_clean_trade"

    major_source = daily_context if daily_context.get("available") else dominant
    structure_layers = derive_structure_layers(timeframe_contexts)
    timeframe_weights = {"daily": 0.4, "four_hour": 0.3, "hourly": 0.2, "thirty_minute": 0.1}
    signed = {"uptrend": 1.0, "pullback_in_uptrend": 0.7, "range": 0.0, "weak_breakdown_risk": -0.65, "downtrend": -1.0}
    available_weight = sum(timeframe_weights[name] for name in available)
    weighted_alignment = sum(
        timeframe_weights[name] * signed.get(str(timeframe_contexts[name].get("trend") or "range"), 0.0)
        for name in available
    )
    alignment_score = 5.0 if available_weight <= 0 else _clip(5.0 + 5.0 * weighted_alignment / available_weight, 0.0, 10.0)

    result = {
        "available": True,
        "available_timeframes": available,
        "missing_timeframes": missing,
        "execution_timeframe": execution_name,
        "dominant_timeframe": dominant_name,
        "timeframes": timeframe_contexts,
        "four_hour_source": four_hour_source,
        "daily_trend": timeframe_contexts["daily"].get("trend") if timeframe_contexts["daily"].get("available") else None,
        "four_hour_trend": timeframe_contexts["four_hour"].get("trend") if timeframe_contexts["four_hour"].get("available") else None,
        "one_hour_trend": timeframe_contexts["hourly"].get("trend") if timeframe_contexts["hourly"].get("available") else None,
        "thirty_minute_trend": timeframe_contexts["thirty_minute"].get("trend") if timeframe_contexts["thirty_minute"].get("available") else None,
        "multi_timeframe_alignment_score": round(alignment_score, 3),
        "current_price": price,
        "dominant_trend": trend,
        "current_structure": current_structure,
        **structure_layers,
        "price_location_context": price_location,
        "local_high": execution.get("local_high"),
        "local_low": execution.get("local_low"),
        "major_high": major_source.get("local_high"),
        "major_low": major_source.get("local_low"),
        "active_range_high": execution.get("local_high"),
        "active_range_low": execution.get("local_low"),
        "local_range_position": execution.get("range_position"),
        "range_position_1m": daily_context.get("range_position_1m") if daily_context.get("available") else execution.get("range_position_1m"),
        "range_position_3m": daily_context.get("range_position_3m") if daily_context.get("available") else execution.get("range_position_3m"),
        "nearest_support_zone": nearest_support,
        "secondary_support_zone": secondary_support,
        "nearest_resistance_zone": nearest_resistance,
        "major_resistance_zone": major_resistance,
        "breakout_trigger_zone": breakout_trigger,
        "current_execution_anchor": nearest_support,
        "atr": atr,
        "momentum_state": execution.get("momentum_state"),
        "volume_state": execution.get("volume_state"),
        "volume_ratio": execution.get("volume_ratio"),
        "rsi": execution.get("rsi"),
        "rsi_state": execution.get("rsi_state"),
        "breakout_state": breakout,
        "pullback_state": execution.get("pullback_state"),
        "extension_state": extension,
        "short_term_reversal_state": execution.get("short_term_reversal_state"),
        "preferred_trade_shape": preferred_shape,
    }
    return _json_safe(result)


# Friendly aliases for callers that use engine-oriented naming.
analyze_chart_context = build_chart_context
build_multi_timeframe_context = build_chart_context
