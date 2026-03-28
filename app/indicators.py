from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    """Normalize daily bars into a sorted DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(bars).copy()
    if "bar_date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["bar_date"])
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    else:
        frame["date"] = pd.NaT

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["open"] = frame["open"].fillna(frame["close"])
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1))
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1))
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return frame


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def rolling_return(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def realized_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    returns = series.pct_change()
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)


def add_indicator_columns(frame: pd.DataFrame, *, atr_window: int = 14, volume_window: int = 20) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    out["ema20"] = ema(out["close"], 20)
    out["sma50"] = sma(out["close"], 50)
    out["sma100"] = sma(out["close"], 100)
    out["sma200"] = sma(out["close"], 200)
    out["atr"] = atr(out, window=atr_window)
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["avg_volume"] = out["volume"].rolling(window=volume_window, min_periods=5).mean()
    out["volume_ratio"] = out["volume"] / out["avg_volume"].replace(0, np.nan)
    out["ret_5"] = rolling_return(out["close"], 5)
    out["ret_20"] = rolling_return(out["close"], 20)
    out["vol_20"] = realized_volatility(out["close"], 20)
    return out


def latest_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    val = frame[column].iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def percentile_rank(values: Iterable[float], value: float) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 0.5
    less = sum(1 for v in vals if v <= value)
    return less / max(len(vals), 1)
