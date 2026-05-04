"""Data loading, feature engineering and windowing for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_prices(ticker: str | None = None, csv: str | None = None,
                start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Load OHLCV data from a CSV (Date, Open, High, Low, Close, Volume) or yfinance."""
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
    if len(df) < 300:
        raise ValueError("Need at least ~300 rows of data")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical indicators. All are computed from past data only (no look-ahead)."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    out["log_ret"] = np.log(close).diff()
    out["hl_range"] = (df["High"] - df["Low"]) / close
    out["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    out["ma10_gap"] = close / close.rolling(10).mean() - 1
    out["ma50_gap"] = close / close.rolling(50).mean() - 1
    out["volatility20"] = out["log_ret"].rolling(20).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100
    out["target"] = out["log_ret"].shift(-1)  # predict next day's log return
    out["close"] = close
    return out.dropna()


class Scaler:
    """Standard scaler fitted on training rows only, to avoid leaking test statistics."""

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def make_windows(features: np.ndarray, target: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(features) + 1):
        X.append(features[i - lookback:i])
        y.append(target[i - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


FEATURES = ["log_ret", "hl_range", "vol_z", "ma10_gap", "ma50_gap", "volatility20", "rsi14"]


def chronological_split(df: pd.DataFrame, lookback: int, train=0.7, val=0.15):
    """Split by time (never shuffle time series), scale with train stats, then window."""
    n = len(df)
    i_train, i_val = int(n * train), int(n * (train + val))
    scaler = Scaler().fit(df[FEATURES].values[:i_train])
    feats = scaler.transform(df[FEATURES].values)
    target = df["target"].values
    splits = {}
    # Each split includes `lookback` rows of history before its start for the first window.
    for name, (a, b) in {"train": (0, i_train), "val": (i_train, i_val), "test": (i_val, n)}.items():
        a0 = max(0, a - lookback + 1) if name != "train" else 0
        X, y = make_windows(feats[a0:b], target[a0:b], lookback)
        splits[name] = (X, y, df.index[a0 + lookback - 1:b], df["close"].values[a0 + lookback - 1:b])
    return splits
