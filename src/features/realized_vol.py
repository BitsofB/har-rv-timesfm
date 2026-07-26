"""
Realized volatility feature construction.

All functions here must only use information available up to time t
(no look-ahead). Intraday input is assumed to be a DataFrame indexed by
timestamp with a single 'price' column, already cleaned and resampled
to a fixed grid (e.g. 5-min).
"""

import numpy as np
import pandas as pd


def intraday_log_returns(prices: pd.Series) -> pd.Series:
    """Log returns within a trading day at the resampled grid frequency."""
    return np.log(prices).diff().dropna()


def daily_realized_variance(intraday_returns: pd.Series) -> pd.Series:
    """RV_t = sum of squared intraday log returns, grouped by date."""
    return intraday_returns.pow(2).groupby(intraday_returns.index.date).sum()


def daily_bipower_variation(intraday_returns: pd.Series) -> pd.Series:
    """
    Jump-robust volatility measure (Barndorff-Nielsen & Shephard).
    BV_t = (pi/2) * sum(|r_i| * |r_{i-1}|)
    """
    abs_r = intraday_returns.abs()
    prod = abs_r * abs_r.shift(1)
    return (np.pi / 2) * prod.groupby(intraday_returns.index.date).sum()


def jump_component(rv: pd.Series, bv: pd.Series) -> pd.Series:
    """J_t = max(RV_t - BV_t, 0)."""
    return (rv - bv).clip(lower=0)


def har_lag_features(rv: pd.Series) -> pd.DataFrame:
    """
    Build the three HAR regressors: daily, weekly (5d avg), monthly (22d avg).
    Uses only past values (shift(1) applied at the point of use, not here —
    caller is responsible for aligning X_t -> y_{t+1}).
    """
    return pd.DataFrame({
        "rv_d": rv,
        "rv_w": rv.rolling(5).mean(),
        "rv_m": rv.rolling(22).mean(),
    })


def build_feature_table(prices: pd.Series) -> pd.DataFrame:
    """Full pipeline: prices -> RV, BV, jumps, HAR lag features."""
    r = intraday_log_returns(prices)
    rv = daily_realized_variance(r)
    bv = daily_bipower_variation(r)
    jumps = jump_component(rv, bv)

    feats = har_lag_features(rv)
    feats["bv"] = bv
    feats["jump"] = jumps
    feats.index = pd.to_datetime(feats.index)
    return feats.dropna()


if __name__ == "__main__":
    raise SystemExit(
        "This module is a library of feature functions — "
        "call build_feature_table() from a pipeline script, not directly."
    )
