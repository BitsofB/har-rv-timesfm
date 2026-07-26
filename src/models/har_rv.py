"""
HAR-RV baseline: RV_{t+1} = b0 + b_d*RV_t + b_w*RV_t_week + b_m*RV_t_month + e

Fit with a rolling/expanding window to avoid full-sample leakage — never
call .fit() once on the whole dataset and evaluate in-sample.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class HARRVResult:
    predictions: pd.Series
    residuals: pd.Series
    coefficients: pd.DataFrame  # per-window coefficients, indexed by fit date


def rolling_har_rv(
    features: pd.DataFrame,
    target: pd.Series,
    min_train_size: int = 250,
    window: str = "expanding",  # "expanding" or "rolling"
    rolling_size: int = 500,
) -> HARRVResult:
    """
    Walk forward one step at a time: fit on data up to t, predict t+1.

    features: DataFrame with columns ['rv_d', 'rv_w', 'rv_m'] indexed by date,
              already shifted so that row t's features only use info <= t.
    target:   RV_{t+1}, aligned so target.index == features.index (the value
              being predicted for the *next* period).
    """
    assert features.index.equals(target.index), "features/target must be aligned"

    preds, resids, coefs = [], [], []
    n = len(features)

    for i in range(min_train_size, n):
        if window == "expanding":
            train_slice = slice(0, i)
        else:
            train_slice = slice(max(0, i - rolling_size), i)

        X_train = features.iloc[train_slice].values
        y_train = target.iloc[train_slice].values
        X_test = features.iloc[[i]].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        idx = features.index[i]
        preds.append((idx, pred))
        resids.append((idx, target.iloc[i] - pred))
        coefs.append((idx, *model.coef_, model.intercept_))

    pred_series = pd.Series(dict(preds)).rename("har_pred")
    resid_series = pd.Series(dict(resids)).rename("har_resid")
    coef_df = pd.DataFrame(
        coefs, columns=["date", "b_d", "b_w", "b_m", "b0"]
    ).set_index("date")

    return HARRVResult(pred_series, resid_series, coef_df)


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QLIKE loss — standard evaluation metric for volatility forecasts
    (penalizes underprediction of variance more than squared error does).
    Assumes y_true, y_pred are variances (not vol / std), both > 0.
    """
    ratio = y_true / y_pred
    return float(np.mean(ratio - np.log(ratio) - 1))
