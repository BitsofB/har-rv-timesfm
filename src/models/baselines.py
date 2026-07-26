"""
Baseline forecasters compared against HAR-RV: naive persistence and
GARCH(1,1). Both walk forward one step at a time -- never fit on the
full sample and evaluate in-sample (same rule as rolling_har_rv in
src/models/har_rv.py).
"""

from dataclasses import dataclass

import pandas as pd
from arch import arch_model


@dataclass
class BaselineResult:
    predictions: pd.Series


def naive_persistence(features: pd.DataFrame, min_train_size: int = 250) -> BaselineResult:
    """
    RV_{t+1}_pred = RV_t. Uses the same evaluation window (starting at
    min_train_size) as rolling_har_rv so metrics are computed on identical
    dates across models.
    """
    preds = features["rv_d"].iloc[min_train_size:]
    return BaselineResult(preds.rename("naive_pred"))


def rolling_garch_11(
    daily_returns: pd.Series,
    min_train_size: int = 250,
    window: str = "expanding",
    rolling_size: int = 500,
) -> BaselineResult:
    """
    Walk forward one step at a time: fit GARCH(1,1) on returns up to t,
    forecast the 1-step-ahead conditional variance for t+1.
    """
    preds = []
    n = len(daily_returns)

    for i in range(min_train_size, n):
        if window == "expanding":
            train = daily_returns.iloc[:i]
        else:
            train = daily_returns.iloc[max(0, i - rolling_size):i]

        # arch_model wants returns scaled to roughly O(1) percent for
        # numerically stable optimization
        model = arch_model(train * 100, vol="Garch", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        forecast = fit.forecast(horizon=1, reindex=False)
        variance_pct2 = forecast.variance.values[-1, 0]
        preds.append((daily_returns.index[i], variance_pct2 / 100**2))

    pred_series = pd.Series(dict(preds)).rename("garch_pred")
    return BaselineResult(pred_series)
