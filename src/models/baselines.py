"""
Baseline forecasters compared against HAR-RV: naive persistence and
GARCH(1,1). Both walk forward one step at a time -- never fit on the
full sample and evaluate in-sample (same rule as rolling_har_rv in
src/models/har_rv.py).
"""

from dataclasses import dataclass

import pandas as pd
from arch import arch_model

from src.models.har_rv import rolling_har_rv


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
            train = daily_returns.iloc[:i + 1]
        else:
            train = daily_returns.iloc[max(0, i + 1 - rolling_size):i + 1]

        # arch_model wants returns scaled to roughly O(1) percent for
        # numerically stable optimization
        model = arch_model(train * 100, vol="Garch", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        forecast = fit.forecast(horizon=1, reindex=False)
        variance_pct2 = forecast.variance.values[-1, 0]
        preds.append((daily_returns.index[i], variance_pct2 / 100**2))

    pred_series = pd.Series(dict(preds)).rename("garch_pred")
    return BaselineResult(pred_series)


def fit_naive_har_garch(
    feats: pd.DataFrame,
    daily_ret: pd.Series,
    min_train_size: int = 250,
) -> tuple[pd.Series, dict]:
    """
    Shared walk-forward fit for naive/HAR-RV/GARCH(1,1) -- used both by the
    live-data pipeline (scripts/pull_spy_data.py) and by scripts comparing
    predictions from cached data (scripts/run_dm_tests.py), so an alignment
    fix only has to happen in one place.

    feats: the RV/HAR feature table (build_feature_table output), must have
    an 'rv_d' column. daily_ret: close-to-close daily log returns.

    Returns (target, predictions) where target is RV_{t+1} indexed by day t,
    and predictions is {"naive": Series, "har_rv": Series, "garch11": Series},
    all sharing the same "key t = forecast for t+1" index convention.
    """
    target = feats["rv_d"].shift(-1).dropna().rename("target")
    features_aligned = feats.loc[target.index]
    daily_ret_aligned = daily_ret.loc[daily_ret.index.intersection(target.index)]

    har = rolling_har_rv(features_aligned[["rv_d", "rv_w", "rv_m"]], target, min_train_size)
    naive = naive_persistence(features_aligned, min_train_size)
    garch = rolling_garch_11(daily_ret_aligned, min_train_size)

    return target, {
        "naive": naive.predictions,
        "har_rv": har.predictions,
        "garch11": garch.predictions,
    }
