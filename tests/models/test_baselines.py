import numpy as np
import pandas as pd
from arch import arch_model

from src.models.baselines import naive_persistence, rolling_garch_11


def test_naive_persistence_shifts_rv_d():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    features = pd.DataFrame({"rv_d": range(10)}, index=idx)

    result = naive_persistence(features, min_train_size=5)

    assert list(result.predictions.index) == list(idx[5:])
    assert list(result.predictions.values) == list(range(5, 10))


def test_rolling_garch_11_produces_positive_variance_forecasts():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, size=30), index=idx)

    result = rolling_garch_11(returns, min_train_size=20)

    assert len(result.predictions) == 10
    assert (result.predictions > 0).all()
    assert list(result.predictions.index) == list(idx[20:])


def test_rolling_garch_11_key_i_uses_info_through_i_inclusive():
    # Prediction keyed at index[i] must be "info through day i, forecast for
    # day i+1" -- the same convention rolling_har_rv and naive_persistence
    # use. That means fitting on daily_returns.iloc[:i+1] (day i INCLUDED),
    # not daily_returns.iloc[:i] (day i excluded, a one-day offset bug).
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, size=30), index=idx)

    result = rolling_garch_11(returns, min_train_size=20)

    i = 20
    train = returns.iloc[:i + 1]
    model = arch_model(train * 100, vol="Garch", p=1, q=1, dist="normal")
    fit = model.fit(disp="off")
    forecast = fit.forecast(horizon=1, reindex=False)
    expected = forecast.variance.values[-1, 0] / 100**2

    assert result.predictions.loc[idx[i]] == expected
