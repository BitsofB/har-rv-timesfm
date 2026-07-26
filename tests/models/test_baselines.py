import numpy as np
import pandas as pd

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
