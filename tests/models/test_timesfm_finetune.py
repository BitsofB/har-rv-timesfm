import numpy as np
import pandas as pd
import pytest

from src.models.timesfm_finetune import (
    assemble_hybrid_forecast,
    build_windows,
    forecast_residuals,
    rolling_zero_shot_forecast,
)


class _FakePersistenceModel:
    """Stand-in for TimesFM_2p5_200M_torch: mimics the real `.forecast()`
    signature (`horizon`, `inputs` -> `(point_forecast, quantile_forecast)`)
    without needing network access or real weights. Forecasts the last
    context value repeated `horizon` times -- a stand-in "persistence"
    zero-shot model -- and records every context array it was called with,
    so tests can assert exactly what information reached the model at each
    step (i.e. that no future value leaked into the context).
    """

    def __init__(self):
        self.seen_contexts = []

    def forecast(self, horizon, inputs):
        self.seen_contexts.append(inputs[0].copy())
        point = np.array([[inputs[0][-1]] * horizon])
        quantiles = np.zeros((1, horizon, 1))
        return point, quantiles


def test_forecast_residuals_calls_model_and_unwraps_batch():
    model = _FakePersistenceModel()
    context = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = forecast_residuals(model, context)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(3.0)


def test_rolling_zero_shot_forecast_indexing_matches_har_rv_convention():
    # Mirrors test_rolling_garch_11_key_i_uses_info_through_i_inclusive and
    # test_naive_persistence_shifts_rv_d: prediction keyed at index[i] must
    # be produced using series values through index i INCLUSIVE (the "as of"
    # date), forecasting the next day -- never using index i+1 or later.
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    series = pd.Series(range(10), index=idx, dtype=float)
    model = _FakePersistenceModel()

    preds = rolling_zero_shot_forecast(
        series, model, context_length=512, min_train_size=5
    )

    # i ranges over [5, 8] (n=10, loop stops at n-1=9 exclusive -> last i=8)
    assert list(preds.index) == list(idx[5:9])
    # our fake model's forecast = last context value = series.iloc[i]
    assert list(preds.values) == [5.0, 6.0, 7.0, 8.0]

    # No leakage: context at step i must never contain series.iloc[i+1] or
    # later. With this monotonically increasing series, the max value seen
    # in each context must equal series.iloc[i], not series.iloc[i+1].
    for i, context in zip(range(5, 9), model.seen_contexts):
        assert context.max() == series.iloc[i]
        assert len(context) == i + 1  # through index i inclusive


def test_rolling_zero_shot_forecast_respects_context_length_cap():
    idx = pd.date_range("2023-01-01", periods=20, freq="D")
    series = pd.Series(range(20), index=idx, dtype=float)
    model = _FakePersistenceModel()

    rolling_zero_shot_forecast(series, model, context_length=5, min_train_size=10)

    # every context window should be capped at context_length once warmed up
    assert all(len(c) <= 5 for c in model.seen_contexts)


def test_build_windows_shapes():
    series = pd.Series(range(10), dtype=float)
    X, y = build_windows(series, context_length=4, horizon=1)

    assert X.shape == (6, 4)
    assert y.shape == (6, 1)
    assert list(X[0]) == [0, 1, 2, 3]
    assert list(y[0]) == [4]


def test_assemble_hybrid_forecast_adds_aligned_series():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    har_pred = pd.Series([1.0, 2.0, 3.0], index=idx)
    resid_pred = pd.Series([0.1, 0.2, 0.3], index=idx)

    hybrid = assemble_hybrid_forecast(har_pred, resid_pred)

    assert list(hybrid.values) == pytest.approx([1.1, 2.2, 3.3])


def test_assemble_hybrid_forecast_rejects_misaligned_index():
    idx_a = pd.date_range("2023-01-01", periods=3, freq="D")
    idx_b = pd.date_range("2023-01-02", periods=3, freq="D")
    har_pred = pd.Series([1.0, 2.0, 3.0], index=idx_a)
    resid_pred = pd.Series([0.1, 0.2, 0.3], index=idx_b)

    with pytest.raises(AssertionError):
        assemble_hybrid_forecast(har_pred, resid_pred)
