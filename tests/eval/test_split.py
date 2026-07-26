import numpy as np
import pandas as pd
import pytest

from src.eval.split import split_with_embargo


def _synthetic_series(n: int = 1000) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.arange(n, dtype=float), index=idx, name="residual")


def test_splits_are_disjoint_and_ordered():
    series = _synthetic_series()
    train, val, test = split_with_embargo(series, context_length=20)

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) < len(series)


def test_embargo_gap_excludes_context_length_sessions_at_each_boundary():
    series = _synthetic_series()
    context_length = 30
    train, val, test = split_with_embargo(series, context_length=context_length)

    gap_train_val = series.index.get_loc(val.index.min()) - series.index.get_loc(train.index.max())
    gap_val_test = series.index.get_loc(test.index.min()) - series.index.get_loc(val.index.max())

    assert gap_train_val > context_length
    assert gap_val_test > context_length


def test_no_val_or_test_window_context_reaches_into_train():
    """
    Guards the embargo's actual contract: IF a window's context were ever
    read as `context_length` prior positions in the full original series
    (global-position indexing, not build_windows()'s current per-split
    local indexing), that context must still not reach into train. This
    is deliberately stricter than what build_windows() needs today under
    the mandated split-then-window pattern, where it cannot leak
    regardless of embargo size -- see split.py's module docstring.
    """
    series = _synthetic_series()
    context_length = 25
    train, val, test = split_with_embargo(series, context_length=context_length)

    train_end_pos = series.index.get_loc(train.index.max())
    for split in (val, test):
        first_window_start_pos = series.index.get_loc(split.index.min())
        context_start_pos = first_window_start_pos - context_length
        assert context_start_pos > train_end_pos


def test_embargo_below_context_length_raises():
    series = _synthetic_series()
    with pytest.raises(ValueError, match="embargo"):
        split_with_embargo(series, context_length=30, embargo=10)


def test_too_large_embargo_raises_on_empty_split():
    series = _synthetic_series(n=100)
    with pytest.raises(ValueError, match="empty"):
        split_with_embargo(series, context_length=90)


def test_unsorted_index_raises():
    series = _synthetic_series()
    shuffled = series.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        split_with_embargo(shuffled, context_length=20)
