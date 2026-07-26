import numpy as np
import pandas as pd

from src.features.realized_vol import daily_close_returns, daily_signed_semivariance


def _two_day_prices():
    idx = pd.to_datetime([
        "2023-01-03 09:30", "2023-01-03 09:35", "2023-01-03 09:40",
        "2023-01-04 09:30", "2023-01-04 09:35", "2023-01-04 09:40",
    ])
    return pd.Series([100.0, 101.0, 100.0, 100.0, 99.0, 99.5], index=idx)


def test_daily_signed_semivariance_splits_up_and_down():
    prices = _two_day_prices()
    returns = np.log(prices).diff().dropna()
    out = daily_signed_semivariance(returns)

    day1 = pd.Timestamp("2023-01-03").date()
    day2 = pd.Timestamp("2023-01-04").date()

    # day1: up then down -> both rv_pos and rv_neg > 0
    assert out.loc[day1, "rv_pos"] > 0
    assert out.loc[day1, "rv_neg"] > 0
    # day2: down then up -> both rv_pos and rv_neg > 0
    assert out.loc[day2, "rv_pos"] > 0
    assert out.loc[day2, "rv_neg"] > 0
    # rv_pos + rv_neg == plain RV for the day
    rv_day1 = returns.loc["2023-01-03"].pow(2).sum()
    assert np.isclose(out.loc[day1, "rv_pos"] + out.loc[day1, "rv_neg"], rv_day1)


def test_daily_close_returns():
    prices = _two_day_prices()
    out = daily_close_returns(prices)

    assert len(out) == 1  # first day has no prior close to diff against
    expected = np.log(99.5 / 100.0)
    assert np.isclose(out.iloc[0], expected)
