import numpy as np
import pandas as pd

from src.features.realized_vol import (
    daily_bipower_variation,
    daily_close_returns,
    daily_realized_variance,
    daily_signed_semivariance,
    intraday_log_returns,
)


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


def _gapped_two_day_prices():
    """Day 2 opens at 150, a huge gap up from day 1's close of 100.

    The overnight gap (100 -> 150) must NOT be diffed into an intraday
    return, and day 2's RV/BV must reflect only the small intraday moves
    within day 2 (150 -> 151 -> 150).
    """
    idx = pd.to_datetime([
        "2023-01-03 09:30", "2023-01-03 09:35", "2023-01-03 09:40",
        "2023-01-04 09:30", "2023-01-04 09:35", "2023-01-04 09:40",
    ])
    return pd.Series([100.0, 101.0, 100.0, 150.0, 151.0, 150.0], index=idx)


def test_intraday_log_returns_drops_overnight_gap_at_session_boundary():
    prices = _gapped_two_day_prices()
    returns = intraday_log_returns(prices)

    day2 = pd.Timestamp("2023-01-04").date()
    day2_returns = returns.loc[returns.index.date == day2]

    # Only 2 intraday returns for day 2 (first bar of the session has no
    # valid within-day prior price to diff against).
    assert len(day2_returns) == 2
    # None of day 2's returns should be the overnight gap log(150/100).
    overnight_gap = np.log(150.0 / 100.0)
    assert not np.isclose(day2_returns.iloc[0], overnight_gap)
    # First retained return of day 2 is 151 vs 150 (intraday-only).
    assert np.isclose(day2_returns.iloc[0], np.log(151.0 / 150.0))


def test_daily_realized_variance_excludes_overnight_gap():
    prices = _gapped_two_day_prices()
    returns = intraday_log_returns(prices)
    rv = daily_realized_variance(returns)

    day2 = pd.Timestamp("2023-01-04").date()
    r1 = np.log(151.0 / 150.0)
    r2 = np.log(150.0 / 151.0)
    expected_rv_day2 = r1 ** 2 + r2 ** 2  # intraday-only, no overnight term
    assert np.isclose(rv.loc[day2], expected_rv_day2)

    # Sanity: contaminated (buggy) RV would have included the overnight
    # gap term and been much larger.
    overnight_gap = np.log(150.0 / 100.0)
    contaminated_rv_day2 = overnight_gap ** 2 + r2 ** 2
    assert rv.loc[day2] < contaminated_rv_day2


def test_daily_bipower_variation_does_not_pair_across_session_boundary():
    prices = _gapped_two_day_prices()
    returns = intraday_log_returns(prices)
    bv = daily_bipower_variation(returns)

    day2 = pd.Timestamp("2023-01-04").date()
    r1 = np.log(151.0 / 150.0)
    r2 = np.log(150.0 / 151.0)
    # Day 2 has only 2 intraday returns, so only one adjacent product
    # (|r2| * |r1|), computed within the session -- not paired with
    # day 1's last return.
    expected_bv_day2 = (np.pi / 2) * (abs(r2) * abs(r1))
    assert np.isclose(bv.loc[day2], expected_bv_day2)
