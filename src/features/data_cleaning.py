"""
Session-grid alignment and data-quality flagging for intraday bars.

Never interpolate over gaps here — flag them and let the caller decide
whether a session is usable. Interpolating price data before computing RV
inflates or deflates the variance estimate in ways that are hard to detect
downstream (CLAUDE.md section 7).
"""

import pandas as pd
import pandas_market_calendars as mcal


def session_grid(
    start: str,
    end: str,
    freq_minutes: int = 5,
    calendar: str = "NYSE",
) -> pd.DatetimeIndex:
    """
    Expected regular-trading-hours timestamp grid for every valid session
    between start and end, at freq_minutes spacing. Holidays and half-days
    are handled automatically by the exchange calendar.
    """
    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start, end_date=end)

    day_grids = [
        pd.date_range(
            row.market_open, row.market_close, freq=f"{freq_minutes}min", inclusive="left"
        )
        for row in schedule.itertuples()
    ]
    return pd.DatetimeIndex(pd.concat([pd.Series(g) for g in day_grids])).sort_values()


def reindex_to_grid(bars: pd.DataFrame, grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """
    Reindex intraday bars onto the expected session grid. Missing slots
    become NaN rows -- never filled here.

    Returns (reindexed_df, missing_report) where missing_report maps
    each session date -> count of missing bars that day.
    """
    reindexed = bars.reindex(grid)
    missing_mask = reindexed["close"].isna()
    counts = missing_mask.groupby(missing_mask.index.date).sum().astype(int)
    missing_report = counts[counts > 0].to_dict()
    return reindexed, missing_report


def flag_bad_days(
    missing_report: dict,
    expected_bars_per_day: int,
    max_missing_frac: float = 0.05,
) -> list:
    """
    Dates whose missing-bar fraction exceeds the threshold -- these should
    be excluded (and logged), never interpolated.
    """
    bad = [
        date
        for date, missing in missing_report.items()
        if missing / expected_bars_per_day > max_missing_frac
    ]
    return sorted(bad)
