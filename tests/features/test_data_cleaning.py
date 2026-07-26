import pandas as pd

from src.features.data_cleaning import flag_bad_days, reindex_to_grid, session_grid


def test_session_grid_two_trading_days():
    # 2023-01-03 and 2023-01-04 are both regular NYSE trading days
    grid = session_grid("2023-01-03", "2023-01-04", freq_minutes=60)
    # 9:30-16:00 ET at 60-min spacing, right-open -> 7 bars/day (9:30..15:30)
    assert len(grid) == 14
    assert grid.date.tolist().count(pd.Timestamp("2023-01-03").date()) == 7


def test_reindex_flags_missing_bars():
    grid = session_grid("2023-01-03", "2023-01-03", freq_minutes=60)
    # Drop the 11:30 bar to simulate a gap
    present = grid.delete(2)
    bars = pd.DataFrame({"close": range(len(present))}, index=present)

    reindexed, missing_report = reindex_to_grid(bars, grid)

    assert len(reindexed) == len(grid)
    day = pd.Timestamp("2023-01-03").date()
    assert missing_report[day] == 1


def test_flag_bad_days_threshold():
    missing_report = {"2023-01-03": 1, "2023-01-04": 5}
    bad = flag_bad_days(missing_report, expected_bars_per_day=7, max_missing_frac=0.3)
    # 1/7 ~= 0.14 (ok), 5/7 ~= 0.71 (bad)
    assert bad == ["2023-01-04"]
