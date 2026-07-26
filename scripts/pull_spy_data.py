"""
End-to-end pipeline: pull SPY intraday bars, align to the NYSE session
grid, build RV/HAR features, run all three baselines walk-forward, and
write reports/baseline_metrics.md.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment. Run:
    python scripts/pull_spy_data.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report
from src.features.data_alpaca import fetch_intraday_bars
from src.features.data_cleaning import flag_bad_days, reindex_to_grid, session_grid
from src.features.realized_vol import build_feature_table, daily_close_returns
from src.models.baselines import fit_naive_har_garch

EXPECTED_BARS_PER_DAY = 78  # 6.5h regular session / 5-min bars


def main(symbol: str, start: str, end: str, min_train_size: int = 250) -> None:
    bars = fetch_intraday_bars(
        symbol, datetime.fromisoformat(start), datetime.fromisoformat(end)
    )

    grid = session_grid(start, end)
    reindexed, missing_report = reindex_to_grid(bars, grid)
    bad_days = flag_bad_days(missing_report, EXPECTED_BARS_PER_DAY)
    if bad_days:
        print(f"WARNING: excluding {len(bad_days)} sessions with excessive "
              f"missing bars: {bad_days}")
        keep = ~pd.Series(reindexed.index.date, index=reindexed.index).isin(bad_days)
        reindexed = reindexed[keep]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    reindexed.to_parquet(f"data/raw/{symbol}_5min.parquet")

    prices = reindexed["close"].dropna()
    feats = build_feature_table(prices)
    daily_ret = daily_close_returns(prices)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    feats.to_parquet(f"data/processed/{symbol}_features.parquet")

    target, predictions = fit_naive_har_garch(feats, daily_ret, min_train_size)

    common_idx = (
        predictions["har_rv"].index
        .intersection(predictions["naive"].index)
        .intersection(predictions["garch11"].index)
    )
    metrics = compute_baseline_metrics({
        name: (target.loc[common_idx], preds.loc[common_idx])
        for name, preds in predictions.items()
    })

    if bad_days:
        excluded_range = f"{min(bad_days)} to {max(bad_days)}"
    else:
        excluded_range = "none"

    notes = (
        "## Notes\n\n"
        "- **Data feed**: Alpaca free tier, IEX feed (a single venue), not the "
        "full consolidated SIP tape. Absolute RV magnitudes here are not "
        "directly comparable to published academic figures built on "
        "TAQ/SIP data.\n"
        f"- **Evaluation window**: {common_idx.min().date()} to "
        f"{common_idx.max().date()} (n={len(common_idx)} sessions).\n"
        f"- **Excluded sessions**: {len(bad_days)} session(s) dropped by "
        f"`flag_bad_days` for excessive missing intraday bars, spanning "
        f"{excluded_range}.\n"
        "- **GARCH(1,1) caveat**: `garch11` forecasts the conditional "
        "variance of close-to-close *daily* returns, not intraday realized "
        "variance. It's included as a standard volatility baseline for "
        "comparison, not because its target matches `naive`/`har_rv` "
        "exactly -- treat cross-model comparisons involving `garch11` with "
        "that in mind.\n"
    )

    write_baseline_report(metrics, notes=notes)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-07-24")
    parser.add_argument("--min-train-size", type=int, default=250)
    args = parser.parse_args()
    main(args.symbol, args.start, args.end, args.min_train_size)
