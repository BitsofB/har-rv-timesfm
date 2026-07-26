"""
Diebold-Mariano pairwise significance tests across all four baselines
(naive, HAR-RV, GARCH(1,1), TimesFM zero-shot) -- required by CLAUDE.md
section 2.2 before any "model A beats model B" ranking in
reports/baseline_metrics.md / reports/zeroshot_timesfm_metrics.md is
treated as skill rather than noise.

Reads the already-cached data/raw/{symbol}_5min.parquet and
data/processed/{symbol}_features.parquet (produced by
scripts/pull_spy_data.py) -- does not re-pull from Alpaca. Recomputes all
four models' predictions on the shared evaluation window so their
per-observation QLIKE losses can be compared pairwise.

Run:
    python scripts/run_dm_tests.py --symbol SPY
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from itertools import combinations

import pandas as pd

from src.eval.diebold_mariano import diebold_mariano_test
from src.features.realized_vol import daily_close_returns
from src.models.baselines import fit_naive_har_garch
from src.models.har_rv import qlike_loss
from src.models.timesfm_finetune import load_pretrained_timesfm, zeroshot_forecast


def load_cached_data(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    """Read the cached parquet files scripts/pull_spy_data.py produces."""
    features_path = Path(f"data/processed/{symbol}_features.parquet")
    raw_path = Path(f"data/raw/{symbol}_5min.parquet")
    for path in (features_path, raw_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `python scripts/pull_spy_data.py "
                f"--symbol {symbol}` first to produce the cached data this "
                f"script reads."
            )

    feats = pd.read_parquet(features_path)
    prices = pd.read_parquet(raw_path)["close"].dropna()
    return feats, prices


def compute_all_predictions(feats: pd.DataFrame, prices: pd.Series, min_train_size: int):
    """Fit naive/HAR-RV/GARCH(1,1) and run TimesFM zero-shot on the same series."""
    daily_ret = daily_close_returns(prices)
    target, predictions = fit_naive_har_garch(feats, daily_ret, min_train_size)

    print("Running TimesFM zero-shot...")
    model = load_pretrained_timesfm()
    predictions["timesfm_zeroshot"] = zeroshot_forecast(feats["rv_d"], min_train_size, model=model)

    return target, predictions


def pairwise_dm_tests(y_true, losses: dict) -> pd.DataFrame:
    """Run a DM test for every unordered pair of models in `losses`."""
    rows = []
    for name_a, name_b in combinations(losses, 2):
        stat, p_value = diebold_mariano_test(losses[name_a], losses[name_b])
        favored = name_a if stat < 0 else name_b
        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "dm_statistic": stat,
            "p_value": p_value,
            "significant_at_5pct": p_value < 0.05,
            "favors": favored if p_value < 0.05 else "neither (not significant)",
        })
    return pd.DataFrame(rows)


def write_dm_report(results: pd.DataFrame, common_idx: pd.DatetimeIndex, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Diebold-Mariano pairwise test results (QLIKE loss)\n\n")
        f.write(f"Evaluation window: {common_idx.min().date()} to "
                f"{common_idx.max().date()} (n={len(common_idx)}).\n\n")
        f.write(results.to_markdown(index=False))
        f.write(
            "\n\nH0: equal predictive accuracy between model_a and model_b's "
            "QLIKE loss series. `favors` is only reported when p_value < 0.05 "
            "-- CLAUDE.md section 2.2 requires this before treating a QLIKE "
            "ranking as evidence of skill rather than noise.\n"
        )


def main(symbol: str, min_train_size: int = 250) -> None:
    feats, prices = load_cached_data(symbol)

    print("Refitting HAR-RV, naive, GARCH(1,1)...")
    target, predictions = compute_all_predictions(feats, prices, min_train_size)

    common_idx = predictions["naive"].index
    for preds in predictions.values():
        common_idx = common_idx.intersection(preds.index)
    print(f"Common evaluation window: {common_idx.min().date()} to "
          f"{common_idx.max().date()} (n={len(common_idx)})")

    y_true = target.loc[common_idx].values
    losses = {
        name: qlike_loss(y_true, preds.loc[common_idx].values)
        for name, preds in predictions.items()
    }

    results = pairwise_dm_tests(y_true, losses)
    print(results.to_string(index=False))

    out_path = "reports/dm_test_results.md"
    write_dm_report(results, common_idx, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--min-train-size", type=int, default=250)
    args = parser.parse_args()
    main(args.symbol, args.min_train_size)
