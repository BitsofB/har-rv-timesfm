"""
TimesFM 2.5 zero-shot sanity baseline (TODO.md step 4): forecast raw RV
directly with the pretrained checkpoint, no HAR involved and no
fine-tuning. Confirms the foundation model gets anywhere near HAR-RV
before spending compute on fine-tuning (CLAUDE.md sections 2.3/6).

Reads data/processed/{symbol}_features.parquet, already produced by
scripts/pull_spy_data.py -- this script does not pull from Alpaca itself.

Run:
    python scripts/run_zeroshot_timesfm.py --symbol SPY
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report
from src.models.timesfm_finetune import load_pretrained_timesfm


def zeroshot_forecasts(
    rv: pd.Series,
    min_train_size: int = 250,
    context_length: int = 512,
    horizon: int = 1,
) -> pd.Series:
    """
    Walk forward one step at a time: at index i, feed the model rv[i-context_length+1 : i+1]
    (info through day i inclusive) and take its 1-step-ahead point forecast as
    the prediction for day i+1 -- same "key i = forecast for i+1" convention
    already used by rolling_har_rv/naive_persistence/rolling_garch_11.

    Unlike those baselines, TimesFM is not refit per step (it's a pretrained
    foundation model run zero-shot) -- only the context window advances.
    """
    model = load_pretrained_timesfm()

    contexts, idxs = [], []
    n = len(rv)
    for i in range(min_train_size, n):
        start = max(0, i - context_length + 1)
        contexts.append(rv.values[start:i + 1].astype("float32"))
        idxs.append(rv.index[i])

    point_forecast, _ = model.forecast(horizon=horizon, inputs=contexts)
    return pd.Series(point_forecast[:, 0], index=idxs, name="timesfm_zeroshot_pred")


def main(symbol: str, min_train_size: int = 250, context_length: int = 512) -> None:
    feats = pd.read_parquet(f"data/processed/{symbol}_features.parquet")
    rv = feats["rv_d"]
    target = rv.shift(-1).dropna().rename("target")

    zeroshot_preds = zeroshot_forecasts(rv, min_train_size, context_length)

    common_idx = target.index.intersection(zeroshot_preds.index)
    metrics = compute_baseline_metrics({
        "timesfm_zeroshot": (target.loc[common_idx], zeroshot_preds.loc[common_idx]),
    })
    print(metrics)

    notes = (
        f"TimesFM 2.5 zero-shot (no fine-tuning) on raw daily RV, no HAR involved. "
        f"Evaluation window: {common_idx.min().date()} to {common_idx.max().date()} "
        f"(n={len(common_idx)}). context_length={context_length}, horizon=1. "
        f"Compare against reports/baseline_metrics.md's naive/har_rv/garch11 rows "
        f"on the overlapping window before judging zero-shot skill (CLAUDE.md section 2.3)."
    )
    write_baseline_report(
        metrics, out_path="reports/zeroshot_timesfm_metrics.md", notes=notes
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--min-train-size", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()
    main(args.symbol, args.min_train_size, args.context_length)
