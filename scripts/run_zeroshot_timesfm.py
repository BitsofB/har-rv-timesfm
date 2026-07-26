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
from src.models.timesfm_finetune import zeroshot_forecast


def main(symbol: str, min_train_size: int = 250, context_length: int = 512) -> None:
    feats = pd.read_parquet(f"data/processed/{symbol}_features.parquet")
    rv = feats["rv_d"]
    target = rv.shift(-1).dropna().rename("target")

    zeroshot_preds = zeroshot_forecast(rv, min_train_size, context_length)

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
