"""
Residual construction for the HAR-RV + TimesFM hybrid (TODO.md step 5):
generate HAR-RV's out-of-sample rolling predictions, compute the residual
series resid_t = RV_t - HAR_pred_t, check what structure (if any) is left
for TimesFM to correct, and confirm the supervised-window format works.

Reads the cached data/processed/{symbol}_features.parquet produced by
scripts/pull_spy_data.py -- does not re-pull from Alpaca.

Run:
    python scripts/build_residuals.py --symbol SPY
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from src.models.har_rv import rolling_har_rv
from src.models.timesfm_finetune import build_windows


def load_features(symbol: str) -> pd.DataFrame:
    path = Path(f"data/processed/{symbol}_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/pull_spy_data.py "
            f"--symbol {symbol}` first to produce it."
        )
    return pd.read_parquet(path)


def compute_residuals(feats: pd.DataFrame, min_train_size: int) -> pd.Series:
    """
    resid_t = RV_t - HAR_pred_t, using HAR-RV's own out-of-sample rolling
    predictions (never in-sample fitted values -- CLAUDE.md section 2.1).
    rolling_har_rv already computes this as HARRVResult.residuals; this
    wrapper just names the walk-forward call for this script's purpose.
    """
    target = feats["rv_d"].shift(-1).dropna().rename("target")
    features_aligned = feats.loc[target.index]
    har = rolling_har_rv(features_aligned[["rv_d", "rv_w", "rv_m"]], target, min_train_size)
    return har.residuals.rename("residual")


def diagnose_residuals(residual: pd.Series, lags: int = 10) -> str:
    """
    ADF test (is the residual stationary?) and Ljung-Box (is there
    autocorrelation left for a model to exploit?). If HAR-RV had already
    captured everything, we'd expect a stationary, white-noise residual --
    Ljung-Box significance here is what justifies spending compute on
    fine-tuning at all (there's real structure left to correct).
    """
    adf_stat, adf_pvalue, *_ = adfuller(residual)
    lb = acorr_ljungbox(residual, lags=[lags], return_df=True)
    lb_stat = lb["lb_stat"].iloc[0]
    lb_pvalue = lb["lb_pvalue"].iloc[0]

    return (
        f"# Residual diagnostics\n\n"
        f"n = {len(residual)}, evaluation window "
        f"{residual.index.min().date()} to {residual.index.max().date()}\n\n"
        f"- **ADF test** (H0: unit root / non-stationary): "
        f"statistic={adf_stat:.4f}, p-value={adf_pvalue:.4g} "
        f"-- {'stationary' if adf_pvalue < 0.05 else 'NOT stationary'} at 5%.\n"
        f"- **Ljung-Box test** (H0: no autocorrelation up to lag {lags}): "
        f"statistic={lb_stat:.4f}, p-value={lb_pvalue:.4g} -- "
        f"{'residual has significant remaining autocorrelation (real signal left for TimesFM)' if lb_pvalue < 0.05 else 'no significant autocorrelation left (HAR-RV may already capture the exploitable structure)'} "
        f"at 5%.\n"
    )


def main(symbol: str, min_train_size: int = 250, context_length: int = 60) -> None:
    feats = load_features(symbol)
    residual = compute_residuals(feats, min_train_size)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    residual.to_frame().to_parquet(f"data/processed/{symbol}_residuals.parquet")

    diagnostics = diagnose_residuals(residual)
    print(diagnostics)
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/residual_diagnostics.md", "w") as f:
        f.write(diagnostics)

    # Smoke-test the supervised-window format build_windows() will feed
    # TimesFM fine-tuning (TODO.md step 6) -- on the full series here only
    # to confirm shapes are sane; the real train/val/test split with an
    # embargo gap happens in step 6, not before this window call, per
    # build_windows()'s own leakage warning.
    X, y = build_windows(residual, context_length=context_length, horizon=1)
    print(f"build_windows(context_length={context_length}, horizon=1) -> "
          f"X.shape={X.shape}, y.shape={y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--min-train-size", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=60)
    args = parser.parse_args()
    main(args.symbol, args.min_train_size, args.context_length)
