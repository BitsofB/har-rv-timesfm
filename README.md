# HAR-RV + TimesFM Hybrid Volatility Forecasting

End-to-end pipeline for forecasting realized volatility using a HAR-RV
residual-correction hybrid: HAR-RV supplies the persistence structure,
a fine-tuned TimesFM model corrects the residuals and captures
nonlinear / regime effects.

## Approach

1. Compute realized volatility (RV) and related realized measures
   (bipower variation, jumps, signed semi-variance) from intraday data.
2. Fit HAR-RV (daily / weekly / monthly lags) as the linear anchor model.
3. Compute HAR-RV's residuals: `resid_t = RV_t - HAR_RV_pred_t`.
4. Fine-tune TimesFM to forecast `resid_t` (optionally with auxiliary
   covariates: jump component, leverage term, overnight return, etc.)
5. Final forecast = HAR-RV prediction + TimesFM residual correction.
6. Benchmark against: HAR-RV alone, GARCH(1,1), zero-shot TimesFM on
   raw RV, and a naive persistence baseline.

## Data sources

- **Alpaca** (primary) — free tier gives real historical intraday minute bars
  (IEX feed) with generous rate limits. Used to build true realized
  volatility (RV) from intraday returns.
- **Alpha Vantage** (secondary) — free tier is daily-bars-only for price data
  (intraday is Premium-gated), so it's used for auxiliary/fundamental data
  (e.g. economic indicators, earnings, sentiment) rather than the core RV
  pipeline. Kept optional; not required for the main HAR-RV/TimesFM path.

## Structure

```
data/
  raw/            # untouched intraday/price data
  processed/      # RV measures, HAR features, train/val/test splits
src/
  features/       # RV construction, HAR feature engineering
  models/         # HAR-RV, GARCH baseline, TimesFM fine-tuning wrapper
  eval/           # backtests, metrics, forecast combination
configs/          # yaml/json configs per experiment
checkpoints/       # fine-tuned model weights
notebooks/        # exploration, diagnostics
reports/          # write-ups, benchmark tables, plots
```

## Status

See (`TODO.md`)[~/TODO.md] for the working task list (current focus: fine-tuning stage).
