# Baseline metrics

| model   |    qlike |         mse |         mae |          r2 |
|:--------|---------:|------------:|------------:|------------:|
| naive   | 0.30363  | 2.28167e-08 | 3.96175e-05 | -0.00140927 |
| har_rv  | 0.226904 | 2.59612e-08 | 3.98317e-05 | -0.139423   |
| garch11 | 0.399032 | 2.23579e-08 | 6.85377e-05 |  0.0187266  |

## Notes

- **Data feed**: Alpaca free tier, IEX feed (a single venue), not the full consolidated SIP tape. Absolute RV magnitudes here are not directly comparable to published academic figures built on TAQ/SIP data.
- **Evaluation window**: 2021-08-24 to 2026-07-22 (n=1230 sessions).
- **Excluded sessions**: 649 session(s) dropped by `flag_bad_days` for excessive missing intraday bars, spanning 2018-01-02 to 2026-07-24.
- **GARCH(1,1) caveat**: `garch11` forecasts the conditional variance of close-to-close *daily* returns, not intraday realized variance. It's included as a standard volatility baseline for comparison, not because its target matches `naive`/`har_rv` exactly -- treat cross-model comparisons involving `garch11` with that in mind.

