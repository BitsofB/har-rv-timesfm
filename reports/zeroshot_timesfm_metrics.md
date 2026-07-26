# Baseline metrics

| model            |    qlike |         mse |         mae |       r2 |
|:-----------------|---------:|------------:|------------:|---------:|
| timesfm_zeroshot | 0.247968 | 1.68566e-08 | 3.33173e-05 | 0.260173 |

TimesFM 2.5 zero-shot (no fine-tuning) on raw daily RV, no HAR involved. Evaluation window: 2021-08-24 to 2026-07-22 (n=1230). context_length=512, horizon=1. Compare against reports/baseline_metrics.md's naive/har_rv/garch11 rows on the overlapping window before judging zero-shot skill (CLAUDE.md section 2.3).
