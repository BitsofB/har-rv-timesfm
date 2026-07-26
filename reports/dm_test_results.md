# Diebold-Mariano pairwise test results (QLIKE loss)

Evaluation window: 2021-08-24 to 2026-07-22 (n=1230).

| model_a   | model_b          |   dm_statistic |     p_value | significant_at_5pct   | favors                    |
|:----------|:-----------------|---------------:|------------:|:----------------------|:--------------------------|
| naive     | har_rv           |        4.36645 | 1.36903e-05 | True                  | har_rv                    |
| naive     | garch11          |       -4.12105 | 4.02415e-05 | True                  | naive                     |
| naive     | timesfm_zeroshot |        4.19981 | 2.86417e-05 | True                  | timesfm_zeroshot          |
| har_rv    | garch11          |      -16.2323  | 0           | True                  | har_rv                    |
| har_rv    | timesfm_zeroshot |       -1.87584 | 0.0609135   | False                 | neither (not significant) |
| garch11   | timesfm_zeroshot |        7.91847 | 5.32907e-15 | True                  | timesfm_zeroshot          |

H0: equal predictive accuracy between model_a and model_b's QLIKE loss series. `favors` is only reported when p_value < 0.05 -- CLAUDE.md section 2.2 requires this before treating a QLIKE ranking as evidence of skill rather than noise.
