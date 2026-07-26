"""
Diebold-Mariano test for equal predictive accuracy between two forecasts'
loss series. CLAUDE.md section 2.2 requires this before any "model A beats
model B" claim is treated as skill rather than noise -- a lower average
loss on its own is not evidence.
"""

import numpy as np
from scipy import stats


def diebold_mariano_test(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """
    Two-sided DM test on d_t = loss_a_t - loss_b_t (Diebold & Mariano 1995).

    H0: E[d_t] = 0 (equal accuracy). A negative statistic means model A had
    the lower average loss; positive means model B did -- the p-value is
    what determines whether that difference is distinguishable from noise.

    loss_a and loss_b must be the same length and already aligned to the
    same evaluation dates (e.g. via qlike_loss(y_true, y_pred) from
    src/models/har_rv.py on a common index across both models).

    Long-run variance uses a Newey-West estimator truncated at h-1 lags,
    matching the forecast horizon (h=1 for the 1-day-ahead baselines in
    this project, so no autocorrelation adjustment is added by default).
    Applies the Harvey-Leybourne-Newbold (1997) small-sample correction,
    which pairs a shrunk statistic with the t(n-1) reference distribution
    used below -- using t(n-1) alone, without this correction, understates
    significance relative to the test it's meant to approximate.

    Returns (dm_statistic, p_value).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_bar = d.mean()

    gamma0 = np.sum((d - d_bar) ** 2) / n
    var_d = gamma0
    for lag in range(1, h):
        # divide by n (not n - lag) -- matches gamma0's normalization, per
        # the standard Newey-West/DM long-run-variance estimator
        gamma_k = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        var_d += 2 * (1 - lag / h) * gamma_k

    var_d_bar = var_d / n
    dm_stat = d_bar / np.sqrt(var_d_bar)

    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= hln_correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)
