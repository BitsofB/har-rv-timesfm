# HAR-RV Baseline Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a complete, leakage-safe out-of-sample baseline comparison (naive persistence, HAR-RV, GARCH(1,1)) for SPY 5-min realized volatility, with results written to `reports/baseline_metrics.md` — the gate CLAUDE.md requires before any TimesFM fine-tuning work starts.

**Architecture:** Pull SPY 5-min bars via the existing Alpaca wrapper, align them to the NYSE regular-session grid (flagging, never interpolating, missing bars), extend the existing RV feature pipeline with signed semi-variance and daily close-to-close returns, add naive-persistence and rolling-GARCH(1,1) baselines alongside the existing `rolling_har_rv`, and aggregate all three into one out-of-sample metrics table (QLIKE/MSE/MAE/R²).

**Tech Stack:** pandas, numpy, scikit-learn (existing), `arch` (GARCH, already in requirements.txt), `pandas-market-calendars` (new — NYSE session calendar), `tabulate` (new — `DataFrame.to_markdown()`), pytest (new — test runner).

## Global Constraints

- Every feature at time `t` uses only information available at `t` — no exceptions (CLAUDE.md §2.1).
- Never fit a model once on the full sample and evaluate in-sample — all baselines walk forward one step at a time (CLAUDE.md §2.1).
- QLIKE is the primary metric; report MSE/MAE/R² alongside it, never rank on MSE alone (CLAUDE.md §2.2).
- Never interpolate over gaps/stale bars — flag loudly and let a threshold decide whether a session is usable (CLAUDE.md §7).
- Alpaca free tier is IEX-only, not consolidated SIP tape — note this in the report (CLAUDE.md §5).
- Target asset: **SPY** (confirmed with user 2026-07-25 — most liquid, spans the 2020 regime shift needed later for the stress-window ablation).
- `TODO.md` is the source of truth for project state — update its checkboxes as sections complete (CLAUDE.md §4).

---

### Task 1: Test infrastructure + new dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/features/test_smoke.py`

**Interfaces:**
- Produces: a working `pytest` invocation (`pytest tests/ -v`) that later tasks' tests plug into.

- [ ] **Step 1: Add new dependencies to `requirements.txt`**

Add these lines under the appropriate existing sections:

```
# Testing
pytest

# Calendar alignment
pandas-market-calendars

# Markdown table export (DataFrame.to_markdown)
tabulate
```

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Expected: installs succeed, no errors.

- [ ] **Step 3: Create test package scaffold**

Create empty `tests/__init__.py`, `tests/features/__init__.py`, `tests/models/__init__.py`, `tests/eval/__init__.py`.

- [ ] **Step 4: Write a smoke test**

```python
# tests/features/test_smoke.py
def test_pytest_runs():
    assert 1 + 1 == 2
```

- [ ] **Step 5: Run it to verify the harness works**

Run: `pytest tests/ -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "test: add pytest harness and new baseline-pipeline dependencies"
```

---

### Task 2: NYSE session-grid alignment and data-quality flagging

**Files:**
- Create: `src/features/data_cleaning.py`
- Test: `tests/features/test_data_cleaning.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `session_grid(start: str, end: str, freq_minutes: int = 5, calendar: str = "NYSE") -> pd.DatetimeIndex`
  - `reindex_to_grid(bars: pd.DataFrame, grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]` — returns `(reindexed_df, missing_report)` where `missing_report` maps `date -> missing_bar_count`.
  - `flag_bad_days(missing_report: dict, expected_bars_per_day: int, max_missing_frac: float = 0.05) -> list` — sorted list of `date` objects to exclude.
  - These three are consumed by Task 6's pipeline script.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_data_cleaning.py
import pandas as pd

from src.features.data_cleaning import flag_bad_days, reindex_to_grid, session_grid


def test_session_grid_two_trading_days():
    # 2023-01-03 and 2023-01-04 are both regular NYSE trading days
    grid = session_grid("2023-01-03", "2023-01-04", freq_minutes=60)
    # 9:30-16:00 ET at 60-min spacing, right-open -> 7 bars/day (9:30..15:30)
    assert len(grid) == 14
    assert grid.date.tolist().count(pd.Timestamp("2023-01-03").date()) == 7


def test_reindex_flags_missing_bars():
    grid = session_grid("2023-01-03", "2023-01-03", freq_minutes=60)
    # Drop the 11:30 bar to simulate a gap
    present = grid.delete(2)
    bars = pd.DataFrame({"close": range(len(present))}, index=present)

    reindexed, missing_report = reindex_to_grid(bars, grid)

    assert len(reindexed) == len(grid)
    day = pd.Timestamp("2023-01-03").date()
    assert missing_report[day] == 1


def test_flag_bad_days_threshold():
    missing_report = {"2023-01-03": 1, "2023-01-04": 5}
    bad = flag_bad_days(missing_report, expected_bars_per_day=7, max_missing_frac=0.3)
    # 1/7 ~= 0.14 (ok), 5/7 ~= 0.71 (bad)
    assert bad == ["2023-01-04"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_data_cleaning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.data_cleaning'`

- [ ] **Step 3: Implement `src/features/data_cleaning.py`**

```python
"""
Session-grid alignment and data-quality flagging for intraday bars.

Never interpolate over gaps here — flag them and let the caller decide
whether a session is usable. Interpolating price data before computing RV
inflates or deflates the variance estimate in ways that are hard to detect
downstream (CLAUDE.md section 7).
"""

import pandas as pd
import pandas_market_calendars as mcal


def session_grid(
    start: str,
    end: str,
    freq_minutes: int = 5,
    calendar: str = "NYSE",
) -> pd.DatetimeIndex:
    """
    Expected regular-trading-hours timestamp grid for every valid session
    between start and end, at freq_minutes spacing. Holidays and half-days
    are handled automatically by the exchange calendar.
    """
    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start, end_date=end)

    day_grids = [
        pd.date_range(
            row.market_open, row.market_close, freq=f"{freq_minutes}min", inclusive="left"
        )
        for row in schedule.itertuples()
    ]
    return pd.DatetimeIndex(pd.concat([pd.Series(g) for g in day_grids])).sort_values()


def reindex_to_grid(bars: pd.DataFrame, grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """
    Reindex intraday bars onto the expected session grid. Missing slots
    become NaN rows -- never filled here.

    Returns (reindexed_df, missing_report) where missing_report maps
    each session date -> count of missing bars that day.
    """
    reindexed = bars.reindex(grid)
    missing_mask = reindexed["close"].isna()
    counts = missing_mask.groupby(missing_mask.index.date).sum().astype(int)
    missing_report = counts[counts > 0].to_dict()
    return reindexed, missing_report


def flag_bad_days(
    missing_report: dict,
    expected_bars_per_day: int,
    max_missing_frac: float = 0.05,
) -> list:
    """
    Dates whose missing-bar fraction exceeds the threshold -- these should
    be excluded (and logged), never interpolated.
    """
    bad = [
        date
        for date, missing in missing_report.items()
        if missing / expected_bars_per_day > max_missing_frac
    ]
    return sorted(bad)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features/test_data_cleaning.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/data_cleaning.py tests/features/test_data_cleaning.py
git commit -m "feat: add NYSE session-grid alignment and gap flagging"
```

---

### Task 3: Signed semi-variance and daily close-to-close returns

**Files:**
- Modify: `src/features/realized_vol.py`
- Test: `tests/features/test_realized_vol.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on the same `prices`/`intraday_returns` conventions already used by `build_feature_table`).
- Produces:
  - `daily_signed_semivariance(intraday_returns: pd.Series) -> pd.DataFrame` with columns `rv_pos`, `rv_neg`.
  - `daily_close_returns(prices: pd.Series) -> pd.Series` — close-to-close daily log returns, consumed by Task 4's `rolling_garch_11`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_realized_vol.py
import numpy as np
import pandas as pd

from src.features.realized_vol import daily_close_returns, daily_signed_semivariance


def _two_day_prices():
    idx = pd.to_datetime([
        "2023-01-03 09:30", "2023-01-03 09:35", "2023-01-03 09:40",
        "2023-01-04 09:30", "2023-01-04 09:35", "2023-01-04 09:40",
    ])
    return pd.Series([100.0, 101.0, 100.0, 100.0, 99.0, 99.5], index=idx)


def test_daily_signed_semivariance_splits_up_and_down():
    prices = _two_day_prices()
    returns = np.log(prices).diff().dropna()
    out = daily_signed_semivariance(returns)

    day1 = pd.Timestamp("2023-01-03").date()
    day2 = pd.Timestamp("2023-01-04").date()

    # day1: up then down -> both rv_pos and rv_neg > 0
    assert out.loc[day1, "rv_pos"] > 0
    assert out.loc[day1, "rv_neg"] > 0
    # day2: down then up -> both rv_pos and rv_neg > 0
    assert out.loc[day2, "rv_pos"] > 0
    assert out.loc[day2, "rv_neg"] > 0
    # rv_pos + rv_neg == plain RV for the day
    rv_day1 = returns.loc["2023-01-03"].pow(2).sum()
    assert np.isclose(out.loc[day1, "rv_pos"] + out.loc[day1, "rv_neg"], rv_day1)


def test_daily_close_returns():
    prices = _two_day_prices()
    out = daily_close_returns(prices)

    assert len(out) == 1  # first day has no prior close to diff against
    expected = np.log(99.5 / 100.0)
    assert np.isclose(out.iloc[0], expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_realized_vol.py -v`
Expected: FAIL with `ImportError: cannot import name 'daily_signed_semivariance'`

- [ ] **Step 3: Implement the additions in `src/features/realized_vol.py`**

Add these two functions after `jump_component` (keep everything else in the file unchanged):

```python
def daily_signed_semivariance(intraday_returns: pd.Series) -> pd.DataFrame:
    """
    Split RV into upside/downside components (Barndorff-Nielsen, Kinnebrock
    & Shephard 2010): RV_t = RV_pos_t + RV_neg_t.
    """
    pos = intraday_returns.clip(lower=0).pow(2)
    neg = intraday_returns.clip(upper=0).pow(2)
    return pd.DataFrame({
        "rv_pos": pos.groupby(intraday_returns.index.date).sum(),
        "rv_neg": neg.groupby(intraday_returns.index.date).sum(),
    })


def daily_close_returns(prices: pd.Series) -> pd.Series:
    """Close-to-close daily log returns (for GARCH, not RV)."""
    daily_close = prices.groupby(prices.index.date).last()
    daily_close.index = pd.to_datetime(daily_close.index)
    return np.log(daily_close).diff().dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features/test_realized_vol.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/realized_vol.py tests/features/test_realized_vol.py
git commit -m "feat: add signed semi-variance and daily close-to-close returns"
```

---

### Task 4: Naive persistence and rolling GARCH(1,1) baselines

**Files:**
- Create: `src/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Consumes: `daily_close_returns` output from Task 3 (for `rolling_garch_11`'s input).
- Produces:
  - `BaselineResult` dataclass with field `predictions: pd.Series`.
  - `naive_persistence(features: pd.DataFrame, min_train_size: int = 250) -> BaselineResult`
  - `rolling_garch_11(daily_returns: pd.Series, min_train_size: int = 250, window: str = "expanding", rolling_size: int = 500) -> BaselineResult`
  - Both consumed by Task 6's pipeline script, alongside the existing `rolling_har_rv` from `src/models/har_rv.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_baselines.py
import numpy as np
import pandas as pd

from src.models.baselines import naive_persistence, rolling_garch_11


def test_naive_persistence_shifts_rv_d():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    features = pd.DataFrame({"rv_d": range(10)}, index=idx)

    result = naive_persistence(features, min_train_size=5)

    assert list(result.predictions.index) == list(idx[5:])
    assert list(result.predictions.values) == list(range(5, 10))


def test_rolling_garch_11_produces_positive_variance_forecasts():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, size=30), index=idx)

    result = rolling_garch_11(returns, min_train_size=20)

    assert len(result.predictions) == 10
    assert (result.predictions > 0).all()
    assert list(result.predictions.index) == list(idx[20:])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.baselines'`

- [ ] **Step 3: Implement `src/models/baselines.py`**

```python
"""
Baseline forecasters compared against HAR-RV: naive persistence and
GARCH(1,1). Both walk forward one step at a time -- never fit on the
full sample and evaluate in-sample (same rule as rolling_har_rv in
src/models/har_rv.py).
"""

from dataclasses import dataclass

import pandas as pd
from arch import arch_model


@dataclass
class BaselineResult:
    predictions: pd.Series


def naive_persistence(features: pd.DataFrame, min_train_size: int = 250) -> BaselineResult:
    """
    RV_{t+1}_pred = RV_t. Uses the same evaluation window (starting at
    min_train_size) as rolling_har_rv so metrics are computed on identical
    dates across models.
    """
    preds = features["rv_d"].iloc[min_train_size:]
    return BaselineResult(preds.rename("naive_pred"))


def rolling_garch_11(
    daily_returns: pd.Series,
    min_train_size: int = 250,
    window: str = "expanding",
    rolling_size: int = 500,
) -> BaselineResult:
    """
    Walk forward one step at a time: fit GARCH(1,1) on returns up to t,
    forecast the 1-step-ahead conditional variance for t+1.
    """
    preds = []
    n = len(daily_returns)

    for i in range(min_train_size, n):
        if window == "expanding":
            train = daily_returns.iloc[:i]
        else:
            train = daily_returns.iloc[max(0, i - rolling_size):i]

        # arch_model wants returns scaled to roughly O(1) percent for
        # numerically stable optimization
        model = arch_model(train * 100, vol="Garch", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        forecast = fit.forecast(horizon=1, reindex=False)
        variance_pct2 = forecast.variance.values[-1, 0]
        preds.append((daily_returns.index[i], variance_pct2 / 100**2))

    pred_series = pd.Series(dict(preds)).rename("garch_pred")
    return BaselineResult(pred_series)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_baselines.py -v`
Expected: 2 passed. (The GARCH test fits 10 tiny models — a few seconds is normal.)

- [ ] **Step 5: Commit**

```bash
git add src/models/baselines.py tests/models/test_baselines.py
git commit -m "feat: add naive-persistence and rolling GARCH(1,1) baselines"
```

---

### Task 5: Baseline metrics aggregation and markdown report

**Files:**
- Create: `src/eval/baseline_report.py`
- Test: `tests/eval/test_baseline_report.py`

**Interfaces:**
- Consumes: `qlike` from `src/models/har_rv.py` (existing).
- Produces:
  - `compute_baseline_metrics(predictions: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame` — indexed by model name, columns `qlike`, `mse`, `mae`, `r2`.
  - `write_baseline_report(metrics: pd.DataFrame, out_path: str = "reports/baseline_metrics.md") -> None`
  - Both consumed by Task 6's pipeline script.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_baseline_report.py
import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report


def test_compute_baseline_metrics_perfect_prediction_is_zero_qlike():
    y = pd.Series([1.0, 2.0, 3.0])
    metrics = compute_baseline_metrics({"perfect": (y, y)})

    assert list(metrics.columns) == ["qlike", "mse", "mae", "r2"]
    assert metrics.loc["perfect", "qlike"] == 0.0
    assert metrics.loc["perfect", "mse"] == 0.0
    assert metrics.loc["perfect", "r2"] == 1.0


def test_write_baseline_report_writes_markdown_table(tmp_path):
    metrics = compute_baseline_metrics({
        "perfect": (pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])),
    })
    out_path = tmp_path / "baseline_metrics.md"

    write_baseline_report(metrics, out_path=str(out_path))

    content = out_path.read_text()
    assert "qlike" in content
    assert "perfect" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_baseline_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.baseline_report'`

- [ ] **Step 3: Implement `src/eval/baseline_report.py`**

```python
"""
Aggregate out-of-sample metrics for all baselines onto one table and write
it to reports/baseline_metrics.md -- required before any fine-tuning work
starts (CLAUDE.md section 3).
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.har_rv import qlike


def compute_baseline_metrics(predictions: dict) -> pd.DataFrame:
    """
    predictions: {model_name: (y_true, y_pred)} -- each pair must already
    be aligned to the same evaluation dates by the caller.
    """
    rows = []
    for name, (y_true, y_pred) in predictions.items():
        rows.append({
            "model": name,
            "qlike": qlike(y_true.values, y_pred.values),
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        })
    return pd.DataFrame(rows).set_index("model")[["qlike", "mse", "mae", "r2"]]


def write_baseline_report(metrics: pd.DataFrame, out_path: str = "reports/baseline_metrics.md") -> None:
    with open(out_path, "w") as f:
        f.write("# Baseline metrics\n\n")
        f.write(metrics.to_markdown())
        f.write("\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_baseline_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/baseline_report.py tests/eval/test_baseline_report.py
git commit -m "feat: add baseline metrics aggregation and markdown report writer"
```

---

### Task 6: End-to-end SPY pipeline script

**Files:**
- Create: `scripts/pull_spy_data.py`
- Modify: `TODO.md` (checkbox updates)

**Interfaces:**
- Consumes: `fetch_intraday_bars` (`src/features/data_alpaca.py`, existing), `session_grid`/`reindex_to_grid`/`flag_bad_days` (Task 2), `build_feature_table`/`daily_close_returns` (existing + Task 3), `rolling_har_rv`/`qlike` (existing), `naive_persistence`/`rolling_garch_11` (Task 4), `compute_baseline_metrics`/`write_baseline_report` (Task 5).
- Produces: `data/raw/SPY_5min.parquet`, `data/processed/SPY_features.parquet`, `reports/baseline_metrics.md`.

This task requires live `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` and network access — it is a manual/integration run, not unit-tested.

- [ ] **Step 1: Implement `scripts/pull_spy_data.py`**

```python
"""
End-to-end pipeline: pull SPY intraday bars, align to the NYSE session
grid, build RV/HAR features, run all three baselines walk-forward, and
write reports/baseline_metrics.md.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment. Run:
    python scripts/pull_spy_data.py
"""

import argparse
from datetime import datetime

import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report
from src.features.data_alpaca import fetch_intraday_bars
from src.features.data_cleaning import flag_bad_days, reindex_to_grid, session_grid
from src.features.realized_vol import build_feature_table, daily_close_returns
from src.models.baselines import naive_persistence, rolling_garch_11
from src.models.har_rv import rolling_har_rv

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

    reindexed.to_parquet(f"data/raw/{symbol}_5min.parquet")

    prices = reindexed["close"].dropna()
    feats = build_feature_table(prices)
    daily_ret = daily_close_returns(prices)
    feats.to_parquet(f"data/processed/{symbol}_features.parquet")

    target = feats["rv_d"].shift(-1).dropna().rename("target")
    features_aligned = feats.loc[target.index]
    daily_ret_aligned = daily_ret.loc[daily_ret.index.intersection(target.index)]

    har = rolling_har_rv(features_aligned[["rv_d", "rv_w", "rv_m"]], target, min_train_size)
    naive = naive_persistence(features_aligned, min_train_size)
    garch = rolling_garch_11(daily_ret_aligned, min_train_size)

    common_idx = (
        har.predictions.index
        .intersection(naive.predictions.index)
        .intersection(garch.predictions.index)
    )
    metrics = compute_baseline_metrics({
        "naive": (target.loc[common_idx], naive.predictions.loc[common_idx]),
        "har_rv": (target.loc[common_idx], har.predictions.loc[common_idx]),
        "garch11": (target.loc[common_idx], garch.predictions.loc[common_idx]),
    })
    write_baseline_report(metrics)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-07-24")
    parser.add_argument("--min-train-size", type=int, default=250)
    args = parser.parse_args()
    main(args.symbol, args.start, args.end, args.min_train_size)
```

- [ ] **Step 2: Confirm Alpaca credentials are set**

Run: `python -c "import os; assert os.environ.get('ALPACA_API_KEY') and os.environ.get('ALPACA_SECRET_KEY'), 'missing keys'"`
Expected: no output (assertion passes). If it fails, stop and get real Alpaca paper-trading keys into `.env` before continuing — this step needs live credentials, which is a user prerequisite, not something to fake or skip.

- [ ] **Step 3: Run the pipeline**

Run: `python scripts/pull_spy_data.py`
Expected: prints any excluded-session warnings, then prints the metrics table with rows `naive`, `har_rv`, `garch11` and columns `qlike`, `mse`, `mae`, `r2`. Runtime will be dominated by the GARCH refit loop (one MLE fit per trading day) — expect several minutes for a multi-year history.

- [ ] **Step 4: Sanity-check the output**

Open `reports/baseline_metrics.md` and confirm:
- All three rows are present with finite, positive `qlike`/`mse`/`mae` values.
- If any excluded-session warning printed, note the count and dates — do not silently ignore it.

- [ ] **Step 5: Update `TODO.md` checkboxes**

Mark these items `[x]` in `TODO.md`:
- Section 0: `requirements.txt` / `pyproject.toml`
- Section 1: all bullets (intraday pull, calendar filtering, cleaning, resampling, storage)
- Section 2: all bullets except the two marked "optional" (VIX-equivalent covariate, HAR-J/HAR-RS stretch variants) and except signed semi-variance which — since it's now implemented — should also be checked off
- Section 3: HAR-RV, GARCH(1,1), naive persistence, rolling re-estimation scheme, and the `reports/baseline_metrics.md` bullet

Leave the "Immediate next actions" section's item 3 checked off by implication once the metrics file exists; add a new top-level note if any bullet doesn't cleanly map (e.g. Alpha Vantage auxiliary data stays unchecked — out of scope for this plan).

- [ ] **Step 6: Commit**

```bash
git add scripts/pull_spy_data.py data/raw/SPY_5min.parquet data/processed/SPY_features.parquet reports/baseline_metrics.md TODO.md
git commit -m "feat: run end-to-end SPY baseline pipeline, record HAR-RV/GARCH/naive metrics"
```

Note: `data/raw/*` and `data/processed/*` are gitignored except `.gitkeep` — if they're still ignored when you reach this step, that's correct per CLAUDE.md (regenerable from the API); drop those two paths from the `git add` and commit only the script, report, and TODO.md update.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers dependency/test setup (TODO §0 partial). Task 2 covers TODO §1 (calendar filtering, gap flagging, resampling — resampling is already effectively handled by requesting 5-min bars directly from Alpaca; Task 2 completes it by reindexing to a complete grid). Task 3 covers the remaining TODO §2 items (signed semi-variance; HAR lags/RV/BV/jump were already implemented). Task 4 covers TODO §3's naive and GARCH(1,1) baselines (HAR-RV OLS and the rolling re-estimation scheme were already implemented). Task 5 + Task 6 cover TODO §3's `reports/baseline_metrics.md` requirement. HAR-J/HAR-RS variants and Alpha Vantage auxiliary data are explicitly marked optional/out-of-scope per TODO.md itself.
- **Out of scope (intentionally):** Section 4+ (TimesFM zero-shot/fine-tuning) — CLAUDE.md §2.3 blocks that work until `reports/baseline_metrics.md` exists, which is exactly what this plan produces. Creating an Alpaca account and setting real API keys are user actions outside what an agent can do; Task 6 Step 2 gates on them explicitly rather than faking success.
- **Type/interface consistency checked:** `BaselineResult.predictions`, `HARRVResult.predictions` (existing) and the plain `pd.Series` from `naive_persistence`/`rolling_garch_11` all share the same index convention (indexed by the date whose *next-day* RV is being predicted), verified against `rolling_har_rv`'s existing indexing so Task 6 can align all three on `common_idx` without silent misalignment.
