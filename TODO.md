# TODO — HAR-RV + TimesFM Hybrid

Legend: [ ] not started · [~] in progress · [x] done

## 0. Project setup
- [x] Scaffold project folder structure
- [x] Decide data source: **Alpaca** (primary — free intraday minute bars via
      IEX feed) + **Alpha Vantage** (secondary — daily-only on free tier, used
      for auxiliary/fundamental data, not core RV)
- [x] Pin TimesFM version: **2.5** (`google/timesfm-2.5-200m-pytorch`, 200M
      params, 16k max context, torch backend, `pip install timesfm[torch,xreg]`)
      — repo: https://github.com/google-research/timesfm
- [ ] Pin remaining environment details (Python version, CUDA/MPS availability
      for local Mac mini M4 fine-tuning — see open question on compute below)
- [x] `requirements.txt` / `pyproject.toml`
- [x] Create Alpaca account, generate API key/secret (paper-trading keys work
      fine for market data access, no funding needed)
- [x] Confirm Alpaca free-tier data feed (IEX, not full SIP consolidated tape)
      and note the coverage/quality difference vs. paid SIP feed

## 1. Data acquisition & cleaning
- [x] Pull intraday minute-bar data for target asset(s) via Alpaca (start with
      1 liquid equity index ETF or large-cap name before multi-asset)
- [x] Handle exchange calendar / trading-hours filtering, holidays, half-days
- [x] Detect gaps/missing bars via the session grid (`flag_bad_days` /
      `reindex_to_grid` in `src/features/data_cleaning.py`) — NOTE: this is
      gap detection only. Bad-tick / crossed-quote / outlier detection is
      NOT implemented anywhere in `src/` and remains a gap; IEX being a
      single venue makes this lower-priority but it's still unaddressed.
- [x] Resample to fixed intraday grid (e.g. 5-min) for RV estimation
- [x] Store cleaned intraday data in `data/raw/`
- [ ] (Optional) Pull Alpha Vantage fundamentals/economic-indicator data as
      auxiliary covariates — separate from the core price pipeline

## 2. Realized volatility feature engineering (`src/features/`)
- [x] Compute daily realized variance (sum of squared 5-min returns)
- [x] Compute realized volatility (sqrt of RV), and log(RV) for modeling
- [x] Compute bipower variation (BV) for jump-robust volatility
- [x] Compute jump component: `J_t = max(RV_t - BV_t, 0)`
- [x] Compute realized signed semi-variance (upside/downside split)
- [x] Compute HAR lag features: RV_t (daily), avg(RV_{t-5:t}) (weekly),
      avg(RV_{t-22:t}) (monthly)
- [ ] Optional covariates: overnight return, leverage term, implied vol
      (VIX-equivalent) if available, volume/turnover
- [x] Save processed feature table to `data/processed/`
- [x] Sanity-check for look-ahead leakage (every feature at time t must only
      use information available up to t)

## 3. Baseline models (`src/models/`)
- [x] Implement HAR-RV (OLS) — this is the anchor/base model
- [ ] Implement HAR-RV variants for comparison: HAR-J (with jumps),
      HAR-RS (with signed semi-variance) — optional stretch
- [x] Implement GARCH(1,1) / EGARCH baseline on daily returns
- [x] Implement naive persistence baseline (RV_{t+1} = RV_t)
- [x] Rolling/expanding-window re-estimation scheme for all baselines
      (avoid full-sample refit leakage)
- [x] Log baseline out-of-sample metrics (QLIKE, MSE, MAE, R²) to
      `reports/baseline_metrics.md`

## 4. TimesFM integration — zero-shot pass first
- [x] Install `timesfm[torch,xreg]` (2.5, see pinned version in step 0),
      verify `load_pretrained_timesfm()` in src/models/timesfm_finetune.py
      actually loads and runs inference — done, see note below
- [x] Run TimesFM zero-shot directly on raw RV series (no HAR involved) as a
      sanity baseline — confirms whether foundation model adds value at all
      before investing in fine-tuning — done, see note below
- [x] Compare zero-shot TimesFM vs HAR-RV vs GARCH on same test window —
      done, see `reports/zeroshot_timesfm_metrics.md` vs
      `reports/baseline_metrics.md`
- [ ] **Compute for fine-tuning is still an open question** (deferred — not
      yet decided whether local Mac mini M4, cloud GPU, or something else;
      revisit before starting step 6 in earnest — zero-shot pass above can
      run on CPU/MPS regardless)

## 5. Residual construction for hybrid
- [ ] Generate HAR-RV rolling out-of-sample predictions across full history
      (not in-sample fitted values — must match the leakage-safe re-estimation
      scheme from step 3)
- [ ] Compute residual series: `resid_t = RV_t - HAR_pred_t`
- [ ] Check residual stationarity / remaining autocorrelation (confirms HAR
      hasn't already captured everything, i.e. there's signal left for the NN)
- [ ] Build supervised windows (context window → next-step residual) for
      TimesFM fine-tuning input format

## 6. Fine-tuning TimesFM (main focus)
- [ ] Define fine-tuning objective: forecast `resid_t+1` (point) — decide
      point vs. quantile/probabilistic loss (TimesFM supports quantile heads)
- [x] Fine-tuning strategy: **LoRA via HuggingFace Transformers + PEFT** —
      this is TimesFM's officially supported/documented fine-tuning path
      (see `timesfm-forecasting/examples/finetuning/` in the repo); adapt
      that reference notebook rather than writing a custom training loop
- [ ] ⚠️ Caution noted: a community benchmark found LoRA-adapted TimesFM had
      no directional skill on equities (different target — price direction,
      not vol residuals — but keep the zero-shot-vs-fine-tuned ablation
      below mandatory, don't assume fine-tuning helps without checking)
- [ ] Set up train/val/test split with strict time-ordering (no shuffling
      across the boundary) and an embargo gap to prevent leakage through
      overlapping windows
- [ ] Decide context length and forecast horizon (align with HAR's own
      1-day-ahead horizon initially, extend to multi-day later)
- [ ] Add auxiliary covariates as side-channel inputs if supported by the
      TimesFM covariate/regressor interface (jump component, leverage term)
- [ ] Set up training loop: batch size, learning rate schedule, early
      stopping on validation QLIKE/MSE
- [ ] Regularization: dropout, weight decay — watch for overfitting given
      residuals are a low-signal target
- [ ] Track experiments (config + metrics) in `configs/` + simple experiment
      log — even a CSV/markdown table is fine for this project size
- [ ] Save best checkpoint(s) to `checkpoints/`
- [ ] Ablation: fine-tuned TimesFM-on-residuals vs. fine-tuned TimesFM-on-raw-RV
      (confirms residual-correction framing is actually better than direct)

## 7. Hybrid assembly & evaluation (`src/eval/`)
- [ ] Combine: `final_pred_t = HAR_pred_t + TimesFM_resid_pred_t`
- [ ] Backtest hybrid vs. all baselines from step 3 on held-out period,
      including at least one high-volatility stress window
      (e.g. 2020 COVID crash or comparable regime shift in your asset)
- [ ] Metrics: QLIKE (standard for vol forecasting), MSE, MAE, R²,
      Diebold-Mariano test for statistical significance of improvement
      over HAR-RV alone
- [ ] Check calibration if using quantile/probabilistic outputs
      (coverage rates at each quantile)
- [ ] Sensitivity check: does the hybrid's edge come mainly from stress
      periods or is it consistent in calm regimes too?

## 8. Forecast combination alternative (optional, cheap to test)
- [ ] As a comparison point: simple/weighted ensemble of independently
      trained HAR-RV and TimesFM (no residual framing) — cheaper to build,
      useful as another baseline

## 9. Write-up
- [ ] Summarize architecture, data, and results in `reports/`
- [ ] Document known limitations (data window, single-asset scope,
      generalization risk) and next steps for multi-asset extension

---

## Immediate next actions
1. Pick target asset + confirm intraday data source
2. Build RV + HAR feature pipeline (steps 1–2)
3. Get HAR-RV baseline numbers before touching TimesFM at all — **done**,
   see `reports/baseline_metrics.md` (SPY, `scripts/pull_spy_data.py`)

## Note (2026-07-25, Task 6 end-to-end run)
- Alpha Vantage auxiliary-data pull (§0/§1 optional bullets) remains
  unchecked — out of scope for the baseline pipeline plan.
- HAR-J / HAR-RS stretch variants (§3) remain unchecked — explicitly optional.
- **Data coverage gap found and worth tracking**: fetching SPY 5-min bars for
  2018-01-01..2026-07-24 from Alpaca's free/IEX feed returned unusable
  (>5% missing) sessions for effectively the entire 2018-01-02 through
  2020-07-24 window (649 sessions excluded by `flag_bad_days`), so the
  baseline numbers in `reports/baseline_metrics.md` are computed on
  ~2020-07-27 onward only, not the full requested 8+ year history. Revisit
  if a longer backtest window (e.g. for the 2020 COVID stress-window
  ablation in CLAUDE.md §6) is needed — may require a different data
  source/tier for the pre-2020-07 period.

## Note (2026-07-26, RV-target fix + TimesFM zero-shot pass)
- **RV target was contaminated by the overnight (prev-close-to-open) return**:
  `intraday_log_returns`/`daily_bipower_variation` diffed/shifted across day
  boundaries, inflating reported RV by ~1.6x on average (up to >50% of a
  day's "realized volatility" on some sessions). Fixed to diff/shift within
  each session only (`src/features/realized_vol.py`); `reports/baseline_metrics.md`
  regenerated with the corrected target. Not a leakage bug (the gap is known
  at time t), but the target definition was wrong — every earlier number
  produced before this fix is stale.
- Current baseline numbers (SPY, corrected target, n=1230,
  2021-08-24..2026-07-22): QLIKE — naive 0.304, **har_rv 0.227 (best)**,
  garch11 0.399. Note MSE/MAE/R² disagree with the QLIKE ranking (R² is
  negative for both naive and har_rv).
- **TimesFM 2.5 zero-shot** (`load_pretrained_timesfm()` implemented,
  `scripts/run_zeroshot_timesfm.py`, same evaluation window): QLIKE 0.248
  (close to HAR-RV's 0.227), and best of all four models on MSE
  (1.69e-08), MAE (0.000033), and R² (0.260 — the only meaningfully
  positive R² among naive/har_rv/garch11/timesfm_zeroshot). This clears
  the CLAUDE.md §2.3/§4 bar ("if zero-shot can't get near HAR-RV, say so
  before fine-tuning") — zero-shot is competitive, so fine-tuning work
  (§5/§6) is justified to attempt next, pending the compute decision
  still open in §4/§8.
- **Diebold-Mariano test implemented** (`src/eval/diebold_mariano.py`,
  `scripts/run_dm_tests.py`, see `reports/dm_test_results.md`) — the QLIKE
  claims above are now checked, not just point estimates:
  - HAR-RV beats naive and GARCH(1,1) on QLIKE, and both differences are
    statistically significant (p=1.4e-05, p≈0) — the "HAR-RV wins" claim
    holds up.
  - TimesFM zero-shot beats naive and GARCH(1,1) on QLIKE, both
    significant (p=2.8e-05, p=5.1e-15).
  - **HAR-RV vs. TimesFM zero-shot is NOT statistically significant**
    (p=0.061 at the 5% threshold) — despite TimesFM's better point-estimate
    R², the two models are statistically indistinguishable on QLIKE. Any
    claim that either "wins" between these two specifically is unsupported
    by this test; report both as tied on the primary metric.
