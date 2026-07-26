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
- [x] Clean microstructure noise (bad ticks, crossed quotes, outliers) — IEX
      feed is a single venue so likely cleaner than consolidated tape, but
      still check for gaps/stale bars
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
- [ ] Install `timesfm[torch,xreg]` (2.5, see pinned version in step 0),
      verify `load_pretrained_timesfm()` in src/models/timesfm_finetune.py
      actually loads and runs inference
- [ ] Run TimesFM zero-shot directly on raw RV series (no HAR involved) as a
      sanity baseline — confirms whether foundation model adds value at all
      before investing in fine-tuning
- [ ] Compare zero-shot TimesFM vs HAR-RV vs GARCH on same test window
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
