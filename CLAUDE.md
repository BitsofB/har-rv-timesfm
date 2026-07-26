# CLAUDE.md — HAR-RV + TimesFM Hybrid Volatility Forecasting

Instructions for an agent working in this repo. Read this before touching code.

---

## 1. What this project is

Forecast **1-day-ahead realized volatility** of a single liquid US equity /
ETF using a **residual-correction hybrid**:

```
final_pred_t = HAR_RV_pred_t  +  TimesFM_residual_pred_t
```

HAR-RV (Corsi 2009) is the linear anchor — it captures volatility persistence
across daily/weekly/monthly horizons with three regressors. TimesFM 2.5 is
fine-tuned (LoRA) to predict **only what HAR-RV gets wrong**, i.e. the
nonlinear / regime-shift structure the linear model structurally cannot reach.

**Why residual-correction and not "feed everything into the deep model":**
HAR-RV is a notoriously hard benchmark to beat, and neural nets trained on raw
RV frequently *underperform* it out-of-sample. Constraining the neural net to
the residual shrinks its target variance and reduces overfitting risk. This
framing is a deliberate decision, not an accident — do not silently replace it
with direct RV forecasting. (Testing it as an *ablation* is required; see §6.)

---

## 2. Non-negotiable methodology rules

These are the rules that make or break this project's validity. Violating any
of them silently invalidates every number downstream.

### 2.1 Leakage is the primary failure mode
- **Every feature at time `t` must use only information available at `t`.**
  No exceptions, no "it's just a small refit."
- **Never fit a model once on the full sample and evaluate in-sample.**
  All baselines use rolling or expanding-window walk-forward re-estimation
  (see `rolling_har_rv()` in `src/models/har_rv.py` for the reference pattern).
- **HAR residuals must come from out-of-sample HAR predictions**, not
  in-sample fitted values. If you use fitted residuals, the neural net is
  learning to correct errors that were computed with future information.
- **Train/val/test splits are strictly time-ordered.** Never shuffle. Add an
  **embargo gap** between splits — overlapping context windows leak across the
  boundary even when the split date looks clean.

### 2.2 Metrics
- **QLIKE is the primary metric** for volatility forecasting, not MSE. QLIKE
  penalizes under-prediction of variance asymmetrically, which matters for
  risk applications. `qlike()` is implemented in `src/models/har_rv.py`.
- Report MSE / MAE / R² alongside it, but do not rank models on MSE alone.
- Claimed improvements over HAR-RV must be checked with a **Diebold-Mariano
  test**. A lower average loss is not evidence of skill on its own.

### 2.3 Baselines come first
Do **not** start fine-tuning before HAR-RV baseline numbers exist in
`reports/baseline_metrics.md`. The whole point of the design is that HAR-RV is
the thing to beat. Without its number, "the hybrid works" is unfalsifiable.

Required baselines before any fine-tuning work:
1. Naive persistence (`RV_{t+1} = RV_t`)
2. HAR-RV
3. GARCH(1,1)
4. TimesFM **zero-shot** on raw RV (no HAR involved)

If zero-shot TimesFM can't get near HAR-RV, say so plainly and flag it before
spending compute on fine-tuning.

---

## 3. Environment & setup

```bash
pip install -r requirements.txt
```

**Required environment variables** (never hardcode, never commit):
```
ALPACA_API_KEY
ALPACA_SECRET_KEY
ALPHAVANTAGE_API_KEY     # only if pulling auxiliary/fundamental data
```

Key pinned versions:
- **TimesFM 2.5** — `google/timesfm-2.5-200m-pytorch`, 200M params, 16k max
  context, torch backend. Install: `timesfm[torch,xreg]`.
- Fine-tuning is **LoRA via HuggingFace Transformers + PEFT** — the officially
  documented path. Adapt `timesfm-forecasting/examples/finetuning/` from
  https://github.com/google-research/timesfm rather than writing a custom
  training loop from scratch.

---

## 4. Repo layout

```
data/raw/         # untouched intraday bars from Alpaca (parquet)
data/processed/   # RV measures, HAR features, splits
src/features/
  data_alpaca.py         # PRIMARY data source — intraday minute bars
  data_alphavantage.py   # SECONDARY — auxiliary/fundamental data only
  realized_vol.py        # RV, BV, jumps, HAR lag features
src/models/
  har_rv.py              # HAR-RV baseline (working) + qlike()
  timesfm_finetune.py    # TimesFM wrapper (scaffold, guarded)
src/eval/                # backtests, metrics, forecast combination
configs/          # one config per experiment
checkpoints/      # LoRA adapter weights
reports/          # benchmark tables, write-ups
TODO.md           # the working task list — source of truth for progress
```

**`TODO.md` is the source of truth for project state.** Update checkboxes as
work completes. If a decision gets made in conversation, record it there — a
decision that only exists in a chat log doesn't exist.

---

## 5. Data source constraints (learned the hard way)

- **Alpaca is primary.** Free tier gives genuine historical intraday minute
  bars via the **IEX feed** — a single venue, *not* the full SIP consolidated
  tape. This matters: IEX-only prices and volumes differ from consolidated
  data, so absolute RV magnitudes are not directly comparable to published
  academic figures built on TAQ/SIP data. Note this in any write-up.
- **Alpha Vantage is secondary and heavily rate-limited** (~25 requests/day,
  5/min on the free tier). Intraday price data is **Premium-gated**, which is
  why it isn't the primary source. Use it only for auxiliary series
  (treasury yield, CPI, earnings, sentiment). It sleeps 12s between calls by
  design — do not "optimize" that away or remove the rate-limit guard.
- Alpha Vantage returns **HTTP 200 even when rate-limited**, with the error
  buried in the JSON body. The existing code checks for this; keep that check.

---

## 6. Required ablations (do not skip)

A community benchmark found **LoRA-adapted TimesFM had no directional skill on
equities**. That's a different target than ours (price direction vs. volatility
residuals), so it isn't disqualifying — but it means fine-tuning benefit must
be *demonstrated*, never assumed.

Mandatory comparisons before claiming the hybrid works:
1. Fine-tuned TimesFM **on residuals** vs. fine-tuned TimesFM **on raw RV** —
   validates the residual-correction framing itself.
2. Fine-tuned vs. **zero-shot** TimesFM — validates that fine-tuning helped.
3. Hybrid vs. **HAR-RV alone** — validates the whole project premise.
4. Performance split by regime: does the edge come only from stress periods
   (2020-style crashes) or does it hold in calm markets too? Report both.

If an ablation shows the added complexity doesn't help, **report that as the
finding**. A clean negative result is a legitimate and useful outcome here.
Do not tune the evaluation until the answer comes out favorable.

---

## 7. Working style in this repo

- **Small, verifiable steps.** Get one symbol's intraday pull working before
  building multi-asset. Get HAR-RV working before touching TimesFM.
- **Don't leave silent stubs.** `timesfm_finetune.py` raises
  `NotImplementedError` with a pointer to the relevant TODO step. Keep that
  pattern — a guard that fails loudly beats a function that returns garbage.
- **Log experiments.** Config + metrics per run, in `configs/` and a table in
  `reports/`. A markdown table or CSV is sufficient at this scale; don't
  introduce MLflow/W&B unless asked.
- **Prefer boring, inspectable code.** This is a research pipeline where
  correctness of the *methodology* dominates. Clever abstractions that obscure
  the time-ordering of operations are a net negative here.
- **Flag data problems loudly.** Gaps, stale bars, suspicious zero-volume
  periods, or unexpected halts should be surfaced and discussed, not silently
  interpolated over. Interpolating price data before computing RV inflates or
  deflates the variance estimate in ways that are hard to detect downstream.

---

## 8. Open questions (unresolved — ask, don't assume)

- **Fine-tuning compute is undecided.** Local Mac mini M4 (MPS), cloud GPU, or
  something else — deferred. The zero-shot pass runs fine on CPU/MPS, so this
  doesn't block steps 0–5. Raise it before starting step 6 in earnest.
- **Target asset not yet chosen.** Needs to be liquid enough that 5-min bars
  aren't dominated by microstructure noise. Should ideally span at least one
  volatility regime shift for the stress-window evaluation.
- **Point vs. quantile forecasting.** TimesFM 2.5 has an optional quantile
  head. Quantile output enables calibration checks but complicates the
  residual-reconstruction step. Not yet decided.

---

## 9. Quick orientation for a fresh session

1. Read `TODO.md` — it reflects current state and what's next.
2. Check whether `reports/baseline_metrics.md` exists. If not, baselines
   aren't done, and fine-tuning work is premature regardless of what's asked.
3. Confirm env vars are set before attempting any data pull.
4. When in doubt about ordering: **data → features → baselines → zero-shot →
   residuals → fine-tune → evaluate.** Don't jump ahead.
