"""
TimesFM 2.5 fine-tuning wrapper for HAR-RV residual correction.

Pinned version: TimesFM 2.5 (200M params, google/timesfm-2.5-200m-pytorch),
installed via `pip install timesfm[torch,xreg]`.
Repo: https://github.com/google-research/timesfm
Official fine-tuning path: LoRA via HuggingFace Transformers + PEFT, example
notebook at timesfm-forecasting/examples/finetuning/ in the repo — this is
the reference implementation to adapt for loading/fine_tune below.

Caution: a community benchmark ("Base-rate-honest benchmark showing
LoRA-adapted TimesFM has no directional skill on equities") found LoRA
fine-tuning added no directional edge for equity price forecasting. That's
a different target (price direction) than ours (HAR-RV residuals), but it's
a reason to keep the zero-shot-vs-fine-tuned ablation (TODO.md step 6) rather
than assume fine-tuning helps — validate it, don't take it on faith.

Target: residual_t = RV_t - HAR_pred_t  (see src/models/har_rv.py)
Input:  context window of past residuals (+ optional covariates via XReg)
Output: point or quantile forecast of residual_{t+1}
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT = 16_000   # TimesFM 2.5 ceiling; our context_length must be <= this
MAX_HORIZON = 256       # per-call horizon ceiling; loop for longer horizons


@dataclass
class FineTuneConfig:
    context_length: int = 512          # well under the 16k ceiling; our daily
                                        # RV series won't need anywhere near max
    horizon: int = 1                   # start 1-step-ahead, matching HAR
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 5
    strategy: str = "lora"             # "lora" (official path, PEFT) | "full"
    lora_rank: int = 8                 # only used when strategy == "lora"
    use_covariates: bool = False       # jump/leverage side-channels via XReg
    quantile_loss: bool = False        # point vs probabilistic (quantile head)


def build_windows(
    series: pd.Series,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a 1D series into (context, target) supervised windows.
    Strictly time-ordered — caller must split train/val/test by date
    BEFORE calling this, not after, to avoid leakage across the boundary
    via overlapping windows.
    """
    values = series.values
    X, y = [], []
    for i in range(context_length, len(values) - horizon + 1):
        X.append(values[i - context_length:i])
        y.append(values[i:i + horizon])
    return np.array(X), np.array(y)


def load_pretrained_timesfm(
    checkpoint: str = CHECKPOINT,
    max_context: int = 1024,
    max_horizon: int = 1,
):
    """
    Load the pretrained TimesFM 2.5 checkpoint for zero-shot inference
    (step 4) or as the base model to wrap with a LoRA adapter (step 6).

    Verified against `timesfm==2.0.2` (`timesfm[torch]`, no xreg/flax extra)
    installed in a CPU-only environment on 2026-07-26: `from_pretrained` /
    `compile` signatures match the calls below, and `import timesfm` +
    `TimesFM_2p5_200M_torch` are importable.

    NOTE: actually *running* this (downloading the checkpoint) requires
    network access to huggingface.co, which was blocked by this sandbox's
    proxy allowlist (`hf.co`, `huggingface.co`, `cdn-lfs.huggingface.co`,
    `hf-mirror.com` all returned 403 from the proxy — only pypi/github were
    reachable). This function is verified to be *correct* but has not been
    exercised end-to-end in this environment. Run it somewhere with HF Hub
    access (e.g. locally) to confirm the checkpoint actually loads.
    """
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,   # our RV series won't need the 16k ceiling
            max_horizon=max_horizon,   # 1-step-ahead, matching HAR
            normalize_inputs=True,
            use_continuous_quantile_head=False,  # True if quantile_loss
            force_flip_invariance=True,
            infer_is_positive=True,    # RV is >= 0; reconsider if fine-tuning
                                        # on signed residuals rather than raw RV
            fix_quantile_crossing=True,
        )
    )
    return model


def fine_tune(
    model,
    train_windows: tuple[np.ndarray, np.ndarray],
    val_windows: tuple[np.ndarray, np.ndarray],
    config: FineTuneConfig,
    covariates: Optional[dict] = None,
):
    """
    LoRA fine-tuning via HuggingFace Transformers + PEFT — this is the
    officially supported path (see module docstring). Adapt the reference
    notebook at timesfm-forecasting/examples/finetuning/ in
    https://github.com/google-research/timesfm rather than writing a custom
    training loop from scratch; the notebook handles the PEFT wrapping,
    optimizer setup, and checkpoint saving in a way that matches how the
    base model expects inputs.

    Steps to fill in (TODO.md step 6):
      1. Wrap `model` with a LoRA config (rank=config.lora_rank) via PEFT's
         get_peft_model()
      2. Build a Dataset/DataLoader from train_windows / val_windows
      3. Standard train loop: optimizer (AdamW), config.learning_rate,
         early stopping on val QLIKE/MSE (config.early_stopping_patience)
      4. If config.use_covariates, pass jump/leverage series through XReg
         per the Oct 2025 covariate-support update in the repo
      5. Save adapter weights to checkpoints/
    """
    raise NotImplementedError(
        "Adapt timesfm-forecasting/examples/finetuning/ from the TimesFM "
        "repo for our residual-forecasting target (TODO.md step 6)."
    )


def forecast_residuals(model, context: np.ndarray) -> np.ndarray:
    """
    Run inference: context window -> next-step point forecast.

    `context` is a single 1D array (one series). TimesFM's `forecast()`
    takes a *list* of series and returns `(point_forecast, quantile_forecast)`
    arrays of shape (n_series, horizon[, n_quantiles]); we pass a
    single-series batch and pull horizon step 0 of the point forecast.
    """
    point_forecast, _quantile_forecast = model.forecast(
        horizon=1, inputs=[np.asarray(context, dtype=np.float32)]
    )
    return point_forecast[0]


def rolling_zero_shot_forecast(
    series: pd.Series,
    model,
    context_length: int = 512,
    min_train_size: int = 250,
) -> pd.Series:
    """
    Walk-forward zero-shot forecast (TODO.md step 4) — NO fine-tuning here.

    `series` is the raw (unshifted) daily RV series, e.g. `feats["rv_d"]`,
    where `series.iloc[i]` is the value *known as of* date `i`. To match
    `rolling_har_rv` / `naive_persistence`'s indexing convention exactly
    (see `pull_spy_data.py`: `target = feats["rv_d"].shift(-1)`,
    `features_aligned = feats.loc[target.index]`, and predictions keyed by
    the "as of" date, not the target date): at step `i` the context includes
    `series.iloc[start:i+1]` (through day i INCLUSIVE), and the resulting
    forecast is for day i+1 but is keyed at `series.index[i]`. This makes
    the output directly comparable to `har.predictions` / `naive.predictions`
    / `garch.predictions` via the same `target.loc[common_idx]` alignment
    pattern used in `pull_spy_data.py` — do not change this indexing without
    re-checking that alignment, since an off-by-one here would either leak
    day i+1 into its own forecast or silently drop a day of context.

    `min_train_size` matches the HAR-RV baseline's warm-up so the same
    evaluation window can be used across models. Loop stops at `n - 1`
    because day `n-1` has no next-day target to forecast.
    """
    values = series.values.astype(np.float32)
    n = len(values)
    preds = []
    for i in range(min_train_size, n - 1):
        start = max(0, i - context_length + 1)
        context = values[start:i + 1]
        pred = forecast_residuals(model, context)[0]
        preds.append((series.index[i], float(pred)))
    return pd.Series(dict(preds)).rename("timesfm_zeroshot_pred")


def assemble_hybrid_forecast(
    har_pred: pd.Series,
    timesfm_resid_pred: pd.Series,
) -> pd.Series:
    """final_pred_t = HAR_pred_t + TimesFM_resid_pred_t"""
    assert har_pred.index.equals(timesfm_resid_pred.index)
    return (har_pred + timesfm_resid_pred).rename("hybrid_pred")
