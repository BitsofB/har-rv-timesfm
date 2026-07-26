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


def load_pretrained_timesfm(checkpoint: str = CHECKPOINT):
    """
    Load the pretrained TimesFM 2.5 checkpoint for zero-shot inference
    (step 4) or as the base model to wrap with a LoRA adapter (step 6).
    """
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,       # our RV series won't need the 16k max
            max_horizon=1,          # 1-step-ahead, matching HAR
            normalize_inputs=True,
            use_continuous_quantile_head=False,  # True if quantile_loss
            force_flip_invariance=True,
            infer_is_positive=True,  # RV/residuals-of-variance are >= 0-ish;
                                      # reconsider if fine-tuning on raw
                                      # (signed) residuals rather than |resid|
            fix_quantile_crossing=True,
        )
    )
    return model


def zeroshot_forecast(
    rv: pd.Series,
    min_train_size: int = 250,
    context_length: int = 512,
    horizon: int = 1,
    model=None,
) -> pd.Series:
    """
    Walk forward one step at a time: at index i, feed the model
    rv[i-context_length+1 : i+1] (info through day i inclusive) and take
    its 1-step-ahead point forecast as the prediction for day i+1 -- same
    "key i = forecast for i+1" convention already used by rolling_har_rv /
    naive_persistence / rolling_garch_11 in src/models/har_rv.py and
    src/models/baselines.py.

    Unlike those baselines, TimesFM is not refit per step (it's a
    pretrained foundation model run zero-shot) -- only the context window
    advances. Pass an already-loaded model to avoid reloading the
    checkpoint across repeated calls (e.g. one comparison script running
    this alongside the other baselines).
    """
    if model is None:
        model = load_pretrained_timesfm()

    contexts, idxs = [], []
    n = len(rv)
    for i in range(min_train_size, n):
        start = max(0, i - context_length + 1)
        contexts.append(rv.values[start:i + 1].astype("float32"))
        idxs.append(rv.index[i])

    point_forecast, _ = model.forecast(horizon=horizon, inputs=contexts)
    return pd.Series(point_forecast[:, 0], index=idxs, name="timesfm_zeroshot_pred")


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
    """Run inference: context window -> next-step residual forecast."""
    raise NotImplementedError(
        "Fine-tuned residual inference isn't wired up yet -- fine_tune() "
        "above hasn't been implemented either (TODO.md step 6). Use "
        "zeroshot_forecast() for the zero-shot-on-raw-RV pass (TODO.md "
        "step 4); this function is for the post-fine-tuning residual path."
    )


def assemble_hybrid_forecast(
    har_pred: pd.Series,
    timesfm_resid_pred: pd.Series,
) -> pd.Series:
    """final_pred_t = HAR_pred_t + TimesFM_resid_pred_t"""
    assert har_pred.index.equals(timesfm_resid_pred.index)
    return (har_pred + timesfm_resid_pred).rename("hybrid_pred")
