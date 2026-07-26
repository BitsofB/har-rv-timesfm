"""
TimesFM 2.5 zero-shot pass on raw daily RV (TODO.md step 4) — NO fine-tuning,
NO HAR involvement. This is the sanity check CLAUDE.md section 3 requires
before any fine-tuning work: does the foundation model get anywhere near
HAR-RV without adaptation?

Reuses the already-processed feature table from `scripts/pull_spy_data.py`
(`data/processed/{symbol}_features.parquet`) so the evaluation window and
walk-forward warm-up exactly match the existing HAR-RV/GARCH/naive numbers
in `reports/baseline_metrics.md` — run `pull_spy_data.py` first if that file
doesn't exist yet.

Requires network access to huggingface.co to download the
`google/timesfm-2.5-200m-pytorch` checkpoint (~200-800MB) on first run. In
the sandboxed dev environment this was authored in, huggingface.co (and
mirrors: hf.co, cdn-lfs.huggingface.co, hf-mirror.com) were blocked by the
outbound proxy allowlist -- only pypi.org/files.pythonhosted.org and
github.com were reachable. This script has NOT been exercised end-to-end
against the real checkpoint as a result; run it somewhere with HF Hub access
(e.g. locally) and treat the first run as the actual verification step.

Run:
    python scripts/run_zeroshot_timesfm.py --symbol SPY
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report
from src.models.timesfm_finetune import (
    FineTuneConfig,
    load_pretrained_timesfm,
    rolling_zero_shot_forecast,
)


def main(
    symbol: str,
    min_train_size: int = 250,
    context_length: int = 512,
    features_path: str | None = None,
    out_path: str = "reports/zeroshot_timesfm_metrics.md",
) -> None:
    features_path = features_path or f"data/processed/{symbol}_features.parquet"
    if not Path(features_path).exists():
        raise FileNotFoundError(
            f"{features_path} not found -- run scripts/pull_spy_data.py first "
            "so this uses the same processed RV series / evaluation window "
            "as the existing HAR-RV/GARCH/naive baselines."
        )

    feats = pd.read_parquet(features_path)
    rv_d = feats["rv_d"].dropna()

    # Same target convention as pull_spy_data.py: target keyed at the "as of"
    # date, value = next day's rv_d. Zero-shot predictions must be aligned
    # the same way (see rolling_zero_shot_forecast docstring).
    target = rv_d.shift(-1).dropna().rename("target")

    print(f"Loading TimesFM 2.5 ({load_pretrained_timesfm.__module__}) ...")
    model = load_pretrained_timesfm(
        max_context=context_length, max_horizon=FineTuneConfig().horizon
    )

    print(
        f"Running walk-forward zero-shot forecast: "
        f"context_length={context_length}, min_train_size={min_train_size}, "
        f"n_steps={len(rv_d) - min_train_size - 1}"
    )
    zero_shot_preds = rolling_zero_shot_forecast(
        rv_d, model, context_length=context_length, min_train_size=min_train_size
    )

    common_idx = target.index.intersection(zero_shot_preds.index)
    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping dates between target and zero-shot predictions -- "
            "check min_train_size / context_length against the length of "
            f"the rv_d series (n={len(rv_d)})."
        )

    metrics = compute_baseline_metrics({
        "timesfm_zeroshot": (target.loc[common_idx], zero_shot_preds.loc[common_idx]),
    })

    note = (
        "\n## Notes\n\n"
        "- `timesfm_zeroshot` is TimesFM 2.5 run with **no fine-tuning** "
        "directly on raw daily RV (`rv_d`), walk-forward, same evaluation "
        "convention as the HAR-RV/GARCH/naive baselines in "
        "`reports/baseline_metrics.md` (prediction keyed at the 'as of' "
        "date, forecasting the next session) -- compare this file's QLIKE "
        "against that one's directly, they are not merged into one table "
        "on purpose (round-tripping the markdown table back into a "
        "DataFrame to append a row is more fragile than just writing a "
        "second small report).\n"
        f"- Context length: {context_length} sessions "
        f"(model ceiling: 16,000). Horizon: 1 (matching HAR).\n"
        f"- Evaluation window: {common_idx.min().date()} to "
        f"{common_idx.max().date()} (n={len(common_idx)} sessions).\n"
        "- Per CLAUDE.md section 3: if this doesn't get near HAR-RV's "
        "QLIKE, that's the expected/documented finding, not a bug -- report "
        "it plainly and do not proceed to fine-tuning compute spend "
        "without first confirming the foundation model has *some* signal "
        "on this series zero-shot.\n"
    )
    write_baseline_report(metrics, out_path=out_path, notes=note)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--min-train-size", type=int, default=250)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()
    main(args.symbol, args.min_train_size, args.context_length)
