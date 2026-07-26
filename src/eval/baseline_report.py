"""
Aggregate out-of-sample metrics for all baselines onto one table and write
it to reports/baseline_metrics.md -- required before any fine-tuning work
starts (CLAUDE.md section 3).
"""

import os

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


def write_baseline_report(
    metrics: pd.DataFrame,
    out_path: str = "reports/baseline_metrics.md",
    notes: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Baseline metrics\n\n")
        f.write(metrics.to_markdown())
        f.write("\n")
        if notes:
            f.write("\n")
            f.write(notes)
            f.write("\n")
