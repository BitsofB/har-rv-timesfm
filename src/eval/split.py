"""
Train/val/test split with an embargo gap for TimesFM fine-tuning
(TODO.md step 6). CLAUDE.md section 2.1: splits must be strictly
time-ordered, never shuffled, with an embargo gap between them --
overlapping context windows leak across the boundary even when the
split date looks clean.

Embargo size must be >= context_length. This module's own contract is
split-first, window-second: call build_windows() separately on each
returned slice. Under that usage build_windows() indexes locally within
its own split array, so it cannot read another split's values regardless
of embargo size -- the embargo instead enforces the discipline itself
(reject an embargo too small to ever be safe) and leaves room for a
future design that windows across split boundaries directly (e.g. using
trailing train sessions as burn-in context for the first val/test
windows via global-position indexing) without silently leaking. If that
global-indexing scheme is ever built, a val/test window's context reaches
back `context_length - 1` sessions from its own start position, so an
embargo shorter than context_length would then let that context read
into the train period.
"""

import pandas as pd


def split_with_embargo(
    residual: pd.Series,
    context_length: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    embargo: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split a date-sorted series into train/val/test with an embargo gap
    of `embargo` sessions (default: context_length) dropped immediately
    after each cut point, before any windowing happens.

    Split the raw series by date first; call build_windows() separately
    on each returned slice afterward. Never build windows across the
    full series and partition by window index -- adjacent windows share
    context_length - 1 values, so slicing post-hoc leaves windows whose
    context reads across the split boundary.
    """
    if embargo is None:
        embargo = context_length
    if embargo < context_length:
        raise ValueError(
            f"embargo ({embargo}) must be >= context_length ({context_length}) "
            "to prevent val/test windows from reading train-period data."
        )
    if not residual.index.is_monotonic_increasing:
        raise ValueError("residual index must be sorted ascending (time-ordered).")

    n = len(residual)
    train_end_idx = int(train_frac * n)
    val_end_idx = int((train_frac + val_frac) * n)

    train = residual.iloc[:train_end_idx]
    val = residual.iloc[train_end_idx + embargo : val_end_idx]
    test = residual.iloc[val_end_idx + embargo :]

    if len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"embargo ({embargo}) leaves an empty val or test split "
            f"(val={len(val)}, test={len(test)}) -- shrink context_length, "
            "shrink embargo, or reallocate train/val/test fractions."
        )

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

    return train, val, test
