# TimesFM fine-tuning train/val/test split + embargo (TODO.md step 6)

Status: decided, implemented in `src/eval/split.py::split_with_embargo`.

Governing constraint (CLAUDE.md 2.1): splits strictly time-ordered, never
shuffled, embargo gap between splits.

## Decisions

- **context_length = 128** trading sessions (~6 months, several multiples
  of HAR's 22-day monthly window). The existing zero-shot report
  (`reports/zeroshot_timesfm_metrics.md`) used `context_length=512` --
  **not** a valid baseline for the fine-tuned-vs-zero-shot ablation at 128.
  Re-run `scripts/run_zeroshot_timesfm.py --context-length 128` before
  that comparison; save to `reports/zeroshot_timesfm_metrics_ctx128.md`,
  do not overwrite the 512-context report.
- **train/val/test fractions = 60/20/20** (not 70/15/15) -- at 128-session
  embargo, 70/15/15 leaves val/test with only ~56 usable sessions each
  after the embargo tax, too thin for a meaningful QLIKE/DM comparison.
  60/20/20 gives ~118 sessions each after embargo.
- **embargo = context_length = 128** at each boundary. Not a free
  parameter smaller than this -- a shorter embargo still lets a val/test
  window's context read backward into the train period.

## Worked numbers (SPY residual series, n=1230, 2021-08-24 to 2026-07-22)

| Split | Date range | n (post-embargo) |
|---|---|---|
| train | 2021-08-24 to 2024-07-31 | 738 |
| val | 2025-02-06 to 2025-07-29 | 118 |
| test | 2026-02-02 to 2026-07-22 | 118 |

## Stress-period coverage (CLAUDE.md 6.4)

Checked monthly mean `rv_d` for the elevated-vol candidates before locking
the boundary:
- **val** contains April 2025 (monthly mean RV 0.000523, the single
  highest-vol month in the whole 2020-2026 series -- a real stress
  episode, not the 2022 rate-hike stretch as originally guessed).
- **test** (2026-02 to 2026-07) is mostly calm (median RV close to the
  full-series median) but has a mini spike around 2026-06-09
  (RV 0.000252, ~6.5x that window's own median) -- enough for the
  regime-split ablation to have *something* elevated inside held-out test,
  in addition to the stronger stress captured in val.

No date-boundary shift was needed -- the mechanical 60/20/20 cut already
lands stress inside val/test rather than entirely in train.

## Acceptance checklist

- [x] context_length decided explicitly (128, not inherited from the
      zero-shot script's 512 or the step-5 smoke test's 60)
- [ ] Zero-shot baseline re-run at context_length=128, saved as its own
      report file (not yet done -- do before the fine-tuned-vs-zero-shot
      ablation)
- [x] `split_with_embargo()` implemented and unit-tested
      (`tests/eval/test_split.py`): asserts no val/test window's context
      reaches into train's date range, on a synthetic monotonic series
- [x] Val and test each have enough sessions for QLIKE + DM comparison
      (~118 each, above the ~150-session aspirational floor is not quite
      met -- flag this if DM tests on the eventual fine-tuned model come
      out noisy; the alternative is 50/25/25 or a shorter context_length)
- [x] Test (and val) each include at least one identifiable elevated-vol
      subperiod, not an entirely calm stretch
- [x] Split boundary dates, embargo size, and fraction choice written
      down here (this file) and in TODO.md
