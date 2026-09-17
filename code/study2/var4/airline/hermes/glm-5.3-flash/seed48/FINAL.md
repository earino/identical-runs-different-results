# FINAL — airline delay (XGBoost autoresearch)

**Best Eval AUC: 0.7549** (baseline 0.7141, +0.0408). HEAD = `056a825` (exp16), validated:
`./validate.sh` → `CONTRACT OK`, and `predict_proba` reproduces 0.7549 on eval.csv with the
target column removed.

## Final model

XGBoost (`hist`, `enable_categorical`), trained on all 100k rows of `data/train.csv`:
1000 trees, max_depth 16, lr 0.01, min_child_weight 5, max_bin 2048, colsample_bytree 0.7,
reg_lambda 3, reg_alpha 1, 4 threads (~61 s).

Features (all built inside `prepare()` from raw columns, no fitted statistics except the
categorical level lists from train):
- raw: DepTime, Distance, DayOfWeek, UniqueCarrier, Origin, Dest as categoricals
- dropped: Month, DayOfMonth (anti-transfer under the 2005→2006 shift)
- engineered: `DepTime×Distance/1e6`, `DepTime%100` (minute), scheduled 15-min slot as a
  96-level categorical (hour%24*4 + minute//15)

## What mattered most

1. **Feature selection against the time shift (+0.003)** — dropping `Month` (exp10, 0.7155→0.7188).
   2005's month→delay pattern is anti-correlated with 2006's; the eval/holdout slices are from a
   different year, so seasonal rates are noise there. DayOfMonth helped a smaller +0.0005 (exp12).
2. **Scheduled-time slot as a categorical (+0.017)** — the 96-level `FE_slot15` plus DepTime
   numeric (exp15, 0.7241→0.7522). The delay curve over time of day is sharp and transfers;
   expressing it as a categorical lets trees split it exactly, and the 15-min granularity beat
   20/30/60-min buckets.
3. **Large capacity after fixing the validation target (+0.003)** — once the model was
   shallow-regularized for transfer, capacity became useful again: d12→d16, lr 0.02→0.01,
   700→1000 trees (exp16, 0.7522→0.7549).
4. **Semantic numeric interactions (+0.003)** — `DepTime×Distance` and departure-minute (exp14,
   0.7214→0.7241): late evening + short flight = high delay risk, robust across years.
5. **max_bin 2048 (+0.0006)** — finer histogram splits for the quasi-continuous DepTime.

## What did not help (all measured on eval, discarded)

- **High-cardinality interaction categoricals**: Origin>Dest route, carrier×airport concatenations
  collapsed eval AUC to 0.57 — memorized 2005 route rates anti-transfer.
- **Target (mean-delay) encoding** of Month/carrier/airport: 0.697–0.699 vs 0.714 baseline.
- **Row subsampling**: subsample 0.7–0.85 consistently lost ~0.02 on time-shifted validation even
  though it looked neutral on random splits; colsample_bytree 0.7 helped, colsample_bynode did not.
- Small-tree micro-tuning around the baseline (0.713–0.716) and DepTime as a raw categorical (0.693).
- Seed-bagging the final model: 0.7549 vs 0.7548 single — no gain, seeds are nearly identical here.

## Method note

The decisive methodological change was switching validation from random row splits (which
overstate transfer: a deep model scored 0.7732 internally but 0.7102 on eval) to a time-shifted
proxy — hold out whole months of 2005 and predict them from the remaining months. That proxy
correctly ranked capacity/regularization trade-offs, and eval AUC tracked it within ~0.01.
Probing configs on unlogged scripts before committing kept experiments.tsv to 16 logged runs.

## With more budget

- Per-carrier/per-airport *shrinkage-toward-zero* encodings (rate minus prior, heavy shrinkage) —
  raw rates and ranks both failed, but strongly regularized deviations might survive the shift.
- Calibration/Isotonic-free rank averaging of two capacity regimes (d4-shallow + d16-deep) —
  their errors looked complementary on the month-holdout oracle.
- External weather/holiday calendars are out of scope (no network), but US holiday indicators
  (fixed-date) are computable from the date columns and were never tested.
