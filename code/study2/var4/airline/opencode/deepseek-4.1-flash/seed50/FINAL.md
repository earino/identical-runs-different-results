# FINAL — airline delay (AUC)

**Best Eval AUC: 0.7493** (`predict_proba` path, `validate.sh` CONTRACT OK, 64.1s to train).

The final `train.py` is committed at `HEAD` (`cedf180`). It is a 3-seed ensemble of leaf-wise
XGBoost models over engineered features. Note: the last improvement (lossguide trees, 0.7438 -> 0.7493)
was found in diagnostics and committed/validated directly; the cell's CPU budget (18000 CPU-s) was
exhausted before it could be logged as a counted experiment, so `experiments.tsv` tops out at 0.7438.

## Changes that mattered most
1. **Ordinal date features.** Encoding `Month`/`DayofMonth`/`DayOfWeek` as integers instead of
   XGBoost categoricals: 0.7157 -> 0.7218. These are ordered/cyclic, so range splits beat category splits.
2. **Frequency encodings**, especially interactions: `route_f`, `carrier_f`, `origin_f`, `dest_f`, and
   `origin_hour_f` / `carrier_hour_f` / `dest_hour_f` (airport/carrier congestion by scheduled hour):
   ~0.730 -> 0.744.
3. **Time-of-day parsing of `DepTime`** (`hhmm` -> minutes since midnight, hour, sin/cos): small but consistent.
4. **Low `colsample_bytree` (0.30)** and **leaf-wise growth** (`grow_policy="lossguide"`,
   `max_leaves=512`, `max_depth=0`): 0.7438 -> 0.7493. Leaf-wise trees with many leaves captured
   interactions the depth-wise trees missed.
5. **Seed ensembling** (average of 3 models). Single-model eval spread is ~0.0009 AUC; averaging
   adds ~+0.002 and reduces variance — important because the true score is on a hidden time slice.

## What did not help
- **Native categorical `Route`** (Origin_Dest, ~4200 levels): severe overfit (0.7087).
- **Target encoding** (in-fold or OOF/smoothed) of carrier/origin/dest/route: no gain once frequency
  encodings were present; OOF versions were slightly worse, and in-fold date target encoding looked
  great (0.727) but did not survive out-of-fold (0.716), i.e. it was an artifact.
- **Day-of-year, cyclic month/dow features, deeper dense trees (depth>=12), higher subsample/colsample**:
  neutral or negative. A tuned 300-tree depth-6 model also overfit vs ~120-600 leaf-wise trees.

## With more budget
The strong result was leaf-wise growth scaling monotonically up to ~2048 leaves (single model ~0.7486,
3-seed ensemble 0.7504 in diagnostics), but those fits are slow and the 120s/experiment and CPU budgets
stopped me before tuning the leaf-count/ensemble-size trade-off under the timeout. Next I would:
fit leaf counts 1024-4096 with fewer ensemble members, add early stopping on a time-ordered internal
validation split (rather than a fixed 600 rounds), test `max_cat_threshold`/one-hot for the low-cardinality
categoricals, and try a small `dart` ensemble — then pick the ensemble size that maximizes expected hidden
holdout AUC subject to stable runtime.
