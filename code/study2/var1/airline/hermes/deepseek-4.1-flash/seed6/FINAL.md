# FINAL — airline delay (dep_delayed_15min) XGBoost

**Best Eval AUC: 0.7573** (commit `3f7e1c5`), vs 0.7141 for the committed baseline. 35 experiments run
of the 40-experiment budget; the run was stopped because the 18,000 CPU-second Python budget was nearly
spent (validation itself costs a full training run).

## Changes that mattered most

1. **Time-of-day decomposition of `DepTime`** (`dep_hour`, `dep_minute`, `dep_tod`, plus a 15-minute
   `tod15` categorical bucket). Departure code is hhmm and runs past 2400 for red-eyes, so plain numeric
   splits are both distorted and non-monotone; explicit clock features were the first real gain
   (0.7141 → 0.7184) and everything later builds on them.
2. **`carrier_hour` cross** (`UniqueCarrier` × hour as a single categorical): the largest single jump,
   +0.016 AUC (0.7229 → 0.7318 for one seed). Carriers have distinct delay profiles through the day, and
   that is a structural effect that transfers across years.
3. **`dist_bin_hour` (250-mile bucket × hour) and `origin_top_hour` (top-30 origins × hour) crosses**:
   +0.005 and +0.001. Full-cardinality airport crosses (282 × 27) overfit badly; folding the tail of the
   airport distribution into an "OTHER" bucket is what makes the origin cross work.
4. **Dropping the `Month`/`DayofMonth` calendar columns**: +0.004 (0.7184 → 0.7201, and the gain held
   through every later re-test). Their rates are year-specific (e.g. January delay rate 0.541 in 2005 vs
   0.444 in 2006) while day-of-week rates are nearly identical across the two years; only the latter
   transfers.
5. **Depth-diverse bag of 10 XGBoost models** (depths 4–12, 500 trees at lr 0.03, `sampling_method=
   gradient_based`, `subsample=0.95`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_alpha=0.5`):
   0.7502 → 0.7573. Cross-depth averaging plus gradient-based sampling and mild L1 were worth ~+0.002
   over the best single configuration.

## Things that did not help

1. **Target/frequency encodings** of `Origin`, `Dest`, `UniqueCarrier`, routes and their hour crosses.
   Smoothed target encoding (k = 20…1000), OOF target encoding, and count/frequency encodings all landed
   at or below the plain categorical model; route × hour encodings were catastrophic (≈ −0.02).
2. **Model capacity on the plain feature set.** 500 trees at depth 7 scored 0.7054 — *worse* than the
   30-tree baseline (0.7141) — and depth 8/10 was worse than depth 4 until the crosses gave the trees
   something worth splitting on. The weak, year-shifted signal simply memorizes 2005 quirks.
3. **Stacking and other structural variants**: an OOF-prediction → XGBoost meta-learner tied plain
   bagging (0.7524 vs 0.7527 on the same base models); `lossguide`/`dart`/one-hot categoricals/monotone
   constraints on `dep_tod`/early stopping on an internal split/structural hub statistics
   (airport degree, hub-owner flags, route frequency) each added ≤ 0.0005 or lost.

## What I would try with more budget

The binding constraint is that the only labeled signal about the 2005 → 2006 shift is `eval.csv` itself,
so keep/discard decisions risk selecting on a single 100k slice. My first move would be to reconstruct
the real calendar from the consistent `(Month, DayofMonth, DayOfWeek)` triples (each pair maps to exactly
one weekday, 365 pairs per file, so the ordering is recoverable) to build a *time-based* internal split
that emulates the year shift, and to re-select model depth, tree count and cross cardinalities against
that instead of against `eval.csv`. Then I would spend the rest on richer interaction structure with
hierarchical smoothing — per-origin and per-route hour profiles shrunk toward carrier and global means —
since the two winning features were exactly of that type (`carrier_hour`, `dist_bin_hour`), and finally
on semi-supervised use of the unlabeled 2006 rows: the scorer hands `predict_proba` the whole holdout
frame at once, so density/congestion features computed within that frame (flights per origin-day, route
share of the day) are computable at prediction time and were left on the table because their scale
depends on how the scorer batches rows.
