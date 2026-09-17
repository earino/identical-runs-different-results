# Final Report — Airline Delay 15min (binary classification, AUC)

## Bottom line

- **Best committed eval AUC: 0.7557** (eval.csv = time-separated 2006 slice; hidden holdout is a
  fresh 1M-row 2006 slice scored off-machine).
- Best commit: `65d570f` ("subsample 0.8 -> 0.75 at d18") — this is `git HEAD` at finalization.
- `./validate.sh` was run on exactly this tree and printed **CONTRACT OK**
  (train-only fitting, `predict_proba(df)` works with the target column removed,
  returns finite probabilities in [0,1], eval AUC via the contract path: 0.7557).
- The CPU budget (18,000 s) exhausted at experiment #25 (26 attempts total incl. the final
  refusal run); 24 real experiments were executed, all committed one-at-a-time and reverted
  whenever the result did not beat the incumbent.

## Final method (train.py)

1. **Features (17 columns), all fitted on train.csv only:**
   - Parsed numeric `Month`, `DayofMonth`, `DayOfWeek` (from `c-XX` strings).
   - DepTime decomposition: `DepHour`, `DepMinute`, `MinOfDay`, `HourSin`, `HourCos`,
     `DepTimeRaw` (clipped 0–2359; a few eval rows have 2400+ values).
   - Categoricals (native XGBoost): `UniqueCarrier` (20), `Origin`/`Dest` (282 each),
     `DepHourC` (24), `QHourC` (96 = hour×15min), `CarrierHour` (480 interaction levels).
   - `Distance`, `LogDistance`.
   - Category vocabularies and any maps are stored at module level from train.csv;
     unseen eval/holdout categories route to the NaN branch.
2. **Model:** 3-seed ensemble (random_state 0/1/2) of
   `XGBClassifier(n_estimators=300, learning_rate=0.037, max_depth=18, reg_alpha=0.5,
   subsample=0.75, colsample_bytree=0.7, tree_method="hist", enable_categorical=True, n_jobs=4)`.
   `predict_proba` = arithmetic mean of the three members' probabilities
   (validated to return values in [0,1]).
3. Wall-clock fit: ~73 s training + ~23 s eval scoring inside the 120 s experiment cap.

## What moved the needle (AUC trajectory)

| # | Change | Eval AUC |
|---|--------|----------|
| 1 | Baseline (raw columns, defaults) | 0.7141 |
| 3 | Date parsing + hour features + deep, regularized 5-seed ens. | 0.7516 |
| 4 | + QHourC (96-level) and CarrierHour categoricals, 3-seed d16 n300 lr.04 | 0.7549 |
| 6 | reg_alpha 1.0 → 0.5 | 0.7554 |
| 10 | 300 → 350 trees, lr 0.035 | 0.7555 |
| 16 | homogeneous max_depth 16 → 18, n300 lr.037 | 0.7556 |
| 18 | subsample 0.8 → 0.75 | **0.7557** |

## What did not help (all reverted, each = one git commit + `git reset --hard HEAD~1`)

- Route / high-cardinality interaction categoricals (Origin×Dest, Origin×Carrier): −0.01 and slow.
- Calendar×hour interactions (dow×hour, month×hour), holiday flags, 5-minute time buckets.
- Target/count encodings of Origin/Dest/Carrier, distance-deviation features (helped single
  trees at weak settings, never helped the tuned ensemble).
- Early stopping on a same-year split (overfits 2005 noise; fixed tree counts generalized better
  under the 2005→2006 shift).
- dart, lossguide, rank:pairwise, row-bagged / mixed-config / 4th-seed members,
  rank-space blending, subsample=1.0 members (better single, worse ensemble),
  min_child_weight/gamma/lambda tweaks, max_bin reductions, max_cat_to_onehot.
- Seed-stability checks (seeds 10–12) came in at 0.7553 vs 0.7557 → seed noise ≈ ±0.0002.

## Interpretation

The eval split is a *different year* from the training data, so generalization is driven by
stable marginals: time-of-day (the dominant signal), carrier, and airport effects. Deep trees
(d16–d18) with strong L1 (alpha 0.5), modest row sampling (0.75) and column sampling (0.7),
plus 3-seed averaging, capture those marginals without memorizing 2005-specific noise. The
eval AUC plateaued at ≈0.7555–0.7557 — the last several experiments (alpha/col/seed/ensemble-
size/depth-19 micro-variants) tied or regressed, indicating the feature set's signal is
exhausted at this data size rather than the tuning being incomplete.

## Budget accounting

- 24 experiments ran (cap 40); CPU ledger: ~18,026 / 18,000 s (the final "budget check" run
  was refused mid-run by the ledger hook, which is the official BUDGET EXHAUSTED signal).
- Wall clock remaining at finalization: ~128 min (CPU was the binding constraint, not time).
- Cost of best run: ~100 s wall, ~350 CPU-s per experiment, ~420 total incl. data loading.
