# FINAL — airline departure-delay prediction (XGBoost)

**Best Eval AUC: 0.7481** (experiment #18, commit `02e7381`; baseline 0.7141 → +0.0340).
All 18 logged experiments ran `ok` (no crashes, no timeouts, no OOM). Final `./validate.sh`
reports `CONTRACT OK`, with the AUC reproduced through `predict_proba()` on the eval frame with
the target column removed (0.7481).

## What mattered most

1. **Deep, strongly regularized trees instead of shallow ones.** `max_depth=28`,
   `min_child_weight=2`, `reg_lambda=10`, `colsample_bytree=0.3`, `lr=0.02`, 600 rounds.
   This single change was worth ~+0.017 (0.7172 → 0.7359 at the time). With 8 raw columns the
   signal is almost entirely in high-order interactions (carrier x time-of-day x airport x
   distance); deep trees with heavy column subsampling learn them, and K-fold validation on the
   training year *actively misleads* here — it prefers more capacity than generalizes.
2. **`carrier x hour` as a native categorical interaction** (+0.0037). How punctual a carrier is
   depends strongly on the time of day; the trees cannot assemble that 480-level interaction
   cheaply from `UniqueCarrier` + `DepTime` under `colsample_bytree=0.3`.
3. **Traffic-volume / congestion proxy: `log1p(count of (Origin, hour) pairs)`** (+0.0036).
   How busy an airport is at that hour is real information that no raw column contains.
4. **`Distance x minute-of-day` and `Distance x hour` products** (+0.0033 combined). Delay
   accumulation over the day scales with stage length; trees cannot multiply.
5. **Holiday / extreme-part-of-day flags** (Thanksgiving, Dec 18+, July 4, New Year; late/early
   hours) and a final **2-seed bag** — small but real (+0.0003 to +0.0007 each).

Two opposite risk controls both helped: richer explicit interactions *and* more regularization
(20 features, only 30% of columns per tree). Everything is fit on training data only and lives
inside `prepare()`, so the hidden holdout sees exactly the same transformation.

## What did not help (measured, then reverted)

- **Target/statistics encoding** of carrier, origin, dest and route delay rates (−0.009 at the
  time, even with train-side leakage): it encodes last year's delay level, which shifts.
- **A high-cardinality `route` (Origin|Dest) categorical** (−0.012): 4.2k levels memorized
  year-specific routes; `carrier*dest`, `origin*month`, `dest*month` and `carrier*origin`
  interactions all hurt the same way. Only the moderate-cardinality `carrier*hour` paid off.
- **Validation-based early stopping** — the random-split validation AUC curve rises with capacity
  while eval AUC falls (val picked 72-233 trees where 600 deep rounds was better); also
  `rank:pairwise` (0.59), `booster=dart` (0.714), `grow_policy=lossguide`, row bagging at 80 %,
  `subsample<0.9`, `colsample>=0.4`, `max_cat_threshold=480` and count features for
  dest/carrier/route, which were all neutral or worse.

## With more budget

The largest untapped lever is *representation of the time axis and the airport network*: this
scenario ships only 8 columns, and 2005→2006 shift means any feature keyed on identity (route,
specific date) decays, while features keyed on capacity/congestion transfer. I would build
airport-level congestion statistics over several time resolutions (origin-hour flow estimated
with out-of-fold shrinkage toward the airport's own mean, plus its first difference from the
previous hour to capture delay propagation through the day), add carrier rotation/connection
proxies if any join key existed, and tune the interaction-feature family jointly rather than one
at a time. On the model side: a proper multi-seed bag with several depths (28/20/14) using the
richest feature set, and a nested split that mimics the year shift (train on months 1-9 of 2005,
validate on months 10-12) instead of a random split, which is what misled early stopping here.
