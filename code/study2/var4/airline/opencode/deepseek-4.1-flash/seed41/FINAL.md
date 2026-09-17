# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7366** (`d50316c`, experiment #40). Baseline: 0.7141. Net +0.0225.

## Setup
Train = 100k rows (2005), eval = 100k rows (2006). Metric AUC on `dep_delayed_15min`.
Final model: an ensemble of 15 XGBoost classifiers (averaged probabilities) over an engineered
feature set, with all feature engineering contained in `prepare(df)` so `predict_proba` reproduces
it on the hidden holdout. Encoders / target-encoding maps / count tables are fit on `data/train.csv`
only.

## Changes that mattered most
1. **Enough capacity + ensemble.** The baseline's 30 trees underfit. 150 trees @ lr 0.1, depth 6 took
   eval from 0.7141 to 0.7207. Averaging 15 diverse XGBoost configs (depth 5–10, lr 0.04–0.1, lossguide,
   seed bagging) added ~+0.002 and is the single most robust gain.
2. **Airport congestion features.** Origin/destination "share of the airport's daily departures in this
   hour" (`orig_hour_share`, `dest_hour_share`) plus carrier hub shares (`carr_orig_share`,
   `carr_dest_share`) were the biggest single jump (0.7243 → 0.7308). Absolute volume (log counts) helped
   less than normalized share.
3. **Estimated arrival-hour congestion.** Block-time proxy `arr_hour = (DepTime + 0.12*Distance + 30)`,
   then destination/origin arrival-hour shares, captured destination-side congestion (0.7308 → 0.7324).
4. **OOF target encoding of time-conditioned groups.** Smoothed (k=50) out-of-fold mean delay for
   `origin|hour`, `dest|hour`, `carrier|hour`, their `arr_hour` variants, and `carrier|origin`,
   `carrier|dest` (0.7324 → 0.7350). OOF for training rows, full-train map at inference.
5. **Hour as a categorical feature** (partition splits over 24 values) gave the final +0.0016 (0.7366).
   Distance-bin × hour TE was a smaller reliable add.

## What did not help
- **Route (Origin×Dest) as a categorical** was catastrophic (0.7008) — thousands of sparse levels
  overfit the 2005 slice and do not transfer. Route target encoding also hurt.
- **Count/frequency features** (origin/dest/carrier/route counts) were mildly negative.
- **Global (un-conditioned) target encoding** of origin/dest/carrier and weekday/month group TE
  overfit the year shift; only time-conditioned / hub groups helped.
- **Seasonality (day-of-year) features and holiday flags** hurt — 2005 calendar effects did not
  transfer to 2006. Deeper trees, more trees (300–400), DART, early stopping on a random split,
  and pruning `dep_raw`/`hour_sin`/`hour_cos` were all neutral-to-negative.

## With more budget
Push the congestion idea further: model the *schedule graph* rather than isolated counts — e.g.
delay-propagation proxies such as the number of the carrier's flights scheduled to arrive at the
aircraft's next airport, tighter block-time estimates from the distance distribution, and
origin→dest→hour bottleneck features. Also worth trying: stacking the ensemble with an XGBoost
meta-learner on OOF predictions, per-group shrinkage tuned per cardinality, and multi-seed bagging
of the final 15-member ensemble for additional holdout variance reduction (eval was flat but the
1M holdout would likely benefit).
