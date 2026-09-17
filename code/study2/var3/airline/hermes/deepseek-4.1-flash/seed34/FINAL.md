# FINAL — airline departure-delay AUC (XGBoost)

**Best Eval AUC: 0.7372** (experiment #38, commit `d379a1a`; baseline 0.7141, +0.0231).
40/40 experiments used, ~2,369 of 18,000 CPU-seconds, ~11 minutes of the 230-minute wall clock.
`./validate.sh` → `CONTRACT OK` (0.7372 reproduced through `predict_proba` with the target column removed).

## Final model (`train.py` @ `d379a1a`)

All feature engineering lives in `prepare(df)`, so `predict_proba` reproduces it on raw unseen rows. Every
encoder/statistic is fit on `data/train.csv` only.

Features: raw columns (Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest as XGBoost categoricals,
DepTime, Distance) · train-set frequency counts for Carrier/Origin/Dest/Route/CarrierOrigin and for every
interaction key · airport connectivity degrees + route/carrier shares · day-of-year, its two Fourier
harmonics and week-of-year · hour, minute-of-day, log-distance · hour-of-day and 200-minute time-block
categoricals · **interaction categoricals**: DayOfWeek×hour, Carrier×hour, Origin×hour, Dest×hour,
250-mile distance band, distance band×hour.
Model: XGBoost `hist`, 600 trees, depth 6, lr 0.03, `min_child_weight` 30, `subsample` 0.7,
`colsample_bytree` 0.6, `reg_lambda` 10, averaged over 4 seeds.

## The 5 changes that mattered most

1. **Regularization over capacity.** The unregularized high-capacity model (500 trees × depth 8) scored
   0.7021 — worse than the 30-tree baseline. Depth 6 + `min_child_weight` 30 + row/column subsampling +
   L2 won (0.7141 → 0.7192) and survived every follow-up probe.
2. **Re-tuning tree count after the feature set grew.** 250 trees was optimal on the early feature set
   (0.7206); after the interaction features were added the optimum moved to 600 trees (0.7305 → 0.7343),
   and 900 was equal-or-worse.
3. **Explicit categorical interaction partitions (biggest single feature win).** DayOfWeek×hour +
   Carrier×hour = 0.7219 → 0.7256; adding Origin×hour and Dest×hour = → 0.7305; distance band×hour and
   band granularity 500→250 mi = → 0.7362 → 0.7372.
4. **Frequency (count) encodings fit on train only**, for the raw keys and for every interaction key —
   stable "how busy is this airport/route/carrier-hour" signal that transfers across years (0.7159 → 0.7166,
   and a component of every later gain).
5. **Time-block categoricals** (hour-of-day and 200-minute buckets) plus day-of-year seasonality with two
   harmonics, airport connectivity degrees and route/carrier shares (0.7209 → 0.7219 → 0.7305 chain).

## The 3 things that did not help

1. **Raw model capacity**: 500×depth-8 trees (0.7021), depth 8 even with heavier regularization (0.7183),
   800/900 trees on the earlier feature set (0.7162/0.7342 vs 0.7206/0.7343).
2. **Target/mean encodings** of Carrier/Origin/Dest/Route (smoothed, k=100): 0.7065 — 2005 delay
   propensities simply do not carry into 2006, unlike counts and calendar structure.
3. **High-cardinality pair categoricals**: raw Route (0.7279), Carrier×Origin, Route×hour (0.7300) and
   Origin/Dest×Month (0.7139) all lost — sparse *and* year-specific combinations overfit.
   Also neutral-to-negative: cyclical time-of-day sin/cos, one-hot categorical encoding, early stopping on
   an internal 10% split, 8 seeds instead of 4, a diverse depth/colsample ensemble, explicit UNK category
   for unseen levels, and holiday-window flags.

## What I would try with more budget

The binding constraint is the 2005 → 2006 distribution shift, so I would stop selecting on `eval.csv` and
instead build an internal year-robustness split (fit on part of the 2005 slice, validate on a held-out 2005
slice) and only accept changes that improve *both*, keeping eval purely as a final sanity check. With that
in place: shrink the interaction categoricals hierarchically (ordered/leave-one-out target statistics with
strong priors, as in CatBoost) so that airport×calendar combinations can be used at all — raw partitions of
those are exactly what failed here; search for interaction families by that stability criterion rather than
my intuition; try a residual/stacked second XGBoost stage and rank-averaging across depth 4/6/8 members;
and tune the categorical partition hyperparameters (`max_cat_threshold`, `max_cat_to_onehot`) which I never
isolated from the encoder choice.
