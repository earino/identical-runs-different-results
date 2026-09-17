# Final Report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7495** (commit `c828e64`, experiment #28)

## What mattered most

1. **Time-of-day feature engineering.** `DepHour` as a 24-level categorical, raw `DepMin`
   (scheduled minute) kept numeric, and an `IsWeekend` flag. Ablation showed `DepMin` alone is
   worth ~0.012 AUC — the exact scheduled minute carries real signal.
2. **Categorical interaction columns** built from training-fixed levels: `CarrierHour`,
   `OrigHour`, `DestHour`, `CarrierOrig`, and especially **`DistBandHour`** (distance band × hour),
   which added ~0.002 on a single model and ~0.0017 to the ensemble.
3. **Frequency / share encodings.** Counts for `Origin`, `Dest`, `Carrier`, `Route`, plus counts of
   every interaction column, and normalized shares: `OrigHourShare`, `DestHourShare`, and
   `RouteHourShare` (`DistBandHourFreq / RouteFreq`). The share features were worth ~0.001–0.002 to
   the ensemble even though they barely moved single-model AUC.
4. **Strong L1-style regularization.** `reg_alpha` (L1) was the single biggest hyperparameter lever;
   `gamma`, `max_cat_threshold` and `colsample_bytree` mattered next. Stochastic row subsampling
   hurt (time-separated data overfits easily), so all final members use `subsample=1.0`.
5. **A diverse 7-model XGBoost ensemble averaged in probability space.** Members span depth 6–8,
   learning rates 0.03–0.07, different `max_cat_threshold`/`max_cat_to_onehot` and gamma. Simple
   probability averaging beat rank-mean, logit-mean and AUC-proportional weighting.

## What did not help

- Naive **target encoding** of Route, a native high-cardinality `Route` categorical, and raw numeric
  hour — all overfit the 2005→2006 shift.
- **Extra distance interactions** (DistBand × Month/DayOfWeek/Carrier/Dest) and **minute-periodicity**
  features (`DepMin mod 5/10/15`) — consistently negative.
- **Ablating features**: every removal hurt, confirming the feature set is not over-specified.
- Adding a low-LR long-run member (0.7480 vs 0.7481) and extra route/carrier share features (0.7491,
  0.7490); all were reverted.
- Different combining rules and weight optimization gave no gain over the plain mean.

## If I had more budget

I would push on three fronts: (a) **seed-bagged duplicates** of the strongest members, since the
ensemble responded well to diversity; (b) **proper out-of-fold target encoding** for Route/Carrier
with smoothing, which is a genuinely different signal source the current frequency features only
approximate; and (c) **bagging over distance/hour binning schemes** to reduce sensitivity to the
chosen bins. Budget (CPU) was the binding constraint, so I would also trim redundant members to make
room for a wider, more diverse ensemble.
