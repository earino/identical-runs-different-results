# FINAL — Airline departure-delay AUC

**Best Eval AUC: 0.7220** (baseline committed `f52d7a3` was 0.7141; 40 experiments used).

Best commit: `f2246bf` — "lighter regularization mcw5 lam1". `./validate.sh` reports `CONTRACT OK`
(eval AUC via `predict_proba` with the target column removed = 0.7220).

## Changes that mattered most

1. **Time-of-day / calendar feature engineering inside `prepare()`.** Parsing `DepTime` into
   `DepHour`/`DepMinute`, `IsWeekend`, and `DayOfYear` gave the first real improvement (0.7153 → 0.7195).
   Hour-of-day turned out to be by far the most transferable signal (delay rate rises monotonically
   from ~0.02 at 5am to ~0.78 at 11pm, almost identically in 2005 and 2006).
2. **Averaging an ensemble of XGBoost models** with different depths (4–8) and seeds (15 models).
   Ensembling was the single largest gain (single model 0.7153 → 0.7205 for 3 depths → 0.7218 for 15).
   Individual models overfit year-specific noise; averaging cancels much of it.
3. **Strong row/column subsampling** (`subsample=0.7`, `colsample_bytree=0.7`) plus mild
   `min_child_weight`/`reg_lambda`, which decorrelates ensemble members and regularizes against the
   2005→2006 distribution shift.
4. **A few explicit hour interactions** (`HourXDist`, `HourFromNoonSq`, `HourAfter5`), worth ~+0.001
   total but consistent with the fact that hour is the dominant stable feature.
5. **Keeping the `c-<n>` columns and carrier/origin/dest as native XGBoost categoricals** instead of
   coercing them to numeric.

## Things that did NOT help (all reverted)

- **High-cardinality interaction categoricals**: `Origin_Dest` route (0.7042) and `Carrier_Origin`
  (0.7149) both caused large drops — new routes appear in 2006 (91% overlap) and route/carrier delay
  rates shift year-to-year, so memorizing them does not transfer.
- **Out-of-fold smoothed target encoding** of carrier/origin/dest (0.7201) and dropping their
  categoricals in favor of TE (0.7155): no gain, TE is not better than XGBoost's native categorical
  splits on this data.
- **Frequency (traffic-volume) encodings** of carrier/origin/dest/route: exactly neutral (0.7195).
- **Early stopping on a random 10% 2005 validation split**: it selected ~280 rounds at lr=0.05, but
  the 2006 eval prefers far less capacity (0.7119) — direct evidence of the year shift.
- **DART booster** (timed out at 120 s) and **loss-guide growth** (0.7201, slightly worse), plus
  monotone `DepHour` constraint (0.7206) and `max_cat_threshold` 16/128 tuning.

## What I would try with more budget

The dominant limitation is the 2005→2006 distribution shift: every capacity/regularization knob
trades train fit against holdout generalization, and internal 2005 validation is actively misleading
(it wants ~10× more trees than the holdout). With more budget I would (a) build an adversarial
validation / importance-weighted training scheme to identify and down-weight 2005-specific features,
(b) explore stronger per-feature regularization or feature dropout rather than one global setting,
and (c) add genuinely new transferable signal that is absent here (airport congestion at the
scheduled hour, weather, aircraft rotation, holidays). I would also test a larger, more diverse
single-feature-family ensemble; the seed/depth ensemble gains were still positive but had clearly
saturated (~+0.0001 per added diversity). Using the labeled 2006 eval slice as additional training
data would likely help the 2006 holdout substantially but would invalidate the reported eval metric,
so I did not do it.
