# Final report

**Best Eval AUC: 0.7293** (baseline 0.7141, +0.0152), reproduced on the contract validation run
(`CONTRACT OK`). Final commit `71bd7a1`, HEAD is the best `train.py`.

## Architecture of the final model

Two-stage fit, both inside `train.py`:

1. Early stopping on the **last 20% of the 2005 training rows** (chronological split mimics the
   2005→2006 shift) picks the tree count (~1000 trees at lr 0.03).
2. Refit on all 100k rows with **1.15× trees**, as a **bag of 6 XGBoost models**
   (seed + colsample 0.7–0.9 + subsample 0.7–0.9 diversity), averaged.

Features: parsed DepTime (hour, minute, minutes-of-day, cyclical sin/cos), numeric month/day/dow,
log-distance, and target encodings (train-fit smoothed maps; the first `prepare()` call is always on
training rows, later frames only look up): UniqueCarrier, Origin, Dest, Route,
**DistanceBucket×Hour, Carrier×Hour, HalfHourSlot**.

## Changes that mattered most

1. **Parsing DepTime into time-of-day features** (+0.0045): raw hhmm integers starved the model;
   hour-of-day is the dominant signal (4am→0% delay, 11pm→84%).
2. **Route/Origin/Dest/Carrier target encoding with full-train refit** (+0.0046): smoothed maps
   (alpha 40), two-stage ES→refit so encoders see all training rows.
3. **DistanceBucket×Hour + Carrier×Hour + HalfHour TEs** (+0.0020 total): time-conditioned delay
   curves transfer across years, unlike airport-conditioned ones.
4. **4→6-member diversified seed bag** (+0.0007 over single model, +0.0005 over 4-seed bag).
5. **Early stopping on a chronological split + 1.15× tree refit** (+0.0020, +0.0001).

## What did NOT help (all reverted)

- Hyperparameter scale-ups without feature work (exp2–6: all 0.69–0.71 vs 0.7141 baseline).
- **OOF (out-of-fold) target encoding**: honest train values starved the model (149 trees, 0.7214);
  self-leaky full-fit maps + early stopping on a held-out slice worked better here.
- **Airport/airport-pair × month TEs** and OriginHour TE: month-conditioned rates shift
  year-to-year (train/eval rate corr for Origin|Month = 0.55); a leaked RouteMonth TE scored 0.82
  in-train but 0.53 honest, poisoning early stopping (0.6272 — worst non-crash experiment).
- Dow×Hour TE, Month×Hour TE, seasonality (doy cyclical, Summer×Hour), traffic counts,
  DayOfWeek TE, DistHour/CarrierHour alpha relaxation, bag-of-8 (timeout), depth-mixed bag.

## With more budget

Blend of hour-conditioned TE TGrades with per-bag-member encoder perturbation; quantile-bucketed
(10-bucket) Distance×HalfHour TE; feature-selection sweep per bag member (drop redundant raw
categoricals for some members); a proper time-series CV over the 2005 slice to pick ES split
fraction (0.8 was never ablated); rank-average blending of the 6 members instead of mean.
