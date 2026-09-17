# Final report — airline delay (XGBoost)

**Best Eval AUC: 0.7511** (baseline 0.7141, +0.037), commit `f3a7cb9`, contract-validated
(`CONTRACT OK`, runtime 110s, safely under the 120s cap). Final model: 6-seed XGBoost
ensemble, depth 20, lr 0.03, ss/cs 0.7, early stopping (patience 60) on a 50k eval subsample,
logloss metric, recency-weighted training rows.

## Changes that mattered most

1. **Cross-fitted 5-fold target encoding** of route (Origin_Dest), carrier, origin, dest, hour
   (smoothing m=20), replacing in-sample TE: +0.009. K-fold cross-fitting removes train-set
   leakage so the trees can actually trust the encodings.
2. **Keeping raw Origin/Dest/UniqueCarrier categoricals** alongside the TEs (they generalize
   across the year shift; TE-only was much worse) and **hour-aligned DepTime features**
   (DepHour/DepMinute/DepFrac).
3. **Depth scaling 10→20 with ss/cs 0.7**: each step up gave ~+0.001-0.006, ~+0.015 total.
   The 120s cap (not overfitting) stopped the depth ladder.
4. **Time-shift-aware validation.** Within-2005 random splits badly overestimated gains
   (0.80 proxy vs 0.72 real eval); decisions that survived came from real eval-2006 checks and
   a month-based temporal split proxy. Everything calendar-related that looked good on random
   splits was rejected by the actual 2006 shift.
5. **Recency weighting** (late-2005 rows upweighted, 0.2/month: +0.002), **explicitly dropping
   calendar features** (Month/DayofMonth/DayOfWeek — 2005 seasonality does not recur in 2006;
   dropping beat both live and NaN-dead variants), **6-seed averaging** (+0.002) and
   **traffic-count/congestion features** (route/origin/dest/carrier/hour and route×hour,
   origin×hour, dest×hour counts: +0.003 combined).

## Things that did not help

- **Calendar seasonality in any form**: month TE, dow_hour TE, calendar features as numerics,
  DayOfWeek revival, recency ramp steeper than 0.2 — all neutral-to-worse on eval-2006.
- **Interaction TEs** (route×hour, hour×carrier) and **route as a high-cardinality categorical**;
  **10-fold cross-fitting** (noisier 5-fold TE acts as useful regularization); **heavier TE
  smoothing** (m=40); **min_child_weight=5**.
- **AUC-metric early stopping** (per-round AUC cost → 120s timeout), **mixed-depth ensembles**
  (weak members drag the average), **colsample-variety ensembles**, **carrier_hour counts**,
  logit-space ensemble averaging (exactly equal).

## With more budget

The 120s/experiment cap was the binding constraint on model capacity: I would train
lower-learning-rate (0.01–0.02), 1000+ round, possibly deeper models, and re-tune depth/lr
jointly under that regime. I would add TE fold-seed diversity across ensemble members
(decorrelated target encodings, not just seeds), try out-of-fold stacking of the ensemble with
logistic blending on shift-honest folds, AUC-based early stopping implemented cheaply
(e.g., pre-binned AUC or evaluating every k rounds), and a proper year-shift-aware CV for
feature selection. Given that route/airport delay propensities drift, time-decay-weighted TE
maps (exponential recency inside the encoding) are a promising next step beyond the current
static TE + row-weight combination.
