# FINAL — airline departure-delay AUC

**Best Eval AUC: 0.7457** (commit `64e9213`, experiment #38)
Baseline (30 trees, raw features): 0.7141. Final: 0.7457 (**+0.0316**).
40/40 experiments used; `./validate.sh` → `CONTRACT OK` (0.7457 reproduced via `predict_proba`).

## Setup that won
A 12-member XGBoost ensemble (`tree_method="hist"`, `enable_categorical=True`) over depths {5,6,7,8} × seeds {42,7,2024}, `n_estimators=250`, `learning_rate=0.1`, `min_child_weight=2`, `reg_lambda=3.0`. Probabilities are averaged. All feature engineering lives in `prepare(df)`; categorical levels/statistics are fit on `data/train.csv` only, so the hidden holdout is handled correctly.

## The 4 changes that mattered most
1. **Time-of-day decomposition + interactions with stable, low/medium-cardinality fields.** Replacing raw `DepTime` with hour/minute/time-of-day, then crossing it with carrier and day-of-week (`CarrierHour`, `DowHour`) was the single biggest jump (0.7200 → 0.7311). Trees alone failed to recover these interactions at shallow depth.
2. **Restricted-airport interactions (`OriginTopHour`, `DestTopHour`, `OriginTopCarrier`, `DestTopCarrier`).** Keeping only the 25 busiest airports and bucketing the rest into `OTHER` (0.7327 → 0.7416). The top-N choice was non-monotonic: top-10, top-50 and top-100 were all worse than top-25 — this is the main tuned knob.
3. **Coarse route/airport × distance-bin features** (`TopRoute`, `CarrierLogDist`, `DistBinHour`, `OriginTopDistBin`, `DestTopDistBin`): 0.7360 → 0.7453. Distance buckets generalize across years where exact route identity does not.
4. **Seed × depth ensembling and mild regularization** (`min_child_weight=2`, `reg_lambda=3`) added a final, robust +0.002/+0.0004 and stabilized the score.

## The 3 things that did not help (and were reverted)
- **High-cardinality identity features**: native `Route` categorical (4198 levels), `Origin×Hour` (6768), `Origin×DayOfWeek`, and OOF smoothed target encoding all *hurt* (e.g. Route → 0.6986). With one year of train and a different year for eval, memorizing specific O/D identifiers overfits.
- **More capacity per model**: 300–600 trees, depth 7–10, `lr=0.05`+subsample+colsample, and 16-member ensembles all matched or degraded the score. The signal saturates near depth 5–8 / ~250 trees.
- **Extra calendar interactions** (`CarrierMonth`, `DowMonth`, `HourMonth`, calendar × top-airport, 30-min time slots): consistently worse, indicating these seasonal/weekday patterns shift year-over-year.

## What I would try with more budget
The model is saturated on the given raw columns; the remaining headroom is almost certainly in richer, *stable* structure rather than more raw interactions. Specifically: (a) out-of-time validation (train on early 2005 months, validate on later ones) to pick features that survive the 2005→2006 shift instead of trusting the single 2006 eval slice; (b) airport-level exogenous proxies that do not depend on the year — e.g. origin traffic counts, hub flags, or scheduled-flight counts reconstructed from the row distribution — to replace the brittle identity encodings; (c) an aircraft-rotation feature if a tail number or arrival time were available (the strongest real driver of departure delay is the inbound leg); and (d) a stacking/weighted ensemble where member weights are fit on the out-of-time validation rather than simple averaging.
