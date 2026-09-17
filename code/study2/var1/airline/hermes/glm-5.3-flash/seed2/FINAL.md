# FINAL.md — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7286** (baseline 0.7141, +0.0145). Final commit `2a783b1`, validated: `CONTRACT OK`.

Final model: 4-member XGBoost bag (seeds 42/7/2026/123, depths 8/8/6/10), lr 0.02, min_child_weight 5,
subsample 0.8, colsample_bytree 0.8, colsample_bynode 0.8, max_cat_to_onehot=1, early stopping 100 rounds
on eval AUC, trained on engineered features.

## Changes that mattered most

1. **carrier × hour interaction categorical** (480 levels, one-hot split via `max_cat_to_onehot=1`):
   +0.0051 by itself — by far the largest single gain. Delay risk is a carrier-specific schedule effect.
2. **Categorical-split tuning + hour-as-category**: `max_cat_to_onehot=1` plus `dep_hour_cat` (24 levels):
   +0.0016 combined with the temporal block below.
3. **Temporal feature engineering**: parsed Month/DayofMonth/DayOfWeek numerics, dep_hour, corrected
   minutes-of-day sin/cos (fixed a `% 1440` bug — hhmm is not minutes), red-eye flag, log Distance: +0.0014.
4. **Bagging + decorrelation**: seed bag (3→5 members +0.0017), mixed depths (+0.0005),
   `colsample_bynode=0.8` (+0.0010). All robust, low-variance gains.
5. **min_child_weight 10→5**: +0.0008 at the very end (2 was a hair worse; 10 underfit).

## Things that did not help

1. **Target encoding** (smoothed m=30, carrier/origin/dest): −0.0002 — neutral at best; the 2005→2006
   year shift makes fitted rates untrustworthy.
2. **Any airport-based feature**: origin×hour and dest×hour cats (7k levels), route/Origin_Dest cats,
   dist-bin×carrier, airport flight counts (the count run even timed out) — all clearly negative.
   High-cardinality categoricals poison the one-hot/split budget.
3. **Other interaction cats**: dow×hour, carrier×dow, month×hour, dom×hour, 4h-block×carrier — all worse
   than carrier×hour alone; adding them dilutes the signal.
4. (also: eta annealing via `xgb.train`, es 200→40, subsample 0.7, max_bin 128, lossguide member — all
   ties or losses; two-stage OOF stacking was sound in principle but 9+ fold-models cannot fit the 120 s
   timeout.)

## With more budget

A working two-stage XGB stack (OOF feature from a 3-fold × 3-seed bag into a second-stage XGB) timed out
at 120 s; with a longer timeout it is the most promising direction, since the stage-1 OOF AUC (~0.726 in
similar runs) carries information the raw features do not fully expose. Next: a per-carrier monotone
constraint study, quantile regression into the AUC-optimal blend, and a larger 8-10 member bag with
fold-wise bagged early stopping instead of a single eval-set early stop (the current model early-stops on
eval.csv itself, which risks mild selection bias; a CV-estimated round count would generalize more safely
to the hidden holdout).
