# Final Report — autoresearch XGBoost (airline delays)

**Best Eval AUC: 0.7421** (commit c61ff16, experiment #36; baseline 0.7141, +0.028 total)
Validation: `CONTRACT OK` (validate.sh re-ran training and scored 0.7421 through `predict_proba`).

## Final architecture

Diversity ensemble of 11 XGBoost classifiers (n_estimators=100, lr=0.03, hist, enable_categorical)
spanning depths {3, 6, 12, 18, 24} x colsample_bytree {1.0, 0.75, 0.6, 0.5, 0.3}, combined by
**weighted rank averaging** (deep + strong-colsample members weighted 1.5). Features: raw DepTime,
cyclical time-of-day sin/cos (24h and 48h periods), month/dom/dow as numerics plus cyclical sin/cos,
Distance, log Distance, DepTime-x-Distance proxy, carrier/origin/dest as native categoricals, and
smoothed target encodings (smoothing=20, fit on train only, applied inside `prepare()`).

## Changes that mattered most

1. **Ensembling across depths (d6+d3+d12, exp #8–9)**: +0.012 over the single best model. Single deep
   models (d8, d6/lr0.05) all lost to the baseline; averaging diverse structures is where the gain was.
2. **Very deep members (d18, d24; exp #19–20)**: +0.005. Deep trees on 100k rows capture sharp
   carrier/airport/time interactions that depth-6 cannot.
3. **Colsample_bytree diversity (0.5, 0.3; exp #23–24)**: +0.008. Feature-subsampled members
   decorrelate the ensemble — the single largest post-ensemble lever.
4. **Rank averaging + member weighting (exp #22, #32)**: +0.001 combined. Robust to different
   probability scales across depths; mild upweighting of the strongest members helped.
5. **Cyclical time-of-day/calendar features + smoothed target encoding (exp #4, #14)**: +0.004/+0.0004.
   Modest but stable; TE carried some of the later ensemble gains.

## Things that did NOT help

1. **Early stopping on a random 20% validation split** (exp #2): 0.7039 — trained on less data; ES on a
   time-separated problem with random splits is anti-creative here.
2. **Route categorical (Origin_Dest)** (exp #12): 0.7137 — too sparse; native categorical Origin/Dest
   already covers it.
3. **Row bagging / subsample=0.8** (exp #18, #29): both slightly negative; the members are already
   regularized by depth/colsample diversity.
4. **min_child_weight=5 and reg_lambda=10** (exp #31, #35): 0.7380 / 0.7360 — pruning the deep members
   hurts much more than it helps generalization.
5. **hour-x-dayofweek TE interaction, dom==31 flag, deptime-missing flag** (exp #16, #17, #30): all
   equal-or-worse; the deep members already learn these interactions from raw features.

## With more budget

- **Quantile/hazard-style DepTime binning** and airline-specific time-of-day (carrier x hour TE) — the
  DepTime x Distance proxy was the best single new feature of the last 15 experiments, suggesting
  time-context interactions are under-exploited.
- **Stacking**: fit a logistic meta-learner on out-of-fold member predictions instead of fixed rank
  weights (needs CV inside the 120s budget — the current 11-member x 100-tree ensemble already runs at
  ~106s, so members would have to be trimmed or shared across folds).
- **Larger ensemble with lower per-member trees** (e.g. 20 members x 40 trees at lr 0.08) to push
  rank-averaging variance further down within the timeout.
