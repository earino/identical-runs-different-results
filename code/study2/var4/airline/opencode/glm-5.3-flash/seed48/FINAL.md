# Final report — airline delay (XGBoost binary classifier)

**Best Eval AUC: 0.7294** (baseline 0.7141), validated with `CONTRACT OK`.

Final model: ensemble of 16 XGBoost classifiers, heterogeneous members (max_depth 6–9 ×
colsample_bytree 0.7/0.8, 2 seeds per config). Each member: 85/15 random split → early stopping
(patience 50, logloss) → refit on the full training set with the ES tree count. Final prediction =
mean of member probabilities. Features: scheduled time-of-day (minutes-since-midnight, sin/cos,
hour), ordinal month/day-of-month/day-of-week, log-distance, carrier/origin/dest + route as
categoricals, carrier×hour and dow×hour crosses, and log-frequency (volume) counts for
carrier/origin/dest/route.

## Changes that mattered most

1. **Time-of-day feature engineering** (tod_min, sin/cos encoding, hour): +0.009 vs baseline —
   by far the biggest lever; raw hhmm DepTime is a poor split variable.
2. **Seed ensembling with ES + refit** (5 → 8 members): +0.002; averaging members removes the
   variance of single-split early stopping (individual members scored 0.718–0.723).
3. **Heterogeneous members** (depth/colsample mix): +0.001 over same-config seeds (16 members: 0.7294).
4. **Early stopping + refit on full train** with n_estimators=3000 budget: core of every good run.
5. **Log-frequency volume features** and small crosses (carrier×hour, dow×hour): +0.0003–0.0005 each.

## Things that did not help

1. **Target encoding in any form**: LOO encoding leaks catastrophically (val AUC 0.0002 logloss-chasing,
   eval 0.60); K-fold OOF TE bundle scored 0.7172; OOF route-TE alone scored 0.7268 ≈ base.
2. **Route as a raw 3000-level categorical**: accelerated memorization, early stopping collapsed (0.7136).
3. **Regularization knobs**: min_child_weight=20 (0.7268), gamma=1 (0.7288), subsample 0.7 (0.7290),
   lr 0.03 (0.7277), bagged 80%-row refits (0.7282) — all ≤ base.
4. **Seasonal/congestion features**: month×hour (0.7201), origin/dest-hour counts (0.7288),
   month×dow + carrier×distbin crosses (0.7225) — 2005-fitted seasonal patterns don't transfer to 2006.
5. **AUC-based early stopping** (0.7179 vs 0.7228 logloss) and rank-averaging (0.7290 ≈ mean).

## With more budget

I would try: (a) monotonic constraints on time-of-day features to force transferable shapes;
(b) a proper stacking layer (member OOF predictions → small XGBoost meta-model) with nested CV;
(c) per-member time-aware validation splits (train on early 2005, stop on late 2005) to pick
tree counts under a real distribution shift; (d) quantile/huber-style objectives averaged into
the ensemble for probability calibration diversity.
