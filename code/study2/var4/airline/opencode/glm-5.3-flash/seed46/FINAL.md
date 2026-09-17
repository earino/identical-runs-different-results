# Final report — airline / XGBoost autoresearch

**Best Eval AUC: 0.7403** (baseline 0.7141, +0.0262 over 40 experiments). Final config: XGBClassifier,
depth 20, lr 0.01, 1500 trees, max_bin 512, subsample 0.9, colsample_bytree 0.4, min_child_weight 5,
hist, enable_categorical; features: time-of-day (sin/cos, hour, red-eye), cyclic calendar, distance/log,
3 string categoricals. `./validate.sh` → CONTRACT OK.

## Changes that mattered most
1. **Time-of-day feature engineering from DepTime** (hour, sin/cos of time-of-day, red-eye flag) + cyclic
   month/day-of-week encodings: 0.7141 → 0.7211 (+0.0070). Raw hhmm DepTime hides the strong non-linear
   delay pattern by hour; this was the single biggest gain.
2. **Capacity: depth 6 → 16 with subsample/colsample**: 0.7211 → 0.7310 (+0.0099 across exps 3/10-12).
   Deep trees with min_child_weight 5 were consistently better; depth kept helping up to 20.
3. **Learning-rate / tree-count schedule**: ES-based stopping always picked too many trees for the
   2005→2006 shift; fixed counts won. lr 0.05/300 → 0.01/1500: 0.7321 → 0.7360 (+0.004).
4. **max_bin 512**: 0.7310 → 0.7321 (+0.0011) — finer bins help the continuous DepTime/Distance signals.
5. **Column subsampling 0.7 → 0.4 (colsample_bytree)**: 0.7379 → 0.7397 (+0.0018, exps 34-36). Strong
   feature de-correlation per tree; the largest late-stage gain.

## Things that did not help
1. **Target encoding** (OOF, smoothed m=50, Origin/Dest/Carrier/route): 0.7251 vs 0.7270 — noise vs the
   year shift; also frequency counts alone hurt (0.7216).
2. **High-cardinality native categoricals**: route (Origin×Dest) as a category crashed AUC to 0.7148;
   even low-card calendar categoricals (month/dow/dom/hour) hurt (0.7253); carrier×hour was a tie.
3. **Ensembles/grow-policy**: 2-model config ensemble 0.7336, seed bagging 0.7371 (both < single best);
   lossguide (64 leaves) 0.7275. A single deep depthwise model dominated everything.

## With more budget
I would map the depth×colsample×subsample surface jointly (they interacted strongly), try a small
lr/iteration grid around (0.008–0.015) × (1000–2000) at depth 18–22, and revisit per-carrier feature
scaling (e.g. hour-of-day profiles per carrier built on train only). Given how consistently extra
categorical machinery hurt, the biggest remaining wins look like capacity/sampling tuning rather than
new features.
