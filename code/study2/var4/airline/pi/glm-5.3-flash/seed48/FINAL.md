# Final Report — airline delay (XGBoost, autoresearch harness edition)

**Best Eval AUC: 0.7459** (baseline 0.7141, +0.032). Final commit `1a13e11`, contract validated (`CONTRACT OK`).

## Final model
Ensemble of 40 XGBoost members (probability average). Members: depth 6/7/8 (cycled), 300 trees,
lr 0.04/0.055, subsample 0.9, colsample_bytree 0.9, colsample_bynode 0.8, `enable_categorical`,
each trained on a random 70%/85% row subset and 60%/80% column subset (cycled per member), 2 seed streams.
Features (all inside `prepare()`): raw columns as native categoricals; DepTime → hour24, sin/cos of
minutes-since-midnight, minute, 15-min block categorical (96), red-eye/early-morning flags;
day-of-year sin/cos, weekend, dom/dow numerics; log-distance; train-only counts (route, origin, dest).

## What mattered most (3–5 changes)
1. **Bagged ensemble of decorrelated XGBoost members** (row+column subsampling per member): 0.7168 → 0.7203,
   and scaling/diversifying members carried through the whole run (+~0.02 total with member tuning).
2. **Departure-time granularity**: minute, 15-min block categorical, holiday-window flags (+0.0072).
3. **Soft, deep members**: 300–400 trees at lr 0.03–0.055, depth 5–8 — much better than the shallow/fast
   regime the raw shift suggested (+0.006 over 100-tree members).
4. **carrier × hour interaction categorical** (480 levels) (+0.0042).
5. **Time/date features** (hour24, cyclic sin/cos, red-eye flags, doy) (+0.0014).

## What did not help
- **Target encoding** (OOF, carrier/origin/dest/route): 0.7023 — the 2005→2006 year shift makes per-level
  target statistics non-transferable.
- **High-cardinality interaction categoricals** (route 4.6k levels, carrier×qblock 1.9k): 0.7027 / 0.7322.
- **Stacking** (5-fold OOF committees + XGB meta-model): 0.7287 — the meta-model fit 2005-specific structure.
- Smaller misses: dow×hour and month×hour cats, day-traffic count, raw minutes-since-midnight numeric,
  extra member regularization (mcw/gamma/reg_lambda), 32 vs 48 members, logit averaging (tie).

## Theory
Eval (2006) is time-shifted from train (2005). Anything that memorizes 2005-specific level effects
(TE, sparse interactions, meta-models on OOF) degrades on 2006, while robust signals (time of day,
carrier schedule patterns, seasonality) transfer. Heavy averaging of moderately-deep, strongly
subsampled members is the right variance-reduction answer under this shift.

## With more budget
- Sweep the member depth/trees/lr grid more finely around (6/7/8, 300, 0.04–0.055) with repeated seeds
  to separate signal from the ±0.001 eval noise.
- Try per-member feature-family specialization (time-only vs route-only members) for stronger decorrelation.
- Quantile-binned DepTime as ordinal + monotone constraints on hour features.
- A careful, heavily-smoothed carrier×month TE as a *secondary* signal (small weight) — risky under shift.
