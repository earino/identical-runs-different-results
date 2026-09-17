# FINAL — airline delay AUC

**Best Eval AUC: 0.7483** (baseline 0.7141), ensemble of 3 XGBoost models, HEAD = `376d85b`.
Contract validated (`CONTRACT OK`, predict_proba reproduces 0.7483 with target column removed).

## What mattered most

1. **Deep, heavily column-subsampled trees** (max_depth 16→20, colsample_bytree 0.7→0.5, max_bin 512→1024,
   600 trees @ lr 0.05): +0.010 over a sane d8 config. The biggest single lever; shallow/regularized settings
   left far too much signal unexploited.
2. **Removing redundant noisy encodings of month/day-of-week**: dropping the cyclical sin/cos month+dow
   columns (and the month_dow/route native categoricals earlier) gave +0.003. Smooth encodings of
   non-smooth calendar effects hurt; plain numeric month/dom/dow + hour-categorical is what works.
3. **Frequency (count) features** for carrier/origin/dest/route: +0.002, and unlike target encodings they
   transferred across the 2005→2006 time shift.
4. **reg_alpha=1.0** on top of the tuned config: +0.002 (single model 0.7479).
5. **3-model ensemble** (d20/c0.5/bin1024, d12/c0.7/bin512, d20/c0.5 seed 7): +0.0004 on top (0.7483).

## What did not help

- **Target encodings** (carrier/origin/dest/route/carrier×hour, various smoothing): consistently −0.002 to
  −0.005 on the out-of-time eval set despite small CV gains — 2005→2006 drift makes them brittle.
- **High-cardinality native categoricals** (route with ~4200 levels, month×dayofweek): −0.01 to −0.04;
  they let the deep trees memorize noise.
- **Feature bloat that looked promising**: distance bins, hour/month/dow traffic counts, cyclical
  month/dow — all neutral or negative once the strong model config was in place.

## With more budget

I would (a) run proper 5-fold-CV-based early stopping per ensemble member to set tree counts instead of a
fixed 600, (b) grow the ensemble to 6–8 members with more diverse depths/colsample/seeds under a relaxed
time limit, (c) explore per-carrier or per-hour stratified models stacked into the ensemble, and (d) test
DART members for diversity. The 18k CPU-second budget was exhausted by these experiments (diag batches +
official runs), so config search had to stop at the sweet spot found above.
