# Final report — airline (XGBoost autoresearch)

**Best Eval AUC: 0.7448** (experiment #40, commit `0cee0e3`; validated: `CONTRACT OK`, predict_proba reproduces 0.7448 on eval.csv with target removed).

## Final model

Bagged ensemble of 2 XGBoost models (probability-space average), each fit on the full 100k-row train set with a
fixed 523 trees (n tuned by early stopping on an internal 80/20 split):

- Member A: depth 22, lr 0.01, subsample 1.0, colsample_bytree 0.8, seed 42
- Member B: depth 20, lr 0.01, subsample 0.7, colsample_bytree 0.5, seed 7
- Shared: `max_bin=2048`, `min_child_weight=3`, `reg_lambda=1.0`, `tree_method=hist`, `enable_categorical=True`

Features (all inside `prepare()`, statistics fit on train only): Month/DayOfWeek kept as native categoricals
(+ numeric copies), DayofMonth numeric, DepTime parsed to minutes-since-midnight (mod 1440) → `dep_min`,
`dep_hour`, `dep_minute`; UniqueCarrier/Origin/Dest native categoricals; raw Distance.

## Changes that mattered most

1. **Time parsing** (+0.0077, exp 3): replacing the raw hhmm DepTime integer with `dep_min`/`dep_hour`/`dep_minute`
   and the c-N strings with ints was the single biggest feature win.
2. **Depth/capacity ramp** (+0.017 cumulative, exps 8–12): depth 8→18 (with lr 0.05→0.01) — deep trees exploit
   native categorical splits and time features; gains flattened at depth 20–22.
3. **2-member bagging** (+0.004, exp 17): averaging two depth-20 models with different seed/subsample/colsample
   beat every single model; a *weaker-depth* second member (depth 10) hurt instead.
4. **max_bin 256→2048** (+0.0015, exps 19/24/30): finer histograms help the numeric time/distance features.
5. **Member A subsample 1.0** (+0.0003, exp 35) and final member-A depth 22 (+0.0001, exp 40).

## Things that did not help

- **Route (Origin_Dest) categorical + frequency encodings** (exp 4, −0.012): high-cardinality categorical overfit.
- **Target encodings**, both in-fold (exp 6, −0.005) and out-of-fold 5-fold (exp 7, −0.004): native categoricals
  already capture group effects; TE added non-transferable train-year-specific noise.
- **Duplicated/derived time features** — month/dow as extra categoricals, weekend flag, week-of-month, hour_cat
  (exp 14, −0.004); stronger regularization (mcw 5–10, gamma 1, lambda 5, exps 15/27/28); lossguide at matched
  tree budget (exp 22, −0.02); logit-space averaging and weighted averaging (exps 21/32); 3rd ensemble member
  (exp 18).

## With more budget

I would try: (a) k-fold bagging with more members on more data each (needs longer runtime than the 120s cap
allows); (b) stacking the two members' OOF predictions with a small XGBoost meta-learner; (c) systematically
re-testing high-cardinality route/carrier-interaction features at the deep-tree operating point, since the
early capacity-level rejections may not transfer; (d) per-year calibration — the train/eval year shift (2005→2006)
suggests monitoring drift-sensitive encodings only, e.g. through smoothed frequency features rather than TE.
