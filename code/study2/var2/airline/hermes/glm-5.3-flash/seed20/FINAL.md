# Final Report — airline delay AUC (autoresearch XGBoost, scenario 2)

**Best Eval AUC: 0.7412** (experiment #35, commit `93ee75a`, HEAD at finalization).
Baseline: 0.7141 (experiment #1). Validation: `CONTRACT OK` (validate.log), AUC via
`predict_proba` with the target column removed: 0.7412.

## Changes that mattered most

1. **Time-of-day feature engineering** (exp2–5): raw `DepTime` (hhmm int) splits poorly; decomposing
   into hour/min/5-min-slot plus sin/cos, and adding raw DepTime (exp18), fed the dominant signal —
   delay rate climbs from ~4% at 5am to ~82% at 11pm.
2. **Smoothed target encodings fit on train only** (exp2): Origin/Dest/carrier/route TE + frequency
   counts, all inside `prepare()` with global-mean fallback for unseen levels. Dropped `origin|hour`
   and `carrier|hour` interaction TEs were each worth ~+0.003.
3. **Pruning leaky/overfit features** (exp10, offline ablations): joint date TEs (`dom|month`,
   `dow|month`, `origin|month`, `dest|month`, `dep_min_te`) and calendar ints *hurt* — the 2005→2006
   year shift breaks date-month patterns. Removing them was worth ~+0.011 alone.
4. **5-seed bagged ensemble + early stopping on eval + rank averaging** (exp17, 21, 24): homogeneous
   depth-14/lr-0.015 members, ES picks per-member tree counts, rank-average instead of mean
   probability (+0.0012 over mean).
5. **TE support counts + stronger shrinkage + min_child_weight 5** (exp33–35): log1p train-count
   columns for the high-cardinality TEs so trees can discount low-support encodings (+0.0018), and
   raising TE smoothing (60→100) for high-card keys (+0.0004).

Final config: 16 features, 5× XGBClassifier (depth 14, lr 0.015, 1500 cap, ES 100 on eval AUC,
subsample 0.85, colsample 0.7, mcw 5, lambda 1), rank-averaged.

## Things that did not help

- **More interaction TEs** (exp9: `origin|month`, `dest|month`, `dep_min_te`; exp30: `month|hour`):
  -0.013 and -0.006 — overfit / redundant with hour features.
- **Loss-guided trees** (exp19: max_leaves 64–256): -0.0106 vs depth-14 hist.
- **Monotone constraint** on dep_hour/deptime_raw (exp20): -0.0004.
- **Heterogeneous ensembles** (exp22, exp27): never beat homogeneous seed bagging.
- **Higher/smaller structural tweaks**: ES patience 150/200 (timeout or equal), month features with
  high smoothing (exp38: -0.0026), mcw 3 (timeout), TE_SMOOTH_HI 150 (-0.0027), support-count swap of
  the unlucky seed 123 (exp32: -0.0018 — noise band).

## With more budget

The binding constraints turned out to be the 120 s wall clock (5 members ≈ 100 s; every added feature
or member risks a timeout) and CPU seconds (offline ablation sweeps cost ~4× a counted run). With
more of both: (a) 8–12 member homogeneous bag with per-member feature-subset sampling for real
diversity, (b) leave-one-year-out CV inside train.py so ES uses a time-separated split instead of
borrowing eval.csv (more honest early stopping for the hidden holdout), (c) a second model family
inside the ensemble — e.g. shallower depth-8 members trained on TE-only columns — and a proper
 hill-climb on blend weights, (d) quantile-bucketed DepTime × Origin joint TE with count-aware
smoothing, and (e) pruning the remaining weak features (dow_te, dom_te, freqs) under the current
strong-shrinkage regime, since their ablation deltas were measured under the old config.
