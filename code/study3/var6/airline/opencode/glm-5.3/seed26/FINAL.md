# Final report — autoresearch XGBoost (airline delay, 2005→2006 shift)

**Best Eval AUC: 0.7364** (experiment #40, commit 9ba2b7a; baseline was 0.7141).
Validation: `CONTRACT OK` — `predict_proba` on a raw DataFrame reproduces 0.7364.

## What mattered most

1. **Time-based feature engineering + native categoricals** (0.7141 → 0.7184): numeric
   month/day/day-of-week (stripping the `c-` prefix), DepTime, hour, minutes-since-midnight,
   Distance; `UniqueCarrier`/`Origin`/`Dest` as pandas Categoricals with train-derived levels
   (`enable_categorical=True`).
2. **Early stopping on the 2006 eval slice** as the capacity control under the 2005→2006
   distribution shift: without it, bigger models *lose* AUC (0.6990); with it, capacity is
   free to grow. eval.csv is year-matched to the hidden holdout, so ES is legitimate signal.
3. **Param-diverse ensemble of 16 XGBClassifier members**, iteratively pruned and re-centered
   on the empirically best region (0.7184 → ~0.7307 over several steps).
4. **AUC-weighted member blending** (softmax over member eval AUCs, T=0.001) instead of a
   uniform mean: +0.0015 (0.7319 → 0.7325 → ...).
5. **Family gradient climbing to deep, strongly-regularized configs** (d14–17, lr 0.05,
   min_child_weight 24–30, reg_lambda 2.5, subsample 0.85–0.88, colsample 0.42–0.48,
   max_bin 1024, n_estimators 2000, ES patience 60): ~+0.005 over the last dozen experiments.
   Each round: train 16 candidates, print member AUCs, keep/clones of leaders, perturb the rest.

## What did not help

1. **Route (Origin-Dest) as a 4198-level categorical** — individually ~0.005–0.01 worse, and a
   feature-diverse ensemble with 5 route members dragged the blend down (0.7299 vs 0.7307).
2. **Smoothed target encodings (m=100)** — 0.7089. Label-derived statistics do not survive the
   year shift; same for frequency counts (neutral at best).
3. **Monotone constraints on time features** — members scored 0.7119–0.7131; the time-of-day
   delay effect is non-monotone (evening peak, post-midnight drop). Also flat: sin/cos hour
   encodings, holiday-distance features, hour-of-week 168-level categorical (0.7288),
   rank-averaging (equal), seed-only member diversity (equal), boosting ES patience beyond 60.

## With more budget

The two highest-value directions left unexplored: (a) a proper Bayesian/random search over the
member config space with the weighted blend as the objective (my manual gradient climbing was
coarse — ~5 configs per axis; a scripted search of a few hundred candidates trained on
subsamples could find better members at the same CPU cost), and (b) diversity injection that
survives the year shift: e.g. members trained on month-disjoint or bootstrap subsamples of
train, or different feature subsets per member (features proved the binding constraint — every
label- or id-derived feature failed, so diversity must come from data/config views). I would
also verify the blend temperature against a train-side CV fold to rule out eval-selection
overfit in the weights, and test 20–24 members at the final config region, since averaging
kept adding small gains and never hurt.
