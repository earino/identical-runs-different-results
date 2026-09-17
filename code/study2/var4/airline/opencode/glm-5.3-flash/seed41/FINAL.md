# Final Report — airline delay AUC (XGBoost, time-shifted 2005→2006)

**Best Eval AUC: 0.7290** (baseline 0.7141; +0.0149). Final model: 10-seed ensemble of
XGBoost `grow_policy=lossguide, max_leaves=96, 800 trees, lr=0.03, subsample=0.6,
colsample_bytree=0.4, colsample_bynode=0.8, min_child_weight=20, gamma=2, lambda=10,
alpha=1, max_bin=512`.

## Changes that mattered most

1. **Slow/shallow/heavy-regularization regime** (800 trees, lr 0.03, subsample 0.7,
   colsample 0.6, lambda 10, mcw 20, gamma 2): 0.7157 → 0.7220. The 2005→2006 year shift
   punishes sharp fitting; many small steps generalize.
2. **Stronger subsampling** (subsample 0.6, colsample_bytree 0.4): → 0.7259. More
   randomness per tree = better cross-year generalization.
3. **Lossguide grow policy** (max_leaves 96 instead of max_depth 6): → 0.7286.
   Loss-prioritized leaf growth beats depth-limited growth in this regime.
4. **Feature engineering** (dep_min minutes-since-midnight + hour categorical + cyclical
   sin/cos for month/day/dayofweek + log-distance + weekend/dom/distance-bin/midnight
   flags + train-only label-free count features): 0.7141 → 0.7157 at fixed small model,
   and it carried through every later gain.
5. **Ensembling + colsample_bynode**: 10 seeds (+0.0002) and per-split feature
   subsampling 0.8 (+0.0004) → 0.7290.

## Things that did not help

- **Target encoding** (naive and leak-free OOF-half variants, smoothing 30–50): hurt
  badly (−0.005 to −0.011). 2005-derived per-airport/route delay rates do not transfer
  to 2006; raw categorical splits are more robust.
- **Interaction categoricals** (route, carrier×hour, dow×hour high-card; and low-card
  carrier×dist_bin, weekend×hour, month×weekend): all hurt; XGBoost finds interactions
  from base features fine at depth 6.
- **Learning-rate decay, more trees (1200), max_leaves cap on hist, lambda 20,
  min_child_weight 30–40, gamma 3, deeper/shallower trees**: all neutral-to-worse.

## With more budget

I would try: (a) larger diverse ensembles mixing lossguide leaf budgets and feature
subspaces; (b) per-row covariate-shift reweighting between 2005/2006 feature
marginals (label-free); (c) quantile/binned DepTime with interaction-aware binning;
(d) nested CV ensembling with different column subsets; (e) hyperparameter search
around the lossguide regime (lr, leaves, mcw jointly, which single-axis probes
cannot see).
