# Final report — airline delay (XGBoost, hidden-holdout scoring)

**Best Eval AUC: 0.7192** (eval = 2006-slice1-100k; experiment #38, commit 3db6c2d content at HEAD).

## Final model

Rank-averaged bagged ensemble of **200 small XGBoost classifiers** trained on 90% row subsamples,
diversified across members by cycling `max_depth {3,4,5,6}`, `learning_rate {0.05,0.1,0.15}`,
`n_estimators {80,120,180}` (12 distinct combos), plus an engineered `hour = DepTime // 100` feature.
All feature engineering lives inside `prepare()` (target-encoding-free; only `hour` added to the raw
categorical/numeric columns; unseen categorical levels map to NaN, which XGBoost handles natively).

## Changes that mattered most (baseline 0.7141 → 0.7192)

1. **Bagging small members** (7 × 30-tree models on 80% row subsamples, proba-averaged): 0.7141 → 0.7146.
   Single-model capacity increases *hurt* monotonically on this time-shifted task (100 trees mild reg:
   0.7118; 2500 trees depth 8: 0.7023), so ensembling many *small* models was the only capacity axis that paid.
2. **Member diversity**: depth cycling {5,6,7} (+0.0005), learning-rate cycling {0.05,0.1,0.15} (+0.0003),
   n_estimators cycling {30,50,80} → later {50,80,120} → {80,120,180} (+0.0008/+0.0004). Diversity axes were
   the single biggest lever after bagging itself.
3. **Rank-averaging instead of probability-averaging** (+0.0001–0.0002): members at different lr/depth have
   different probability spreads; averaging ranks weights them equally for the AUC metric.
4. **Tuning the member operating point**: bag fraction 0.9 (vs 0.65/0.8: stronger members beat more diversity),
   depth window shifted down to {3,4,5,6} (+0.0003), trees window shifted up to {80,120,180} (+0.0004).
   Consistent theme: members should be shallow but reasonably long-trained.
5. **`hour = DepTime // 100`** (+0.0001, only on the final strong ensemble): quantized time-of-day helps once
   ensemble capacity is sufficient; it never helped the small single models.

## What did not help

1. **Target encoding** of Origin/Dest/route (smoothed m=50, leave-one-out in-train): 0.6697 — much worse.
   2005 airport/carrier delay propensities do not transfer to 2006; baked-in priors actively mislead.
2. **More capacity in a single model**: 100 trees with mild regularization 0.7118; depth-8/2500-tree model
   0.7023. The 2005→2006 shift punishes fitting 2005-specific detail.
3. **Stronger/weaker bag members at the extremes**: bag fraction 0.65 (0.7154), depth-8 members (0.7162),
   max_bin diversity {64,128,256} (0.7153), min_child_weight diversity {1,20,100} (0.7161), lr down-shift
   {0.03,0.05,0.1} (0.7173), depth down to {2,3,4,5} (0.7174) — all below the then-best.

## With more budget

- **Feature work targeted at the shift**: the TE failure suggests year-lag-robust encodings (e.g., encoding
  computed with temporal decay inside 2005, or rank/frequency features instead of raw target means).
- **Time-block bagging**: members trained on different *temporal* slices of 2005 (untested; ran out of slots)
  to further mimic the year shift.
- **Larger member pools via faster training**: at 200 members the run is ~105 s (limit 120 s); a cached DMatrix
  or lower max_bin would buy more members/combos.
- **Greedy member selection** on a pseudo-time validation slice (train on early 2005, validate on late 2005)
  to prune harmful member types instead of averaging all.
