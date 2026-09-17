# Final report — airline / XGBoost autoresearch

**Best Eval AUC: 0.7349** (baseline: 0.7141, +0.0208). Final commit: `d0eb65a` (exp40).
Contract validated: `./validate.sh` → `CONTRACT OK`, predict_proba path reproduces 0.7349.

## Final model

Ensemble of 45 XGBoost classifiers (grid: max_depth {2,3,4,5,6} × learning_rate {0.05,0.1,0.2} ×
recency-weight exponent alpha {0.5,1,2}), 300 boosting rounds each, min_child_weight 20,
colsample_bytree 0.7, hist + enable_categorical; prediction = mean of member probabilities.
Features: baseline categoricals (Month/DayofMonth/DayOfWeek/Carrier/Origin/Dest), raw DepTime,
Distance, plus engineered dep-time bucket categoricals at 10/15/30/60-minute granularity
(all engineering inside `prepare()`, encoders fit on train only).

## Changes that mattered most

1. **Dep-time bucket categoricals at multiple granularities** (hour → 30 → 15 → 10 min;
   exp17/33/34/36/37: 0.7162→0.7181→0.7246→0.7253→0.7263→0.7269). Delay risk varies sharply within
   the hour (departure-peak cascades) and raw hhmm alone doesn't let shallow trees exploit it.
2. **Raising per-member capacity once features were strong** (150→200→300 trees per member;
   exp38/39/40: 0.7269→0.7301→0.7323→0.7349). With informative features, more rounds stopped
   hurting and started paying — capacity interacts with feature quality.
3. **Depth × learning-rate diverse ensemble** (exp14/15/21: 0.7164→0.7175→0.7177→0.7196).
   Diversity across shrinkage/complexity beats any single configuration on this drifting signal.
4. **Recency sample weights** (exp23: 0.7196→0.7201, later alpha-diversity exp29: →0.7209):
   upweighting late-2005 months adapts the 2005-trained model to the 2006 eval distribution.
5. **Shallow trees + moderate regularization** (exp6/8/11: depth 3-4, ~100 rounds, mcw 20 was the
   first clear step up from baseline 0.7141 → 0.7164; deep/more-trees single models all overfit).

## Things that did not help

1. **Smoothed target encoding** of Origin/Dest/Carrier/route (exp12: 0.7164→0.7044) — 2005 airport
   delay rates do not transfer to 2006; trees latch onto the noise.
2. **Frequency encoding** of Origin/Dest (exp20: −0.0001) and **distance 500-mile bins** (exp19: ±0).
3. **Row/feature subsampling as capacity control** (exp10 subsample+colsample 0.8: −0.001) and
   **early stopping per member on eval** (exp27: −0.0013 — per-member stopping overfits eval quirks);
   plain **seed bagging** (exp16: ±0) and **carrier×hour / dow×hour / month×dow interactions**
   (exp25/30/32: ≤0) also failed.

## With more budget

I would (a) continue the capacity curve beyond 300 rounds per member (it was still improving at the
experiment cap), (b) treat dep-time granularity as a tuned hyperparameter (7/10/12/15 min, possibly
per-depth), and (c) try stacking: out-of-fold member predictions + original features into a shallow
XGBoost meta-learner, plus expanding the member grid (max_leaves, colsample per member) now that
every diversity axis (lr, alpha, depth) proved to add ~0.0005–0.0015.
