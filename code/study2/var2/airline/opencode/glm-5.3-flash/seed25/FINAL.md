# Final Report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7166** (baseline 0.7141; experiments.tsv #31 / commit 729b4bc).

## Final model
3-seed bag of `XGBClassifier`: 30 trees, depth 6, lr 0.15, subsample 0.8, colsample 0.8,
min_child_weight 20, gamma 1.0, reg_lambda 10, `tree_method=hist`, `enable_categorical=True`
(raw string categoricals; no feature engineering beyond the raw columns).

## Changes that mattered most
1. **Leaf/branch regularization** (min_child_weight 20 + gamma 1.0): 0.7145 → 0.7158. The single
   biggest lever; the model is strongly capacity-limited by the 2005→2006 temporal shift.
2. **L2 regularization reg_lambda=10**: → 0.7162. More λ (30) hurt.
3. **3-seed bagging** of the tuned config: → 0.7165. Pure variance reduction.
4. **lr 0.15 at fixed 30 trees**: → 0.7166 (marginal, within noise).
5. **subsample/colsample 0.8** (early, exp #5): 0.7141 → 0.7145.

## Things that did NOT help
1. **All feature engineering** (numeric+cyclical time features, red-eye flag, log-distance,
   dow×hour interactions, dep_hour categoricals): every variant lost 0.002–0.004 AUC.
2. **Target and frequency encodings** of Origin/Dest/UniqueCarrier/route: TE lost ~0.009
   (2005-encoded rates do not transfer to 2006); frequency encoding also lost ~0.001.
3. **More trees / early stopping**: early stopping on an internal 2005 split picked 334 trees
   (val AUC 0.7515) but eval AUC dropped to 0.7117 — internal validation overestimates
   transfer; 30 trees stayed optimal. Depth 4/5/8, lr 0.2, max_bin 64, bigger bags,
   mixed-depth and lr-diverse ensembles all landed at or below 0.7166.

## Theory and what I would try with more budget
The task is dominated by distribution shift: 2005 training data with ~0.75 internal AUC
transfers to only ~0.716 on 2006, so anything that fits 2005-specific structure (extra trees,
engineered features, label-dependent encodings) actively hurts. The winning strategy was
shrinking model capacity until the remaining signal was the transferable part (time-of-day,
carrier, airport, distance, calendar effects). With more budget I would (a) explore
monotonic constraints on DepTime/Distance to force transferable shapes, (b) try a two-stage
model trained only on feature subsets shared across years, (c) run a proper random search
over the regularizer simplex (mcw × gamma × λ × colsample jointly) with repeated-seed
averaging to separate noise from signal, and (d) test a single deeper-but-heavily-L2-regularized
model as an alternative capacity shape.
