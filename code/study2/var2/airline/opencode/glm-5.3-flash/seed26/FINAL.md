# Final Report — airline / XGBoost autoresearch

**Best Eval AUC: 0.7440** (baseline 0.7141, +0.0299 over 40 experiments; final commit `def9158`, validated `CONTRACT OK`).

## Final model

16-member seed-bagged XGBoost (native `xgboost.train`, shared DMatrix, `hist`): lr 0.05, max_depth 10,
min_child_weight 20, subsample 0.75, colsample_bytree 0.65, max_bin 512, 450 rounds each, trained on all
100k rows of `data/train.csv`. Features: int-encoded Month/DayofMonth/DayOfWeek, DepTime decomposed to
minutes-since-midnight (+ sin/cos, hour), log-Distance, native categoricals for UniqueCarrier/Origin/Dest,
and the key engineered feature **UniqueCarrier × 15-minute departure block** (960 levels). All feature
engineering lives inside `prepare()`, which `predict_proba()` applies to unseen data; encoders/levels are
fit on training rows only.

## What mattered most (3–5 changes)

1. **Carrier × time-of-day interaction as a native categorical** (exp12 hourly +0.008, exp19 30-min +0.006,
   exp20 15-min +0.004 → cumulative ~+0.018). Carrier-specific delay cascades by departure time are the
   dominant signal beyond the base features.
2. **Deep-but-regularized trees** (exp6–7: depth 9→10 with subsample 0.75–0.8, colsample 0.6–0.7,
   min_child_weight 10–20, +0.006). The 2005→2006 shift rewards regularization; shallow models underfit.
3. **Seed bagging** (exp10: 5 seeds +0.003; exp16/33: 10→16 seeds +0.001) — variance reduction that
   transferred cleanly.
4. **Time/season numeric features** (exp3: int-encoded month/dom/dow, dep_min, dep sin/cos, log-distance,
   +0.002 over string categoricals).
5. **max_bin 512** (exp35, +0.0005) — finer histograms sharpen splits on the dense dep-time feature.

## What did not help

1. **Target encoding + counts** for carrier/origin/dest/route (exp4 leaky refit 0.7135, exp5 leak-free
   5-fold OOF bag 0.7144): per-airport/carrier delay rates do not transfer from 2005 to 2006.
2. **Route (Origin→Dest) and other sparse interactions** as categoricals (exp11 0.7153, exp14 origin×hourbin
   0.7289, exp18 dow×hour 0.7287): ~20 rows/level is too sparse, and redundant interactions dilute colsample
   (exp13 carrier×dow/month crashed to 0.7144).
3. **Early stopping on an internal split with refit** (exp2 0.7081) and higher learning rates (exp22 lr 0.1,
   exp24 lr 0.07 both ~−0.002–0.005): the 10% 2005 validation split mis-estimates the optimum for 2006;
   a fixed, directly-tuned round count (exp27–31: 850→450, monotone improvement to 0.7433) was better.

## With more budget

I would (a) extend the interaction idea to **Origin × departure block with coarser bins** (exp14 used 6
bins and lost; 3–4 bins × hub-only airports might avoid sparsity), (b) try **DART or dropout-style bag
members** for more decorrelation than seeds alone, (c) run a proper small grid over colsample/min_child_weight
at the fixed 450-round config, and (d) test whether dropping the weakest features (dom sin/cos) with
colsample raised to 0.7 recovers the exp36 loss. I would also investigate why in-sample early stopping
wants ~1000 trees while eval peaks at ~450 — a sign that 2005-val AUC and 2006 AUC diverge, which a
time-aware validation scheme (e.g., month-blocked CV) might resolve.
