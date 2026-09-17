# Final Report — airline / XGBoost autoresearch

**Best Eval AUC: 0.7356** (baseline 0.7141, +0.0215). Final model: 5-seed bagged XGBoost,
`n_estimators=300, max_depth=14, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
colsample_bylevel=0.8, reg_alpha=6, max_bin=512, tree_method=hist, enable_categorical=True`.

## Changes that mattered most

1. **Strong L1 (`reg_alpha`) enabling deep trees** — the single biggest axis. Raising alpha from 1
   (default) to 5–10 allowed max_depth 8→14 without year-shift overfitting: 0.7141 → ~0.725.
   Alpha prunes trees to few high-value splits; deeper trees + sparsity beat shallow dense ones on
   the 2005→2006 shift.
2. **Hour-of-day as an explicit categorical** (next to raw hhmm DepTime): +~0.002. The raw integer
   conflates hour and minute; the model couldn't isolate hour effects cleanly.
3. **Minute-of-hour as an atomic numeric feature**: +~0.0034 (0.7317 → 0.7356). Same lesson as (2):
   decompose the composite `DepTime` integer. Categorical/bucketed minute variants were all worse;
   plain numeric minute was best.
4. **Time-of-day features** (Tod bins, sin/cos cyclic time, log-distance, Origin_Dest route
   categorical) added on top of the deep-sparse recipe: +~0.002 (0.7250 → 0.7271). Notably, Route
   as a 4198-level categorical *hurt* shallow models but was neutral-to-positive once trees were
   deep and alpha-regularized.
5. **5-seed bagging + mild subsampling** (subsample/colsample 0.85, colsample_bylevel 0.8):
   +~0.0005-0.001 and more robust to eval noise.

## Things that did not help

1. **Target encoding** of any key (route, origin, dest, carrier, hour, dow). Helped in-year CV
   (+0.008) but hurt or was neutral on 2006 eval: per-level delay rates are unstable across years
   (measured: origin/route level-rate correlation 2005 vs 2006 ≈ 0.3; hour/dow ≈ 0.96).
2. **Count/frequency features** (flights per route/origin/dest/hour): actively harmful (0.6999 vs
   0.7159 with the then-current recipe).
3. **Capacity without sparsity**: depth 7-8 at alpha 1, more trees at lr 0.1, lossguide, DART,
   max_leaves caps — all worse. Also depth-diversity ensembles (d10/14/18) ≈ no gain over seed bags;
   single-depth alpha-mix was better.
4. Day-of-year / holiday / seasonal flags and a Dow×Tod cross: diluted the Month/DoM signal (0.7283
   vs 0.7303).
5. max_bin 1024, max_leaves caps, colsample 0.7, subsample 0.9/0.75, mcw 10-20, gamma, lambda — all
   neutral or worse.

## With more budget

- A proper sweep of the interaction between alpha, depth, lr and n_estimators around the final
  recipe (the alpha×depth frontier was still moving when CPU budget ran out), including 2-3 alpha
  values per depth in the bag.
- Learned-robust encodings for high-cardinality keys: hierarchical/bayesian shrinkage of
  origin/dest toward hub aggregates, using *counts only* (stable) not outcome rates.
- Isotonic-calibrated blend of a deep-sparse model and a shallow-dense model (feature-view
  diversity), and a larger bag (9-15 seeds) since single-experiment runtime (≈90-115 s) was the
  binding constraint, not CPU.
- DepTime anomaly values (2400-2620) as an explicit flag; minute-wave interaction features
  (minute × hour-slot congestion proxies built from counts, not outcome rates).
