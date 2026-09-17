# FINAL REPORT

Best Eval AUC: **0.7431** (baseline 0.7141, +0.029)

Final model: probability-average of 4 XGBoost models — {d13_bl95_n600 x seeds 42,43} + {d3_bl90_n500 x seeds 42,43}
— on the raw columns minus Month and DayofMonth. Config shorthand: d = max_depth, bl = colsample_bylevel,
n = n_estimators; shared: learning_rate 0.03, subsample 0.95, colsample_bytree 0.95, max_bin 512, hist,
native categoricals. All feature engineering lives inside `prepare()`; encoders fitted on train only.

## Changes that mattered most

1. Dropping the Month column (+0.0023 at fixed model): 2005's seasonal delay pattern actively hurts on 2006.
   DayofMonth also dropped later (+0.0003, a simplification win).
2. Cross-depth ensemble with subset search (0.7200 -> 0.7431, the single biggest lever): averaging a very
   deep (d12/d13) and a very shallow (d3) model's predictions beats any single model by ~0.007. Group-level
   subset search mattered — redundant mid-depth configs diluted the pair.
3. Deep trees work when each config is *itself* seed-averaged: depth climbed monotonically d4 -> d13 once
   variance was controlled (d13 group alone: 0.7352 vs 0.7202 at d4). Early single-model tuning at d4/d6
   was entirely misleading about this.
4. max_bin=512 (+0.0005): finer bins resolve DepTime (hhmm) better; the model leans almost entirely on
   scheduled departure time.
5. Seed-averaging (+0.0003-0.0005, saturating at ~5 seeds): robust and cheap; all selection used the
   averaged-prediction AUC to avoid picking seed luck.

## Things that did not help

1. Engineered features of every kind — cyclical time encodings, hour flags, Origin-Dest route pairs
   (catastrophic, -0.015), DepTime-as-categorical (-0.014) — anything beyond raw columns overfit 2005.
2. Train-fitted target encodings (k=50 smoothing) on airports/carrier/route/time parts: all variants
   reduced AUC; 2005-specific delay rates do not transfer.
3. Covariate-shift correction (frequency-ratio sample weights on carrier/airports): -0.001; the drift is
   in P(y|X), not P(X). Early stopping on an internal 2005 split was also counterproductive (chose 466
   trees while eval-2006 wanted ~30); eval-driven capacity selection was the fix.

## With more budget

The depth frontier was still open at d13 (d12 -> d13 group gain +0.0009, no sign of a peak). I would map
d14-d18 with seed-averaging and re-search the shallow partner (d2-d5, lr, n_estimators) against the best
deep config, then sweep learning_rate (0.02-0.05) and n_estimators (400-1000) jointly at the winning depth
pair, and finally scale each group to 8+ seeds. A 3-way {d13, d8, d3} stack is the next natural subset to
test. CPU used: ~5900 of 18000 s; the binding constraint was the 120 s per-experiment wall clock.
