# Final report — airline dep_delayed_15min (XGBoost)

**Best Eval AUC: 0.7440** (experiment #40, commit `45b1ae5`, 94 s)
Baseline: 0.7141 (experiment #1). Net gain: **+0.0299 AUC**, 40/40 experiments used, ~9,900 of 18,000 CPU-s.

## Setup that won

A 3-model ensemble of regularized `hist`-based `XGBClassifier`s, all with
`grow_policy="lossguide"`, 255/191/255 leaves, 1300 trees each, `lr=0.019`, `min_child_weight`
60–80, `reg_lambda` 10–20, `colsample_bytree` 0.25–0.35 — averaged on `predict_proba`.
Simple categoricals (`pd.Categorical` on train-derived levels, XGBoost native handling) plus the
engineered numeric features below. `predict_proba(df)` calls `prepare(df)`, so every transformation
reproduced on the hidden holdout is the same code path used in training; all encoder statistics
(frequencies, schedule stats, shares) are fitted on `data/train.csv` only.

## The 3–5 changes that mattered most

1. **Heavy regularization + many slow trees** (+0.021 by itself, exps 8→11: 0.7197→0.7298). Depth 6→10,
   `min_child_weight` 5→80, `lambda` 1→20, `lr` 1.0→0.008 with thousands of trees. Because the eval slice
   is the *next year*, under-regularized models overfit 2005 and lost AUC; this was the single largest axis.
2. **`grow_policy="lossguide"` with ~255 leaves** (0.7298→0.7327). Leaf-wise growth beat depth-wise on this
   high-cardinality categorical mix; 511 leaves gave nothing extra.
3. **Schedule-structure features fitted on train** (0.7327→0.7434, three steps):
   time-of-day z-score within route/carrier/origin/dest (how early or late this flight sits in its market's
   day), departure-hour density per route/airport (schedule peaks), and rotation position —
   `pos01` (first→last flight of the operational day) plus `is_first`/`is_last` flags. First flights of the
   day are rarely delayed, last ones inherit accumulated delay; this generalizes across years because the
   schedule shape is stable.
4. **Carrier dominance shares** (0.7402→0.7411): the carrier's share of flights on the route / at the
   origin / at the dest, a competition proxy.
5. **Averaging diverse XGBoost models** (0.7434→0.7440): three models differing in seed, leaf count and
   column/row sampling. Small on eval but the variance-reduction is the part most likely to transfer to the
   hidden holdout.

## Three things that did NOT help (all discarded)

1. **High-cardinality categoricals.** Adding `route` (Origin_Dest, ~4.2k levels) as a native categorical
   cost 0.011–0.015 AUC (0.7327→0.7222 even under heavy regularization) and nearly doubled runtime;
   `carrier_dow` and route×weekday/month share features behaved the same way — these behave like
   per-key IDs and fit 2005 idiosyncrasies. Frequency/share encodings of the same keys are fine; the raw
   high-cardinality category is not.
2. **Out-of-fold target encoding** (carrier/origin/dest/route/carrier-route): 0.7173→0.7155. The tree's own
   categorical splits already extract that signal, and the encoder adds year-specific noise.
3. **Calendar/cyclical expansions**: day-of-year and month sin/cos hurt (0.7298→0.7262), as did making
   hour/minute explicit categoricals (0.7310). Raw `tod` + `minute` numerically were strictly better —
   dropping them cost 0.003 (0.7327→0.7297), so `DepTime` is used both as hhmm and decomposed.
   (Also discarded: `min_child_weight=160`/`lambda=40`, and very fine-grained schedule-rank features, which
   were both useless and slow enough to hit the 120 s cap.)

## What I would try with more budget

The wins came from *market- and schedule-structure* features, so I would push further in that direction
rather than tuning: per-key departure ranks were cut short by the 120 s limit, but a cheaper sampling of
them (rank of this flight among the route's flights in the same 6-hour window, or minutes since the route's
previous scheduled departure) plausibly adds another +0.002–0.005 by capturing rotation propagation
directly. Second, I would build a proper time-based validation split inside 2005 (last quarter as
validation) to pick `n_estimators` and `lambda` per model rather than fixing them on eval, since eval-based
keep/discard at a 0.0005 granularity is already in the noise. Third, I would tune the ensemble composition
under a stricter wall-clock clock budget (3–4 models × 1000–1300 trees), and check a stacked blend
(out-of-fold XGBoost predictions as meta-features) — allowed within the XGBoost-only constraint. Finally,
the last three timeouts (rank01, carrier-hour density, 3× large models) were all cost-driven rather than
idea-driven: capping each member at ~1300 trees and precomputing all lookups as merge-able tables would
let those ideas be evaluated instead of being lost to the timeout.
