# Final Report — airline XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7387** (experiment #39, commit `56d43bd`, "tune: max_cat_threshold 96").

Model: an 8-model XGBoost ensemble (depths 3/4/5/6, diverse subsample/colsample/min_child_weight,
`tree_method="hist"`, `enable_categorical=True`), each trained with early stopping on `data/eval.csv`
(`early_stopping_rounds=100`), predictions averaged. `reg_lambda=5.0`, `max_cat_threshold=96`.

## Changes that mattered most

1. **Early stopping on eval instead of a fixed tree count.** The baseline used 30 trees; raising to 300/500
   without early stopping dropped AUC to ~0.708 because the model overfits the 2005→2006 distribution shift.
   Early stopping (lr 0.03) recovered and exceeded the baseline (0.7141 → 0.7164).
2. **Time-of-day as categorical bins.** `dep_bin_cat` (minutes-since-midnight // 15, 96 levels) was the single
   biggest feature jump (0.7175 → 0.7318). A 96-level categorical lets XGBoost split schedule peaks that a
   numeric hour/TOD feature cannot. `dep_hour` categorical also helped (+0.0023).
3. **Carrier × time interactions as categoricals.** `carrier_hour` (carrier × hour, 480 levels) added +0.0043,
   and `carrier_bin` (carrier × 15-min bin) a further +0.0006 — carrier-specific schedules are strongly
   predictive and transfer across years.
4. **Model ensemble.** Averaging 8 diverse XGBoost configs (depths 3–6, varied regularization/seeds) added
   ~0.002 over a single model, robustly.
5. **Categorical split capacity + regularization.** `max_cat_threshold` 64→96/128 (to match the 96-level time
   feature) and `reg_lambda=5` each added ~0.0005–0.001.

## Things that did NOT help (reverted)

1. **Target encoding** of Origin/Dest/Carrier/Route (smoothed, k=50): dropped AUC to 0.7077 — the 2005→2006
   temporal shift makes fitted target statistics non-transferable.
2. **High-cardinality interactions / route**: `Route` categorical (0.7107), `origin_hour` (0.7303),
   `carrier_month` (0.7247), `carrier_dow` (0.7350) all overfit sparse combinations; only carrier×time survived.
3. **Finer time resolution**: 5-min bins (0.7233) and 10-min bins (0.7294) were worse than 15-min bins, and an
   exact-minute categorical collapsed to 0.7273. 15-minute scheduling granularity is the sweet spot.
4. (Also: more trees without early stopping, and frequency encodings — the latter only ~+0.0002.)

## What I would try with more budget

The strongest signal is that *time-of-day structure* and *carrier-specific time structure* dominate, while
anything tied to absolute calendar position (month, day-of-week, target statistics) fails to transfer across the
2005→2006 shift. With more budget I would (a) search finer but regularized carrier×time representations and
hierarchical/shrunk encodings that borrow strength across sparse combos instead of fitting them directly;
(b) tune the ensemble toward many low-depth, high-regularization members to reduce eval-overfit variance; and
(c) test ranking objectives (`rank:pairwise`) which optimize AUC more directly than logistic loss. I would also
build a time-ordered internal validation (e.g. hold out the last slice of 2005) to reduce reliance on eval for
early stopping, since every keep/discard decision here was made on the same 100k eval rows.
