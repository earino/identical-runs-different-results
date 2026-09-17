# Final report — airline departure-delay classifier (XGBoost)

**Best Eval AUC: 0.7600** (baseline 0.7141, +0.0459). Final commit: `ebecc69` ("6-seed ensemble").
Validated: `./validate.sh` → `CONTRACT OK`, predict_proba AUC on eval = 0.7600.

## Changes that mattered most

1. **Time-of-day features from DepTime** (hour, minutes-of-day, sin/cos of the daily cycle) instead of the raw hhmm
   integer alone — first solid jump (0.7141 → 0.7232).
2. **Smoothed out-of-fold target encodings of the delay rate by (route | origin | dest | carrier) × time-of-day
   bucket** — the single biggest lever. The winner combination uses 15-minute buckets for route/origin, 30-minute
   for dest/carrier, plus coarser hour-bucketed route/origin companions (OOF 5-fold for training rows, full-train
   stats for new rows, smoothing m = 25–100 by cardinality).
3. **Flight-volume "congestion" counts** per origin / dest / route / carrier / (origin×hour) / (route×hour),
   counted on the training data only (~+0.003).
4. **Deep, heavily column-subsampled trees + seed ensembling**: max_depth=20, min_child_weight=1,
   colsample_bytree=0.3, subsample=1.0, lr=0.05, λ=5, early stopping on a 20% train split, then retraining 6
   full-data models with different seeds and averaging their probabilities (+~0.004 over a single model).
   Ensembling members that differ only by seed beat a mixed-depth ensemble.

## Things that did not help

1. Target-encoding route/origin/dest/carrier **without** the time-of-day interaction, and route as a native
   high-cardinality categorical (partition-based splits were clearly worse than the TE).
2. **Day-of-week / month / seasonality features**: route×dow, origin×dow, carrier×dow, route×month TEs and
   day-of-year sin/cos all reduced AUC; a 2nd (12 h) harmonic of the daily cycle also hurt.
3. **Regularization/protocol tweaks**: reg_alpha (L1), 10-fold OOF (5-fold was better), 90/10 ES split,
   colsample 0.2–0.5 vs 0.3 at subsample 0.7–0.8, min_child_weight 1→5/20, max_bin 512, lr 0.02–0.03,
   finer (15-min) congestion counts — all flat or worse.

## With more budget

The remaining signal is probably in better use of the time structure: encodings keyed on *scheduled* sequence
within an airport-day (e.g. rank of departure among flights from the same origin that day, distance to the
previous scheduled departure), a hierarchical shrinkage of the bucket TEs (15-min ← 30-min ← hour ← route marginal),
and year-shift-robust variants of the counts. I would also ensemble across *feature* subsets, not just seeds,
and select the tree count by averaging early-stopping curves over several seeds instead of one split.
All of this is CPU-cheap to try per idea; the binding cost here was CPU budget, so I stopped with ~950 CPU-s
held back to guarantee a clean validation.
