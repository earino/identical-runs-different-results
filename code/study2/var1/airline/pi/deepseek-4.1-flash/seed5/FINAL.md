# Final report — autoresearch XGBoost on `airline`

## Best result

- **Best Eval AUC: 0.7481** (experiment #40, commit `64df0a1`, "dep_min + slightly leaner max_bin512").
- Baseline: 0.7141 (experiment #1).
- Contract: `python train.py` prints `Eval AUC: 0.7481` and `./validate.sh` prints `CONTRACT OK`
  (AUC via `predict_proba` on `data/eval.csv` with the target removed: 0.7481).
- Training time ~94s in-run (the benchmark caps an experiment at 120s), 3-model XGBoost ensemble, 4 threads.

## Changes that mattered most

1. **One-hot encoding instead of XGBoost native categorical splits** (exp #26: 0.7228 vs 0.7205).
   Encoding `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest` as indicator columns
   generalised across the 2005→2006 boundary far better than partition-based categorical splits, and it
   unlocked the next two changes.
2. **Deep trees with `min_child_weight=1`** (exp #27 → #28: 0.7379 → 0.7460, exp #40: 0.7481).
   With one-hot features the model wanted depth 20–36 and almost no minimum child weight. Sweeps showed a
   clean monotone response: depth 5 → 0.723, depth 10 → 0.736, depth 16 → 0.738, depth 24/36 + `mcw=1`
   → 0.742–0.744 for a single model.
3. **Airport congestion / "peak-ness" features** (exp #18, #20, #22, #32: 0.7172 → 0.7464 overall).
   Flight counts per `(Origin, scheduled hour)` and `(Dest, scheduled hour)`, their ratios to each airport's
   total traffic, destination congestion at the *estimated* arrival hour (`DepTime + Distance/500 mph`), and
   origin *inbound* congestion at the departure hour. These were the only engineered features that produced
   consistent, sizeable gains.
4. **`max_bin=512`** (exp #34: 0.7464 → 0.7472). Finer histogram bins for the continuous `DepTime`,
   `Distance` and ratio features helped a little.
5. **Small deep ensemble (3 models, probability average)** (exp #27/#28). Seeds/depths 24/24/36 gave a
   reproducible ~+0.001 over the best single model. `dep_min` (minutes since midnight) plus a ~10% tree
   reduction was the final improvement (exp #40: 0.7472 → 0.7481) while keeping the run under the time cap.

## Things that did not help (and were reverted)

1. **Target encoding / route interactions / native categorical route.** Smoothed target encoding of
   carrier/origin/dest/route leaked and dropped to 0.7075; adding a `Route` categorical dropped to 0.7056.
   Native categorical splits already capture those effects without the leakage.
2. **Cyclic encodings, raw frequency totals, and a monotone constraint on `dep_min`.** Cyclic hour/month/dow
   (0.7195 vs 0.7205), frequency encodings, and a monotone-increasing constraint on minutes-since-midnight
   were all neutral-to-negative — the trees already recover the strong time-of-day gradient directly.
3. **Scaling the ensemble up / rank averaging / categorical smoothing.** 8 members, rank-averaging instead of
   probability averaging, and `cat_smooth=50` all gave exactly the same or slightly worse AUC; returns from
   extra members saturated after ~3.

## What I would try with more budget

The biggest remaining lever is better use of the high-cardinality structure without leaking: out-of-fold
(or CatBoost-style ordered) target statistics for carrier × airport × hour, and explicit carrier/route
"delay propensity" features computed only from hold-out folds, evaluated against a time-aware validation
split (train on early 2005, validate on late 2005, test on 2006) so model selection reflects the 2005→2006
drift rather than one 2006 slice. I would also add a proper estimated block/flight-time model (to sharpen
the arrival-hour congestion features), run a constrained Bayesian/random hyperparameter search over
depth, `min_child_weight`, `max_bin`, subsample and colsample, and explore multi-level stacking with an
XGBoost meta-learner. Because the 120s cap binds, the search should be paired with feature pruning and
2–3-member ensembles so more effective capacity fits the budget.
