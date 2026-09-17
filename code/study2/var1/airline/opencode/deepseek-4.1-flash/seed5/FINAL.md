# Final Report — airline XGBoost (AUC)

**Best Eval AUC: 0.7537** (10-seed XGBoost ensemble, commit `5ebac74`).
Baseline was 0.7141. Contract validated (`CONTRACT OK`, eval AUC via `predict_proba` = 0.7537).

## Changes that mattered most

1. **Regularization + depth.** The single biggest jump came from turning the tiny
   baseline (30 trees, lr 0.1, depth 6) into a regularized deep model
   (`subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, then
   `reg_alpha=1`). Deep trees (depth 6→16) kept improving Eval AUC, and
   extra feature/row subsampling (`subsample=0.7`, `colsample_bytree=0.5`)
   added another ~0.002. L1 (`reg_alpha=1`) added ~0.0014.
2. **Time-of-day as a native categorical (`c_DepHour`) plus carrier×hour
   (`c_CarrierHour`).** This was the largest feature win (~+0.0075 on top of the
   tuned model). Scheduled hour is the dominant delay signal.
3. **Frequency encodings.** Counts of `Origin`, `Dest`, `UniqueCarrier`,
   origin/dest/carrier × hour, `Origin_Dest`, route×hour, and carrier×airport
   (all target-free) added ~+0.005 cumulatively.
4. **Structural share ratios.** `freq(carrier,airport)/freq(airport)` and
   `freq(route)/freq(airport)` gave a small, robust +0.0004.
5. **Seed ensembling.** Averaging 10 XGBoost models with identical params but
   different `random_state` added ~+0.002 (4→8→10 seeds: 0.7507 → 0.7516 → 0.7537).
   This is the most robust component because it only reduces variance.

## Things that did not help (reverted)

1. **High-cardinality categorical interactions.** `Route` (Origin_Dest) as a
   native categorical was strongly negative (−0.008), and `Origin×Hour` also hurt;
   the model overfits these partitions. `Dow×Hour`, `Month`, `Dow`, and
   `Weekend×Hour` categoricals each hurt as well.
2. **OOF target encoding** for Origin/Dest/Carrier/Route was slightly negative
   (−0.0008): 2005 delay propensities do not transfer cleanly to 2006.
3. **Extra capacity without regularization.** 400 trees at lr 0.05 (and 500 at
   lr 0.03) were worse than the tuned setting — the raw baseline overfits the
   2005 training year.
4. Day-of-year seasonality features were negative (−0.0025).

## With more budget

The strongest remaining lever is more/better ensembling: the 10-seed homogeneous
ensemble was still edging up. With more compute I would (a) scale to 20–30 seeds
and blend with a second XGBoost variant (`dart` or `grow_policy=lossguide`) for
genuine diversity, (b) search `max_cat_threshold` / `max_cat_to_onehot` around
the categorical hour features, and (c) explore carefully regularized target
encodings fit only on 2005 with strong smoothing and out-of-fold construction,
since the failure of the naive version looks like temporal shift rather than a
lack of signal. Any changes would be judged on the time-separated eval and the
hidden 2006 holdout, favoring robust, low-variance modifications.
