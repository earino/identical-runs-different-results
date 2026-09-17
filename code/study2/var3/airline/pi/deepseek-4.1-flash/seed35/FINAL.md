# Final report — airline delay XGBoost (AUC)

**Best Eval AUC: 0.7446** (experiment #37, commit `608a49e`, 108s).
Baseline was **0.7141**. Total: 40 experiments run, budget exhausted.

## What mattered most

1. **Out-of-fold target encoding (10-fold) for carrier / origin / dest / route and their
   hour interactions.** This was the single biggest lever early on (0.7141 → 0.7229, then
   0.7257). Encodings are fit on 2005 train only; training rows use OOF values while
   `predict_proba` uses the full-train maps.
2. **Airport/route congestion counts.** Hourly departure/arrival counts at the origin, dest
   and route, plus fractions of the airport/route total (0.7276 → 0.7304 → **0.7324**).
   Traffic density around the scheduled hour is highly predictive and stable across years.
3. **Cumulative daily flight share and peak-relative load.** Fraction of the airport's/route's
   daily flights already scheduled before the departure hour, and hourly load relative to the
   airport peak (0.7411 → **0.7446**). Strong, robust structural signal.
4. **Aggressive column subsampling + regularization.** `colsample_bytree=0.2`,
   `subsample=0.5`, `reg_lambda=3`, `reg_alpha=1`, `min_child_weight=40`, depth 8, lr 0.03 with
   early stopping (0.7334 → 0.7407 → 0.7410). With many correlated target-encoded features,
   forcing feature diversity per tree was worth more than any single new feature.
5. **3-model heterogeneous seed ensemble** (varying colsample/subsample) averaged for a small,
   reliable gain (+0.0007–0.001) and for smoothing over eval-based early stopping.

## What did not help

- **Raw high-cardinality `route` as a native XGBoost categorical** with deep trees overfit hard
  (0.7092 vs 0.7141 baseline) and peaked at iteration 73.
- **Day-level traffic counts, airport × month target encodings, and net arrival/departure
  congestion** — overfit the 2005 slice and failed to transfer (0.7195, 0.7373, neutral).
- **Lower/aggressive target-encoding smoothing (k=5) or per-key smoothing**, low-cardinality
  temporal TEs (month/dow/hour), and adding more seeds with `max_bin=128` all gave no gain.
- Cyclical dow/month features were removed with no loss (simpler and marginally better).

## With more budget

I would push time-aware validation (e.g. blocked/rolling splits over months) instead of
eval-based early stopping, so model selection is less coupled to the 2006 slice. On the feature
side I would try hierarchical/empirical-Bayes target encoding with per-key priors, interactions
of the strongest congestion features with carrier, and a stacking level trained on OOF
predictions. On the model side, a slower learning rate (0.02) with a faster hist config, and
bagging over several OOF fold seeds, look most promising. The main remaining risk is that the
recent gains lean on target-encoded 2005 rates; an ensemble weighted toward the pure structural
congestion features would likely generalize best to the hidden holdout.
