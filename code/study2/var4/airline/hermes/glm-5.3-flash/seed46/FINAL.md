# FINAL — autoresearch XGBoost (airline delays, szilard scenario 2)

**Best Eval AUC: 0.7506** (baseline: 0.7141, +0.0365). Final `train.py` = commit `6a79f44`
(bagged XGBoost ensemble, `CONTRACT OK` in validate.log).

## What mattered most

1. **Feature selection against the time shift**: dropping `Month` (+0.0035) and later `DayofMonth`
   (+0.005) from the model — 2005's calendar patterns do not transfer to 2006 rows, and deeper trees
   amplified the leak. After removal, depth 8 became optimal (it had looked harmful before).
2. **Hour-of-day × carrier and hour × origin categorical interactions** (`hhxcar`, `hhxorg` from
   `DepTime//100`): the single biggest feature win (+0.013 with depth-8, n500, L1=5).
3. **Recency-weighted training**: weight ∝ 1 + 5·(month/12) so late-2005 flights dominate the loss —
   a direct lever on the 2005→2006 covariate shift (+0.0025; `alpha`=5 and an L1-heavy depth-8
   model absorb it best).
4. **Seed-bagged ensemble of XGBoost models** (8× base, colsample 0.85, n400) blended with 5
   *unweighted* Month-feature diversity models at weight 0.7 (+0.004). The Month-models are kept
   unweighted on purpose: their 2005 month pattern is useful as diversity, but only partially transfers.
5. **L1 regularization (alpha=5)** at depth 8 (+0.002 over alpha=0); XGBoost categoricals with
   unseen-level→NaN handling for the 2 unseen carriers / 7-8 unseen airports in 2006 data.

## What did not help

- **Target/likelihood encodings** (OOF-smoothed, smooth=100): never beat one-hot/categorical splits;
  month-of-year×day-of-month TE was catastrophic (−0.013).
- **Engineered calendar/time features** (sin/cos month, dom, dow, dep_min, late-night flag, route
  pairs, frequency counts, congestion rates, dep-time 5-min buckets, quarter×carrier): all neutral
  or negative; DepTime+interactions already carry the time signal.
- **Complexity moves**: early stopping on a 15% holdout (−0.001), row subsampling (−0.01 alone),
  deeper/longer single models (depth ≥ 6 loses at n>100), target-encoding stacks, stacking
  meta-models, isotonic recalibration, rank averaging — all ≤ 0.

## With more budget

- A proper **time-based validation design**: my keep/discard decisions used eval.csv (2006 slice 1);
  the hidden holdout is a different 2006 slice. I would build an internal 2005→2006 pseudo-shift
  validator to stop relying on eval.csv selection (final config was chosen to be robust: every
  component positive across both 2005-CV and 2006-eval).
- Diagnose the residual 0.0025 gap between my inline sweeps (~0.7516) and the packaged train.py
  (~0.7488) for the 3-family variant; if closed, the 3-family ensemble
  (Month, Month×carrier, Month×origin diversity families) looked consistently stronger.
- Larger member counts and per-family weight optimization on a proper time-split CV.

## Notes

- 26 counted experiments (+2 refused), ~18,000/18,000 CPU-seconds used.
- Final contract check: `CONTRACT OK`, eval AUC via `predict_proba` 0.7506, runtime ~77 s.
