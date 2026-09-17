# Final report — airline XGBoost

**Best Eval AUC: 0.7350** (experiment #40, commit `414cb81`; baseline 0.7141).

## Changes that mattered most

1. **Heavy regularization + low learning rate.** Going from the baseline (lr 0.1, 30 trees,
   depth 6) to `lr=0.005`, 5000 trees, `colsample_bytree=0.4`, `min_child_weight=50`,
   `reg_lambda=10` lifted AUC from 0.7141 to ~0.722. A 3-seed ensemble added a small, robust bump.
2. **Dropping non-transferring calendar features.** `Month` and `DayofMonth` have year-specific
   delay patterns (2005 vs 2006 differ a lot), so the model over-fit them on train and generalized
   poorly. Removing both: 0.7229 → 0.7269 (+0.0040).
3. **Deeper trees with fewer boosting rounds.** Once the noisy calendar features were gone, depth
   could be increased safely. Ladder depth 6→13 (5000→1800 trees) gave 0.7285 → 0.7350.
   `DayOfWeek` stayed (weekly pattern is stable across years); all three models share the config,
   only the seed differs.
4. **Departure-time representation.** `DepTime` → minutes-of-day plus sin/cos cyclic encoding is the
   single strongest predictor (delay rate rises monotonically 05:00→23:00).
5. **Frequency (count) encodings** of carrier, origin, dest and route: small but repeatable gain
   (+0.002 when ablated at 0.7269). Target-independent, so they transfer across years.

## What did not help

1. **High-cardinality categoricals / target encoding.** Route (4198 levels), carrier×origin and
   carrier×dest, and OOF target encoding all *hurt* badly (0.70–0.71) — they memorize 2005 route
   effects that do not transfer to 2006.
2. **Early stopping on a random 2005 holdout** and adding engineered `is_weekend` / day-of-year /
   malformed-time flags: 0.708 and 0.7214 respectively — no gain over fixed low-lr boosting.
3. **Very low lr / very many trees** (lr 0.003, 8000 trees, mcw 80) and aggressively low
   `subsample=0.6` were both slightly worse than the 0.005 / 5000 / 0.8 setting.

## With more budget

The dominant failure mode here is the 2005→2006 covariate shift, so I would invest in
*shift-robust* modelling rather than more capacity: adversarial/importance-weighted validation,
per-feature stability selection across a time-based split, and cross-year target encoding computed
only from the most recent slice. I would also sweep depth vs. `min_child_weight` jointly with a
smaller grid and try adding a monotone constraint on minutes-of-day (delay is monotone increasing
during the day), plus bagging over bootstrap resamples to further reduce variance. Finally, a
2-model average of depth-13 and a shallower, more-regularized learner might buy a final ~0.001.
