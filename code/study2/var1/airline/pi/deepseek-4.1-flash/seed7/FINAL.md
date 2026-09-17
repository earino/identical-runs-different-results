# Final report — airline delay XGBoost

**Best Eval AUC: 0.7411** (experiment #40, commit `b21f984`), up from the 0.7141 baseline.

Final model: 5-seed averaged `XGBClassifier`, `max_depth=20`, `n_estimators=200`,
`learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, native categorical handling,
trained on 2005 and evaluated on 2006.

## Changes that mattered most

1. **Calendar as numeric ordinals, not categoricals** (+0.0034, exp #14). The `c-<n>` strings
   encode real integers (Month, DayofMonth, DayOfWeek). Mapping them back to ordered numbers lets
   the trees model seasonality/weekday trends smoothly instead of treating each level independently;
   this generalizes far better across the 2005→2006 year shift.
2. **Grouping rare categorical levels** (+0.0024 → +0.009 overall as the threshold and depth were
   tuned, exp #22–#39). Origin/Dest levels with too few 2005 examples (threshold raised
   progressively to `RARE_MIN=800`) are collapsed into a single `__RARE__` level. This removes
   year-specific airport noise and was the single largest source of gains once combined with depth.
3. **Deep trees** (exp #26–#33). Once categories are grouped, increasing `max_depth` from 4 to 20
   monotonically improved cross-year AUC (0.7245 → 0.7381). `min_child_weight=10` was catastrophic
   (0.7300), confirming the model depends on fine-grained deep leaves; `depth=24` was past the peak.
4. **Frequency encoding** (+0.0001, exp #13) of Origin/Dest/carrier/route — a small but free
   robust proxy for airport/carrier volume.
5. **Seed averaging** (exp #19, #40), 5 seeds — a small, reliable variance reduction (+0.0003–0.0004)
   with no overfitting risk.

## Things that did not help

- **DepTime-derived features** (hour, time-of-day, cyclic sin/cos) on top of the raw `DepTime`
  integer: consistently −0.0005 AUC (exp #11, #18). Raw `DepTime` splits already capture the
  intraday delay build-up.
- **Out-of-fold target encoding** of Origin/Dest/carrier/route (exp #20): −0.0003. The native
  XGBoost categorical splits plus rare grouping already capture the useful signal, and 2005
  per-category delay rates transfer poorly to 2006.
- **Cyclic calendar encodings** (sin/cos of month/dom/dow, exp #15): −0.0011 once numeric ordinal
  features were present.
- **Dropping Origin/Dest** (exp #5): −0.0141 — they carry real signal, so the goal was to regularize
  them, not remove them.
- Also no help: `max_cat_threshold=16`, `min_child_weight=20`, 500 trees at `lr=0.03`.

## What I would try with more budget

The strongest remaining direction is to tune the deep-tree configuration more finely and to exploit
the fact that eval is drawn from the same year as the hidden holdout. I would grid over
`max_depth ∈ {18,19,20,21,22}`, `n_estimators ∈ {150,200,300}`, and `RARE_MIN ∈ {600,700,900}`,
then test a heterogeneous ensemble that mixes depth-18/20/22 models (rather than only seeds) to add
diversity, and re-check `reg_lambda`/`colsample_bytree`. Beyond tuning, I would add grouped
interaction features (carrier×hub, route-distance bins) with the same rare-level collapse, which was
the mechanism that unlocked the big gains here. All selection so far used eval, so a final sanity
check on a held-back slice of 2006 (rather than eval) would guard against selection overfitting.
