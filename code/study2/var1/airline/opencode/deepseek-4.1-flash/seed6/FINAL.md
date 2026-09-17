# Final report — airline (dep_delayed_15min)

**Best Eval AUC: 0.7529** (commit `538c113`, experiment #38) vs. baseline 0.7141 (+0.0388).
Final `train.py` passes `./validate.sh` (`CONTRACT OK`), and `predict_proba(df)` re-runs all
feature engineering and encoders fit on training data only.

## What mattered most

1. **Turning capacity *down*, not up.** The baseline (`30` trees, depth 6, lr 0.1) is a strong local
   optimum on this time-separated split, and a bigger model generalizes *worse* (500 trees/depth 7 =
   0.7054). A shallow, regularized model (250 trees, depth 4, lr 0.05, `min_child_weight=5`,
   `colsample_bytree=0.8`) reached 0.7171. The train (2005) → eval (2006) shift punishes memorization.
2. **Cross-year-stable interaction target encodings.** EDA showed delay-vs-*hour* is almost perfectly
   stable across years (corr 0.96) while raw `Origin`/`Dest` delay propensities are not (corr ≈0.38),
   yet `(airport × hour)` is stable (corr 0.87–0.90). Encoding Origin/Dest/Carrier × hour took the
   model from 0.7182 to 0.7221, and layering `(carrier × airport × hour)` triples reached 0.7246–0.7271.
3. **Minute-of-hour ("scheduling bank") interactions** were the single biggest jump: 0.7321 → 0.7421
   (and 0.7421 → 0.7485 with `(airport × hour × minute × distance-bucket)`). Delay risk is strongly
   structured around clock banks of departures, and that structure reproduces year over year.
4. **Frequency/count encodings** for every target-encoded key (log1p of training frequency) gave a
   robust +0.0020 (0.7502 → 0.7524): support size tells the trees how much to trust each encoding.
5. **XGBoost ensemble**: seed × config averaging (depth 3/4/5 plus deeper regularized depth-6/8/10
   configs) and 5-minute buckets finished at **0.7529**.

## What did not help

- More raw model capacity / deeper unregularized trees (500 trees d7: 0.7054; d6 config before rich
  features: 0.7229).
- High-cardinality `Route` categorical as-is (0.7056) and one-hot encoding (0.7102) — partition
  categorical splits win.
- Plain (no time-interaction) target encoding of Carrier/Origin/Dest/Route (0.7137), day-of-week
  interactions (0.7228), heavier smoothing (0.7480), and `grow_policy="lossguide"` (0.7504).

## With more budget

I would push the interaction-encoding idea further and make it robust: finer/adaptive time buckets
with hierarchical (per-key Bayesian) smoothing instead of one global prior, aircraft-rotation features
(same tail number's previous-leg delay) and origin/destination weather proxies if data were available,
plus a proper stacking meta-learner over the XGBoost ensemble. I would also validate encoder stability
on a held-out slice of 2005 with a simulated year gap, since eval.csv selection is noisy (~0.0017 SE)
and all keep/revert decisions were made on it.
