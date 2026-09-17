# Final report — airline delay (XGBoost, szilard scenario 2)

**Best Eval AUC: 0.7265** (baseline 0.7141, +0.0124). Final model: 9-member XGBoost ensemble
(hist + lossguide boosters), all trained on `data/train.csv` with early stopping against `data/eval.csv`.

## Final recipe (HEAD = f0bdbff)

- Features (all inside `prepare()`, applied identically by `predict_proba`):
  native pandas categoricals for Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest,
  smoothed target encoding for carrier/origin/dest (m=100, fit on train only),
  `hour`, `h2` (hours since 5am), cyclical sin/cos of time-of-day, `late_sched` flag, raw DepTime + Distance.
- Ensemble members share the core regularization `{colsample_bynode: 0.3, colsample_bylevel: 0.7, subsample: 1.0}`
  with diversity across grow policy (hist/lossguide-64), min_child_weight (10/20), lr (0.05/0.03),
  colsample_bytree 0.8, and seeds; `max_depth` 6, `min_child_weight` 10–20, tree count (300–500) picked
  per member by early stopping (patience 30) on eval.csv (2006-slice1; hidden holdout is 2006-slice2).

## Changes that mattered most

1. **Strong per-node/per-level column subsampling** (`colsample_bynode` 0.3, later `colsample_bylevel` 0.7):
   the single biggest win (+0.005 and +0.0026 respectively on singles). Prevents the model from latching onto
   2005-specific feature combinations — the key defense against the 2005→2006 distribution shift.
2. **subsample 1.0** (+0.002 with the column subsampling in place): row subsampling 0.85 was hurting once
   column decorrelation did the regularization work.
3. **9-member ensemble** of the strongest, most diverse families (+0.0014 over the best single 0.7251):
   seed variance on singles is ±0.002, so averaging is pure variance reduction.
4. **h2 = (hour − 5) mod 24** (+0.0011 single, +0.0004 ensemble): an ordered feature aligned with the actual
   daily delay ramp (5am minimum, rising all day, red-eye wraparound) complements hour + cyclical features.
5. **Early stopping against eval.csv (2006)** instead of a random 2005 split: 2005-internal validation chose
   ~2–3× more trees than optimal for 2006 (2005 val AUC 0.76 vs 2006 0.71 — strong year shift).

## Things that did not help

1. **Interaction categoricals** (Origin×Dest route, carrier×origin, carrier×dest, month×hour, dow×hour):
   all hurt (−0.005 to −0.010) — route-level 2005 propensities (~17 flights/route) do not transfer to 2006.
2. **Holiday-window flags** (Christmas/Thanksgiving/summer): exactly zero effect.
3. **max_bin 256/512/1024, TE smoothing strength (m=50–400), season sin/cos, ES patience (25–150)**:
   all neutral; freq_ features and log/bin distance transforms were droppable at equal AUC.
4. **max_delta_step 1.0**: helped singles on the old recipe (+0.0008) but was neutral in the final ensemble.
5. **AUC-weighted / top-8 / rank blending**: uniform mean of probabilities was best (weighted 0.7237 vs 0.7240).

## What I would try with more budget

With ~20 more experiments I would (a) sweep `colsample_bylevel` × `colsample_bynode` jointly on the new recipe
(0.7/0.3 was found greedily, one axis at a time); (b) grow the ensemble toward 16–20 members with the faster
patience-30 recipe while keeping runtime < 120 s (the 14-member attempt timed out; a slower-lr, smaller-tree
variant would fit); (c) probe `max_leaves` 32–96 for lossguide under the new recipe; (d) test a 2-fold OOF
stacker (XGBoost-only meta-model on member predictions) if runtime allowed; (e) re-check TE interaction with
the new subsampling recipe (TE was inert only under the old one).
