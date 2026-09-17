# Final report — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7369** (baseline: 0.7141, +0.0228). Contract validated (`CONTRACT OK`);
`predict_proba` reproduces all feature engineering on unseen rows (all engineering lives in
`prepare(df)`, encodings/count tables fit on `data/train.csv` only, OOF variants used for the
training rows).

Final model: 12-model bagged XGBoost ensemble (d=12, n=150, min_child_weight=30, lr=0.1,
subsample=0.9, per-model colsample 0.5/0.65/0.8/1.0), OOF target encodings, periodic harmonic
bases, and congestion counts.

## The 3-5 changes that mattered most

1. **Rich periodic bases on time-of-day (k=2..24 harmonics of hour), day-of-year (k=1..4) and
   day-of-week (k=1..3).** The single biggest lever: +0.005 cumulative. The delay-vs-time-of-day
   curve is smooth but multi-modal; harmonics let even axis-aligned trees represent it exactly,
   and the annual cycle (summer/winter, holidays) generalized from 2005 to 2006.
2. **Target-encoded time-of-day (30-min and 10-min bins) + origin TE, computed out-of-fold for
   training rows.** te_time carries ~40% of gain in early models; OOF computation removed the
   self-leak and added robustness for free.
3. **Congestion counts**: flights per origin in the same 30-min window, per origin x dow,
   per carrier x 30-min, origin x carrier. Proxy for queueing/airport load; fit on train only.
4. **Capacity re-tuning after the feature changes.** With sparse/binned features, deep+many trees
   overfit the 2005->2006 shift (0.708 at d=6/n=800 on raw features). Once harmonics gave the
   trees dense, smooth structure, deeper trees became a win: the optimum moved from d=3/n=200 to
   d=12/n=150 (+0.010 over the shallow optimum).
5. **Bagging with real diversity**: 12-16 models averaging over subsample + per-model
   colsample_bytree. Worth +0.0005-0.001 consistently.

## Three things that did NOT help

1. **High-cardinality interaction encodings**: route (origin x dest) TE, origin x hour TE,
   dest TE, origin x 6h-block TE all *hurt* (up to -0.01). Sparse groups memorize 2005 noise.
2. **More/bigger trees on weak features** and early stopping via internal validation split
   (adaptive n_estimators picked the wrong point; fixed 150-200 was better).
3. **Over-extending the harmonic basis** (k up to 32, doy k up to 6) diluted the model
   (-0.0016); monotone constraints on time features and interaction constraints also lost.

## What I would try with more budget

- Blend the XGBoost bag with a second model family (logistic regression / GAM on the harmonic
  basis) — the time signal is nearly additive, so a spline-GAM may complement the trees.
- Learn a per-airport time-of-day curve with hierarchical smoothing (partial-pooling toward the
  national curve by airport size) instead of the crude te_O x harmonic products.
- Nested/stacked ensembling: OOF predictions from the 12-model bag as a feature for a
  calibration layer; also rank-averaging across heterogeneous depth/colsample configs.
- Systematic multi-seed evaluation of the top-3 configs (each single eval has ~±0.0005 noise).
