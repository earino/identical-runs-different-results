# Final report — airline delayed-departure classifier

**Best Eval AUC: 0.7273** (experiment #39, commit `bfe003c`, `train.py` at HEAD).
Baseline was **0.7141** (experiment #1), so the loop gained **+0.0132 AUC** in 40 experiments.

## Final approach

XGBoost binary classifier trained on `data/train.csv`, evaluated on `data/eval.csv`.
Everything is computed inside `prepare(df)`, so `predict_proba(df)` reproduces the exact
feature pipeline on the hidden holdout:

- Native categoricals for `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`
  (XGBoost `enable_categorical=True`, which beat one-hot and target encoding here).
- Time-of-day: `dep_hour` as a categorical, plus `tod` (minutes since midnight) and its
  sine/cosine.
- Calendar flags: `travel_rush` (Dec 18–31, Jan 1–4, late Jun–early Jul, Nov 20–30) and
  `holiday_day` (the actual holiday dates, which behave differently — delays drop on the
  day itself), plus `is_weekend`.
- High-value interactions as native categoricals: `carrier_hour`, `origin_hour`, `dest_hour`.
- A **5-model ensemble** (varied depth 3–6, trees 150–400, lr 0.03–0.07, subsample/colsample,
  seeds) averaged at prediction time, all with `reg_lambda=5`, `reg_alpha=5`,
  `sampling_method="gradient_based"`.

## Changes that mattered most

1. **Regularization (`reg_alpha=5`, `reg_lambda=5`)** — +0.0022 over the unregularized ensemble.
   With a 2005→2006 distribution shift, strong L1/L2 was the single biggest lever against
   overfitting year-specific noise.
2. **`origin_hour` / `dest_hour` / `carrier_hour` interactions** — +0.0043 combined
   (0.7230 → 0.7273). Airport- and carrier-specific delay behaviour by departure hour is a
   genuine, transferable signal.
3. **Hour-of-day representation** (categorical hour + `tod` + cyclic) — +0.0013 over the
   ensemble (0.7176 → 0.7189). Hour-of-day is the dominant feature (delay rate rises from
   ~0.04 at 05:00 to ~0.85 at midnight).
4. **Moderate capacity**: depth 4 with ~200 trees at lr 0.05 was the sweet spot; depth 6 and
   500+ trees overfit (0.7098–0.7150).
5. **Multi-model ensembling** across depths/seeds/rates — +0.0006–0.0009, a robust variance
   reduction.

## Things that did NOT help (all reverted)

- **Target encoding** (OOF, smoothed) of carrier/origin/dest/route/hour: neutral-to-worse;
  native categorical treatment already captures it.
- **`route` as a native categorical** (4198 levels): clearly worse (0.7046) — route delay
  rates are unstable year-to-year (corr ≈ 0.4 for origins).
- **One-hot encoding** of all categoricals (0.7162) and `max_cat_to_onehot=32` (0.7182):
  worse than native categorical splits.
- **Frequency-count features**, **day-of-year cyclic seasonality**, and **`carrier_origin`**:
  no gain (the last, experiment #40, was the only wasted final run: 0.7270 vs 0.7273).

## What I would try with more budget

The remaining headroom is almost certainly in richer, *stable* interaction structure rather
than more trees. I would (a) build a small validation split from late-2005 rows to select
`n_estimators`/regularization against the year shift instead of eval.csv, reducing the risk of
overfitting the visible eval; (b) add route-level encodings conditioned on carrier and hour
with hierarchical shrinkage toward the (stable) carrier/origin priors, rather than raw route
rates; (c) try a two-level XGBoost stack over the existing ensemble's out-of-fold predictions;
and (d) sweep `reg_alpha`/`reg_lambda`/`max_depth` jointly now that the feature set is much
larger. Given every added interaction improved eval, the interaction route still looks the
most promising.
