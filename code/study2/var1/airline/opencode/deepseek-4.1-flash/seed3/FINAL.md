# Final Report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7385** (experiment #37, commit `1f6897c`, contract validation: `CONTRACT OK`).

## Setup
- Data: `data/train.csv` (2005, 100k) → `data/eval.csv` (2006, 100k), target `dep_delayed_15min`, balanced.
- Model: ensemble of 5 `XGBClassifier` models (seeds 42/7/2024/1337/99), probabilities averaged, `tree_method="hist"`,
  native categorical support, `n_estimators=400`, `max_depth=24`, `learning_rate=0.02`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=20`, `reg_lambda=20`.
- All feature engineering lives in `prepare(df)`; category levels / target-encoding maps are fit on training data only,
  so `predict_proba` reproduces everything on the hidden holdout.

## The changes that mattered most
1. **`carrier_hour` interaction categorical** — `UniqueCarrier` × 30-minute departure-time bucket. This was the single
   largest win (+0.009 over the base ensemble), and finer time resolution mattered: 30-min buckets (0.7321) beat both
   20-min (0.7305) and 15-min (0.7292). Time-of-day × carrier is the dominant delay signal.
2. **Seed ensembling** — averaging 5 XGBoost seeds gave a robust +0.002 over a single model.
3. **Deeper trees** — with the strong `carrier_hour` feature, depth 10→24 monotonically helped (0.7231→0.7351).
4. **Lower learning rate + more trees** — depth 24 with `n_estimators=400, lr=0.02` added another +0.002 (0.7374).
5. **Stronger L2 regularization** — `reg_lambda=20` at depth 24 lifted 0.7374→0.7385 (L2 was the sweet spot; 100 hurt).

## What did NOT help (reverted)
- Raw time features (`dep_hour`, `hour_sin/cos`, `dep_minutes` as numeric): neutral to slightly negative.
- Other high-cardinality interaction categoricals: `route` (−0.009), `origin_hour`/`dest_hour` (−0.002),
  `carrier_month`/`carrier_dow` (−0.011), `carrier_dist` (−0.007). Only the low-cardinality `carrier_hour` helped.
- Count/frequency encoding of carrier/origin/dest, and global (non-carrier) 30-min time bucket: no gain.
- Too-aggressive regularization (`reg_lambda=100`, `min_child_weight=50`) or L1 (`reg_alpha=1`): small losses.

## What I would try with more budget
The dominant lever was clearly the departure-time signal, so I would keep mining moderate-cardinality time
interactions (e.g. `Origin` × coarser 2-hour buckets, or `carrier` × time-of-day smoothed target encodings rather
than raw categoricals) with the deep, well-regularized model. I would also run a proper internal time-based
validation (train on early 2005, validate on late 2005) to tune `max_depth`/`reg_lambda` against the 2005→2006
distribution shift instead of relying on eval, and possibly add early stopping to trim the 400-tree depth-24 models.
Given the balanced target and the year shift, a small amount of additional smoothing/regularization is likely to
generalize better than any further eval-only gain.
