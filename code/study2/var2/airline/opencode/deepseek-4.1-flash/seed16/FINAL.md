# Final Report — airline departure-delay AUC

**Best Eval AUC: 0.7578** (experiment #25, `depth 11 bagged`, commit `de1c325`).
Contract validated on the final `train.py`: `CONTRACT OK`.

## Setup
`train.py` builds features in `prepare()` (so `predict_proba` applies the identical path to
unseen rows), fits statistics only on `data/train.csv`, and trains an ensemble of XGBoost
boosters with `objective="reg:squarederror"`, `max_depth=11`, `eta=0.02`, 1600 rounds,
`colsample_bytree=0.6`, `subsample=0.9`, a monotone-increasing constraint on departure
hour, and 5 seeds averaged.

## Changes that mattered most
1. **Feature-bagged ensemble** — each of 5 members trains on a random 80 % subset of the
   feature columns (with the matching monotone-constraint vector). This was the single
   biggest gain: 0.7527 → 0.7575. Decorrelating members via feature sampling beat merely
   adding seeds.
2. **Traffic-share ("congestion") features** — `oh_share`/`dh_share` (an airport-hour's
   share of that airport's daily volume) and `co_share_c`/`co_share_o`/`ch_share_c`
   (carrier-origin / carrier-hour share of carrier volume). Together these lifted the
   model from ~0.7466 to 0.7527 and were the only engineered features that transferred.
3. **Objective + schedule** — squared error for a ranking/AUC metric, low `eta=0.02` with
   many trees, and adding capacity to 1600 rounds once bagging regularized the fit
   (0.7527 → 0.7576).
4. **Encoding choices** — ordinal codes for low-cardinality columns (Month, DayofMonth,
   DayOfWeek, UniqueCarrier) plus frequency encoding for high-cardinality ones, and a
   smoothed (alpha=5) target encoding of Origin/Dest.
5. **Monotone constraint on hour** — a small, principled regularization that matched or
   beat unconstrained fits in ensembles.

## Changes that did not help
- **Native XGBoost categorical encoding** (~0.70, far below frequency encoding).
- **Extra target encodings** (route, origin-hour, dest-hour, residualized variants) and
  **other share/ratio features** (route-hour, carrier-route, node-diversity counts) — all
  neutral or harmful.
- **Calendar flags** (weekend/Friday/holiday-season/summer) and **neighborhood-hour volume**
  features — no gain.
- **Simply enlarging the ensemble** (7–8 seeds) or micro-tuning hyperparameters
  (eta/colsample/lambda/min_child_weight) — flat within eval noise.

## What I would try with more budget
The remaining headroom is in averaging rather than more raw features. I would (a) run a
proper 2-level stack: K-fold out-of-fold XGBoost predictions from several feature-bagged
base configs, then a small meta-XGBoost on those OOF scores plus a few raw features; (b)
sweep bagging fraction and member depth jointly (e.g. half depth-10, half depth-13 members)
with more members, since member diversity — not member count — drove the gains; and (c)
test per-member `colsample`/`subsample` randomization. Feature engineering on the eight raw
columns appears saturated: only volume-normalized congestion shares helped, so further work
should focus on better aggregation of the existing signals and on probability calibration
before ranking.
