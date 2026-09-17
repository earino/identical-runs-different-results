# Airline delay prediction — final report

**Best Eval AUC: 0.7436** (baseline: 0.7141) — commit `2994345`, experiment #40.
Metric: ROC-AUC on `data/eval.csv` (2006 slice), hidden holdout is 2006 slice2.

## What mattered most

1. **Out-of-fold target encoding of interaction keys** — the single biggest lever (+0.012 alone).
   Smoothed (k=40) delay-rate encodings of `Route`, `route_hour`, `origin_hour`, `carrier_hour`,
   `carrier_origin`, `carrier_route`, `origin_dow`, `carrier_dow`, and 3-way keys
   (`carrier_route_hour`, `route_dow_hour`, `carrier_dest_hour`, `carrier_route_dow`, `origin_dow_hour`, …).
   Train rows use out-of-fold encodings (no leakage); `predict_proba` applies the full-train map.
2. **Count/volume features for every encoded key** (traffic at an airport/hour, route frequency, group size)
   — +0.001, and gives the trees a reliability signal for each target encoding.
3. **Bagged ensemble of 20 diverse XGBoost models** (depth 5–7, 450 trees, lr 0.02, varied
   subsample / colsample / min_child_weight / reg_lambda, different seeds) averaged in probability space
   — +0.003 over a single model, and much more stable under the 2005→2006 distribution shift.
4. **Strong regularization + aggressive feature subsampling** with the wide TE feature set:
   `min_child_weight` 20–44, `reg_lambda` 8–20, `colsample_bytree` 0.2–0.4 — final +0.003.

## What did not help

1. **More single-model capacity**: 200 / 500 trees and higher depth overfit 2005 badly (internal holdout AUC
   kept rising to 0.75+ while eval AUC peaked near 30–75 trees). The A/B year gap is the dominant problem.
2. **`Route` as a native categorical feature** — steep drop (0.708); high-cardinality identity features
   destabilize rather than help. Consistent with this, the smoothed TE version *does* help.
3. **Plain / temporal / high-order target encodings** (carrier, origin, dest, month, dayofmonth, dayofweek,
   distance-bin interactions, 4-way keys) — all neutral or slightly negative; the useful signal is already
   captured by the 2/3-way volume interactions.

## With more budget

I would attack the temporal shift directly rather than adding capacity: (a) multi-scale target encoding
(several smoothing strengths as separate features so the model can weigh reliability adaptively);
(b) a proper time-aware internal validation split (e.g. hold out a pseudo-year slice) to select the number
of trees / smoothing without touching eval; (c) a stacked XGBoost meta-model over out-of-fold predictions of
the base ensemble; (d) monotonic constraints on smooth features such as `origin_hour` delay propensity to
reduce variance under shift. The data carries limited raw signal, so the returns are now small
(~±0.001), and the most reliable remaining gains look like variance reduction, not feature discovery.
