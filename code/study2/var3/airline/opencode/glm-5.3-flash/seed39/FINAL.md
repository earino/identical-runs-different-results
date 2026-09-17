# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7375** (baseline 0.7141, +0.0234). Commit `e89cd8a`, validated (`CONTRACT OK`).

## Final model

28-member heterogeneous XGBoost bag (depths 3–9 × subsample 0.6–1.0, lr 0.05, 400 trees each,
max_bin 512, full-train fits, no early stopping), averaged probability predictions.

## Changes that mattered most

1. **Interaction target encodings with multi-scale time bins** — smoothed, OOF-fitted TE of
   Carrier/Origin/Dest × dep_time-bin (15/30/60 min): 0.7242 → 0.7375. By far the biggest win
   (+0.0133); delay propensity is a sharp function of (entity, time-of-day).
2. **Seed/hyperparameter bagging** — 5-seed → 28-member hetero bag over depths/subsamples:
   0.7167 → 0.7242 (+0.0075). Variance reduction is the main lever under 2005→2006 drift.
3. **Stochastic regularization** — subsample 0.8 + colsample 0.8: 0.7158 → 0.7167 first gain.
4. **Time-of-day feature engineering** — dep_min (hhmm→minutes), dep_hour, cyclical sin/cos,
   numeric calendar: 0.7141 → 0.7158.
5. **max_bin 512** — finer histograms for dep_min/Distance (+~0.0001, principled).

## Things that did NOT help (all reverted)

- Random-split early stopping (0.7092): loses 20% of train, and 2005-only validation tunes into drift.
- More capacity: 1500 trees/depth 8 (0.7093–0.7141); more/deeper members dilute or overfit.
- High-cardinality additions: Route as categorical (0.6987!), raw-feature target encodings
  (0.7151), day-of-year features (0.7123), DomMonth TE (0.7288), interaction TEs beyond the
  3×3 multi-scale set (0.7262–0.729), mcw diversity (0.7224), dep bins (0.7221),
  colsample 0.9 (0.7370), month-proximity weights (+0.0001 noise).

## Theory of the data

Delay risk is dominated by scheduled-departure time (0.04 at 5am → 0.98 at 24h+), modulated by
carrier/hub scheduling patterns; 2005→2006 temporal drift makes complexity the enemy: everything
that memorized 2005 specifics (routes, exact days, deep trees) transferred negatively, while
shrunk interaction statistics and ensembled medium trees transferred well.

## With more budget

- Scale the multi-scale TE idea to other pairings chosen by OOF value (Origin×Dow, Carrier×Origin
  were neutral-to-noise, so selection would need care), and tune alpha per cardinality.
- Two-layer stacking: OOF bag predictions + raw features into a shallow meta-XGB.
- 40–56 members at 300 trees within the 120s cap (needs faster prepare, e.g. polars), plus
  per-member colsample diversity.
