# FINAL — autoresearch XGBoost (airline dep-delay, AUC)

## Result

- **Best kept Eval AUC: 0.7580** (experiment #14, commit 9bde5d7, 85 s runtime; contract-validated, `CONTRACT OK`).
- A 4-seed variant measured 0.7581 (#15) but ran 114 s — within 6 s of the 120 s hard cap; reverted for
  time-margin robustness (the +0.0001 is inside eval noise of ±0.0005–0.001).
- Baseline (#1): **0.7141** → final **0.7580** (+0.0439) over 15 experiments; every kept step was a
  structured, multi-seed-verified change.

## The 5 changes that mattered most

1. **Kernel-smoothed circular time-of-day delay-rate profiles** (the core feature family). For each key
   (Origin, Dest, UniqueCarrier) build a 1440-min circular histogram of the positive rate on train only,
   Gaussian-smooth it (σ 8–60 min), shrink toward the global prior (m 3–10), and look it up at the flight's
   DepTime. Training rows use 5-fold OOF values (no self-leakage); unseen rows use full-train profiles.
   Replaced hard hour-bins/plain target encoding: 0.7141 → ~0.7393.
2. **Arrival-time-indexed profiles.** Index the same smoothed curves at the flight's *approximate arrival
   time* (DepTime + 45 min + 0.15·Distance) instead of departure time, for Dest/Origin/Carrier with fine σ
   (8–15 min). Captures congestion at the arrival airport around landing time: +0.005 → ~0.7448.
3. **Composite-key profiles**: route (Origin×Dest, σ15/m2), Origin×Carrier (σ10/m2), Dest×Carrier
   (σ10/m2) at departure time — sparse keys made safe by heavy kernel smoothing and shrinkage:
   +0.002 each-ish → ~0.7578.
4. **Deep trees once features became smooth**: with raw features depth >6 badly overfit the 2005→2006
   shift (d8: 0.6958); with 13 smoothed profile features the optimum moved to depth 18 (450 rounds,
   lr 0.04, max_bin 128, subsample .9, colsample .6, α .5, 3-seed ensemble): d6 → d18 ≈ +0.004 → 0.7580.
5. **XGBoost gblinear member blended at w=0.25** with the tree ensemble average (same features, cats as
   integer codes): ranking diversity, +0.0015–0.002.

## 3 things that did not help (tested, reverted)

- **Month/DayofMonth features and month/day interactions**: year-specific noise; dropping them helped.
- **Route as a native categorical / plain target encodings / hard time-bin counts / monotone constraints /
  rank:pairwise objective / DOW-conditioned profiles / product-interaction features / depth-mixed
  ensembles / residual-profile boosting**: all flat or worse.
- **Bigger rounds at fixed depth and early stopping on 2005 splits**: no gain — 2005-validation rounds
  underestimate the time-shifted optimum; fixed 450–600 low-lr rounds were better.

## Theory of the data

2005→2006 is a distribution shift: raw high-cardinality cats and calendar features carry year-specific
noise, while *smoothed conditional delay-rate curves* (time-of-day × airport/carrier/route) encode stable
operational structure that transfers. Once features are smoothed statistics, capacity helps again — deep
trees can exploit them without memorizing 2005.

## With more budget

Estimate per-route flight durations from the data (e.g., route × Distance quantiles) to sharpen arrival-time
indexing; add arrival-time variants of the composite keys (route/oc/dc at arrival); learn the shrinkage m
per key family by CV instead of hand-setting; larger seed ensembles under a self-imposed 100 s budget;
calibration-aware weighting of the tree/linear blend; feature-wise `max_bin`; and a proper 2005-month →
2006 transfer study (recency weighting) instead of the quick negative test.
