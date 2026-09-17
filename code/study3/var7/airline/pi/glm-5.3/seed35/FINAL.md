# Final Report

**Best Eval AUC: 0.7914** (experiment #15, commit `5f9d0d7`), validated end-to-end: `./validate.sh` prints
`CONTRACT OK` and reproduces `Eval AUC: 0.7914` through the module-level `predict_proba(df)` on data/eval.csv.
Budget at finalization: CPU compute exhausted (18,022 / 18,000 s); 24 experiment slots and ~40 min wall clock
remained unused. Baseline (experiment #1) was **0.7141**, so the loop gained **+0.0773 AUC**.

## The changes that mattered most

1. **OOF-honest interaction target encodings (TE)** (+~0.015 over plain features): smoothed target means of
   `Origin×hour`, `Route×hour`, `Route×30min`, `Route×15min`, `Carrier×Route×hour`, etc. The critical detail is
   honesty: train rows get 5-fold out-of-fold TE values, unseen rows get full-train maps. Any in-sample TE
   destroyed AUC (trees over-trust inflated correlations). TE *difference* features (route-vs-origin, carroute-vs-route,
   dest-vs-origin) added another +0.003 because a single tree split cannot compute them.

2. **Same-day traffic-density features (+0.028, the single biggest step)**: label-free counts computed **within the
   incoming dataframe** — `Origin×day`, `Origin×hour`, `Dest×day`, `Dest×hour`, `Carrier×day`, `Carrier×hour`,
   whole-slice `day×hour` load, hour-share-of-day (schedule peakedness), and airport-relative busy-ness ratios.
   These capture *actual operational congestion on that very day* (weather days, schedule banks), which
   train-2005-fitted statistics cannot express. Each count is `rank(pct=True)`-transformed, making the features
   invariant to the sampling density of the slice — essential because the hidden holdout has 10x the rows of
   train/eval. Raw train-fitted volume counts were actively harmful in comparison (0.7740 vs 0.7776).

3. **Two-family seed-bagged ensemble (+0.005)**: 6 full-feature models + 3 "stable" models (numeric, calendar,
   categorical and density features only, all TE and TE-difference features dropped). The stable family hedges
   the drift-prone route-level TEs; averaging 9 XGBoost models (different random_state) is worth +0.002 by itself.

4. **Lossguide growing + capacity** (+0.003–0.005): `grow_policy="lossguide"`, `max_leaves=64→96`,
   `n_estimators=200→300`, lr 0.06→0.05, `colsample_bytree=0.4`, `subsample=0.8`, `reg_lambda=20` with native
   categoricals (`enable_categorical=True`) for Carrier/Origin/Dest.

5. **Simple calendar/numeric hygiene** (part of the early gains): integer month/day/dow, dep_hour/dep_min,
   minutes-since-midnight, Distance as numeric.

## Things that did not help (all reverted)

- **In-sample (non-OOF) target encoding** — catastrophic AUC loss; over-trust of inflated correlations.
- **Route as a native categorical** — the 1-bit-per-split representation of 6k routes is far weaker than TEs.
- **Window/rolling and residual TEs, seasonal/dow/month TEs, hierarchical priors** — flat or worse.
- **TE products** (e.g. `te_orighour × te_desthour`) — actively harmful (−0.01); they create spurious interactions.
- **Rolling within-day origin/dest load (±45/90 min), prev/next-hour counts, position-in-day ranks** — the same-hour
   density already carries the congestion signal; finer/lagged variants added only noise.
- **dart, deeper/wider trees, 10-fold OOF, more folds-seeds for TE, colsample_bynode, max_bin=512, family
   weight tuning, rank-averaging, multi-family depth mixes, >9-model bags** — all within noise (±0.0003).

## What I would try with more budget

The strongest remaining direction is closing the train/serve gap of the TE features on the *hidden* slice: fit
TEs on a temporally reweighted train (e.g. upweight late-2005 to match 2006 seasonality), and shrink route-level
TEs toward their airport-level parents by *cross-time* stability rather than sample size alone (buckets whose
2005H1 and 2005H2 rates disagree should borrow more strength). I would also build a proper time-machine
benchmark — train on Jan–Jun 2005, validate on Jul–Dec 2005 — to measure which features actually transfer across
time before trusting them, and use it to prune the drift-prone route TEs. Finally, the same-day density family
suggests modelling *system state*: per-airport departure-bank timing (share of the day's flights departing in
this flight's ±90-min window as a fraction, which was flat here but only tested raw) and, if allowed, a larger
seed bag (20+ models) at the current 300×96 capacity, which was just beginning to pay (+0.0004 at the last rung)
when the CPU budget ran out.

## Reproducibility

- Final model: `train.py` at commit `5f9d0d7` (experiment #15: two-family 6+3 XGBoost seed bag, 10 OOF-honest
  interaction TEs + 5 TE-differences + 12 within-df density features + calendar/numeric/categorical features,
  300 lossguide trees, lr 0.05, 96 leaves, float32 matrices).
- `validate.log`: `CONTRACT OK`, eval AUC 0.7914 via `predict_proba` with the target column removed.
- All kept experiments are on branch `experiment`; every non-improving commit was reverted per protocol.
