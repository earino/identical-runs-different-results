# Final Report — airline delay (AUC)

**Best Eval AUC: 0.7232** (experiment #39, commit `6c95584`), up from the 0.7141 baseline.

## Changes that mattered most
1. **Heavy regularization + deep trees.** This was the single biggest lever. Adding
   `reg_lambda=20`, `reg_alpha=5`, `gamma=3` and raising `min_child_weight` to 10–20 let the
   ensemble grow much deeper trees (final depths 9–11 across members) without overfitting;
   each depth increment kept adding a little (0.7198 → 0.7209 → 0.7219 → 0.7223 → 0.7227 →
   0.7228 → 0.7230 → 0.7232).
2. **Day-of-year cyclic features** (`doy_sin`, `doy_cos`) — the only feature that produced a
   clean, repeatable gain (+0.0015). Aligning seasonality across the 2005→2006 split matters.
3. **Seed / hyperparameter-diverse XGBoost ensemble** (12 members varying depth, subsample,
   colsample, min_child_weight, seed). Small but consistent variance reduction
   (0.7170 → 0.7189 as diversity grew, then combined with regularization).
4. **Network / frequency features** — `freq_*` of origin/dest/route/carrier,
   `route_ncarriers`, and carrier airport-share. Small but positive (+0.0002–0.0003).
5. **Time-of-day engineering** (`hour`, `tod_sin`, `tod_cos`) on top of the raw `DepTime`.
   Neutral-to-slightly-positive alone, but `DepTime`/`tod_sin` are by far the top-importance
   features, so the time representation is central.

## Things that did not help
1. **`route` (Origin_Dest) as a raw categorical** — catastrophic (−0.013); it overfits the
   2005 route set and does not transfer to 2006.
2. **Out-of-fold target encoding** of carrier/origin/dest/route — slightly negative
   (0.7162 vs 0.7170 at the time); native XGBoost categorical handling already captures it.
3. **Train target-rate features** (hour/carrier/origin/dest/month means) — marginally negative.
4. **More trees at low regularization** (800–1200 trees, lr 0.015–0.03, depth 6) — overfits the
   2005 slice; 400 trees lr 0.03 was already past the optimum until regularization was added.
5. **`grow_policy=lossguide`**, `dow` cyclic, numeric `tod`, dropping Month/DayofMonth/DayOfWeek
   categoricals — all neutral or worse.

## What I would try with more budget
The time-separated split (train 2005, eval/holdout 2006) means the dominant failure mode is
year-to-year drift in delay rates, so I would focus on drift-robust methods rather than more
capacity: (a) a proper time-series validation split inside 2005 (e.g. train on months 1–9,
validate on 10–12) to pick iteration count and regularization without touching eval; (b)
calibrating/renormalizing per-month or per-carrier predicted rates to the target-year marginal;
(c) monotonic constraints on `hour`/`DepTime` delay propensity; and (d) bagging many
strongly-regularized deep models with different feature subsets. The regularization-vs-depth
trade-off was still improving at the budget cap, so a slightly deeper/regularized configuration
plus drift correction is the most promising direction.
