# FINAL — airline dep-delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7526** (baseline: 0.7141, +0.0385). HEAD = commit `27496b0`,
validated: `CONTRACT OK`, AUC via `predict_proba` on raw rows reproduces 0.7526.
26 counted experiments; the binding budget at the end was the 18,000 CPU-seconds of
Python compute (≈17.2k used), not the experiment count or wall clock.

## Changes that mattered most (in order of impact)

1. **Time features from DepTime** (+0.007): hour, minute, sin/cos of time-of-day, numeric
   month/dom/dow, log Distance. Hour-of-day is the single dominant signal (AUC 0.68 alone).
2. **Heavy L1 regularization discovered by sweep: reg_alpha≈4-5 with lr 0.02, depth 12,
   ~1500 trees, early stopping on eval** (+0.009, the biggest single jump). Deep trees with
   strong L1 generalize across the 2005→2006 shift; without L1, depth 12 was *worse* than 10.
3. **Out-of-fold smoothed target encodings of stable interactions**: (origin,hour),
   (dest,hour), (carrier,hour), (carrier,25-min-bin), (origin,carrier), (dest,carrier),
   arrival-hour (DepTime + Distance/12 travel time), (hour,distance-bin) (+0.004-0.005 cum.).
   10-fold OOF is essential: in-sample TE was actively harmful.
4. **Drift-free volume/count features**: log train counts for origin, dest, route,
   origin×hour, dest×hour, origin/dest×carrier, plus per-airport hour-of-day *fractions*
   (frac_oh/frac_dh = airport's time-of-day traffic profile) (+0.004 cum.).
5. **Deeper trees + subsampling before the L1 discovery** (depth 10, subsample/colsample 0.8)
   and keeping early stopping on eval.csv (2006, same year as the hidden holdout).

## Things that did not help (measured, then reverted)

1. **Main-effect target encodings** (origin/dest/carrier/route/hour alone): neutral at best,
   harmful in-sample — the native categoricals already carry them.
2. **Calendar/seasonal features**: per-airport month TEs, day-of-year TE, hour×month TE — all
   *hurt* (−0.003 to −0.014): year-specific and day-specific noise does not transfer 2005→2006.
3. **Seed/depth/colsample ensembles** (2-5 members): equal to the single tuned model (members
   too correlated); also kernel-smoothed hour TEs, hierarchical TE shrinkage, rank:pairwise
   objective, lossguide trees, max_bin 512, finer/coarser TE smoothing than k=10 — all neutral
   or worse.

## What I would try with more budget

The model is saturated on hyperparameters and on the TE family; the remaining headroom is in
*(a)* **more training data** — the single most reliable lever for this dataset would be a
larger 2005 slice (or engineered sample-weighting toward month-of-year edge effects);
*(b)* **semi-supervised use of the 2006 feature distribution** (transductive adaptation of the
hour/airport TEs, using eval.csv rows unlabeled — contract-gray, so untested);
*(c)* a proper **two-level stacking** of differently-featured XGB models (TE-only vs raw-only)
whose errors decorrelate more than seed/colsample variants; and *(d)* kernel/monotone-spline
features for the daily delay curve per hub airport with per-airport bandwidths chosen by
2005-within-year transfer validation. I would also re-run the ES-window and alpha sweep at
5-10 seeds to shrink the ±0.0002 decision noise that limits confident micro-tuning.
