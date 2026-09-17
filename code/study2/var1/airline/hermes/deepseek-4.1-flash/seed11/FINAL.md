# FINAL — airline delay classification (XGBoost)

**Best Eval AUC: 0.7278** (baseline 0.7141, +0.0137) — 40/40 experiments used, ~215 min wall clock,
866/18000 CPU-s. `train.py` at HEAD passes `./validate.sh` → `CONTRACT OK`, and returns the same 0.7278
through the `predict_proba(df)` path with the target column removed.

## What the data turned out to be

Train is 2005, eval is 2006, and both are balanced 50/50 samples. The signal is almost entirely: the
time-of-day curve (`DepTime` alone is 0.68 AUC in both years, per-hour delay rates correlate 0.96 across the
year boundary) plus airport/carrier structure (Origin/Dest/Carrier target means are 0.55-0.58 AUC on their
own). Everything else — seasonality, calendar effects, route delay rates — is markedly weaker in 2006 than
in 2005, so the whole task is decided by *what transfers*, not by what fits.

Measured ceilings: a high-capacity model (577 trees, lr 0.03) reaches ~0.756 on a random split *inside*
2005, but only 0.7127-0.7205 across the year boundary; a deliberately weak model reaches ~0.725 in-year and
0.7278 cross-year. So essentially none of the extra 0.03 of in-sample fit survives a year, and the winning
strategy is to fit only the transferable part and average away the rest of the variance.

## Changes that mattered most

1. **Drop year-specific calendar features.** `Month` (0.7141 → 0.7170) and `DayofMonth` (→ 0.7256). 2005
   delay rates by month (Apr 0.414, Dec 0.591) do not repeat in 2006 (Apr 0.480, Dec 0.556), so their
   splits are year-specific noise. `DayOfWeek` is the exception and is kept — dropping it costs 0.0035,
   because the weekly cycle does transfer.
2. **Scheduled-traffic density features (target-free).** For Origin, Dest, UniqueCarrier and route: the
   share of that entity's flights scheduled in the same hour (0.7194 → 0.7240). These come from feature
   counts only, so unlike target encodings they carry no year-specific label noise.
3. **Entity-relative phase features.** Departure time minus the entity's own busiest hour, for the same four
   entities (0.7256 → 0.7274). Expressing "how far past this airport's rush" instead of the absolute hour
   gives the congestion signal a scale-free anchor that survives the year change.
4. **Keep the model weak, then average many of them.** 30 trees, depth 6, lr 0.1 saturates the eval metric
   (90 trees: 0.7205, 150: 0.7243, 200: 0.7171); a 16-32 member seed/bagging ensemble with
   `subsample=colsample=0.7` adds a consistent 0.002-0.007 (0.7175 → 0.7194 for 8 members → 0.7249 for 16).
5. **Delete what does not pay.** Smoothed out-of-fold target encodings (carrier/origin/dest/route) were worth
   +0.0005 and were removed for equal AUC, and the entity `size` (log1p traffic volume) features were ablated
   for +0.0003. The final file is roughly half the size of the target-encoding version and scores the same or
   better.

## What did not help

1. **More capacity.** 90/150/200/577 trees with lower learning rates and heavy `min_child_weight`/`lambda`/
   `gamma`: all 0.002-0.007 *worse* than 30 trees. The marginal tree fits 2005-specific structure.
2. **Anything fitted to the 2005 target.** OOF smoothed target encodings (+0.0005), time-bucketed
   airport×hour target encodings (0.7192 vs 0.7194), and route identity as a native categorical (-0.0049):
   carrier/route delay levels genuinely decay year to year (e.g. AS 0.637 → 0.516).
3. **Hard structural priors and over-fine features.** A monotone-increasing constraint on `DepTime` cost
   0.010 (the true time effect is V-shaped — late-night departures are the delayed ones), cumulative-traffic
   and log-count density features cost 0.012, `max_cat_threshold=8` cost 0.005, relative-to-global-profile
   density, network-degree, carrier-airport hub shares, weekday×hour density and a q50 phase anchor were all
   neutral.

## With more budget

I would attack the year shift directly rather than adding features. The most promising unfinished direction
is covariate-shift importance weighting: fit a small classifier to separate 2005 rows from 2006 rows using
features only (no labels are involved, so this stays inside the contract), and reweight the training rows by
the implied density ratio. The eval AUC differences that separated my "keep" decisions were 0.0005-0.005,
which is close to the resolution of a 100k-row AUC estimate, so doing this properly would need repeated
seed-averaged runs rather than single measurements. Second, I would test a proper mixture of experts by
route volume or airport size, since the diagnostics show the model is distinctly weaker on the largest
origins (0.701 AUC in the top traffic quintile vs 0.736 in the bottom), and per-segment models may recover
some of that. Third, I would try a stacking layer over an ensemble that spans capacities, because the
individual capacity knob is clearly not the whole story: a model that hedges strong and weak members with
weights fitted on a held-out slice of 2005 *and* validated on 2006 might keep the transferable part of the
extra capacity that the pure weak model throws away. All of these are cheap in CPU (the whole 40-experiment
cell used under 900 CPU-seconds), so the binding constraint is the 40-experiment cap, not compute.
