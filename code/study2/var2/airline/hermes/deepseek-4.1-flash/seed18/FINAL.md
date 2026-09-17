# FINAL — airline departure-delay classifier (XGBoost)

**Best Eval AUC: 0.7480** (experiments.tsv #36, commit `1ceffb6`; the final HEAD adds only a
dead-code cleanup, re-validated at 0.7480 via `predict_proba` with the target column removed).

Baseline (unmodified `train.py`, 30 trees): **0.7141**. 40/40 experiments used, no crashes, OOMs or
timeouts; ~7.3k of the 18k CPU-seconds budget consumed.

## What mattered most

1. **Schedule-structure features instead of identity features** (biggest single move, +0.0055 combined).
   Departure time parsed into hour / minutes-of-day / cyclic encodings, plus lookup tables built *only*
   from training rows: airport/route/carrier frequencies, carrier-vs-airport hub shares, share of an
   airport's departures in that hour, and the cumulative share of an airport's daily departures up to
   that hour (a congestion proxy). These describe the *schedule*, which is stable from 2005 to 2006.
2. **Dropping the high-cardinality `Origin`/`Dest` categoricals** (+0.0044, `b64eee3`). At 282 levels each
   they let the model memorise 2005 airport-specific weather; their information is carried much more
   robustly by the numeric surrogates above. Interesting asymmetry: `UniqueCarrier` (20 levels) *does*
   transfer and was kept — dropping it cost 0.002.
3. **Dropping absolute-calendar features** (`Month`, then `DayofMonth`, +0.0019 total). Same mechanism:
   calendar position encodes year-specific weather, not a recurring pattern. Day-of-week and time-of-day
   stayed (they recur weekly/daily).
4. **An 8-model XGBoost ensemble** (varied depth 6–10, lossguide variant, different subsample/colsample
   and seeds, probabilities averaged): +0.0032 over the best single model at the time, and it is the most
   robust lever tried — averaging tolerates the heavy per-model overfitting (train AUC ≈ 0.98).
5. **Capacity / bugs**: going from 30 to ~600–1200 trees was worth +0.011 early on; two real bugs were
   found and fixed (swapped lookup keys made the hub-share features constant 0, +0.0010; and route-level
   categoricals had to be removed outright).

## What did not help

- **Route-level identity**: `route` categorical (4198 levels, −0.016), `route_hour_cum` (−0.016),
  `route_hour_share`/`origin_dow_share` (−0.005). Anything keyed on the specific city pair memorises 2005.
- **Aggressive regularisation**: shallow depth-5 trees with `min_child_weight=10`, `lambda=2`
  (−0.002) and a lr-0.02/3000-tree run (−0.003) both lost. The optimum is a *narrow* capacity band
  (d9, ~600–1200 trees, lr 0.03–0.04); train AUC ≈ 1.0 is not itself the problem.
- **Smoothed target encoding of identity keys** (carrier/origin/dest/route, out-of-fold for training and
  train-fitted for prediction): ±0.0005 twice — fully redundant with the categorical features, and later
  with the numeric surrogates. Also neutral: monotone constraint on time-of-day (−0.0003), 10- and
  12-model ensembles (0.0000), `colsample_bynode=0.8` (−0.0001).
- **Extra structural features** hurt more often than they helped (dest-hour cumulative, carrier-hour
  share, arrival-side hub shares, extended ratio set: −0.001 to −0.005 each). The feature set is at a
  sharp local optimum; single additions are close to coin flips.

## With more budget

The score is bounded by 2005→2006 distribution shift, not by model capacity: every model reaches
train AUC ≈ 0.98 while eval sits at 0.748, and the winning moves were all *removals* of features that
encode year-specific events. I would attack that directly — train each ensemble member on a
time-respecting internal split (fit on earlier 2005 rows, early-stop on later ones) so tree counts are
chosen against the shift rather than against an unshifted validation set, and add per-model sample
weighting that down-weights exactly the memorised-perfect 2005 rows. On the feature side, the only
untried family with a plausible persistent signal is relative-time structure — e.g. each airport's
scheduled departure "slot index" and inter-departure gaps, which proxy bank structure — and
bagged/subsampled versions of the airport-level encodings to cut their variance. Finally, two or three
more evaluation folds would help: with hidden-holdout scoring, differences below ~0.001 are noise, and
several keep/discard calls in the log sit inside that band.
