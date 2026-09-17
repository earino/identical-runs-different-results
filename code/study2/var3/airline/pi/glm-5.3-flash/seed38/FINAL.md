# Final report — airline dep_delayed_15min (XGBoost)

**Best Eval AUC: 0.7563** (baseline 0.7141, +0.042). Hidden-holdout scorer: `predict_proba(df)` in `train.py`
(HEAD = `4eae04b`, CONTRACT OK in validate.log).

## Final model

Depth-diverse bagged pair of XGBoost models (d20 seed7 + d24 seed13; n=1200, lr=0.035, colsample_bytree=0.4,
reg_alpha=2, hist, 4 threads), probabilities averaged. Trains in ~110 s.

## What mattered most (ordered by impact)

1. **DepTime decomposition** (hour, minute, time-of-day sin/cos): the raw hhmm integer is nearly opaque to
   trees; the decomposition is the single most predictive stable signal (+~0.002 with early configs, and it
   unlocked capacity growth later).
2. **Numeric-only feature set — drop the raw categorical columns** once their encodings exist (+0.0030).
   Native categorical splits memorize 2005-specific level sets; smoothed encodings transfer to 2006.
3. **Smoothed target encoding + frequency counts for Origin/Dest/carrier** (TE M=100, counts for
   origin/dest/route/carrier-hour/dest-hour/origin-dow) (+~0.015 cumulative): volume counts are year-stable;
   TE rates carry drift, hence smoothing.
4. **Feature subsampling (colsample 0.25→0.4) + L1 (reg_alpha)**: the main overfitting brakes. Row subsampling
   *hurt* consistently (-0.01) — never use it here.
5. **Capacity re-tuned after each feature wave** (lr 0.1/n30 → lr 0.03/n1500, depth 6→20-24): rich features
   made deep trees profitable (+0.014 from depth alone at the numeric-only set); depth tuning had to be
   *repeated* because the optimum moved every time features changed.

## What did not help

- Row subsampling (0.8), deeper trees *before* the feature waves, gamma/min_child_weight/reg_lambda bumps.
- Route/interaction columns as raw categoricals (Origin_Dest: -0.02); carrier_month TE; doy cyclicals;
  carrier mean distance; hub TEs; extra count features beyond the chosen set; hour/month/dow/dom TEs.
- Same-depth and cross-depth ensembles of >2 models (+0.0001 at best) and rank-encoded TE: plateau noise.

## Theory of the data

The 2005→2006 split introduces real drift (carrier delay rates move by ±0.1 AUC-relevant points), so the
winner is the model that encodes *stable structure* (schedule shape, volumes, spatial load) with smoothing
and L1, and avoids raw high-cardinality memorization. eval.csv (2006 slice1) is from the same year as the
hidden holdout (2006 slice2), so eval-based selection is a fair proxy — but gains <0.0005 were treated as noise.

## With more budget

- A proper time-based CV protocol (multiple year-splits) to tune against drift directly instead of one eval slice.
- Third-level target encodings with hierarchical shrinkage (route→origin→prior), OOF-fit to remove self-leakage.
- A small hardcoded airport geography table (timezone/region) for red-eye and congestion features.
- Larger seed-bagged ensembles with a bigger time budget (the 4-model variant timed out at 120 s).
