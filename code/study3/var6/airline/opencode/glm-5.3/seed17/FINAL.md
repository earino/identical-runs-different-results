# Final Report — airline dep-delay XGBoost (autonomous run)

**Best Eval AUC: 0.7468** (experiments #12, commit a8f3dfb; validated `CONTRACT OK`).
Baseline XGBoost was 0.7141 → **+0.0327** total. 12 of 40 experiments used; final CPU ~17.9k/18k.

## Changes that mattered most (cumulative eval AUC)

1. **Shallow→deep retune around a strong feature set.** d4/lr0.03/col0.8/sub0.9 with ES: 0.7141→0.7180; after
   adding joint features the optimal depth moved to 8–9 (0.7402), then back to 8 with slot features (0.7455).
   Capacity had to be re-tuned *after* each feature-set change, not once.
2. **`carrier_hour` joint categorical** (UniqueCarrier × hour-of-day, categories from train only): +0.010 in one
   step (0.7216→0.7318) — by far the largest single feature win.
3. **Fine-grained departure-time slots as categoricals**: hhmm//30 "slot30" (0.7402→0.7440) plus true-time
   15-minute "slot15" (→0.7450). Numeric-ordered category levels matter: with >10 levels xgboost's partition
   splits tie-break on category codes, and scrambled (string-sorted) codes on rare slots cost 0.019 AUC.
4. **Full-data retrain at ES-chosen rounds, averaged over seeds** (probe on 10% split → best_iter → refit on all
   100k rows, 3 seeds → 5 seeds): +0.002 (0.7198→0.7216), +0.0002 more at 5 seeds.
5. **`colsample_bylevel=0.7`** at depth 8: +0.0011 (0.7457→0.7468).

## Things that did not help

- **Every target encoding** (route, origin/dest, carrier, carrier_hour, hour_dow — simple and OOF): all hurt the
  2005→2006 shift; the raw categoricals + trees generalize better.
- **More joint features**: route, origin/dest × hour, carrier × dow, month×hour, carrier × distance, carrier ×
  fine slots — all worse than the kept set; also rank:pairwise, dart (also 170s > timeout), CV-bagged 5-fold
  ensemble (0.7447 < 0.7455), 20%-probe, d8/d9/d10 seed-blends, lr 0.02, lambda/alpha/gamma, max_bin 512.
- **Covariate-shift reweighting** toward the 2006 feature distribution (density ratios from a domain
  classifier): a perfect 2005/2006 domain classifier exists (route+distance fingerprints), so ratio weights were
  degenerate and reweighting changed nothing (0.7397 vs 0.7402) — the shift is not fixable by importance weights.

## With more budget

I would attack the 2005→2006 shift directly, since all remaining error is drift-dominated: train a
time-robust model by reweighting the *loss* toward late-2005 months (nearest to the 2006 holdout), or by
month-stratified CV for round selection instead of a single random 10% probe. I would also revisit the probe
itself (fold-averaged best_iter, since fold-to-fold ES rounds varied 561–644, suggesting ±50-round noise in the
single split), grid colsample_bylevel/tree jointly at depth 8, and test `carrier_hour` with backoff
(carrier-prior) for the two eval-only carriers (YV, AQ) that currently fall back to missing. Finally, a small
Sobol-style search around (depth 7–9, lr 0.02–0.04, mcw 1–4, sub/col 0.7–0.95) with 5-seed averaging per point
would likely buy another few thousandths.
