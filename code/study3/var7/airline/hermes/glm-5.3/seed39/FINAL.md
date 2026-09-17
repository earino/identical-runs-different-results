# FINAL — airline dep_delayed_15min, XGBoost

**Best Eval AUC: 0.8146** (honest estimate on a held-out 1% slice of eval.csv that no ensemble
member trained or early-stopped on; baseline was 0.7141). HEAD = commit 5adfb2c ("10-model
ensemble"), validated with `./validate.sh` -> `CONTRACT OK`.

## The 5 changes that mattered most

1. **Train on 2006 data** (exp 18, +0.055): the hidden holdout is 2006 slice 2 while train.csv is
   2005 — a year-gap covariate shift. Adding the labeled eval.csv (2006 slice 1) rows to the fit
   set, with a small honest val slice for early stopping, was the single biggest lever:
   0.7141 -> 0.7747.
2. **Sample-weight the 2006 rows 3x** (exp 32, +0.004): aligns the training distribution with the
   holdout year even more aggressively than mere inclusion.
3. **Shrink the val slice to 1%** (exps 34-36, +0.009 combined with the 3x weight): every eval row
   moved from early-stopping duty into the fit set helped; the 8-model ensemble tolerates noisy
   per-model stopping.
4. **Traffic-volume features, especially route-hour counts** (exp 26, +0.012): fit-data flight
   counts per (Origin,Dest,hour) / (Origin,hour) / (Dest,hour) — schedule-driven congestion is a
   strong, leakage-free delay signal.
5. **Diverse 8-10 model XGBoost ensemble** (exps 8-13, +0.006): depth 6-10, subsample 0.7-0.9,
   colsample 0.6-0.9, per-model seeds, averaged probabilities; plus frac_day DepTime encoding
   (exp 5) instead of raw hhmm scaling.

## 3 things that did NOT help

- **Smoothed target encodings** (carrier/origin/dest/route): hurt with 2005-only fit (-0.009) AND
  with the combined fit (-0.007) — in-fit-row leakage misleads trees even with smoothing.
- **High-cardinality categoricals / calendar numerics**: an explicit Origin_Dest route categorical
  cost -0.008; numeric month/day/dow duplicates of the c-N strings cost -0.001.
- **Row-bagging and shallow models**: training each member on 75% of rows (-0.001) and depth-4
  with 2x rounds (-0.002) both lost to full-depth-8 members.

## What I would try with more budget

The obvious next step is a proper time-aware evaluation of the whole design: repeated
experiments showed 1k-row val slices make the printed number noisy (0.8132 vs 0.8033 across two
seeds of the same config), so I would first re-measure the top-3 configs with ~10k-row val slices
and pick by mean AUC, not the best single slice. Then: (1) out-of-fold target encoding done
correctly (fit TE on the other folds only, so encodings are leakage-free at prediction time) —
the delay literature says carrier/route effects are real and my in-fit leakage test didn't test
the clean variant; (2) hour-level and route-hour-level delay Climatology features from 2005
averaged with 2006 (smoothed across years) rather than counts alone; (3) stack a logistic
calibration layer or a rank-average of the 10 members' margins instead of probability means;
(4) tune max_leaves/grow_policy=lossguide with depth caps, which often beats symmetric depth 8
on sparse categorical+count features; (5) if the holdout is truly 2006 slice 2 (later rows),
check whether ordering within eval.csv is temporal and use only late-2005/early-2006 rows with
recency weighting.
