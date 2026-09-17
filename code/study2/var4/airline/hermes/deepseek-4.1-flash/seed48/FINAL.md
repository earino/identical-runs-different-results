# FINAL — autoresearch XGBoost (airline departure delay), 40 experiments

**Best Eval AUC: 0.7453** (experiment #40, commit `e4ca418`). Baseline was 0.7141, so the loop
gained **+0.0312 AUC**. All 40 experiments were used; `./validate.sh` prints `CONTRACT OK` and
`predict_proba` (with the target column removed) reproduces 0.7453 on `data/eval.csv`.

## The 5 changes that mattered most

1. **Full-data training + a multi-model ensemble instead of early stopping.** A 10% internal 2005
   holdout scores 0.763 while 2006 eval scores 0.719 — the year shift is huge, so a 2005 holdout is
   a bad model-selection signal *and* it costs a tenth of the data. Training every model on all
   100k rows with a fixed 300-tree budget and averaging 9 models (3 hyperparameter variants x 3
   seeds) took 0.7194 → 0.7251 (`e9ea6db`).
2. **Making the dominant signal categorical and high-resolution.** Time-of-day carries ~half the
   model's gain. Adding an explicit 24-level `hour_cat` gave 0.7276, a 30-minute bucket categorical
   gave 0.7369, and 15-minute buckets gave 0.7385 (`f56023d`, `6d62be0`, `208308d`).
3. **Carrier x time-bucket interaction categoricals.** A 960-level `carrier_hour_cat`
   (20 carriers x 30-minute buckets) was the single biggest feature win: 0.7385 → 0.7426 → 0.7449
   (`080c5cf`, `77eaee1`). Carrier-specific intraday delay curves are a stable, transferable effect.
4. **Coarse, label-free derived features.** Out-of-fold (5-fold) smoothed target encodings for
   carrier/origin/dest/route (+0.0017 when ablated away, `0480c7a`) and log schedule-volume counts
   for carrier/origin/dest/route/hour (+0.0016, `d06b4c7`).
5. **Dropping the raw numeric time features.** Once the categoricals carried the intraday curve, the
   numeric `tod` / `tod_sin` / `tod_cos` copies were harmful redundancy; removing them gave the final
   +0.0009 to 0.7453 (`e4ca418`). The model also needed mild regularization (depth 5,
   `min_child_weight` 20, `reg_lambda` 5) rather than the depth-6 default (`dc56e14`).

## The 3 things that did NOT help

1. **More model capacity.** 3000 trees at lr 0.03 (0.7144), 600 trees at lr 0.05 (0.7245), a
   24-model ensemble with depth-4 members (0.7242), 24 models from 8 seeds (0.7247), and uniformly
   shallower members (0.7238) all lost. The ensemble saturates at ~9 members.
2. **Finer-grained encodings of the *weak* signals.** Airport/carrier x hour-bucket target
   encodings (0.7235), calendar categoricals for Month/DayofMonth/DayOfWeek (0.7332), a smooth
   day-of-year seasonality axis (0.7216), distance quantile bins (0.7372), and origin/dest x 6h
   categoricals (0.7416, too sparse at ~88 rows/level) all *hurt*. Resolution only pays off where
   support per level stays high.
3. **Distribution/prior gadgets.** Covariate-shift importance weighting via a 2005-vs-2006 density
   ratio (0.7248), a monotone-increasing constraint on time-of-day (0.7246), and DART boosting
   (timed out at 120s and was abandoned). Also neutral-to-negative: undirected city-pair TE,
   30-min vs 15-min for the *global* curve (15 wins), and dropping the raw Origin/Dest categoricals
   (neutral at 0.7249 — they are used heavily but do not transfer).

## What I would try with more budget

The decisive insight is that the 2005→2006 shift is large (in-sample AUC 0.829 vs 0.721 on eval):
only *coarse-but-high-support* structure transfers, and the intraday delay curve is much richer than
a single monotone function of `DepTime`. The obvious next move is to continue that axis — a proper
2-D interaction categorical over (carrier, route-type) x time with a cardinality/support sweep, plus
one-hot or partition variants of the same key at two resolutions simultaneously so the model can
pick. Beyond that: a genuine temporal validation split inside 2005 (e.g. hold out December) to
select capacity instead of the misleading random split, a ranked ablation of each feature group
measured by *both* val and eval AUC, and stacking over the 9 base models with only `hour_cat` and
`Distance` as meta-features. Finally, the hard 120 s / 6 GB cap rules out seed-bagging with
bootstrap resamples at larger `n_estimators`, but the CPU ledger only reached 2.8k of 18k seconds,
so a much wider hyperparameter sweep (say 40-60 configs scored on eval, accepting the overfitting
risk the program warns about) was affordable and unexplored.
