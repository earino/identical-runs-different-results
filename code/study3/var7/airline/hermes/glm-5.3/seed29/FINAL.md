# Final report — airline dep-delay XGBoost (autoresearch benchmark)

## Best Eval AUC: 0.7397 (commit f418210, experiment #38)

Baseline was 0.7141, so the final model gains +0.0256 AUC on the 2006 eval slice, using a
contract-conforming `train.py` (validated: `CONTRACT OK`, `predict_proba` reproduces the score on a
target-free DataFrame).

Important honesty note: experiments #34-36 logged 0.7666-0.7855, but that was a bug, not a gain.
`evald.index` happens to equal `train.index` (both RangeIndex 0..99999), so `df.index.equals(OOF_INDEX)`
fired for eval and handed eval row *i* the out-of-fold TE values of train row *i* — a row-alignment
artifact that would score ~0.679 on shuffled rows and would not transfer to the 1M hidden holdout
(different index). It was diagnosed via a row-shuffle probe, and the corrected framework (proper float
target encodings, OOF values used only for the training matrix, full-train smoothed maps inside
`prepare()` for any unseen rows) supersedes those rows. 0.7397 is the honest number.

## The 3-5 changes that mattered most

1. **Time-of-day and calendar features** (+0.004 over baseline, #6): numeric Month/DayofMonth/DayOfWeek
   (stripping the `c-` prefix), Hour, Minute, DepTimeMin; later refined with DayOfYear, Hour-as-categorical,
   and 15-minute TimeBin (#28-29, +0.002). Departure hour is the single dominant risk factor (delay rate
   rises from ~4% at 5am to ~98% after midnight); calendar position carries a seasonal signal.
2. **Heavy stochastic regularization** (+0.007, #10-15): subsample=0.5, colsample_bytree=0.5-0.6 with
   depth-8 trees, plus min_child_weight=5, reg_lambda=3, reg_alpha=2 (#21, #32). The 2005→2006 shift
   punishes confident memorization; unpruned deep trees scored 0.698 while the regularized setup reached
   0.729.
3. **Smoothed target encodings done correctly** (+0.006, #32-38): smoothed (empirical-Bayes) train-only
   encodings for Origin, Dest, Carrier, Route(Origin_Dest), Hour, TimeBin, DayOfWeek, DayofMonth, Month,
   and Origin×TimeBin, with the training matrix using out-of-fold (K=10) values to avoid self-leakage
   and `predict_proba` mapping unseen rows through the full-train tables. This was the single most
   delicate part: naive full-train TEs on the training rows make the model over-trust them (the
   leak-inflated 0.7666/0.7855 showed how seductive that artifact is).
4. **Bagged XGBoost ensemble** (+0.002, #23-27): 12 models averaged, diversified over seed,
   max_depth (6/7/8/9/10) and subsample (0.4/0.5/0.6), each ~350 trees at lr 0.03. Averaging tames the
   variance that the temporal shift amplifies.
5. **DistPerMin interaction** (part of #32): Distance / (DepTimeMin+1) — a cheap proxy that lets the
   trees separate short-haul evening flights from long morning legs.

## 3 things that did NOT help

1. **Route (Origin_Dest) as a raw categorical feature**: -0.011 both without (#4) and with (#13)
   sampling — high-cardinality identity features memorize 2005-specific route delays that don't transfer.
   The smoothed RouteTE was fine, but the raw categorical was noise.
2. **More raw capacity without regularization**: 600 deep trees at lr 0.06 scored 0.698 vs the 0.714
   baseline (#2); even 120 trees alone was worse than 30 (#5). Everything had to be bought back with
   sampling and leaf penalties.
3. **Cyclical sin/cos encodings** (#19, #30), **month×hour interaction categorical** (#31, -0.007),
   **frequency/count features** (#18), **carrier×hour and carrier×month TEs**, and the
   **airport-union fallback TE** (#39) all scored equal or worse; they were reverted.

## What I would try with more budget

The biggest open direction is fitting the *shift* rather than the snapshot: (a) importance-weighted or
recent-month-upweighted training (weight late-2005 rows higher to better match 2006), and a small
temporal-CV grid to pick hyperparameters that maximize last-quarter-2005 AUC rather than eval AUC;
(b) leakage-free richer flight-state features — per-airport-per-hour and per-route-per-hour OOF TEs with
hierarchical shrinkage (route → airport → global), and distance-to-nearest-hub style geodesic features
computed from the train coordinates implicitly via TE pools; (c) a wider ensemble: the current 12 bags
saturated, but adding column-subset-diversified models (different feature blocks per member, e.g.
TE-only vs raw-categorical-only members) decorrelates errors in a way seed/depth variation cannot;
(d) monotone constraints on DepTimeMin (later ⇒ higher risk, mostly true) to regularize the dominant
effect; and (e) two-stage stacking inside XGBoost-only rules — a first round predicting per-route noise
residuals, a second round on those as features.
