# FINAL — airline delay (XGBoost autoresearch)

**Best Eval AUC: 0.7512** (baseline: 0.7141, +0.0371). Final commit: `7a12e48`
("15-min windows, fine incoming, asymmetric broad"). 40/40 experiments used; validation passes
(`CONTRACT OK`), `predict_proba` reproduces the score on a target-dropped frame.

## What mattered most

1. **Phase-aligned time-of-day features (+~0.002 each step).** Departure minute plus a day cycle anchored at
   ~5am (`depmin_shift`, sin/cos of the shifted cycle) — 5am is the daily delay minimum and delays accumulate
   through the operating day. `depmin_shift` became the single most important feature (importance ~0.32).
2. **Dropping the native Origin/Dest/UniqueCarrier categoricals (+0.0047).** Raw category codes made the
   models memorize 2005-specific airport/carrier delay rates that do not transfer to 2006. Identity is now
   carried by stable schedule-volume features instead. This was the single largest AUC jump of the run.
3. **Unsupervised schedule-volume/congestion features (fit on train only, no target).** Counts of scheduled
   flights per (origin, hour), (dest, arrival-hour), route, carrier-bank — and especially *windowed densities*:
   how many train flights depart the origin / arrive at the destination / land at the origin (incoming aircraft
   supply) within ±15/30/60/90/120-minute windows around this flight's times. Multiple window sizes each kept
   adding small gains (+0.0006–0.0027).
4. **Estimated arrival time from distance** (`depmin + 35 + dist/8`): unlocks arrival-side congestion
   (destination queue at the time you land, dest operating-window position) (+0.0016).
5. **Ensembling diverse XGBoost configs** (8 configs × depths 4–12, lr 0.03–0.15, varied subsample/colsample,
   3 seeds each = 24 members, probability-averaged): +0.009 over a single model. Diversity across configs
   mattered; adding more seeds beyond 3 added nothing.

## What did NOT help

1. **Target encoding in any form** — plain (route/origin/dest/carrier/hour), interaction
   (carrier/origin/dest × hour-block), heavy smoothing (M=100): all hurt by 0.001–0.008. 2005 delay rates do not
   transfer to 2006; AUC is rank-based so the leakage-prone features actively mislead.
2. **DART boosters** in the ensemble: no gain, ~5× slower. **Monotone constraints** on the aligned dep time:
   -0.001. **Rank-averaging** instead of probability-averaging: slightly worse.
3. **Capacity/regularization micro-tuning**: more/larger trees than ~200@0.1 (or the equivalent) overfit the
   year shift; higher min_child_weight hurt; day-of-week- and month-conditioned volume features and
   day-of-year harmonics also hurt (year-fragile, sparse cells); route as a native categorical and carrier
   volume-identity features hurt as well.

## What I would try with more budget

The clear remaining vein is *physical delay-propagation modeling on the hidden-holdout scale*: per-airport
scheduled-departure *cumulative curves* (rank of this flight within the airport's operating day is already in;
a smooth "hours since the airport's first departure" per calendar quarter would be next), arrival-queue
interaction features (incoming-aircraft supply × carrier bank dominance), and congestion windows conditioned on
route direction rather than raw counts. Second, the ensemble could be made stronger where it is provably
diverse: feature-subspace bagging (each member trained on a different ~85% of columns) and a proper 2-fold
OOF stack (an XGBoost meta-model over member OOF predictions) — both fit the compute budget but need runtime
care under the 120s cap. Third, I would revisit early stopping with a *2005→2006-style* internal split
(November–December validation) now that the feature set is dominated by stable structural statistics rather
than year-fragile encodings — the earlier neutral/negative results were obtained when noisy features dominated.
Finally, cheap robustness checks: dropping each feature family for one run to confirm every family earns its
place on 2006 data, since the hidden holdout is a different slice of the same year.
