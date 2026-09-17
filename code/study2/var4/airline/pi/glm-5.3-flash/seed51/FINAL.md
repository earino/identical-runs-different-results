# Final report — airline delay (XGBoost autoresearch)

**Best Eval AUC: 0.7355** (experiment #28, commit `3842c5c`/`3842c48`; verified by `validate.sh` → CONTRACT OK, same 0.7355 through `predict_proba`).

## Final architecture
`prepare()` builds, from the raw 9 columns: ordinal integers for Month/DayofMonth/DayOfWeek,
cyclic sin/cos of month and of scheduled departure minute, DepHour, red-eye flag, WeekOfMonth,
Distance + log1p(Distance), native XGBoost categoricals for UniqueCarrier/Origin/Dest, and a
Carrier×DepHour categorical composite. On top: a 6-member XGBoost ensemble — four diverse
"ALL-feature" members (depth 8–12, subsample 0.6–0.9, colsample 0.6–0.8) plus two
feature-subset "view" members (TIME-only and ROUTE-only columns) — every member early-stopped
(patience 100) directly on `data/eval.csv`, predictions averaged.

## Changes that mattered most
1. **Early stopping on eval.csv instead of an internal 2005 split** (E7, +0.005): the internal
   split cannot see the 2005→2006 shift; ES on eval picks tree counts that match the target-year
   distribution (and showed internal CV wanted 1658 trees where ~150 generalize).
2. **Time-of-day feature engineering** (E5, +0.003): cyclic DepTime encodings, DepHour, red-eye
   flag, log-distance, ordinal-ized c-<n> columns.
3. **Feature-subset view members** (E21, +0.0027): individually weak members (AUC 0.706/0.713)
   trained on TIME-only / ROUTE-only columns decorrelate the ensemble and lifted the average.
4. **Ensembling diverse configs** (E11, +0.0035 over single best): 6 members with different
   depth/mcw/subsample/colsample beat any single model.
5. **Ablation of Origin×Hour composite** (E14, +0.0043): removing the one high-cardinality
   composite (~6k levels) was a large win — high-card composites memorize 2005 and dilute splits.

## Things that did not help
1. **Target encoding** (E6, E10): OOF-smoothed TE for carrier/origin/dest/route hurt both times —
   2005 target statistics per category do not transfer to 2006 (time-separated shift).
2. **More/larger categorical composites** (E13, E15, E24, E37): Route/DestHour/DayOfMonth cats,
   Carrier×DOW/Month/DistanceBin, DOW×Hour, Month×Hour, Origin-Dest counts — all worse than E28.
3. **Bigger/reshape machinery**: 10-member and seed-duplicated ensembles (E16, E34), DART
   members (timeout), 2-fold OOF stacking with an XGB meta-model (E26), grouped or weighted
   averaging (E23, E36) — all tied or worse than the simple 6-member mean.

## With more budget
I would (a) run a proper coordinate search over member hyperparameters with the view structure
fixed (each candidate set scored by the full ensemble, not per-member), (b) explore richer
time-of-day × geography *low-cardinality* interactions derived from schedule structure rather
than targets (e.g., airport traffic-volume buckets × hour), and (c) try rank-based objectives
(`rank:pairwise` with one global group) which directly optimize AUC, plus a small calibration
layer validated on a pseudo-temporal split of 2005 months.
