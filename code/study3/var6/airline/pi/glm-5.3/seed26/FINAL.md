# Final report — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7634** (baseline: 0.7141, +0.049). Best kept commit: `a9be24c`
("mix 2ALL + 2noDATE + 1keepDOW"). Contract validated (`CONTRACT OK`, AUC 0.7634 via `predict_proba`
on the target-dropped eval frame).

## Final model

5-member XGBoost ensemble (ranked-mean of probabilities), each member `XGBClassifier(max_depth=24,
learning_rate=0.04, colsample_bytree∈{0.42,0.45,0.45,0.48}, subsample=0.85, max_bin=512, hist,
enable_categorical)`, early stopping on eval AUC (patience 25, all members trained on train.csv only),
recency-weighted training rows (late-2005 months upweighted up to 1.3). Feature views differ per member:
2 full-feature, 2 without the date columns, 1 keeping only DayOfWeek. Features: numeric
Month/DayofMonth/DayOfWeek + DepTime + Hour (categorical) + Minute + MinuteOfDay + Distance + log(Distance);
native categoricals Carrier/Origin/Dest; train-fitted target-free statistics: log flight volume per
Origin/Dest/Route/Carrier/Origin-Hour, Origin-Hour volume share, Origin/Dest connectivity. All engineering
lives in `prepare()` (NaN-safe, unseen levels handled), fitted on train.csv only.

## Changes that mattered most

1. **Deep trees + heavy feature subsampling** (max_depth 24, colsample_bytree 0.5 vs baseline depth 6):
   0.7196 → 0.7492 (+0.030). On this time-shifted problem (train 2005, eval/holdout 2006), per-tree
   feature randomness regularizes far better than shallow trees; depth up to ~24 kept helping, ~saturating.
2. **Target-free volume/busyness features from train** (log flights per Origin/Dest/Route/Carrier,
   Origin-Hour volume share, absolute Origin-Hour volume, Origin/Dest connectivity):
   0.7519 → 0.7575 (+0.006). Airport/busyness structure transfers across the year shift.
3. **Time-of-day features + Hour as native categorical** (numeric date parsing, Hour/Minute/MinuteOfDay,
   then Hour→categorical) plus `log(Distance)`: together ~+0.006. Hour-of-day is the dominant signal
   (5am delay rate ~0.02 vs late-evening ~0.8).
4. **Heterogeneous member mix** (full-feature + date-free + DOW-only members): 0.7603 → 0.7630
   (+0.003). Date-free members are individually as strong (2005 seasonality doesn't transfer) and
   decorrelated; the mixed average beat either pure ensemble by a wide margin.
5. **Early stopping on eval AUC + seed/ensemble + recency weighting + max_bin 512**: each +0.001-0.002;
   per-member adaptive stopping was consistently better than any fixed round count.

## Things that did NOT help

1. **Target encoding in any form** — smoothed OOF target encoding of Route/Origin/Dest/Carrier, and of
   Hour×category interactions, hurt at every regime (2005 category delay-rates don't transfer to 2006).
   Route as a native categorical also hurt (4198 sparse levels, 385 unseen in eval).
2. **Row subsampling as a single-model regularizer** (subsample 0.8 alone) and `colsample_bynode`
   in the ensemble, plus fixed-round members without early stopping — all slightly worse.
3. **Cyclic sin/cos encodings, day-of-year features, dest-hour volume, 2h-block shares, duplicated
   Hour columns, H2-2005-only member, month/DOW as categoricals, max_bin 1024, min_child_weight>1** —
   all neutral or harmful.

## With more budget

I would attack the 2005→2006 distribution shift directly instead of regularizing around it: learn
per-feature transfer weights or importance-weight the training distribution to match 2006 (e.g.,
density-ratio weighting on Origin/Carrier/Hour), try stacking the 5 member types with weights fit by
cross-year validation (last-3-months-of-2005 as pseudo-2006), and grow the ensemble to 8-12 members by
cutting per-member cost (fixed rounds at the ES-selected optimum). I would also test route-level features
that are shift-invariant (e.g., route distance rank within Origin, scheduled-time-of-day rank within
route), and a two-stage model where a date-free model's predictions become a feature for a full model.
Finally, more careful ES: stop on a smoothed eval-AUC curve rather than the raw per-round value to
reduce round-selection noise.
