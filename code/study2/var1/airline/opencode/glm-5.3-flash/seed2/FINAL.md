# Final Report — airline delay XGBoost (autoresearch, harness edition)

**Best Eval AUC: 0.7335** (baseline 0.7141 → +0.0194). Final model: 2-member XGBoost ensemble
(max_depth 10 + 12, lr 0.03, AUC-based early stopping, refit on full train), validated
(`CONTRACT OK`, `predict_proba` reproduces 0.7335 with target removed).

## Changes that mattered most

1. **Minute-of-hour features** (+0.0052, exp34): `minute` numeric + `is_round_slot` flag. Schedule
   texture separates commuter/mainline flights and transfers across the 2005→2006 year shift.
2. **Congestion counts** (+0.0030, exp10): log flight counts per carrier, origin, dest, origin×hour,
   dest×hour, plus 15-min departure-slot categorical. Label-free structural volumes.
3. **Strong regularization** (+0.0013, exp15): subsample 0.7, colsample_bytree 0.7, colsample_bynode 0.7,
   min_child_weight 20, reg_alpha 0.5, lambda 2, max_bin 512. The 2006 shift punishes memorization.
4. **Gamma pruning** (+0.0020, exp25): min_split_loss 2.0 prunes low-gain splits; also made runs ~30% faster.
5. **Capacity + early stopping** (+0.0043 total, exp3/7/8): depth 8–12, lr 0.03, early stopping on AUC
   over a random 20% of train, refit on 100%; ensembling d10+d12 (+0.002, exp12/24), geography features
   (origin/dest mean distance, distance buckets, +0.001, exp17).

## Things that did not help (all reverted)

1. **Target encoding** of any kind — airports/carrier (OOF or full-map), even stable time buckets
   (exp4–6: 0.7131/0.7084/0.7055 vs 0.7173). 2005-specific delay rates do not transfer to 2006.
2. **High-cardinality categorical additions** (route = Origin_Dest as categorical, day-of-year/holiday
   flags, month/day volume counts, slot5, minute categorical) — all added 2005-specific noise.
3. **Bigger/cheaper ensembles**: 3rd member (d8 or d12), DART booster (timeout), bagging row subsets,
   lr decay schedule, lr 0.04, mcw 35, colsample 0.7 re-probe, rank:pairwise member (tie at 2× cost).

## With more budget

I would (a) run a small systematic sweep around the current optimum (gamma 1–3 × subsample 0.6–0.8 ×
depth 9–11) since several re-probes flipped sign after the minute features landed; (b) mine more
label-free schedule/texture features (carrier×slot congestion, per-slot national rhythm, DepTime
rounding patterns) — that family produced the two largest gains; (c) try 4-seed averaging of the
winning d10/d12 pair with a reduced ES cap to fit the 120 s limit; (d) validate stability of keep
decisions with a second eval-style split, since several winning margins (+0.0001–0.0002) are within
run-to-run noise.
