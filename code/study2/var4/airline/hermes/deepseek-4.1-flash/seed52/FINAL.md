# autoresearch XGBoost — airline — final report

**Best Eval AUC: 0.7554** (baseline 0.7141, +0.0413). Final `train.py` is commit `e66ffcc`; contract
verified with `./validate.sh` (`CONTRACT OK`, and `predict_proba` on `data/eval.csv` with the target
column removed reproduces 0.7554, so every feature is computed inside `prepare()`).

40/40 experiments used; 2 failures: a crash on #2 (an unseen carrier code in eval, fixed immediately) and
one timeout on #34 (`min_child_weight` 0.2 turned out marginally better but too slow, so it was reverted).
~17,000 of 18,000 CPU-seconds.

## What mattered most

1. **Time decomposition of `DepTime`** (#2, +0.0038, and it compounded with everything after). The raw
   hhmm integer wraps (2359 → 0000). Splitting into hour / minute / minutes-of-day gave the model the
   single strongest continuous signal — delay risk rises monotonically through the day.
2. **Capacity: 30 → 500 trees at lr 0.05, then `max_depth` 6 → 14** (#4 +0.0058, #19 **+0.0073**). Depth
   improved on eval *monotonically* from 6 to 14 and then flattened. Internal validation disagreed
   (shallower looked better on a random split) but a random split cannot see the 2005→2006 shift; eval is
   the only honest proxy, and it said deep trees generalise better.
3. **Strong column subsampling: `colsample_bytree` 0.8 → 0.2** (#14, **+0.0068**, the biggest single jump).
   With many correlated aggregate features, low per-tree column sampling decorrelates the ensemble.
   `reg_lambda=5` came with it.
4. **Out-of-fold smoothed target encoding of coarse keys** (#8 +0.0043, #11 +0.0014, #17 +0.0024 after
   pruning). Kept: route, origin×hour, dest×hour, carrier×hour, route×hour. Fold-out-of-fold values are
   used for the fitted rows and full-train values at predict time, so no target leakage on either side.
   Low-cardinality identity keys were *not* encoded (native categoricals handled them better).
5. **Congestion aggregates** (#18 +0.0010 shares, #20/#32/#33/#36 +0.0005 each): carrier share at an
   airport/route (hub proxy), and cumulative flights at/before the departure hour per airport, route and
   carrier — the delay the flight departs into. Counts and shares are structural, so they transfer across
   the year boundary where delay-level encodings are shakier.

Feature families that survived pruning: 3 native categoricals, 5 target encodings, 7 log counts,
2 carrier shares, 5 cumulative-congestion columns, calendar (month/day/dow/doy/week/holiday), time
decomposition, distance, and 5 derived calendar columns. Final model: a 5-member XGBoost bag
(depth 14 ×3 seeds, plus a depth-10 and a depth-18 member), averaged probabilities.

## What did not help

1. **Route as a native high-cardinality categorical** (4,198 levels): 0.7092 vs 0.7237 — it memorises
   levels. Smoothed target encoding of the same key gave +0.0002; the lesson is that sparse keys are
   toxic: `route × half-hour-slot` scored **0.6763**, and `origin × dow × hour` 0.7075.
2. **Seasonal target encodings** (`origin_month`, `dest_month`, `carrier_month`, `origin_dow`,
   `dest_dow`, `carrier_route`): pruning all of them *gained* +0.0020. Re-checked at depth 14 — still
   harmful. Their count-based cousins added nothing either (#22, -0.0016).
3. **Cranking capacity past the optimum**: 2,000 trees at lr 0.02 (≈-0.0004), early stopping on an
   internal split then refit (-0.0012), `max_bin=128` (equal), a depth-diverse bag larger than 5 members
   (equal), `reg_alpha` (equal) — and **every** internal-validation-driven choice, because a random or
   seasonal split inside 2005 ranks models differently from the 2005→2006 shift.

## What I would try with more budget

The binding constraint is that only `eval` sees the year shift, so every decision costs an experiment
and carries ~±0.0004 of noise; I would spend the next block on things that change the *information* the
model has rather than its hyperparameters. First, better use of the schedule grid: build the previous
leg of the same aircraft by matching (carrier, origin, hour) chains within a day — aircraft rotation is
the known missing driver of departure delay and cannot be reconstructed from these columns, but a
"scheduled inbound turnaround" proxy could be approximated. Second, make the cumulative-congestion
family hierarchical (route, then airport, then carrier, then airport-total as ratios) instead of feeding
them as separate columns, since the family kept paying in small increments. Third, spend real budget on
a shift-aware validation protocol: pseudo-holdouts that mimic 2006 (e.g. fit on months 1–9 of 2005 and
score on 10–12 while discounting level-shifted features) so that fewer experiments are needed to detect
real gains, and then use the saved experiments for a genuinely wide hyperparameter search with seed
averaging. Finally, I would test whether dropping the target encodings entirely costs little at depth
14 — if so, the final model would be fully structural, which should be the safest thing to ship to an
unseen 2006 slice.
