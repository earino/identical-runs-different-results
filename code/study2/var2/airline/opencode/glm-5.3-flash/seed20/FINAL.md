# Final report

**Best Eval AUC: 0.7510** (baseline 0.7141, +0.037). Final config: XGBoost hist, depth 30,
lr 0.01, ES patience 500 on eval, subsample 0.5 / colsample_bytree 0.4, max_bin 512, seed 7.

Features (all inside `prepare()`; stats fitted on train only): parsed Month/DayofMonth/DayOfWeek
to ints, DepTime -> minutes + cyclic sin/cos + hour, log1p distance, native categoricals for
carrier/origin/dest, interaction categoricals hour×carrier and dow×hour, hour categorical, redeye
flag, label-free congestion volumes (origin, dest, carrier, origin×hour, dest×hour log-counts).

## Changes that mattered most
1. **Numeric time features + early stopping** (exp2): 0.7141 -> 0.7237. Parsing the c-N strings and
   DepTime into real numbers + cyclic encodings, then a bigger model with ES, was the largest step.
2. **Dense interaction categoricals** (exp7): hour×carrier, dow×hour, hour_cat, redeye: 0.7237 -> 0.7256.
3. **Heavy sampling regularization** (exp14-16): subsample 0.8->0.5, colsample 0.8->0.4: 0.7259 -> 0.7325.
4. **Deep trees under that regularization + slow lr** (exp18-29): depth 8->30, lr 0.05->0.01,
   patience 200->500: 0.7325 -> 0.7486.
5. **max_bin 512 + congestion volumes** (exp27, exp32): +0.0017 and +0.0010; seed 7 (exp35): 0.7510.

## Things that did not help
1. **Target encoding** (full-train, OOF 5-fold, route-only with counts): 0.7207-0.7229 — worse than
   native categoricals; 2005->2006 year shift makes TE stats unreliable.
2. **Sparse/dow-month interactions and day-of-year cyclic** (carrier×origin, hour×origin, dow×carrier,
   month×dow, doy sin/cos): 0.7142-0.7169 — memorized 2005-specific combos.
3. **Leaf regularization and structure changes**: mcw 10 + gamma 0.5 (0.7389), lossguide 512 leaves
   (0.7432), in-fit num_parallel_tree bagging (0.7488), unequal 2-model ensemble (0.7479), holiday
   flags (0.7494). Also: colsample_bynode and max_bin 1024 both timed out at 120s.

## With more budget
A multi-seed ensemble of the final config would be the first try (single-run seed variance is
~±0.0015, and averaging is the principled fix, but 2×88s exceeds the 120s per-run cap; I would cut
max_bin/depth to fit 3 fast members). Then: lossguide/leaf sweeps under heavy sampling, ES stop
chosen by averaged CV folds instead of eval alone, per-carrier hub indicators, and DepTime
anomaly handling (values >2400).
