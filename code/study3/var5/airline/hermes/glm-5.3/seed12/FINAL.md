# Final Report — airline dep-delay XGBoost (autoresearch benchmark)

## Result

Best Eval AUC (2006 slice): **0.7475** (baseline: 0.7141, +0.0334)

Final commit: `46bc9bc` — d20/n200/lr0.05/mcw1/l2/s1.0/cs0.4, 5-seed bagged ensemble,
categorical hour + minute/dep_minutes/min_bin/log_dist/dom_bin features.

Validation: `CONTRACT OK` (predict_proba path reproduces 0.7475 on the raw-DataFrame path).

## The 5 changes that mattered most

1. **Deep trees + heavy column subsampling** (n200, max_depth 20, colsample_bytree 0.4). The single
   biggest jump: eval AUC went from ~0.721 to ~0.739. Deep trees let the model carve the fine
   time-of-day / route / carrier interactions that dominate delay risk; colsample 0.4 keeps the
   ensemble members decorrelated, which also stacks with bagging.
2. **Seed-bagged ensemble (5 seeds, predictions averaged).** Consistent +0.001 to +0.002 on top of
   any single config, and it reduces variance on the hidden holdout, which is the score that counts.
3. **Categorical hour-of-day (categories taken from train only).** +0.001-0.0015. Hour-of-day is the
   strongest single signal (single-feature AUC ~0.68); giving it its own categorical (rather than
   only raw DepTime) lets trees split on it directly. Note: hour categories must be train-derived —
   eval/holdout contain hours 24-26 that must map to NaN, not to a fabricated category.
4. **Fine time features: minute, dep_minutes, min_bin (5-minute bins), dom_bin, log_dist.** Together
   +0.001-0.002. The minute-of-hour and 5-minute-bin granularity feeds the deep trees; log_dist
   stabilizes long distances.
5. **Regularization retuning after depth change (mcw 10→1, lambda 10→2, subsample 0.7→1.0).** The
   optimal regularization profile flips once trees go deep: min_child_weight=1 and full-row
   subsampling let deep trees fit the fine interactions; row subsampling now hurt.

## The 3 things that did not help (and were reverted)

1. **Target-encoded route delay-rate** (Origin×Dest smoothed rate): in-sample 0.65 AUC collapsed to
   0.57 across the 2005→2006 shift; adding it as a feature *hurt* eval. Route-specific delay
   probabilities do not transfer across years.
2. **More trees at fixed depth via early stopping on a within-2005 split** (n2000, best_iter 747):
   0.7018. Within-year validation rewards fitting 2005 idiosyncrasies; the year shift punishes it.
3. **Circular/sin-cos hour encodings, hour×carrier and dow×hour interaction categoricals,
   route-pair categorical, frequency features** — all neutral-to-negative (0.709-0.719). Deep
   trees learn these interactions better from the raw columns.

## What I would try with more budget

The two directions that diagnostic runs showed still promising but that did not fit the time/CPU
caps: (a) 8-seed bagging at the final config (0.74585 in diagnostics vs 0.74528 at 5 seeds, and
more seeds monotonically improved up to the 8 I tested); (b) d24 with n300 at lr0.04 (best single
0.74207), which needs the per-run wall clock to fit 5 seeds under 120s — feasible with fewer
features or with `xgboost.train` on a DMatrix reused across seeds. Beyond that: a two-model
ensemble averaging the deep-tree config with a lossguide/max_leaves config for structural
diversity, and OOF-based stacking of the seed predictions rather than a plain mean.

## Experiment log (kept commits)

| # | commit | Eval AUC | change |
|---|--------|----------|--------|
| 1 | dff744d | 0.7141 | baseline |
| 5 | 3509f1b | 0.7170 | regularization (lr.05 mcw10 l10 s.7 cs.7 n100) |
| 6 | 46a19ef | 0.7174 | n200 lr.025 |
| 7 | f243f4b | 0.7188 | 5-seed bagged ensemble |
| 8 | 928d376 | 0.7191 | minute feature |
| 9 | 053189d | 0.7206 | hour categorical |
| 10 | 3a053c8 | 0.7211 | dep_minutes + min_bin + hour categories from train |
| 11 | 7bc7733 | 0.7215 | log_dist + dom_bin |
| 12 | 1ec4850 | 0.7388 | deep trees d20 cs.4 lr.05 |
| 13 | a1c3829 | 0.7424 | mcw1 |
| 14 | 46bc9bc | 0.7475 | subsample 1.0 + reg_lambda 2 |

Reverted: route target-encoding (0.7036), within-year early stopping (0.7018), temporal sin/cos
features (0.7134), d24 n300 5-seed (timeout).
