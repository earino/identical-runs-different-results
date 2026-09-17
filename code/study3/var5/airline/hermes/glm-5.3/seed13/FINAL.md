# Final report — autoresearch XGBoost (airline, scenario 2)

## Result

- Best Eval AUC (2006 slice 1): **0.7585** (baseline: 0.7141, +0.0444)
- Final commit: `2893176` (colsample_bytree 0.6 in the no-date regime), preceded by `d80c751` (drop date columns), `f9fdcb4` (sin/cos + traffic counts), `6bf0277`/`363861a` (deep regularized ensemble), `2370965` (colsample ensemble), `3abdbed` (numeric dates).
- 13 experiments logged, all `ok`; budget ended on the 18,000 CPU-second cap with 134 wall-clock minutes and 27 experiment slots unused.
- `./validate.sh` prints `CONTRACT OK` (eval AUC via predict_proba: 0.7585).

## The 5 changes that mattered most

1. **Numeric encoding of the c-N string columns** (Month/DayofMonth/DayOfWeek -> ints): +0.002 on its own and
   the prerequisite for everything later; string categoricals forced memorizing 2005-specific partitions.
2. **Deep trees + heavy per-node regularization** (max_depth 24, gamma=0, reg_alpha=1, colsample_bynode=0.8):
   single biggest jump (+0.02). In this drift-heavy task the winning axis is depth-with-regularization, not
   more boosting rounds — n_estimators stayed at 30 and every increase of it hurt.
3. **Seed-jittered colsample ensemble** (10 members, mean of probabilities): +0.003 alone and it compounds
   with the deep/regularized regime; averaging 10 jittered models is what makes depth 24 safe.
4. **Dropping the calendar columns entirely** (Month/DayofMonth/DayOfWeek): +0.007. The strongest single
   anti-drift move: 2005 seasonality/weather does not repeat in 2006, so calendar features only gave the
   model more rope to memorize the training year.
5. **Traffic-count features + sin/cos of minute-of-day** (route/origin/dest counts from 2005, sin/cos
   cyclical encoding): +0.005 together. Deep trees exploit these smooth, year-invariant signals where the
   original 30-tree shallow model could not.

## 3 things that did not help (and why)

1. **Target encodings** (carrier/origin/dest/route/origin-hour, K=10-20 smoothing): -0.002 to -0.012.
   2005 delay rates per airline/airport drift by 2006; they inject in-year label noise.
2. **More boosting capacity** (n_estimators 300-500, lr 0.05, early stopping on a 2005 holdout): all worse
   (-0.006 to -0.007). A same-year validation split selects for year-specific fit — the opposite of what
   the time-separated eval rewards.
3. **Route categorical (Origin_Dest) and row subsampling**: -0.009 and -0.003. Route level is too fine
   (5000+ levels, most seen once per year); subsample=0.8 just adds variance that colsample already covers.

## What I would try with more budget

The one structural idea I did not exhaust is ensemble *diversity engineering*: feature-subset members
(each member drops one feature group) scored 0.7550 with a date-polluted feature set, beating the plain
x10 ensemble, but adding them to the final no-date model diluted it (0.7521 mixed vs 0.7584 plain).
With budget I would re-run the feature-subset scheme in the no-date regime with equal-weight blending
tuned per member class, add 20-30 members (x16 gained +0.0003 over x10 and was still not saturated),
and re-screen the small surrogate winners (hour as a 24-level categorical, +0.001 on the surrogate, that
came out neutral at full depth) — those three threads all pointed up when the CPU budget cut them off.

## Method notes

- All keep/discard decisions used eval.csv (2006 slice 1); the risk of overfitting the eval was managed by
  (a) requiring gains > ~0.001 across disjoint seed sets, (b) preferring anti-drift changes (dropping
  features, more regularization) over eval-positive/complexity-positive ones, and (c) never stacking more
  than one idea per logged experiment.
- Offline screening used a cheap surrogate (d10, 2 seeds) to rank candidates, with full-config (d24 x10)
  confirmation before logging; the surrogate correctly ranked 8 of 10 confirmed changes.
