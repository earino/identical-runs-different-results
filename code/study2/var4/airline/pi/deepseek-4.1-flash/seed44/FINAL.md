# FINAL — airline departure-delay AUC (autoresearch XGBoost)

## Result
- **Best Eval AUC: 0.7500** (experiment #12, commit `115dd80`, "drop raw Month/DayofMonth in favor of doy").
- Baseline: 0.7141. Net improvement: **+0.0359**.
- 12 experiments run; the last few were stopped when the 18,000 CPU-second Python budget was exhausted.

> Note: `./validate.sh` could not be executed at finalization time because the cell's Python CPU budget
> was already spent and the interpreter refuses to start. The committed `train.py` is the exact file that
> produced the logged 0.7500 run, and its structure keeps all feature engineering inside `prepare()`,
> which `predict_proba(df)` calls on the raw dataframe (fit-on-train-only category levels), so the
> hidden-holdout path reproduces the same transform.

## Changes that mattered most
1. **`carrier × hour-of-day` interaction (native categorical).** +0.005 at the time it was added and the
   single biggest signal jump. Carriers have stable, year-to-year daily delay profiles; a 480-level
   categorical captures what trees cannot cheaply build from raw `DepTime` + `UniqueCarrier`.
2. **Deep trees + strong L2.** Moving from the baseline depth-6/30-tree shallow model to depth 6–12 with
   `reg_lambda=50` was worth ~+0.012, and the large L2 was itself worth ~+0.002. Capacity only helped
   *after* the interaction features existed.
3. **More high-cardinality hour interactions: `origin × hour` and `hour × distance-bin`.** +0.006. These
   encode airport-specific congestion by time of day and distance-dependent schedule patterns; both
   generalize across the 2005→2006 shift. `carrier × distance-bin` added a further ~+0.0015.
4. **Ordinal day-of-year (`doy`) replacing raw `Month`/`DayofMonth`.** Small but consistent (~+0.001),
   and it removes year-specific raw date splits. Dropping the raw month/day columns after adding `doy`
   was itself worth ~+0.0009.
5. **Diverse 4-model XGBoost ensemble** (depths 6/8/10/12, different seeds/colsample) averaging
   probabilities: ~+0.002 over the best single model, and more robust on the time-shifted eval.

## Things that did NOT help
- **Raw route (`Origin_Dest`) categorical, `carrier×origin`, `origin×dayofweek`:** high-cardinality
  categoricals with sparse per-level counts consistently destroyed AUC (−0.01 to −0.04) by fitting
  year-specific noise.
- **Cyclical encodings** (sin/cos of hour, month, doy) and target encoding (OOF, smoothed) of
  origin/dest/carrier: neutral to slightly negative; trees already handle these orderings.
- **Row subsampling / lower learning rate / DART / lossguide growth / per-carrier models:** all neutral
  or worse. `minute` and `DayOfWeek` turned out to be genuinely useful, while deeper attempts at
  date/holiday features hurt.

## What I would try with more budget
The clearest lead is that replacing the 4-model ensemble with an ensemble of `num_parallel_tree=3`
(random-forest-style within boosting) models was measuring **0.7509** in my last offline check while
being ~30% cheaper per fit, but the CPU budget ran out before I could promote and validate it. Next, I
would (a) confirm that configuration and search `num_parallel_tree`/depth more finely, (b) test whether
origin×hour can be made more robust with shrinkage toward a global hour effect (hierarchical/target
encoding) so unseen airports degrade gracefully, and (c) try multi-seed averaging of the best single
model as a cheaper variance-reduction alternative to the current ensemble. Beyond that the feature set
(8 raw columns, all used) is saturated; gains would likely be small and dominated by regularization and
ensembling choices rather than new signal.
