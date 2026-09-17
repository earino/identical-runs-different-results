# Final report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7222** (experiment #39, commit `f040d8c`; validated `CONTRACT OK`)

Baseline was 0.7141. Total: 40 experiments, ~1373 CPU-seconds.

## Final model

An averaged ensemble of 15 `XGBClassifier` models (depths 3–7 × seeds 42/7/2024),
`n_estimators=300`, `learning_rate=0.05`, `colsample_bytree=0.6`, `max_bin=512`,
hist tree method, native categoricals for `Month`/`DayOfWeek`/`UniqueCarrier`/`Origin`/`Dest`,
plus smoothed out-of-fold target encodings (smoothing 20) for `Origin`, `Dest`,
`UniqueCarrier`, `Month`, `DayOfWeek`, `Origin_Dest` (route), `Origin×departure-hour`
and `Carrier×Origin`. Encoders are fit on training data only (OOF for train rows,
full-train statistics at inference), so `predict_proba(df)` reproduces everything.

## Changes that mattered most

1. **Smoothed target encodings** (+0.0019): OOF target-mean encodings of Origin/Dest/
   route/Hour interactions were the single largest gain, confirming that airport/route
   delay propensity is stable across the 2005→2006 shift while raw high-cardinality
   categorical splits overfit it.
2. **Shallow, well-regularized trees** (+0.0021 over baseline): depth 3 beat depth 6;
   depth 2 and deep trees were worse. The temporal shift strongly favours low model capacity.
3. **Multi-depth ensemble** (+0.0011): averaging depths 3–7 consistently beat any single
   depth (0.7167 → 0.7222), with diminishing but real gains as depths were added.
4. **Feature subsampling (`colsample_bytree=0.6`)**: +0.0003 over no subsampling, and
   it made seed-bagging meaningful for diversity.
5. **`max_bin=512`**: +0.0004 — finer numeric splits for `DepTime`/`Distance`.

## Changes that did not help (reverted)

1. **Raw route categorical / dropping Origin & Dest**: route-as-categorical collapsed to
   0.7027, and removing Origin/Dest entirely gave 0.7007. Raw high-cardinality group
   identifiers do not transfer across years; only their smoothed target rate does.
2. **Time-of-day decomposition** (`hour`, `tod`, `is_weekend`) and **frequency encodings**:
   neutral at best (`DepTime` in hhmm is already monotone in time), so they were dropped.
3. **Heavy regularization bundles** (`subsample=0.8` + `mcw=5` + `reg_lambda=2`) and
   **more trees at lower LR** (800 @ 0.02, 700 @ 0.025): equal or worse. Extra capacity
   just overfits 2005; the signal is capacity-limited, not round-limited.

## What I would try with more budget

The largest remaining lever is better use of the time dimension: the eval/holdout are a
different year, so I would build explicit seasonality/trajectory features (month-over-month
delay-rate deltas per airport/carrier) and test recency-weighted training (weighting late-2005
months more) to track the 2006 regime. I would also try a proper stacked second level using
out-of-fold base-model predictions instead of simple averaging, and a hierarchical/shrunk
target-encoding scheme with per-cardinality smoothing tuned by internal time-based CV rather
than eval. Finally, with a reliable inner time split I would search over more aggressive
histogram/regularization settings and consider monotonic constraints on `DepTime`.
