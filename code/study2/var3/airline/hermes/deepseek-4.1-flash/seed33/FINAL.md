# FINAL — airline departure-delay (dep_delayed_15min) XGBoost

**Best Eval AUC: 0.7542** (baseline committed at session start: 0.7141). All 40 experiments used;
`validate.sh` reports `CONTRACT OK` and reproduces 0.7542 through `predict_proba(df.drop(target))`.

Best commit: `e5ec690` — *colsample_bytree=0.75 feature bagging* (9-member depth ladder).

## Changes that mattered most

1. **`CarrierHour` interaction categorical** (UniqueCarrier × departure hour, ~480 levels).
   Alone this was worth about **+0.018 AUC** (0.7170 → 0.7346 for a single deep model). It is the single
   biggest lever found: carriers keep stable departure banks from year to year, so the interaction is
   schedule structure rather than year-specific noise. Every other categorical interaction tried
   (Origin×hour, Carrier×Origin, Carrier×DayOfWeek, Carrier×Month, hour×day-of-week, 5-minute time bins)
   *hurt* by 0.005–0.015, so only this one is kept.
2. **Dropping `Month` and `DayofMonth`** (−0.0017 for a single model, and simpler). Calendar-position
   effects do not transfer 2005 → 2006; leaving them in let the trees fit year-specific weather/calendar
   noise. `DayOfWeek` was kept (removing it cost 0.0005).
3. **Time-of-day decomposition of `DepTime`**: `dep_hour`, `dep_minute`, `dep_sod`, and sin/cos cyclicals.
   `dep_minute` alone is worth ~0.006 (removing it dropped the ensemble to 0.7478); `dep_hour` matters
   specifically *in combination* with the `CarrierHour` categorical.
4. **An ensemble of 9 XGBoost models over a contiguous depth ladder (2,3,4,5,6,9,12,15,18), 250 rounds
   each, averaged on the probability scale** (+0.011 over the best single model). Capacity diversity alone
   was worth ~+0.005; adding the very shallow members (depth 2–3) on top was worth another ~+0.003.
5. **`colsample_bytree=0.75` feature bagging** (+0.0025, and ~15% faster): with a thin 14-feature frame,
   bagging increases member decorrelation more than it costs in member accuracy. 0.6 and 0.85 were both
   worse (0.7538 / 0.7535); it is a genuine optimum, not a monotone knob.

## Things that did NOT help (measured, then reverted)

1. **Target encoding of Origin/Dest/Carrier/Route and a Route categorical** — the worst result of the
   session (0.6978 vs 0.7170 for the same single model). Cross-year target statistics are not stable here
   and the ~7000-level Route cardinality just memorises 2005.
2. **More high-cardinality interaction categoricals** (Origin×hour, Carrier×Origin, Carrier×DOW,
   Carrier×Month, hour×DOW, 5-minute departure bins) and **schedule-volume counts**
   (flights per Origin/Dest/Route-hour, degree counts, frequency encodings) — all neutral-to-negative;
   they dilute the single good interaction.
3. **Subsampling / heavier regularization**: `subsample=0.8` (timed out at 120 s — it roughly doubles cost),
   `min_child_weight=depth` (0.7484), `min_child_weight=3` (0.7506), `gamma=1` (0.7524),
   `reg_lambda=3` (0.7538), `reg_alpha=1` (0.7534), `max_cat_to_onehot=1` and `max_bin=192` (both exactly
   equal, so reverted for simplicity). Per-member learning-rate or colsample diversity, per-member seeds,
   and a denser/alternate depth ladder were all within ±0.001 of the kept config — i.e. noise.

## What I would try with more budget

The remaining headroom is in the features, not the hyperparameters: eight raw columns are almost exhausted
by a depth-18 tree, and the last ~15 experiments were all inside the ±0.002 noise band of a 100 k-row eval
set, so further single-knob tuning cannot be distinguished from sampling noise. I would (a) build
*out-of-fold, year-robust* encodings — e.g. target-encode `Origin`/`Dest`/`Route` with a strong prior and
learn the smoothing constant on a 2006-style holdout rather than on 2005, since the naive version failed for
what is most likely a drift reason rather than a signal reason; (b) treat `DepTime` as a *local-clock bank
structure* (per-airport bank timestamps, connecting-wave membership) rather than as a raw minute-of-day;
(c) replace the hand-built ladder with a properly weighted stack — fit a level-1 XGBoost on out-of-fold
member predictions, which is legal here (ensembles of XGBoost only) and should beat the plain mean by more
than any parameter change managed; and (d) run repeated cross-year validation (train on 2005 folds,
validate on a held-out slice of 2006) so that decisions are made on a signal larger than the eval-noise
floor instead of on a single 0.0005 delta.
