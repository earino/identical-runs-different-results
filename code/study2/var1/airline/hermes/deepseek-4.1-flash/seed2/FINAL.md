# FINAL — airline departure-delay (15+ min), XGBoost

**Best Eval AUC: 0.7432** (baseline 0.7141, +0.0291)
Champion commit: `fa51943` — bag of 3 XGBoost classifiers. `./validate.sh` prints `CONTRACT OK`
(`predict_proba` reproduces 0.7432 on `data/eval.csv` with the target column removed).
Budget: 40/40 experiments used, ~180 min of the 230 min wall clock, 8,656 of 18,000 CPU-seconds.

## What the final model does

Features (21), all built inside `prepare(df)` from the raw row (train-fit lookup tables only):
`Month`, `DayofMonth`, `DayOfWeek` (decoded from the `c-<n>` strings to plain integers), `Distance`,
`log_distance`, `DepHour`, `DepMinOfDay`, `DepMinute`, `UniqueCarrier`/`Origin`/`Dest` (categorical),
`CarrierHour` (categorical carrier x departure-hour interaction), 7 log scheduled-traffic counts
(carrier, origin, dest, route, origin x hour, dest x hour, carrier x hour), `carrier_origin_share`, `dist_x_hour`.

Model: 3 x `XGBClassifier(n_estimators=2500, max_depth=7, learning_rate=0.01, min_child_weight=10,
subsample=0.7, colsample_bytree=0.4, reg_lambda=10, tree_method="hist", enable_categorical=True)`
with different seeds, probabilities averaged.

## The changes that mattered most

1. **Decomposing `DepTime`** (hour / minute-of-day / raw minute). The initial baseline's feature-set
   ablation without `DepTime` scored 0.5886 vs 0.7132 with it — it is by far the dominant signal.
   Splitting it into `DepHour` + `DepMinOfDay` + `DepMinute` reached 0.7420 as a single model: the hist
   builder bins each feature into ≤256 quantile buckets, so minute-of-day alone loses the fine
   round-minute structure of scheduled departures. (+0.0066 from `DepMinute` alone, measured in isolation,
   and making minute *categorical* instead cost -0.007, so the effect is ordinal, not discrete.)
2. **`CarrierHour` categorical + log traffic counts** (exp #6, 0.7200 -> 0.7289). Carrier x hour is the
   second-strongest feature group: removing it costs -0.0063, removing the traffic counts -0.0020.
3. **Dropping the sin/cos "cyclical" encodings** (#16/#17, +0.0018). They let the trees memorise
   2005-specific seasonality that does not transfer to the 2006 evaluation year. Calendar effects
   themselves do help (removing `Month`/`DayofMonth`/`DayOfWeek` costs -0.005), just not in periodic form.
4. **Depth 7 + low learning rate** (#28/#29): a randomised search moved the model from
   `depth 6, lr 0.02, 1500 trees` to `depth 7, lr 0.01, 2500 trees` (+0.0009). A paired 3-seed check
   confirmed this is real (d7: 0.73458 ± 0.00006, d6: 0.73365 ± 0.00020 — all three d7 seeds beat all
   three d6 seeds), which justified treating later ~0.0005 differences as signal.
5. **Rare-level bucketing** (#21, +0.0006): categorical levels with <10 training rows are folded into a
   single `RARE` bucket, so unseen holdout levels land somewhere sensible instead of NaN.
   Plus **seed bagging** of 3 models (+0.0004) and, from the same "multiple views of one variable"
   principle as `DepMinute`, **`log_distance`** alongside `Distance` (+0.0009).

## What did NOT help

1. **Target encoding** (out-of-fold, smoothed, k=30/100 over carrier/origin/dest/route/hour keys): best
   variant +0.0003 over the base — pure noise. 2005 delay rates do not carry to 2006.
2. **High-cardinality interaction categoricals**: `route` (-0.008), origin x hour (-0.003), dest x hour
   (-0.006), carrier x origin (-0.006), route x hour (-0.006) — all clearly harmful, and still harmful
   after rare-level bucketing. XGBoost's categorical splits overfit them even at depth 4-6 with
   `reg_lambda` 10-20.
3. **Capacity/regularisation micro-tuning past the depth-7 point**: `max_bin` 512/1024, `reg_alpha`,
   `colsample_bylevel`, `lossguide`/`max_leaves`, `max_cat_threshold`, `cat_smooth`/`cat_l2`
   (silently ignored by XGBoost 3.4.1), `min_data_per_group` — every variant landed inside ±0.0005 of the
   reference. Similarly, windowed (±1 h) congestion counts, origin evening-peak share, global and
   month x hour peak counts, route distance aggregates, and structural features (carrier-route counts,
   origin competition) were all flat or negative.

Also worth recording: sweep runs that train several variants in one `run_experiment.sh` call log their
best variant's AUC into `experiments.tsv`, so a few "best_so_far" entries (#31, #36, #37, #39) belong to
diagnostic scripts, not to a committed model; the committed champions are #18, #23, #29, #33, #35, #38, #40.
Two experiments crashed on trivial bugs inside throwaway sweep scripts (#20, #24) and were re-run.

## What I would try with more budget

The remaining headroom looks like it is in the *representation of the departure-time variable*, not in the
model. `DepMinute` and `log_distance` both paid off purely by giving the histogram builder a second,
differently-scaled view of a variable it was already given, which suggests systematically enumerating
alternative encodings of the existing columns (minute-of-hour x carrier, minute-of-hour interaction with
traffic counts, quantile/rank transforms of the distance and count features, per-carrier distance
residuals) rather than hunting for new columns — the remaining unmodelled signal is likely aircraft
turnaround cascades, which this feature set cannot express. I would also spend a few experiments on a
proper paired-seed protocol for *every* promotion decision (as in #30): seed noise is only ±0.0002, but
column-order/colsample jitter is ~±0.0005, so single-run promotions below that margin are not
distinguishable, and a 3-seed paired comparison is the cheapest way to tell. Finally, with more
CPU budget I would raise the bag from 3 to 8-16 members: bagging still showed a small monotone gain
(1 -> 3 -> 5 members: 0.7331 -> 0.7335 -> 0.7336 on an earlier feature set) and it is the one change
that is essentially guaranteed not to overfit the evaluation year.
