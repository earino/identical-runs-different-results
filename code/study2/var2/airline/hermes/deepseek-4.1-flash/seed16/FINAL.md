# Final report — autoresearch XGBoost on airline delay prediction

**Best Eval AUC: 0.7448** (experiment #40, commit `e3416c1`) — baseline was 0.7141 (+0.0307).
**Contract:** `./validate.sh` prints `CONTRACT OK`; `predict_proba` on `data/eval.csv` with the target
column removed reproduces exactly 0.7448. Budget: 40/40 experiments, ~4,016 of 18,000 CPU-s, ~22 min of the
230-minute wall clock. Final `train.py` runs in ~75 s, inside the 120 s per-experiment cap.

## What mattered most

1. **Time-of-day feature extraction + native categorical calendar features** (+0.0138 together).
   `DepTime` (hhmm int) was split into hour / minute / minutes-since-midnight plus cyclic sin-cos, and
   Month / DayOfWeek / hour were fed to XGBoost as *categorical* features rather than only as ordinals.
   Categorical treatment alone was worth +0.0031 — the delay-vs-hour curve and the seasonality are strongly
   non-monotone, so partition splits beat threshold splits. (Day-of-month as a category, by contrast, hurt.)
2. **`UniqueCarrier` × hour-of-day as a single categorical** (+0.0049, the largest single feature win).
   Carrier scheduling/pad-time behaviour is hour-specific (bank structure, hub banks); trees at depth 6 rarely
   build that interaction explicitly. The same trick at higher cardinality (Dest×hour, 6.7k levels,
   or Origin/Dest routes) overfits badly — ~480 levels is about the useful ceiling here.
3. **`grow_policy="lossguide"` with `max_leaves=256`** (+0.0031 over depthwise). Replacing depth-limited
   growth with a leaf budget lets the trees spend capacity on the categorical partition splits instead of on
   uniformly deep numeric ones. 128 leaves gave +0.0031 too; 384 was slightly worse than 256.
4. **`colsample_bynode=0.4`** (+0.0042 over the default 1.0, discovered after the leaf budget went up).
   With many correlated categorical columns, sampling columns per split rather than per tree was the single
   most reliable regularizer; 0.6/0.8/0.25/0.3 were all worse than 0.4. `max_cat_threshold=128` (+0.0015)
   belongs to the same family: capping the categories per split partition reduces greedy categorical overfit.
5. **Averaging 8 jittered XGBoost models** (+0.0018). Seed/`colsample_bytree`/depth/min_child_weight jitter;
   cheap (adds ~60 s), monotone in size (3 → 5 → 8 members each added a little), and structurally the safest
   change with respect to the hidden holdout.

## What did not help

1. **Target encodings of carrier / airport / route / carrier×hour** (0.7224, −0.002). Smoothed 2005 delay
   rates do not transfer to 2006 rows; the 2005→2006 time separation punishes any statistic that encodes a
   year-specific level. Plain traffic counts (frequency, congestion) were neutral (+0.000, 0.7388).
2. **High-cardinality crossed categoricals**: route (`Origin_Dest`, 4k levels) cost −0.012, Dest×hour −0.010,
   Day-of-month as a category −0.006, month×hour and dow×hour −0.008. Four low-cardinality crosses added at
   once cost −0.032 — categorical split search overfits aggressively once there are many such columns.
3. **More capacity without more regularization**: 400 depth-6 trees unregularized (0.7164 < 30-tree baseline
   config), 800 trees with mcw 50 / lambda 5 (0.7210), 700 trees at lr 0.035 (0.7351 at 2× the runtime).
   Shallow-but-regularized (depth 4) beat deep, and the useful direction was always *more* stochasticity.

## What I would do with more budget

The remaining headroom looks like it is in feature representation, not tuning. The dataset has only eight
columns, so the ~0.75 AUC neighbourhood is probably a soft ceiling for this information, but three directions
seem promising. First, an out-of-fold target-encoding scheme (K-fold inside `prepare`-compatible code paths,
full-map at predict time) instead of the single smoothed map I tried, which would let airport- and
carrier-route-level delay propensity be used without the leakage/drift that killed the simple version.
Second, a proper early-stopping split inside 2005 (temporally ordered, if the slice ordering can be
recovered) to choose `n_estimators` per ensemble member, plus a drift-aware reweighting of training rows
towards later 2005 months, which should help the cross-year transfer specifically. Third, a wider search over
the categorical representation — `max_cat_threshold` swept jointly with `colsample_bynode` and the leaf
budget, and a rare-level collapse (levels below ~50 training rows mapped to a shared `RARE` token) so that
Origin×hour-scale interactions can be tried without the overfit that killed them here. A cheap fourth lever
would be a larger ensemble (16+ members) with per-member feature subsets, which has been monotone so far but
ran into the 120 s per-experiment cap at 8 members and 75 s.
