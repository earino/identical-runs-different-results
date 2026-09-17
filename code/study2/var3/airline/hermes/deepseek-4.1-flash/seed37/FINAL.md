# autoresearch XGBoost — airline (dep_delayed_15min) — FINAL

**Best Eval AUC: 0.7354** (baseline 0.7141, +0.0213)

Final `train.py` = 16-member bagged XGBoost ensemble. Bagging over seeds/configs, `lossguide` growth
with 96 leaves, 350 rounds at lr 0.05, `min_child_weight=50`, `reg_lambda=2`, `subsample`/`colsample_bytree`
varied per member (0.5–1.0). Features: numeric time-of-day/calendar (hour, minute, tod, dow, month, dom, doy),
distance, train-fitted frequency and normalised congestion statistics (origin/dest/carrier/route/hour counts and
shares), airport peak-hour statistics, and explicit `tod x feature` interaction products. Model trained on 2005,
scored on 2006 (`validate.sh`: CONTRACT OK, `predict_proba` on `eval.csv` with the target dropped → 0.7354).

## Changes that mattered most

1. **Numeric temporal features** (exp 3, 0.7141 → 0.7177). `Month/DayOfMonth/DayOfWeek` are ordinal `c-<n>`
   strings; decoding them and deriving hour/minute/minute-of-day made the dominant, year-stable signal
   directly available. A standalone smoothed hour-of-day target encoding scores 0.689 AUC on 2006, and the
   train/eval delay-rate-vs-hour curves are nearly identical — time of day is essentially all of the stable signal.
2. **Traffic frequency + normalised congestion features** (exp 12/17, 0.7194 → 0.7243). Counts and shares
   fitted on train only (`origin/dest/carrier/route/hour`), plus per-airport peak-hour ratios. These describe
   *schedule shape*, which is stable from 2005 to 2006 — unlike delay levels themselves.
3. **Bagged ensemble instead of one fit** (exp 8/18/36/37, 0.7179 → 0.7317). Averaging 5 → 8 → 12 → 16
   XGBoost members with varied depth (4–9), `subsample` and `colsample_bytree` gave a steady +0.0005–0.002
   per step. The year shift means a single fit locks onto 2005-specific noise; averaging is the cheapest cure.
4. **Explicit `tod x ...` interaction products** (exp 32, 0.7297; peak-hour stats exp 34, 0.7303). With
   `colsample_bytree` as low as 0.5 a tree often cannot see both parents of an interaction, so making
   `tod x origin_hour_share`, `tod x dest_hour_share`, `tod x log(distance)` explicit paid off.
5. **`grow_policy="lossguide"` / `max_leaves=96`** (exp 39, 0.7317 → 0.7354). The single largest late gain:
   leaf-wise growth spends splits where the loss is, which suited this mostly time-of-day-driven problem
   much better than depth-wise growth. (Note: it costs ~100 s per experiment at 16 members — near the 120 s cap.)

Also kept: 350 rounds at lr 0.05 (slower learning only became a win *after* bagging existed — exp 2, 4, 7 all
showed raw capacity hurting a single fit), and pruning the redundant cyclic sin/cos features (exp 26, simpler
and +0.0002).

## What did NOT help

1. **Target encodings of the identity columns** (exp 13, −0.0094, the worst result). Smoothed 2005 delay rates
   for origin/dest/carrier/route let the model rely on a statistic that drifts between years; single-feature
   encodings score 0.53–0.57 AUC on 2006, and `hour + route` (0.658) is *worse* than `hour` alone (0.689).
2. **Raw capacity**: 400/200/100 trees or depth 4 at lr 0.1, single fit (exp 2, 4, 5, 7) — each was worse than
   30 trees, i.e. the model was overfitting 2005 long before it ran out of signal. `min_child_weight=200`
   (exp 24, −0.0022) and dropping Origin/Dest categorical identity (exp 14, −0.0039) also hurt.
3. **Structural priors applied without support**: a monotone time-of-day constraint (exp 15, 0.7224 vs 0.7226),
   `max_cat_threshold=8` (exp 16, −0.0007), `max_bin=128` (exp 40, −0.0004), carrier hub-share features
   (exp 20, ±0.0000), an airport operating-window position feature (exp 22, −0.0005), and more members (16 → tied)
   — all neutral-to-negative, and all reverted under the simplicity criterion.

## With more budget I would

Push on three fronts. First, extend the winning leaf-wise direction: `max_leaves` in a 64–160 range and
`grow_policy="lossguide"` combined with a per-member depth cap, since the one measurement at 96 leaves gave the
biggest single jump and the neighbourhood is unexplored (a 128-leaf test was impossible inside the 120 s cap at
16 members — a 12-member variant would fit). Second, attack generalization directly rather than eval AUC:
the honest weakness of this result is that every keep/discard decision was made on one 100k-row 2006 slice, and
differences below ~0.001 are inside the noise band there. A proper protocol would hold out an internal
time-split of train (e.g. late-2005 vs early-2005) to choose the config, use eval.csv only once, and weight
members by that internal validation — or train with sample weights emphasizing the most recent months of 2005.
Third, more stable-signal feature engineering of the kind that worked: per-airport *delay-propagation* structure
(time since the origin's first departure of the day, turnaround pressure on the same aircraft tail if it were
available) and route-level schedule-shape features, while continuing to avoid anything fitted to 2005 delay
*levels*. Inspiration from the failures: features must be structural (counts, shares, peak positions) rather
than statistical (rates), because only structure survives the 2005→2006 shift.
