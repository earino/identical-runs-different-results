# FINAL — airline delay (XGBoost, scenario 2: train 2005 / eval & holdout 2006)

**Best Eval AUC: 0.7336** (experiment #36, commit `c6480c6`), vs **0.7141** for the unmodified baseline.
Final `train.py` is a single `XGBClassifier` with `max_depth=0` (unlimited), `learning_rate=0.0125`,
`n_estimators=800`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_lambda=2.0`,
`tree_method="hist"`, `enable_categorical=True`, on the raw columns minus `Month` and `DayofMonth`.
`./validate.sh` → CONTRACT OK; `predict_proba(df)` (target column removed) reproduces 0.7336.
40/40 experiments used, 34 of them `ok`, 3 crashes (all trivial bugs, fixed in place), no timeouts/OOM.

## Changes that mattered most

1. **Dropping the calendar-identity features (`Month`, `DayofMonth`; only `DayOfWeek` survives) — the single
   biggest win, ~+0.004.** `Month`, `DayofMonth` and `DayOfWeek` are `c-<n>` strings, and *jointly* they
   fingerprint the year: the 2005 weekday/day-of-month alignment differs from 2006, so a domain classifier on
   the features separates the two years with **AUC 1.0**. That gave the trees a channel to memorise
   date-specific 2005 effects (e.g. weather days) that cannot transfer. Removing it lifted eval AUC
   0.7171 → 0.7197 (and 0.7163 → 0.7166 even at depth 4). `DayOfWeek` alone is still informative (removing it
   cost 0.0026), so weekday effects are real; month/day-of-month effects were mostly year-specific noise.
2. **Spending the freed capacity on depth: `max_depth` 4 → 12 → 16 → unlimited (`0`), +0.0139 total.**
   With the memorisation channel closed, deeper trees became strictly better: 0.7197 (d4) → 0.7292 (d12) →
   0.7313 (d16) → **0.7331** (unlimited), even though train AUC climbs to 0.966. The signal lives in
   high-order interactions (airport × carrier × time-of-day × distance × weekday); with the raw 6-feature
   table this is where the gains were, and depth is what unlocked them.
3. **Slow learning: `learning_rate` 0.05 → 0.0125 with more trees (+0.0005 at the optimum).** 800 trees at
   lr 0.0125 beat 400 at 0.025 (0.7336 vs 0.7331); 1200 trees at the same lr overshot and fell back to 0.7310.
4. **The intermediate regularisation phase (shallow + heavy regularisation) was worth +0.002 on its own**
   and set up everything after it: depth 4, 150 trees, lr 0.05, `min_child_weight=10`, `subsample`/`colsample`
   0.8, `reg_lambda=2` gave 0.7163 while depth 6–7 with 400–500 trees gave 0.7115–0.7125. Train AUC was the
   diagnostic that pointed at overfitting long before the two fixes above were found.
5. **Keeping the model set simple.** Multi-member ensembles were consistently no better than the best single
   model at these settings (#37: 0.7335, #40: 0.7336 with two members vs 0.7336 single), so the final artifact
   is one booster — fewer moving parts for the same score.

## Things that did not help

1. **Target-rate encodings** (smoothed mean `dep_delayed_15min` per carrier/origin/dest/route, fitted on
   train): eval AUC collapsed to 0.6987 with train AUC 0.8397 — in-sample leakage plus weak across-year
   stability of airport rates (origin delay rates correlate only 0.384 between 2005 and 2006 vs 0.99 for
   hour-of-day).
2. **Frequency/volume counters** (`log1p` counts of each carrier, origin, dest and route) and the
   shifted-circadian clock feature (`minutes since 03:00`, so post-midnight flights sort next to 23:59):
   each cost ~0.001–0.002 AUC. The delay-vs-hour curve is nearly monotone and XGBoost already extracts it
   from raw `DepTime`.
3. **Fine-grained category identity — `Route` = `Origin_Dest` (4198 levels) as a categorical**: 0.7060, with
   train AUC jumping to 0.80; it memorises route-level 2005 quirks. The same overfitting pattern hit the
   aggressive hyperparameter directions: heavier regularisation at the optimum (`min_child_weight=30`,
   `reg_lambda=5`) gave 0.7295, and `max_cat_threshold=16` / `max_cat_to_onehot=8` gave 0.7122 (the XGBoost
   defaults, 64/4, are better here).

## What I would try with more budget

The two-year gap is the whole problem, and every remaining lever should be judged by whether it transfers.
First, I would attack the year-shift directly rather than by feature pruning: build a 2006-aware validation
split inside `train.py` (e.g. hold out 2005 months whose weekday alignment matches 2006) and use it for early
stopping, so the tree count stops being tuned on `eval.csv`; then test out-of-fold target encodings of
*interaction* keys (origin × hour bucket, carrier × weekday) — computed fold-wise they cannot leak, and the
earlier failure was a leakage artefact, not evidence that group rates are useless. Second, I would push on
capacity with the calendar channel closed: `colsample_bytree` and `subsample` were never re-tuned after the
feature fix, and 800 trees at lr 0.0125 is a 2-point Newton step on the (trees, lr) surface — a short
coordinate sweep there, plus `max_delta_step` and `grow_policy="lossguide"` with a leaf cap, is the cheapest
remaining capacity. Third, with the extra time I would try knowledge transfer across the seasonal boundary:
train a booster on 2005, use it to pseudo-label the *unlabelled* portion of the holdout year at a high
threshold, and refit on 2005 + confident 2006 rows — the standard self-training trick for covariate shift, and
legitimate here because the hidden holdout is the same year as `eval.csv`. Finally, given that a domain
classifier separates the years perfectly, I would try importance weighting by a softened version of that
classifier's output, and I would re-check whether the strongest surviving `DayOfWeek` effect is really weekday
or just its residual correlation with the dropped calendar features.
