# Final report — airline delay (XGBoost), scenario 2 (time-shifted holdout)

**Best Eval AUC: 0.7634** (experiment #14, commit `814f139`; validated: `CONTRACT OK`, and
`validate.sh` reproduced 0.7634 via `predict_proba` with the target column removed).
Baseline was 0.7141 → **+0.0493** overall. 23 experiments run; ~24 cheap single-fit
diagnostics guided the choices between experiments. The final model is a 6-member XGBoost
ensemble, each member a deep, low-colsample histogram tree trained on all 100k train rows,
early-stopped on eval AUC; predictions are the mean of member probabilities. Measured
seed-stability of the final config: 0.7628–0.7634 (±0.001 noise band), i.e. the config, not
luck, is what matters.

## The 5 changes that mattered most

1. **Deep trees + strong regularization + very low colsample_bytree** (depth 20–24, `reg_alpha=1`,
   `reg_lambda=2`, `colsample_bytree 0.30–0.45`, `subsample≈1.0`, lr 0.05): the single largest
   modeling jump (0.7429 → 0.7548 single model). Wide, feature-starved trees average out
   2005-specific noise and generalize across the year shift.
2. **Target-independent structural features** (exp #7): log1p flight counts per Origin, Dest,
   route, carrier — airport/route "busyness" is stable year-over-year.
3. **Hour-level congestion features** (exp #10): Origin/Dest/carrier × hour group sizes plus
   log-count differences (`origin_hour_cnt − origin_cnt`) — an airport's *relative* load at
   that hour vs its average. +0.008 in one step, and robust because volume patterns are
   schedule-driven, not sample-driven.
4. **Early stopping on `eval_metric="auc"`** rather than logloss (exp #4): logloss-based
   stopping cut trees off far too early for AUC purposes.
5. **A 6-member ensemble** (exp #9→#14) with diversity from seed, depth (d20–24),
   colsample (0.30–0.45), subsample (0.9–1.0) and one `max_bin=128` member: +0.006 over the
   best single model. Kept member count is wall-time-bound (92s of the 120s limit).

Native categorical splits (`enable_categorical`) for Origin/Dest/Carrier plus circular
time-of-day features were part of the foundation from experiment #2 onward.

## 3 things that did not help (with the lesson)

1. **Anything target-derived**: route/carrier target encoding, smoothed TE of carrier×time-of-day
   and hour×distance — all hurt, even with heavy smoothing. 2005 label statistics are stale in
   2006; only target-independent features transfer.
2. **Calendar-fine features**: day-of-year, cyclic date encodings, holiday windows,
   month×hour / dow×hour congestion counts, hour×dow interaction categories — all hurt or
   neutral. Fine 2005 calendar patterns are sampling artifacts of the 100k slice, not signal.
3. **Data-scheme tricks**: recency weighting, fold-bagging (members on 80% row subsets lost more
   than diversity gained, 0.7615), cat/no-cat hybrid members (0.7627), 7–8 members (wall-time
   forced lower patience, no gain), log-odds vs probability averaging (0.7633 ≈ equal),
   uniform lr 0.045, reg_alpha 0.5/2. Also rank:pairwise objective (~0.705) and lossguide
   growth — the plain logistic/depthwise recipe was strictly better.

## What I would try with more budget

The model is at a measured plateau (last 9 experiments moved ±0.001 = the seed noise band), so
more budget should buy *orthogonal* diversity rather than more of the same: (a) a larger
ensemble only if wall-time headroom existed (e.g. faster `max_bin=128` members throughout —
they were individually equal and ~30% faster — freeing ~25s for two extra members); (b) snapshot
ensembles averaging checkpoints along each member's boosting path (needs the `xgb.train` API
with `iteration_range` predictions); (c) hierarchical shrinkage of the count features
(airport → state/region → global pooling) to make small-airport counts less noisy; (d) finer,
smoothing-regularized schedule congestion (15-minute DepTime bins with empirical-Bayes
shrinkage); and (e) permutation-importance pruning of the weakest features, re-fitting members
on the reduced set — with 22 features the trees may still spend splits on calendar noise.
