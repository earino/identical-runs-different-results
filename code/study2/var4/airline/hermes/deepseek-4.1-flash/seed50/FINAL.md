# FINAL — airline (dep_delayed_15min), XGBoost

**Best Eval AUC: 0.7476** (experiment #40, commit `51d417f`; reproduces at 0.7476 through
`predict_proba` with the target column removed — `validate.log` prints `CONTRACT OK`).

Baseline was 0.7141 (#1). 40/40 experiments used, ~8100 of 18000 CPU-seconds.

## The 5 changes that mattered most

1. **Deep trees, very few boosting rounds.** The single biggest structural finding. The eval year is one
   ahead of train (2005 -> 2006), so capacity has to shrink: an early-stopping probe on a held-out slice of
   *train* picks round 532, and the model scores best at ~0.32 of that (170 rounds). Sweeping the fraction:
   0.50 -> 0.7340, 0.35 -> 0.7413, 0.25 -> 0.7413, 0.125 -> 0.7437 (underfit), 0.32 -> 0.7476. Depth went the
   other way: 6 -> 0.7234, 10 -> 0.7293, 14 -> 0.7325, 18 -> 0.7339, unlimited -> 0.7344 (each at 0.25).
   Deep, few, heavily-subsampled trees; `learning_rate=0.03`, `min_child_weight=10`, `reg_lambda=2`.
2. **`colsample_bytree=0.5`** (+0.0059 over 0.8, the best single hyperparameter change). 1.0 -> 0.7266,
   0.3 -> 0.7363, 0.5 -> 0.7413, 0.8 -> 0.7354. Sampling ~half the features per tree decorrelates the
   members of the ensemble, which is what the deep-tree regime needs.
3. **Conjunction categoricals `carrier_hour` and `origin_hour`** (+0.0057, #32). With `colsample_bytree=0.5`
   a tree sees only half the columns, so a (carrier, hour) or (airport, hour) conjunction is often
   unreachable; making it a single feature makes it always reachable. This is the only feature engineering
   that paid off. Note the asymmetry: with depot delays driven by conditions *at the origin*, `origin_hour`
   helps while `dest_hour` (-0.0032) and `route` (-0.0060) hurt.
4. **Refit on 100% of train at the probe's round count** (+0.0030, #8). Refitting without an eval set uses
   the 10% that early stopping had to hold out, and drops the early-stopping constraint on the final model.
5. **Averaging a 6-member bag** (+0.0012 for 3 members over 1; the 6 diverse members added a little more).
   Members vary seed, `colsample_bytree` in [0.40, 0.60] and `subsample` in [0.70, 0.90]. Averaging is the
   cheap, robust part: 10 members was within noise of 6 (+0.0001) at nearly double the runtime, so the
   smaller bag was kept.

## The 3 things that did not help

1. **Every other feature-engineering attempt**: train-set frequency counts for Origin/Dest/carrier/route
   (-0.0022), smoothed target encoding of carrier/Origin/Dest (-0.0008), a `route` categorical (-0.0060),
   `dest_hour` (-0.0032). High-cardinality per-route features do not transfer across the year boundary;
   `predict_proba`-safe encoding alone is not enough.
2. **Heavier regularization / lower learning rate**: `lr=0.015` with 8000 rounds (0.7125), depth 6 with
   `min_child_weight=30`, `colsample=0.7`, `reg_lambda=5` (0.7117), `min_child_weight=50` at unlimited depth
   (0.7271 — it re-imposes a shallow depth limit), `subsample=0.5` (0.7376), `colsample_bynode=0.5` (0.7293),
   `max_cat_threshold=16` (0.7470, equal but 10 s slower). Regularizing by *shrinking the round count* is
   what works here, not by shrinking the trees.
3. **Spending compute on more boosting per member.** Halving the rounds and doubling the bag to keep total
   tree count constant (12 members x 66 rounds) lost 0.0034, so the marginal boosting round is worth more
   than an extra averaged member at this operating point.

## What I would try with more budget

The model is a deep, heavily feature-subsampled, bagged tree ensemble; almost all remaining headroom is in
the *round count versus bag size* tradeoff, which my last two experiments only bracketed. I would map that
2-D surface properly (rounds 100-250 x bag size 6-12 at fixed total compute) and, separately, re-derive the
probe fraction from scratch at the final feature set instead of reusing fractions measured before the
conjunction features existed. Beyond that, the conjunction result suggests the untested direction is
*symmetric* aggregates rather than more cardinality: per-origin and per-carrier historical delay rates
computed as out-of-fold targets (so `predict_proba` can reproduce them), and a second-stage XGBoost trained
on the residuals of the first stage. I deliberately stopped short of per-route features, since every
high-cardinality route-level signal I tried lost AUC on the year shift — but a route *rate* with heavy
shrinkage towards the origin/carrier rates is the one variant I did not get to test and would try next.

## Contract notes

- Only `train.py` was edited; `run_experiment.sh`, `validate.sh`, `validate.py`, `task.json`, `data/` and
  `../bench.env` are untouched. No installs, no network, no web research, `n_jobs` left at `BENCH_THREADS`.
- All encoders (`CAT_LEVELS`) are fit on `data/train.csv` only, inside the module, before `prepare()` is
  ever called; `prepare(df)` derives everything else from the passed-in frame, so the hidden holdout gets
  the identical pipeline. `validate.py` confirms this: 0.7476 with the target column removed.
- Unseen category levels in the holdout map to NaN and are routed as missing by XGBoost.
