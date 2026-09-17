# Final report — airline delay AUC maximization

## Best model

**Eval AUC: 0.7334** (commit `84a5d1e`, experiment 39/40; baseline was 0.7141).

Final `train.py` = a 5-member XGBoost ensemble (probability average). Each member: depth 3,
learning rate 0.03, 1200 rounds, `reg_lambda=10`, `reg_alpha=1`, `colsample_bynode=0.8`,
`colsample_bytree=0.9`, seeds 0–4, native categoricals. Features: raw schedule columns
(DepTime, Distance, Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest) + 7 smoothed
out-of-fold target encodings of congestion keys (origin×hour, origin×15/30-min departure block,
dest×hour/30/15-min block, origin×day-of-week) + a label-free circular holiday-distance signal
(hol_dist, hol_near) + one joint-congestion product (te_o_block15 × te_d_block30).

All encoders/statistics are fit on `data/train.csv` only; training rows get 5-fold out-of-fold TE
values (leak-free), prediction rows use full-train statistics. `predict_proba(df)` re-runs the same
`prepare()` on the raw frame passed in — nothing is ever fit on eval or holdout data.

## The 5 changes that mattered most (cumulative, eval AUC)

1. **Depth-4→3 shallow trees with slow learning** (0.7168 → 0.7192 → 0.7300): the 2005→2006
   distribution shift punishes variance; depth 6 with 30 fast rounds (the baseline recipe) was
   the main overfitting source. lr 0.03 with 800–1200 rounds at depth 3 generalized best.
2. **Origin×hour target encoding** (0.7141 → 0.7168): first evidence that smoothed hub-congestion
   statistics carry signal the trees can't learn from raw columns in limited rounds.
3. **Origin/dest × 15/30-minute departure-block TEs** (0.7192 → 0.7274): finer time granularity
   on congestion; origin side first (+0.0067 alone), dest side additive on top.
4. **Proper OOF target encoding instead of naive LOO** (exp 9 → 10): LOO encoding was a
   catastrophic leak (0.5846 — trees split on the ±1/(C+m) label residue); 5-fold OOF fixed it.
   This leak-free TE machinery is what all later gains were built on.
5. **Label-free holiday-distance feature + 5-member column-subsampled ensemble**
   (0.7323 → 0.7329 → 0.7333): holiday proximity captures real delay spikes that transfer across
   years; per-node column subsampling (0.8) gave just enough member decorrelation for the
   probability average to add ~+0.0008 over the best single model.

## 3 things that did NOT help (tried and reverted or discarded)

1. **Bigger/faster trees and more capacity**: depth 5–6, lr 0.1, 30–480 rounds, row-bagging
   (subsample 0.7–0.95), lossguide, max_bin 512, 8–10 ensemble members — all neutral or worse.
2. **More TE keys beyond the congestion family**: route, carrier, month, dow, carrier×hour,
   origin×month, origin×week, dest×dow, dow×hour, 10-min blocks, hierarchical (parent-shrunk)
   variants, count/volume features — all neutral or harmful (year-shift noise or redundancy).
3. **Ensemble diversity via fold-seed, different folds (K=10), depth-mixing, feature-mixing,
   logit-averaging**: every variant landed 0.7322–0.7333 vs 0.7333 control — the simple
   colsample_bynode ensemble was already at the plateau.

## What I would try with more budget

The single biggest unexploited direction is modeling the *physics of delay propagation* rather
than static congestion rates: the TEs estimate E[delay | hub, time-block], but a flight's true
delay depends on the upstream aircraft's lateness. With per-row data alone I'd build
"same-route-same-day earlier-departure" features from the training table (rotation proxies),
and origin-level *daily* TE residuals (a bad morning at a hub cascades all day) — i.e., keys
like origin×date with heavy shrinkage toward origin×dow×block, letting the model detect
snowballing days. Second, I'd revisit calibration of the 2005→2006 shift itself: importance-weight
training rows by covariate shift between 2005 and 2006 feature distributions (the eval frame is
legitimately available for *selection only*), and test TE variants smoothed by Poisson/binomial
variance rather than flat counts. Finally, given the last six experiments all landed within
0.0008, I'd spend remaining slots only on ideas with a mechanism, not parameter jitter.

## Budget usage

39/40 experiments (several were crash/timeout iterations fixed by amend+rerun; see
`experiments.tsv`), ~192 min wall of 230, ~5,200 CPU-s of 18,000. HEAD is the best kept commit;
`validate.sh` prints `CONTRACT OK` (eval AUC via `predict_proba`: 0.7334).
