# FINAL — Airline departure-delay classifier (XGBoost, maximize AUC)

## Best Eval AUC: **0.7577**

Validation (`./validate.sh`): `CONTRACT OK`, eval AUC via `predict_proba` = 0.7577, `train.py` runs in ~97 s.
Final model = `HEAD` (commit `55a2224`): a 3-model `XGBClassifier` ensemble at `max_depth` 14/16/18,
`n_estimators=350`, `learning_rate=0.023`, `subsample=0.9`, `colsample_bytree=0.5`,
`min_child_weight=0`, `reg_lambda=0.0`, `reg_alpha=0.5`, `tree_method="hist"`, `enable_categorical=True`,
with per-model seeds. `predict_proba` averages the three members' probabilities. Progression: 0.7141 → 0.7577.

## Changes that mattered most (5)
1. **Time handling**: ordinal-encode `Month`/`DayofMonth` (keep `DayOfWeek` categorical), treat scheduled
   departure hour as a 24-level categorical, and add a continuous `DepMinOfDay = hour*60 + minute`.
   This alone moved the baseline from 0.7141 to ~0.7445, and `DepMinOfDay` added ~0.001 later.
2. **Deep trees**: `max_depth` 14–18 decisively beat shallow trees; a 3-model depth ensemble (14/16/18)
   reduced variance and added ~0.001.
3. **Airport traffic-volume features** computed from `train` only: `Origin|hour`, `Dest|hour`, and
   `Origin|hour±1` counts. The neighbor-hour origin traffic gave a clear gain.
4. **Volume/frequency counts** (`f_org`, `f_dst`, `f_car`, `f_route`) mined from train value counts —
   late gain of ~0.0015.
5. **Relaxing regularization on deep trees**: driving `min_child_weight` 5→0 and `reg_lambda` 5→0, and
   `reg_alpha` to 0.5, while using high stochasticity (`subsample=0.9`) and low `colsample_bytree=0.5`.
   Each step gave small, consistent gains on eval.

## Things that did NOT help (3)
1. **Target/mean encoding** of high-cardinality categoricals and route-level features (hurt).
2. **DART / `grow_policy=lossguide`** and route×time interaction features (all neutral-to-worse).
3. **Destination neighbor-hour traffic, hub-ness features, and hour `%24` wrapping** (neutral or slightly worse),
   plus `max_bin=512` (tiny +0.0002 but pushed runtime past the 120 s limit — reverted).

## What I would try with more budget
The dominant remaining uncertainty is runtime: the best configs sit at 95–115 s against a 120 s per-experiment
limit, and I deliberately chose the 0.7577 model at ~97 s over an equally-scoring one at ~115 s because a
timeout on the hidden scorer is catastrophic. With more budget I would (a) build a proper out-of-fold / seed
bagged ensemble to stabilize the ~±0.0015 eval noise instead of chasing sub-0.001 single-config deltas,
(b) control cost so a larger ensemble fits comfortably, e.g. capped `max_bin` plus reduced `n_estimators`
with a slightly higher learning rate, and (c) test a small `rank:pairwise` objective and careful
cross-view-robust encodings, evaluated by repeated CV on eval rather than a single split, since the two
data views appear to be two feature observations of the same underlying labels.
