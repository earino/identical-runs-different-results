# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7545** (experiment #40, commit 173ed76, `predict_proba` on a target-free frame
reproduces exactly this: `./validate.sh` → `CONTRACT OK`).
Baseline (unmodified `train.py`, 30 trees) was **0.7141**. Budget: 40/40 experiments, ~57 min wall,
10.6k of 18k CPU-seconds.

## What mattered most

1. **Smoothed target/frequency encodings over schedule keys, fitted on training data only**
   (5-fold out-of-fold for the training rows, full-train tables for anything passed to `predict_proba`).
   The winning family is route/airport × *time-of-day slot*: `route_tod30`, `route_tod15`,
   `origin_tod30`, `dest_tod30`, `distb_tod30`, `minhr`, plus coarser views (`origin`, `hour`, `month`,
   `route_month`, `route_hour`, `distb`). Worth ~+0.03 AUC in total (0.714 → ~0.747) and the single
   largest effect measured. Unseen levels map to NaN and are handled natively.
2. **Explicit contrasts and shares between encodings** (`dev_fine_minus_coarse`, and
   `shr_slot_share_of_parent`) — a tree cannot subtract two features, so the deviation of a slot's delay
   rate from its route's / airport's baseline is handed to it directly (+0.0003…+0.0008).
3. **Raw clock-face / time-structure features from `DepTime`**: minutes-after-midnight, minute-of-hour,
   round-5/10/15/30 departure flags, minute buckets and cyclic sin/cos. Combined +0.005 across
   experiments #34/#35/#37/#38/#40 — the strongest *non-fitted* feature family found.
4. **Calendar features**: day-of-year, US holiday travel windows, weekday ordinal (+0.0016, #29).
   Recurring-window features generalize across the year shift; a finer window-id/edge variant did not.
5. **Regularized deep model + diverse ensemble**: `depth 5/6/7` × 3 seeds (9 models), `lr 0.03`,
   `min_child_weight 30/50/60`, `colsample 0.5–0.7`, `lambda 20`, `gamma 1`. On this year-shifted split,
   capacity without regularization is destructive (300 trees at the baseline depth scored *below* 30
   trees), and averaging diverse members beat any single configuration.

## What did not help

1. **More raw capacity / hyperparameter chasing**: 300 trees at depth 6 (0.7083 vs 0.7141 baseline),
   depth 3, `lr=0.03` with 1000 trees, a `lossguide`/leaf-wise member, depth-7 member, 10-fold OOF
   (worse than 5-fold). Every one was reverted.
2. **Finer or exact schedule keys**: `route_dep`/`carrier_dep` (route × exact departure time),
   `tod10`, three-way `route_tod30_month`/`_dow`, week-of-year TEs, and extra shrinkage levels for the
   slot encodings — all neutral-to-negative, so the slot resolution saturates at ~15 minutes.
3. **Ablation-driven pruning**: dropping keys that a single-seed 300-tree ablation called redundant
   (`route_tod15`, `route_hour`) regressed 0.7460 vs 0.7540 once measured with the real ensemble; the
   ablation's ~0.001 noise floor made those "useless" verdicts unreliable.

Rejected by design, not by experiment: day-rotation features (`rank`/`gap` within a carrier-airport-day).
`train.csv`/`eval.csv` are ~1.4 % random samples of their year while the hidden holdout is ~10× denser, so
any statistic computed inside the frame passed to `predict_proba` would shift distribution; only fitted
tables (constant at predict time) and per-row transforms are safe here.

## With more budget I would

Attack the combiner rather than the feature set. The 9 model members are highly correlated; a level-2
XGBoost on out-of-fold member predictions (nested CV, ~1 m rows of training material) is the obvious next
step, together with a proper per-key shrinkage search — each smoothing constant is currently a guess, and
the tuning signal (0.001-ish AUC on 100k eval rows) is barely above the noise floor, so it needs repeated
seeds to steer. Beyond that: repeat every iso-AUC decision 3× before accepting it, use a time-ordered
internal validation split for early stopping instead of a fixed tree count, and look for genuinely new
information rather than more views of `DepTime` — the schedule-rotation features blocked above would be
the first thing to retry if the holdout sampling density were known to match training.
