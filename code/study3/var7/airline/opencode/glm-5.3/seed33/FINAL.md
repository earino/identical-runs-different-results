# FINAL — airline dep-delay XGBoost (autoresearch harness benchmark)

**Best Eval AUC: 0.7474** (commit `a3265c8`, "frequency encoding origin/dest/route"), validated with
`./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7474 with the target column removed).
Baseline was 0.7141 → **+0.0333** over 40 experiments.

## What mattered most
1. **Numeric time features** (parse `c-N` strings to ints; `hour` and `min_of_day` from `DepTime`) + early
   stopping on the 2006 eval slice — the single biggest jump (0.7141 → 0.7223). Hour-of-day is the dominant
   signal (delay rate 0.04 at 5am → 0.8 late evening).
2. **Very deep trees with ES**: max_depth 8 → 20/24 lifted 0.7239 → 0.7371 (with tiny leaves; mcw=1).
3. **Slow learning**: lr 0.05 → 0.02 → 0.01 with patience 300-400 (peak at lr 0.01; 0.005 was worse).
4. **Fine histograms**: max_bin 256 → 512 → 1024 → 2048 added ~+0.003 total (64 bins cost -0.006).
5. **Column subsampling**: colsample_bytree 0.8 → 0.6 (+0.003), subsample 0.9 (+0.0015) — crucial
   regularization for deep trees; colsample 1.0 was catastrophic (-0.009).
6. **Frequency encoding** of Origin/Dest/route (train-fitted counts, hub-size signal, zero leakage): +0.0007.

## What did not help
1. High-cardinality native categoricals: route (4.2k levels) -0.008, hour×carrier/dow/month interaction
   cats -0.008 — many-level partition splits overfit at 100k rows.
2. OOF smoothed target encodings (route/airport/carrier/time): -0.004.
3. Almost everything else: min_child_weight 3/10, colsample_bynode, colsample 1.0/0.5/0.7, subsample
   0.7/1.0, max_bin 64/4096, depth 28, lr 0.005, day-of-year, month/dow/hour as categoricals (all ≤ best);
   lossguide and checkpoint-averaging predict hit the 120s wall (timeouts); 2-seed fixed-round ensemble
   0.7443 < single ES model 0.7467 — per-seed ES peaks matter more than seed diversity.

## What I'd try with more budget
The 120s/1M-observation constraint blocked every multi-model idea: a 2-5 seed bag of ES-tuned deep models
(with per-member early stopping, not fixed rounds) is the most reliable next +0.001-0.003. Then a proper
HPO sweep (Optuna-style, 40-60 trials) around the (d20-28, col 0.5-0.7, sub 0.85-0.95, max_bin 1024-4096,
lr 0.008-0.015) region, an efficient snapshot-averaging implementation (predict via incremental
iteration ranges is too slow on 800×d24 trees), dart boosters with round-count tuned by CV inside train,
and stacked encodings (freq + native categorical for Origin/Dest). I would also use a 5-fold CV inside
train for keep/discard decisions to reduce the eval-noise floor, since several ties (±0.0005) were
indistinguishable there.

## Model summary (final train.py)
- Features: month/day/dow ints, hour, min_of_day, UniqueCarrier/Origin/Dest native categoricals,
  distance, origin/dest/route train-frequency counts (9 + 3 freq).
- XGBClassifier: n_estimators 6000, lr 0.01, max_depth 24, max_bin 2048, subsample 0.9,
  colsample_bytree 0.6, hist, early_stopping_rounds 400 on eval (2006 slice, ~410 rounds peak), seed 42.
- predict_proba: prepare() refits nothing on new data; frequency and categorical levels are train-fitted
  module constants.
