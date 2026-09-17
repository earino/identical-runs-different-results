# Final report — XGBoost airline delay (autoresearch benchmark)

**Best Eval AUC: 0.7428** (experiment #16, commit c4adde0; validated via `./validate.sh` → CONTRACT OK).
Baseline was 0.7141, so the campaign gained **+0.0287 AUC** over 16 experiments.

## What mattered most (in order of impact)

1. **Minute-of-hour feature** (`DepTime % 100`, numeric, both model views): +0.0030 in one step
   (0.7398 → 0.7428). Scheduled-departure minutes cluster into flight banks with distinct delay
   behavior; the raw hhmm encoding hides this from shallow trees.
2. **Two-view logit-averaged ensemble**: a native-categorical view (depth 4, enable_categorical)
   averaged with a deep one-hot view (depth 20, dummy-encoded carrier/origin/dest + hour dummies):
   0.7204 → 0.7327. The views' errors decorrelate across the 2005→2006 time shift.
3. **Bagging the deep one-hot members** (subsample 0.85, colsample 0.5, 5-12 seeds, n200 lr 0.075,
   shared DMatrix via `xgb.train`): subsample+colsample acts as regularization against 2005-specific
   noise; each member costs ~7s wall, several members beat any single unbagged model (best single
   model 0.7353 vs 0.7269 unbagged). 0.7327 → 0.7391.
4. **Recency-weighted members** (sample_weight = 1 + alpha·(month recency), alpha 0.5 and 1.0):
   2005→2006 drift means later training months predict 2006 better; recency members are the
   strongest single members and stack in the bag (+0.0005 each).
5. **Blend-weight pruning**: reducing the native view from 5 equal-weight members to 2 lifted
   0.7395 → 0.7398 — the weak-but-different native view helps only at ~15% of ensemble weight.

## What did not help

1. **Route (Origin+Dest) as a categorical or target-encoded feature** — poison (−0.012 AUC):
   high-cardinality route splits memorize 2005 route mix, which shifts by 2006.
2. **Capacity + early stopping on an internal 2005 split** (1500 trees, depth 8, ES on last
   20% of 2005): 0.7054 — internal validation measures 2005 fit, not 2006 generalization, so ES
   picked far too many rounds (overfit to the shift).
3. **Alternative model families / objectives**: sklearn HistGradientBoosting members (0.7239
   single, 26s/fit, slow and weak), `rank:pairwise` XGBoost members (0.5901 single; dragged the
   blend down), LogisticRegression on one-hot (0.7084). Also max_bin 64/128 (AUC loss, no speedup).
4. **Target encoding of carrier/origin/dest** in any honest form (OOF/holdout-calibrated): 0.7322
   vs 0.7334 without; only the leaky full-fit version looked good (in-sample illusion).
5. **Interaction dummies** (hour×day-of-week, 5-minute buckets, native-depth 6): all flat or
   slightly negative; the deep members already learn these splits.

## What I would try with more budget

The two structural wins (deep-bagged one-hot ensemble + recency weighting + minute-of-hour)
suggest the remaining headroom is in *schedule semantics*: features like hour×minute-as-count
(scheduled departure slot index), carrier-specific bank patterns, and origin-departure-bank
congestion proxies computed on 2005 with proper leave-one-out hygiene. I would also (a) raise the
member count to 20+ under a higher wall-clock cap (the 120s/4-thread experiment limit bound the
bag at ~13 members — with faster members, e.g. fewer columns or shallower+more trees, the bag
curve was still rising), (b) optimize blend weights by stacked OOF instead of grid guesswork
(also fixing the native-view weight more principledly), and (c) probe whether the minute-of-hour
effect is stable across carriers/airports with interaction features selected by OOF gain rather
than eval AUC, to keep eval-overfitting risk low.
